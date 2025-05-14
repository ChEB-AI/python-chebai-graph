from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from chebai.preprocessing.reader import DataReader
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from rdkit import Chem
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.collate import GraphCollator
from chebai_graph.preprocessing.fg_detection.fg_aware_rule_based import (
    detect_functional_group,
    get_structure,
    set_atom_map_num,
)
from chebai_graph.preprocessing.properties import MolecularProperty
from chebai_graph.preprocessing.properties.constants import *


class _AugmentorReader(DataReader, ABC):
    """
    Abstract base class for augmentor readers that extend ChemDataReader.
    Handles reading molecular data and augmenting molecules with functional group
    information.
    """

    COLLATOR = GraphCollator

    def __init__(self, *args, **kwargs):
        """
        Initializes the augmentor reader and sets up the failure counter and molecule cache.

        Args:
            *args: Additional arguments passed to the ChemDataReader.
            **kwargs: Additional keyword arguments passed to the ChemDataReader.
        """
        super().__init__(*args, **kwargs)
        self.f_cnt_for_smiles = (
            0  # Record number of failures when constructing molecule from smiles
        )
        self.f_cnt_for_aug_graph = (
            0  # Record number of failure during augmented graph construction
        )
        self.mol_object_buffer = {}
        self._num_of_nodes = 0
        self._num_of_edges = 0

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Returns the name of the augmentor.

        Returns:
            str: Name of the augmentor.
        """
        pass

    @abstractmethod
    def _create_augmented_graph(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict]:
        """
        Augments a molecule represented by a SMILES string.

        Args:
            mol (Chem.Mol): RDKIT molecule.

        Returns:
            Tuple[torch.Tensor, Dict]: Graph edge index and augmented molecule information
        """
        pass

    @abstractmethod
    def _read_data(self, raw_data: str) -> GeomData:
        """
        Reads raw data and returns a list of processed data.

        Args:
            raw_data (str): Raw data input.

        Returns:
            GeomData: `torch_geometric.data.Data` object.
        """
        pass

    def _smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """
        Converts a SMILES string to an RDKit molecule object. Sanitizes the molecule.

        Args:
            smiles (str): SMILES string representing the molecule.

        Returns:
            Chem.Mol: RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rank_zero_warn(f"RDKit failed to parse {smiles} (returned None)")
            self.f_cnt_for_smiles += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                rank_zero_warn(f"RDKit failed at sanitizing {smiles}, Error {e}")
                self.f_cnt_for_smiles += 1
        return mol

    def on_finish(self) -> None:
        """
        Finalizes the reading process and logs the number of failed SMILES and failed augmentation.
        """
        rank_zero_info(f"Failed to read {self.f_cnt_for_smiles} SMILES in total")
        rank_zero_info(
            f"Failed to construct augmented graph for {self.f_cnt_for_aug_graph} number of SMILES"
        )
        self.mol_object_buffer = {}

    def read_property(self, smiles: str, property: MolecularProperty) -> Optional[List]:
        """
        Reads a specific property from a molecule represented by a SMILES string.

        Args:
            smiles (str): SMILES string representing the molecule.
            property (MolecularProperty): Molecular property object for which the value needs to be extracted.

        Returns:
            Optional[List]: Property values if molecule parsing is successful, else None.
        """
        if smiles in self.mol_object_buffer:
            return property.get_property_value(self.mol_object_buffer[smiles])

        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None

        returned_result = self._create_augmented_graph(mol)
        if returned_result is None:
            return None

        _, augmented_mol = returned_result
        return property.get_property_value(augmented_mol)


class GraphFGAugmentorReader(_AugmentorReader):
    """
    A reader class that augments molecules with artificial functional group (FG) nodes and a graph-level node
    to support graph-based molecular learning tasks.

    The FG nodes to connected to its related atoms and graph node is connected to all FG nodes.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the name identifier of the augmentor.

        Returns:
            str: Name identifier.
        """
        return "graph_fg_augmentor"

    def _read_data(self, smiles: str) -> GeomData | None:
        """
        Reads and augments molecular data from a SMILES string.

        Args:
            smiles (str): SMILES representation of the molecule.

        Returns:
            GeomData: A PyTorch Geometric Data object with augmented nodes and edges.
        """
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None

        returned_result = self._create_augmented_graph(mol)
        if returned_result is None:
            rank_zero_info(f"Failed to construct augmented graph for smiles {smiles}")
            self.f_cnt_for_aug_graph += 1
            return None

        edge_index, augmented_molecule = returned_result
        self.mol_object_buffer[smiles] = augmented_molecule

        # Empty features initialized; node and edge features can be added later
        x = torch.zeros((augmented_molecule["nodes"]["num_nodes"], 0))
        edge_attr = torch.zeros((augmented_molecule["edges"]["num_edges"] * 2, 0))

        assert (
            edge_index.shape[0] == 2
        ), f"Expected edge_index to have shape [2, num_edges], but got shape {edge_index.shape}"

        assert (
            edge_index.shape[1] == edge_attr.shape[0]
        ), f"Mismatch between number of edges in edge_index ({edge_index.shape[1]}) and edge_attr ({edge_attr.shape[0]})"

        assert (
            len(set(edge_index[0].tolist())) == x.shape[0]
        ), f"Number of unique source nodes in edge_index ({len(set(edge_index[0].tolist()))}) does not match number of nodes in x ({x.shape[0]})"

        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _create_augmented_graph(
        self, mol: Chem.Mol
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Generates an augmented graph from a SMILES string.

        Args:
            mol (Chem.Mol): A molecule generated by RDKit.

        Returns:
            Tuple[dict, torch.Tensor]: Augmented molecule information and edge index.
        """
        returned_result = self._augment_graph_structure(mol)
        if returned_result is None:
            return None

        edge_index, node_info, edge_info = returned_result

        augmented_molecule = {"nodes": node_info, "edges": edge_info}

        return edge_index, augmented_molecule

    def _augment_graph_structure(
        self, mol: Chem.Mol
    ) -> Optional[Tuple[torch.Tensor, dict, dict]]:
        """
        Constructs the full augmented graph structure from a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[torch.Tensor, dict, dict]: Edge index, node metadata, and edge metadata.
        """
        self._num_of_nodes = mol.GetNumAtoms()
        self._num_of_edges = mol.GetNumBonds()

        self._annotate_atoms_and_bonds(mol)
        atom_edge_index = self._generate_atom_level_edge_index(mol)

        # Create FG-level structure and edges
        returned_result = self._construct_fg_to_atom_structure(mol)

        if returned_result is None:
            return None

        fg_atom_edge_index, fg_nodes, atom_fg_edges, structured_fg_map, bonds = (
            returned_result
        )

        fg_internal_edge_index, internal_fg_edges = self._construct_fg_level_structure(
            structured_fg_map, bonds
        )
        fg_graph_edge_index, graph_node, fg_to_graph_edges = (
            self._construct_fg_to_graph_node_structure(structured_fg_map)
        )

        # Merge all edge types
        full_edge_index = torch.cat(
            [
                atom_edge_index,
                torch.tensor(fg_atom_edge_index, dtype=torch.long),
                torch.tensor(fg_internal_edge_index, dtype=torch.long),
                torch.tensor(fg_graph_edge_index, dtype=torch.long),
            ],
            dim=1,
        )

        node_info = {
            "atom_nodes": mol,
            "fg_nodes": fg_nodes,
            "graph_node": graph_node,
            "num_nodes": self._num_of_nodes,
        }
        edge_info = {
            WITHIN_ATOMS_EDGE: mol,
            ATOM_FG_EDGE: atom_fg_edges,
            WITHIN_FG_EDGE: internal_fg_edges,
            FG_GRAPHNODE_EDGE: fg_to_graph_edges,
            "num_edges": self._num_of_edges,
        }

        return full_edge_index, node_info, edge_info

    @staticmethod
    def _annotate_atoms_and_bonds(mol: Chem.Mol) -> None:
        """
        Annotates each atom and bond with node and edge with certain properties.

        Args:
            mol (Chem.Mol): RDKit molecule.
        """
        for atom in mol.GetAtoms():
            atom.SetProp(NODE_LEVEL, ATOM_NODE_LEVEL)
        for bond in mol.GetBonds():
            bond.SetProp(EDGE_LEVEL, WITHIN_ATOMS_EDGE)

    @staticmethod
    def _generate_atom_level_edge_index(mol: Chem.Mol) -> torch.Tensor:
        """
        Generates bidirectional atom-level edge index tensor.

        Args:
            mol (Chem.Mol): RDKit molecule.

        Returns:
            torch.Tensor: Bidirectional edge index tensor.
        """
        # We need to ensure that directed edges which form a undirected edge are adjacent to each other
        edge_index_list = [[], []]
        for bond in mol.GetBonds():
            edge_index_list[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index_list[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        return torch.tensor(edge_index_list, dtype=torch.long)

    def _construct_fg_to_atom_structure(
        self, mol: Chem.Mol
    ) -> Optional[Tuple[List[List[int]], dict, dict, dict, list]]:
        """
        Constructs edges between functional group (FG) nodes and atom nodes.

        Args:
            mol (Chem.Mol): RDKit molecule.

        Returns:
            Tuple[List[List[int]], dict, dict, dict, list]:
                Edge index, FG node info, FG-atom edge attributes,
                structured FG mapping, and bond list.
        """

        # Rule-based algorithm to detect functional groups
        set_atom_map_num(mol)
        detect_functional_group(mol)
        structure, bonds = get_structure(mol)
        assert structure is not None, "Failed to detect functional groups."

        fg_atom_edge_index = [[], []]
        fg_nodes, atom_fg_edges = {}, {}
        structured_fg_map = (
            {}
        )  # Contains augmented fg-nodes and connected atoms indices

        for fg_group in structure.values():
            structured_fg_map[self._num_of_nodes] = {"atom": fg_group["atom"]}

            # Build edge index for fg to atom nodes connections
            for atom_idx in fg_group["atom"]:
                fg_atom_edge_index[0].extend([self._num_of_nodes, atom_idx])
                fg_atom_edge_index[1].extend([atom_idx, self._num_of_nodes])
                atom_fg_edges[f"{self._num_of_nodes}_{atom_idx}"] = {
                    EDGE_LEVEL: ATOM_FG_EDGE
                }
                self._num_of_edges += 1

            # Identify ring vs. functional group type
            ring_fg = {
                mol.GetAtomWithIdx(i).GetProp("RING")
                for i in fg_group["atom"]
                if mol.GetAtomWithIdx(i).GetProp("RING")
            }

            if len(ring_fg) > 1:
                raise ValueError(
                    "A functional group must not span multiple ring sizes."
                )

            if (
                len(ring_fg) == 1
            ):  # FG atoms have ring size, which indicates the FG is a Ring or Fused Rings
                ring_size = next(iter(ring_fg))
                fg_nodes[self._num_of_nodes] = {
                    NODE_LEVEL: FG_NODE_LEVEL,
                    # E.g.,  Fused Ring has size "5-6", indicating size of each connected ring in fused ring
                    "FG": f"RING_{ring_size}",
                    "RING": ring_size,
                }
            else:  # No connected has a ring size which indicates it is simple FG
                fg_set = {mol.GetAtomWithIdx(i).GetProp("FG") for i in fg_group["atom"]}

                if "" in fg_set and len(fg_set) == 1:
                    # There will be no FGs for wildcard SMILES Eg. CHEBI:33429
                    return None

                if "" in fg_set or len(fg_set) > 1:
                    raise ValueError("Invalid functional group assignment to atoms.")

                for atom_idx in fg_group["atom"]:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetProp("FG"):
                        fg_nodes[self._num_of_nodes] = {
                            NODE_LEVEL: FG_NODE_LEVEL,
                            "FG": atom.GetProp("FG"),
                            "RING": atom.GetProp("RING"),
                        }
                        break
                else:
                    raise AssertionError(
                        "Expected at least one atom with a functional group."
                    )

            self._num_of_nodes += 1

        return fg_atom_edge_index, fg_nodes, atom_fg_edges, structured_fg_map, bonds

    def _construct_fg_level_structure(
        self, structured_fg_map: dict, bonds: list
    ) -> Tuple[List[List[int]], dict]:
        """
        Constructs internal edges between functional group nodes based on bond connections.

        Args:
            structured_fg_map (dict): Mapping from FG ID to atom indices.
            bonds (list): List of bond tuples (source, target, ...).

        Returns:
            Tuple[List[List[int]], dict]: Edge index and edge attribute dictionary.
        """
        internal_fg_edges = {}
        internal_edge_index = [[], []]

        for bond in bonds:
            source_atom, target_atom = bond[:2]
            source_fg, target_fg = None, None

            for fg_id, data in structured_fg_map.items():
                if source_atom in data["atom"]:
                    source_fg = fg_id
                if target_atom in data["atom"]:
                    target_fg = fg_id

            assert (
                source_fg is not None and target_fg is not None
            ), "Each bond should have a fg node on both end"

            internal_edge_index[0].extend([source_fg, target_fg])
            internal_edge_index[1].extend([target_fg, source_fg])
            internal_fg_edges[f"{source_fg}_{target_fg}"] = {EDGE_LEVEL: WITHIN_FG_EDGE}
            self._num_of_edges += 1

        return internal_edge_index, internal_fg_edges

    def _construct_fg_to_graph_node_structure(
        self, structured_fg_map: dict
    ) -> Tuple[List[List[int]], dict, dict]:
        """
        Constructs edges between functional group nodes and a global graph-level node.

        Args:
            structured_fg_map (dict): Mapping from FG ID to atom indices.

        Returns:
            Tuple[List[List[int]], dict, dict]: Edge index, graph-level node, edge attributes.
        """
        graph_node = {NODE_LEVEL: GRAPH_NODE_LEVEL, "FG": "graph_fg", "RING": "0"}

        fg_graph_edges = {}
        graph_edge_index = [[], []]

        for fg_id in structured_fg_map:
            graph_edge_index[0].extend([self._num_of_nodes, fg_id])
            graph_edge_index[1].extend([fg_id, self._num_of_nodes])
            fg_graph_edges[f"{self._num_of_nodes}_{fg_id}"] = {
                EDGE_LEVEL: FG_GRAPHNODE_EDGE
            }
            self._num_of_edges += 1
        self._num_of_nodes += 1

        return graph_edge_index, graph_node, fg_graph_edges
