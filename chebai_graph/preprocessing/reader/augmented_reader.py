from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from chebai.preprocessing.reader import ChemDataReader
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from rdkit import Chem
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.collate import GraphCollator
from chebai_graph.preprocessing.fg_detection.rule_based import (
    detect_functional_group,
    get_structure,
    set_atom_map_num,
)
from chebai_graph.preprocessing.properties import MolecularProperty
from chebai_graph.preprocessing.properties.constants import *


class _AugmentorReader(ChemDataReader, ABC):
    """
    Abstract base class for augmentor readers that extend ChemDataReader.
    Handles reading molecular data and augmenting molecules with functional group
    information.

    Attributes:
        failed_counter (int): Counter for failed SMILES parsing attempts.
        mol_object_buffer (dict): Cache for storing augmented molecular objects.
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
        self.failed_counter = 0
        self.mol_object_buffer = {}
        self.num_nodes = 0
        self.num_of_edges = 0

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
    def _create_augmented_graph(self, smile: str) -> Tuple[Dict, torch.Tensor]:
        """
        Augments a molecule represented by a SMILES string.

        Args:
            smile (str): SMILES string representing the molecule.

        Returns:
            Tuple[Dict, torch.Tensor]: Augmented molecule information and corresponding edge index.
        """
        pass

    @abstractmethod
    def _read_data(self, raw_data: str) -> GeomData:
        """
        Reads raw data and returns a list of processed data.

        Args:
            raw_data (str): Raw data input.

        Returns:
            List[int]: Processed data as a list of integers.
        """
        pass

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Converts a SMILES string to an RDKit molecule object. Sanitizes the molecule.

        Args:
            smiles (str): SMILES string representing the molecule.

        Returns:
            Optional[Chem.Mol]: RDKit molecule object if conversion is successful, else None.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rank_zero_warn(f"RDKit failed to parse {smiles} (returned None)")
            self.failed_counter += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                rank_zero_warn(f"RDKit failed at sanitizing {smiles}, Error {e}")
                self.failed_counter += 1
        return mol

    def on_finish(self):
        """
        Finalizes the reading process and logs the number of failed SMILES.
        """
        rank_zero_info(f"Failed to read {self.failed_counter} SMILES in total")
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
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None

        if smiles in self.mol_object_buffer:
            return property.get_property_value(self.mol_object_buffer[smiles])

        augmented_mol, _ = self._create_augmented_graph(smiles)
        return property.get_property_value(mol)


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

        edge_index, augmented_molecule = self._create_augmented_graph(mol)
        self.mol_object_buffer[smiles] = augmented_molecule

        num_nodes = augmented_molecule["nodes"]["num_nodes"]
        num_edges = augmented_molecule["edges"]["num_edges"]

        # Empty features initialized; node and edge features can be added later
        x = torch.zeros((num_nodes, 0))
        edge_attr = torch.zeros((num_edges, 0))

        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _create_augmented_graph(self, mol: Chem.Mol) -> Tuple[torch.Tensor, dict]:
        """
        Generates an augmented graph from a SMILES string.

        Args:
            mol (Chem.Mol): A molecule generated by RDKit.

        Returns:
            Tuple[dict, torch.Tensor]: Augmented molecule information and edge index.
        """
        edge_index, node_info, edge_info = self._augment_graph_structure(mol)

        augmented_molecule = {"nodes": node_info, "edges": edge_info}

        return edge_index, augmented_molecule

    def _augment_graph_structure(
        self, mol: Chem.Mol
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Constructs the full augmented graph structure from a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[torch.Tensor, dict, dict]: Edge index, node metadata, and edge metadata.
        """
        self.num_of_nodes = mol.GetNumAtoms()
        self.num_of_edges = mol.GetNumBonds()

        self._annotate_atoms_and_bonds(mol)
        atom_edge_index = self._generate_atom_level_edge_index(mol)

        # Create FG-level structure and edges
        fg_atom_edge_index, fg_nodes, atom_fg_edges, structured_fg_map, bonds = (
            self._construct_fg_to_atom_structure(mol)
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
            "num_nodes": self.num_of_nodes,
        }
        edge_info = {
            WITHIN_ATOMS_EDGE: mol,
            ATOM_FG_EDGE: atom_fg_edges,
            WITHIN_FG_EDGE: internal_fg_edges,
            FG_GRAPHNODE_EDGE: fg_to_graph_edges,
            "num_edges": self.num_of_edges,
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
        edge_index = torch.tensor(
            [
                [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
                [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
            ]
        )
        return torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

    def _construct_fg_to_atom_structure(
        self, mol: Chem.Mol
    ) -> Tuple[List[List[int]], dict, dict, dict, list]:
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

        for idx, fg_key in enumerate(structure):
            structured_fg_map[self.num_of_nodes] = {"atom": structure[fg_key]["atom"]}

            # Build edge index for fg to atom nodes connections
            for atom_idx in structure[fg_key]["atom"]:
                fg_atom_edge_index[0] += [self.num_of_nodes, atom_idx]
                fg_atom_edge_index[1] += [atom_idx, self.num_of_nodes]
                atom_fg_edges[f"{self.num_of_nodes}_{atom_idx}"] = {
                    EDGE_LEVEL: ATOM_FG_EDGE
                }
                self.num_of_edges += 1

            # Identify ring vs. functional group type
            ring_fg = {
                mol.GetAtomWithIdx(i).GetProp("RING")
                for i in structure[fg_key]["atom"]
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
                fg_nodes[self.num_of_nodes] = {
                    NODE_LEVEL: FG_NODE_LEVEL,
                    # E.g.,  Fused Ring has size "5-6", indicating size of each connected ring in fused ring
                    "FG": f"RING_{ring_size}",
                    "RING": ring_size,
                }
            else:  # No connected has a ring size which indicates it is simple FG
                fg_set = {
                    mol.GetAtomWithIdx(i).GetProp("FG")
                    for i in structure[fg_key]["atom"]
                }
                if "" in fg_set or len(fg_set) > 1:
                    raise ValueError("Invalid functional group assignment to atoms.")

                for atom_idx in structure[fg_key]["atom"]:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetProp("FG"):
                        fg_nodes[self.num_of_nodes] = {
                            NODE_LEVEL: FG_NODE_LEVEL,
                            "FG": atom.GetProp("FG"),
                            "RING": atom.GetProp("RING"),
                        }
                        break
                else:
                    raise AssertionError(
                        "Expected at least one atom with a functional group."
                    )

            self.num_of_nodes += 1

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

            internal_edge_index[0] += [source_fg, target_fg]
            internal_edge_index[1] += [target_fg, source_fg]
            internal_fg_edges[f"{source_fg}_{target_fg}"] = {EDGE_LEVEL: WITHIN_FG_EDGE}
            self.num_of_edges += 1

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
            graph_edge_index[0] += [self.num_of_nodes, fg_id]
            graph_edge_index[1] += [fg_id, self.num_of_nodes]
            fg_graph_edges[f"{self.num_of_nodes}_{fg_id}"] = {
                EDGE_LEVEL: FG_GRAPHNODE_EDGE
            }
            self.num_of_edges += 1

        return graph_edge_index, graph_node, fg_graph_edges


class RuleBasedFGReader(ChemDataReader):
    """
    A reader which give numeric value for given functional group.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the name of the rule-based functional group reader.

        Returns:
            str: The name of the reader.
        """
        return "rule_based_fg"

    def _read_data(self, fg: str) -> Optional[int]:
        """
        Reads and returns the token index for a given functional group.

        Args:
            fg (str): The functional group to look up.

        Returns:
            Optional[int]: The index of the functional group, or None if not found.
        """
        return self._get_token_index(fg)
