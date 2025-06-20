import textwrap
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from chebai.preprocessing.reader import DataReader
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
        # Record number of failures when constructing molecule from smiles
        self.f_cnt_for_smiles = 0
        # Record number of failure during augmented graph construction
        self.f_cnt_for_aug_graph = 0
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

    @abstractmethod
    def _create_augmented_graph(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict]:
        """
        Augments a molecule represented by a SMILES string.

        Args:
            mol (Chem.Mol): RDKIT molecule.

        Returns:
            Tuple[torch.Tensor, Dict]: Graph edge index and augmented molecule information
        """

    @abstractmethod
    def _read_data(self, raw_data: str) -> GeomData:
        """
        Reads raw data and returns a list of processed data.

        Args:
            raw_data (str): Raw data input.

        Returns:
            GeomData: `torch_geometric.data.Data` object.
        """

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
            print(f"RDKit failed to parse {smiles} (returned None)")
            self.f_cnt_for_smiles += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"RDKit failed at sanitizing {smiles}, Error {e}")
                self.f_cnt_for_smiles += 1
        return mol

    def on_finish(self) -> None:
        """
        Finalizes the reading process and logs the number of failed SMILES and failed augmentation.
        """
        print(f"Failed to read {self.f_cnt_for_smiles} SMILES in total")
        print(
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

        try:
            returned_result = self._create_augmented_graph(mol)
        except Exception as e:
            raise RuntimeError(
                f"Error has occurred for following SMILES: {smiles}\n\t {e}"
            ) from e

        # If the returned result is None, it indicates that the graph augmentation failed
        if returned_result is None:
            print(f"Failed to construct augmented graph for smiles {smiles}")
            self.f_cnt_for_aug_graph += 1
            return None

        edge_index, augmented_molecule = returned_result
        self.mol_object_buffer[smiles] = augmented_molecule

        # Empty features initialized; node and edge features can be added later
        x = torch.zeros((augmented_molecule["nodes"]["num_nodes"], 0))
        edge_attr = torch.zeros((augmented_molecule["edges"][NUM_EDGES], 0))

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

    def _create_augmented_graph(self, mol: Chem.Mol) -> Tuple[torch.Tensor, dict]:
        """
        Generates an augmented graph from a SMILES string.

        Args:
            mol (Chem.Mol): A molecule generated by RDKit.

        Returns:
            Tuple[torch.Tensor, dict]:
                - Augmented graph edge index,
                - Augmented graph (nodes and edges).
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
            Tuple[torch.Tensor, dict, dict]:
                - Augmented graph edge index,
                - Augmented graph node attributes
                - Augmented graph edge attributes.
        """
        self._num_of_nodes = mol.GetNumAtoms()
        self._num_of_edges = mol.GetNumBonds()

        self._annotate_atoms_and_bonds(mol)
        atom_edge_index = self._generate_atom_level_edge_index(mol)

        # Create FG-level structure and edges
        fg_atom_edge_index, fg_nodes, atom_fg_edges, fg_to_atoms_map, bonds = (
            self._construct_fg_to_atom_structure(mol)
        )

        fg_internal_edge_index, internal_fg_edges = self._construct_fg_level_structure(
            fg_to_atoms_map, bonds
        )

        fg_graph_edge_index, graph_node, fg_to_graph_edges = (
            self._construct_fg_to_graph_node_structure(fg_to_atoms_map)
        )

        # Merge all edge types
        directed_edge_index = torch.cat(
            [
                atom_edge_index,
                torch.tensor(fg_atom_edge_index, dtype=torch.long),
                torch.tensor(fg_internal_edge_index, dtype=torch.long),
                torch.tensor(fg_graph_edge_index, dtype=torch.long),
            ],
            dim=1,
        )
        # First all directed edges from source to target are placed, then all directed edges from target to source
        # are placed --- this is needed as it is easier to align the property values in same way
        undirected_edge_index = torch.cat(
            [directed_edge_index, directed_edge_index[[1, 0], :]], dim=1
        )

        total_atoms = sum([mol.GetNumAtoms(), len(fg_nodes), 1])
        assert (
            self._num_of_nodes == total_atoms
        ), f"Mismatch in number of nodes: expected {total_atoms}, got {self._num_of_nodes}"
        node_info = {
            "atom_nodes": mol,
            "fg_nodes": fg_nodes,
            "graph_node": graph_node,
            "num_nodes": self._num_of_nodes,
        }

        total_edges = sum(
            [
                mol.GetNumBonds(),
                len(atom_fg_edges),
                len(internal_fg_edges),
                len(fg_to_graph_edges),
            ]
        )
        assert (
            self._num_of_edges == total_edges
        ), f"Mismatch in number of edges: expected {total_edges}, got {self._num_of_edges}"
        edge_info = {
            WITHIN_ATOMS_EDGE: mol,
            ATOM_FG_EDGE: atom_fg_edges,
            WITHIN_FG_EDGE: internal_fg_edges,
            FG_GRAPHNODE_EDGE: fg_to_graph_edges,
            NUM_EDGES: self._num_of_edges * 2,  # Undirected edges
        }
        return undirected_edge_index, node_info, edge_info

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
            torch.Tensor: Directed edge index tensor.
        """
        # We need to ensure that directed edges which form a undirected edge are adjacent to each other
        edge_index_list = [[], []]
        for bond in mol.GetBonds():
            edge_index_list[0].append(bond.GetBeginAtomIdx())
            edge_index_list[1].append(bond.GetEndAtomIdx())
        return torch.tensor(edge_index_list, dtype=torch.long)

    def _construct_fg_to_atom_structure(
        self, mol: Chem.Mol
    ) -> tuple[list[list[int]], dict, dict, dict, list]:
        """
        Constructs edges between functional group (FG) nodes and atom nodes.
        This method detects functional groups in the molecule and creates edges
        between FG nodes and their connected atom nodes.

        Args:
            mol (Chem.Mol): RDKit molecule.

        Returns:
            tuple[list[list[int]], dict, dict, dict, list]: A tuple containing:
                - Edge index for FG to atom connections.
                - FG node info,
                - FG-atom edge attributes,
                - FG to atoms mapping,
                - Bonds between FG nodes.

        Raises:
            ValueError: If functional groups span multiple ring sizes or if no functional group is assigned to atoms.
        """

        # Rule-based algorithm to detect functional groups
        set_atom_map_num(mol)
        detect_functional_group(mol)
        structure, bonds = get_structure(mol)
        assert structure is not None, "Failed to detect functional groups."

        fg_atom_edge_index = [[], []]
        fg_nodes, atom_fg_edges = {}, {}
        # Contains augmented fg-nodes and connected atoms indices
        fg_to_atoms_map = {}

        molecule_atoms_set = set()
        for _, fg_group in structure.items():
            fg_to_atoms_map[self._num_of_nodes] = fg_group
            is_ring_fg = fg_group["is_ring_fg"]

            connected_atoms = []
            # Build edge index for fg to atom nodes connections
            for atom_idx in fg_group["atom"]:
                # Fused rings can have an atom which belong to more than one ring
                if atom_idx in molecule_atoms_set and not is_ring_fg:
                    raise ValueError(
                        f"An atom {atom_idx} cannot belong to more than one functional group"
                    )
                molecule_atoms_set.add(atom_idx)

                fg_atom_edge_index[0].append(self._num_of_nodes)
                fg_atom_edge_index[1].append(atom_idx)
                atom_fg_edges[f"{self._num_of_nodes}_{atom_idx}"] = {
                    EDGE_LEVEL: ATOM_FG_EDGE
                }
                self._num_of_edges += 1

                atom = mol.GetAtomWithIdx(atom_idx)
                connected_atoms.append(atom)

            if is_ring_fg:
                self._set_ring_fg_prop(connected_atoms, fg_nodes)
            else:
                self._set_fg_prop(connected_atoms, fg_nodes)

            self._num_of_nodes += 1

        return fg_atom_edge_index, fg_nodes, atom_fg_edges, fg_to_atoms_map, bonds

    def _set_ring_fg_prop(self, connected_atoms, fg_nodes):
        # FG atoms have ring size, which indicates the FG is a Ring or Fused Rings
        ring_size = len(connected_atoms)
        fg_nodes[self._num_of_nodes] = {
            NODE_LEVEL: FG_NODE_LEVEL,
            # E.g.,  Fused Ring has size "5-6", indicating size of each connected ring in fused ring
            "FG": f"RING_{ring_size}",
            "RING": ring_size,
        }
        # In this case, all atoms of Ring/Fused Ring are assigned the ring size as functional group
        for atom in connected_atoms:
            ring_prop = atom.GetProp("RING")
            if not ring_prop:
                raise ValueError("Atom does not have a ring size set")
            max_ring_size = max(list(map(int, ring_prop.split("-"))))
            atom.SetProp("FG", f"RING_{max_ring_size}")

    def _set_fg_prop(self, connected_atoms, fg_nodes):
        fg_set = {atom.GetProp("FG") for atom in connected_atoms}
        if not fg_set:
            raise ValueError(
                "No functional group assigned to atoms in the functional group."
            )

        if "" in fg_set and len(fg_set) == 1:
            if len(connected_atoms) == 1:
                # If there is only one atom and one edge connecting this atom to its fg_atom,
                # the functional group will be the symbol of this atom
                # This special case is to handle wildcard SMILES Eg. CHEBI:33429
                atom = connected_atoms[0]
                # TODO: needed or can we set to default fg prop `NO_FG`?
                atom.SetProp("FG", atom.GetSymbol())
            else:
                # If there are multiple atoms connected to the functional group, and no atoms have a functional group property/name
                # assigned, Eg. CHEBI:55388, atom idx 2 and 3 ([C-]#[C-]") have no functional group name, so default FG prop is used
                for atom in connected_atoms:
                    atom.SetProp("FG", "NO_FG")
                    # atom.SetProp("FG", fg_smiles)

        if len(fg_set - {""}) > 1:
            raise ValueError(
                "Connected atoms have different function groups assigned.\n"
                "All Connected atoms must belong to one functional group or None"
            )

        # Select any one connected atom to get FG type and ring size
        representative_atom = next(
            (atom for atom in connected_atoms if atom.GetProp("FG")), None
        )
        if representative_atom is None:
            raise AssertionError("Expected at least one atom with a functional group.")

        fg_nodes[self._num_of_nodes] = {
            NODE_LEVEL: FG_NODE_LEVEL,
            "FG": representative_atom.GetProp("FG"),
            "RING": 0,
        }

    def _construct_fg_level_structure(
        self, fg_to_atoms_map: dict, bonds: list
    ) -> Tuple[List[List[int]], dict]:
        """
        Constructs internal edges between functional group nodes based on bond connections.

        Args:
            fg_to_atoms_map (dict): Mapping from FG ID to atom indices.
            bonds (list): List of bond tuples (source, target, ...).

        Returns:
            Tuple[List[List[int]], dict]:
                - Edge index within fg nodes
                - Edge attributes for edges within fg nodes.
        """
        internal_fg_edges = {}
        internal_edge_index = [[], []]

        def add_fg_internal_edge(source_fg, target_fg):
            assert (
                source_fg is not None and target_fg is not None
            ), "Each bond should have a fg node on both end"
            assert source_fg != target_fg, "Source and Target FG should be  different"

            edge_key = tuple(sorted((source_fg, target_fg)))
            edge_str = f"{edge_key[0]}_{edge_key[1]}"
            if edge_str not in internal_fg_edges:
                # If two atoms of a FG points to atom(s) belonging to another FG. In this case, only one edge is counted.
                # Eg. In CHEBI:52723, atom idx 13 and 16 of a FG points to atom idx 18 of another FG
                internal_edge_index[0].append(source_fg)
                internal_edge_index[1].append(target_fg)
                internal_fg_edges[edge_str] = {EDGE_LEVEL: WITHIN_FG_EDGE}
                self._num_of_edges += 1

        for bond in bonds:
            source_atom, target_atom = bond[:2]
            source_fg, target_fg = None, None
            for fg_id, data in fg_to_atoms_map.items():
                if source_fg is None and source_atom in data["atom"]:
                    source_fg = fg_id
                if target_fg is None and target_atom in data["atom"]:
                    target_fg = fg_id
                if source_fg is not None and target_fg is not None:
                    break
            add_fg_internal_edge(source_fg, target_fg)

        # For Rings belonging to fused rings
        fg_nodes = list(fg_to_atoms_map.keys())
        for i, fg_node_1 in enumerate(fg_nodes):
            fg_map_1 = fg_to_atoms_map[fg_node_1]
            for fg_node_2 in fg_nodes[i + 1 :]:
                fg_map_2 = fg_to_atoms_map[fg_node_2]
                if (
                    (fg_node_1 == fg_node_2)
                    or not fg_map_1["is_ring_fg"]
                    or not fg_map_2["is_ring_fg"]
                ):
                    continue
                if fg_map_1["atom"] & fg_map_2["atom"]:
                    add_fg_internal_edge(fg_node_1, fg_node_2)

        return internal_edge_index, internal_fg_edges

    def _construct_fg_to_graph_node_structure(
        self, fg_to_atoms_map: dict
    ) -> Tuple[List[List[int]], dict, dict]:
        """
        Constructs edges between functional group nodes and a global graph-level node.

        Args:
            fg_to_atoms_map (dict): Mapping from FG ID to atom indices.

        Returns:
            Tuple[List[List[int]], dict, dict]:
                - Graph to FG Edge index
                - Graph-level node attribute
                - FG to Graph Edge attributes
        """
        graph_node = {NODE_LEVEL: GRAPH_NODE_LEVEL, "FG": "graph_fg", "RING": "0"}

        fg_graph_edges = {}
        graph_edge_index = [[], []]

        for fg_id in fg_to_atoms_map:
            graph_edge_index[0].append(self._num_of_nodes)
            graph_edge_index[1].append(fg_id)
            fg_graph_edges[f"{self._num_of_nodes}_{fg_id}"] = {
                EDGE_LEVEL: FG_GRAPHNODE_EDGE
            }
            self._num_of_edges += 1
        self._num_of_nodes += 1

        return graph_edge_index, graph_node, fg_graph_edges
