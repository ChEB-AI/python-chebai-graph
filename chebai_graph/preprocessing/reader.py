import importlib
import os
from typing import List, Mapping, Optional, Tuple

import chebai.preprocessing.reader as dr
import networkx as nx
import pysmiles as ps
import rdkit.Chem as Chem
import torch
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from rdkit.Chem import Mol
from torch_geometric.data import Data as GeomData
from torch_geometric.utils import from_networkx

import chebai_graph.preprocessing.properties as properties
from chebai_graph.preprocessing.collate import GraphCollator
from chebai_graph.preprocessing.fg_detection.rule_based import (
    detect_functional_group,
    get_structure,
    set_atom_map_num,
)


class GraphPropertyReader(dr.ChemDataReader):
    COLLATOR = GraphCollator

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.failed_counter = 0
        self.mol_object_buffer = {}

    @classmethod
    def name(cls):
        return "graph_properties"

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.rdchem.Mol]:
        """Load smiles into rdkit, store object in buffer"""
        if smiles in self.mol_object_buffer:
            return self.mol_object_buffer[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rank_zero_warn(f"RDKit failed to at parsing {smiles} (returned None)")
            self.failed_counter += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                rank_zero_warn(f"Rdkit failed at sanitizing {smiles}")
                self.failed_counter += 1
        self.mol_object_buffer[smiles] = mol
        return mol

    def _read_data(self, raw_data):
        mol = self._smiles_to_mol(raw_data)
        if mol is None:
            return None

        x = torch.zeros((mol.GetNumAtoms(), 0))

        edge_attr = torch.zeros((mol.GetNumBonds(), 0))

        edge_index = torch.tensor(
            [
                [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
                [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
            ]
        )
        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def on_finish(self):
        rank_zero_info(f"Failed to read {self.failed_counter} SMILES in total")
        self.mol_object_buffer = {}

    def read_property(
        self, smiles: str, property: properties.MolecularProperty
    ) -> Optional[List]:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None
        return property.get_property_value(mol)


class GraphReader(dr.ChemDataReader):
    """Reads each atom as one token (atom symbol + charge), reads bond order as edge attribute.
    Creates nx Graph from SMILES."""

    COLLATOR = GraphCollator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirname = os.path.dirname(__file__)

    @classmethod
    def name(cls):
        return "graph"

    def _read_data(self, raw_data) -> Optional[GeomData]:
        # raw_data is a SMILES string
        try:
            mol = ps.read_smiles(raw_data)
        except ValueError:
            return None
        assert isinstance(mol, nx.Graph)
        d = {}
        de = {}
        for node in mol.nodes:
            n = mol.nodes[node]
            try:
                m = n["element"]
                charge = n["charge"]
                if charge:
                    if charge > 0:
                        m += "+"
                    else:
                        m += "-"
                        charge *= -1
                    if charge > 1:
                        m += str(charge)
                m = f"[{m}]"
            except KeyError:
                m = "*"
            d[node] = self._get_token_index(m)
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        for edge in mol.edges:
            de[edge] = mol.edges[edge]["order"]
            for attr in list(mol.edges[edge].keys()):
                del mol.edges[edge][attr]
        nx.set_node_attributes(mol, d, "x")
        nx.set_edge_attributes(mol, de, "edge_attr")
        data = from_networkx(mol)
        return data

    def collate(self, list_of_tuples):
        return self.collator(list_of_tuples)


class GraphFGAugmentorReader(dr.ChemDataReader):
    COLLATOR = GraphCollator
    NODE_LEVEL = {"atom_node": 1, "fg_node": 2, "graph_node": 3}
    EDGE_LEVEL = {
        "within_atoms": 1,
        "within_fg": 2,
        "atom_fg": 3,
        "fg_graphNode": 4,
    }

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.failed_counter = 0
        self.mol_object_buffer = {}

        if "graph_fg" not in self.cache:
            raise KeyError(
                f"Function group `graph_fg` doesn't exits in {self.token_path}. "
                f"It should be manually added to token file (preferably at 0th index)"
            )

    @classmethod
    def name(cls):
        return "graph_fg_augmentor"

    def _read_data(self, raw_data):
        mol = self._smiles_to_mol(raw_data)
        if mol is None:
            return None

        x = torch.zeros((mol.GetNumAtoms(), 0))

        edge_attr = torch.zeros((mol.GetNumBonds(), 0))

        edge_index = self._augment_graph(mol)

        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.rdchem.Mol]:
        """Load smiles into rdkit, store object in buffer"""
        if smiles in self.mol_object_buffer:
            return self.mol_object_buffer[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rank_zero_warn(f"RDKit failed to at parsing {smiles} (returned None)")
            self.failed_counter += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                rank_zero_warn(f"Rdkit failed at sanitizing {smiles}")
                self.failed_counter += 1
        self.mol_object_buffer[smiles] = mol
        return mol

    def _augment_graph(self, mol: Mol):
        edge_index = torch.tensor(
            [
                [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
                [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
            ]
        )
        within_atoms_edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

        num_of_nodes = mol.GetNumAtoms()

        set_atom_map_num(mol)
        detect_functional_group(mol)

        node_features = []
        sorted_atoms = sorted(
            list(mol.GetAtoms()), key=lambda atom: atom.GetAtomMapNum()
        )

        for idx, atom in enumerate(sorted_atoms):
            node_features.append(
                [
                    self.NODE_LEVEL["atom_node"],
                    self._get_fg_index(atom),
                    self._get_ring_size(atom),
                ]
            )

        structure, bonds = get_structure(mol)

        if not structure:
            raise ValueError("")

        # Preprocess the molecular structure to match feature dictionary keys
        fg_to_atoms_edge_index = [[], []]
        new_structure = {}
        for idx, fg in enumerate(structure):
            # new_sm = preprocess_smiles(sm)  # Preprocess SMILES to match the feature dictionary
            new_structure[num_of_nodes] = {
                "atom": structure[fg]["atom"]  # Get atom list for fragment
            }
            for atom in structure[fg]["atom"]:
                fg_to_atoms_edge_index[0].extend([num_of_nodes, atom])
                fg_to_atoms_edge_index[1].extend([atom, num_of_nodes])

            any_atom = next(iter(structure[fg]["atom"][0]))  # any atom related to fg
            node_features.append(
                [
                    self.NODE_LEVEL["fg_node"],
                    self._get_fg_index(any_atom),
                    self._get_ring_size(any_atom),
                ]
            )

            num_of_nodes += 1

        within_fg_edge_index = [[], []]
        for bond in bonds:
            start_idx, end_idx = bond[:2]
            for key, value in new_structure.items():
                if start_idx in value["atom"]:
                    source_fg = key
                if end_idx in value["atom"]:
                    target_fg = key
            within_fg_edge_index[0].extend([source_fg, target_fg])
            within_fg_edge_index[1].extend([target_fg, source_fg])

        node_features.append(
            [self.NODE_LEVEL["global_node"], self._get_token_index("graph_fg"), 0]
        )
        global_node_edge_index = [[], []]
        for fg in new_structure.keys():
            global_node_edge_index[0].extend([num_of_nodes, fg])
            global_node_edge_index[1].extend([fg, num_of_nodes])

        all_edges = torch.cat(
            [
                within_atoms_edge_index,
                torch.tensor(fg_to_atoms_edge_index, dtype=torch.long),
                torch.tensor(within_fg_edge_index, dtype=torch.long),
                torch.tensor(global_node_edge_index, dtype=torch.long),
            ],
            dim=1,
        )

        return all_edges

    def _get_fg_index(self, atom):
        fg_group = atom.GetProp("FG")
        if fg_group:
            fg_index = self._get_token_index(fg_group)
            return fg_index
        else:
            raise Exception("")

    def _get_ring_size(self, atom):
        ring_size_str = atom.GetProp("RING")
        if ring_size_str:
            ring_sizes = list(map(int, ring_size_str.split("-")))
            # TODO: Decide ring size for atoms belongs to fused rings, rn only max ring size taken
            return max(ring_sizes)
        else:
            return 0

    def on_finish(self):
        rank_zero_info(f"Failed to read {self.failed_counter} SMILES in total")
        self.mol_object_buffer = {}

    def read_property(
        self, smiles: str, property: properties.MolecularProperty
    ) -> Optional[List]:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None
        return property.get_property_value(mol)
