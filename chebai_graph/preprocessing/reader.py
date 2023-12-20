import importlib

from torch_geometric.utils import from_networkx
from typing import Tuple, Mapping, Optional, List

import importlib
import networkx as nx
import os
import torch
import rdkit.Chem as Chem
import pysmiles as ps
import chebai.preprocessing.reader as dr
from chebai_graph.preprocessing.collate import GraphCollater
import chebai_graph.preprocessing.properties as properties
from torch_geometric.data import Data as GeomData
from lightning_utilities.core.rank_zero import rank_zero_warn, rank_zero_info


class GraphPropertyReader(dr.ChemDataReader):
    COLLATER = GraphCollater

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
        if not isinstance(mol, Chem.rdchem.Mol):
            try:
                rank_zero_warn(f"Rdkit failed to sanitize {smiles}")
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
            except Exception as e:
                rank_zero_warn(f"Rdkit failed without sanitizing for {smiles}")
                self.failed_counter += 1
                mol = None
            if mol is None:
                rank_zero_warn(
                    f"RDKit failed to without sanitizing for {smiles} (returned None)"
                )
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

    def read_atom_property(self, smiles: str, property: properties.AtomProperty):
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None
        return torch.tensor(
            [property.get_atom_property_value(atom) for atom in mol.GetAtoms()]
        )

    def read_bond_property(self, smiles: str, property: properties.BondProperty):
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None
        return torch.tensor(
            [property.get_bond_property_value(bond) for bond in mol.GetBonds()]
        )


class GraphReader(dr.ChemDataReader):
    """Reads each atom as one token (atom symbol + charge), reads bond order as edge attribute.
    Creates nx Graph from SMILES."""

    COLLATER = GraphCollater

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
        return self.collater(list_of_tuples)
