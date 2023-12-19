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

    def _resolve_property(
        self, property  #: str | properties.MolecularProperty
    ) -> properties.MolecularProperty:
        # split class_path into module-part and class name
        if isinstance(property, properties.MolecularProperty):
            return property
        try:
            last_dot = property.rindex(".")
            module_name = property[:last_dot]
            class_name = property[last_dot + 1 :]
            module = importlib.import_module(module_name)
            return getattr(module, class_name)()
        except ValueError:
            # if only a class name is given, assume the module is chebai_graph.processing.properties
            return getattr(properties, property)()

    def __init__(
        self,
        atom_properties,  #: Optional[List[str | properties.AtomProperty]],
        bond_properties,  #: Optional[List[str | properties.BondProperty]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # atom_properties and bond_properties are given as lists containing class_paths
        if atom_properties is not None:
            atom_properties = [self._resolve_property(prop) for prop in atom_properties]
        if bond_properties is not None:
            bond_properties = [self._resolve_property(prop) for prop in bond_properties]
        self.atom_properties = atom_properties
        self.bond_properties = bond_properties
        self.failed_counter = 0

    @classmethod
    def name(cls):
        return "graph_properties"

    def _read_data(self, raw_data):
        mol = Chem.MolFromSmiles(raw_data)
        if not isinstance(mol, Chem.rdchem.Mol):
            rank_zero_warn(f'RDKit failed to read SMILES "{raw_data}"')
            self.failed_counter += 1
            return None

        x = torch.zeros((mol.GetNumAtoms(), len(self.atom_properties)))
        for i, atom in enumerate(mol.GetAtoms()):
            for j, prop in enumerate(self.atom_properties):
                if not isinstance(atom, Chem.rdchem.Atom):
                    rank_zero_warn(f"Uh oh! atom {atom} is not an Atom object")
                x[i, j] = prop.encode_property_value(prop.get_atom_property_value(atom))

        edge_attr = torch.zeros((mol.GetNumBonds(), len(self.bond_properties)))
        for i, bond in enumerate(mol.GetBonds()):
            for j, prop in enumerate(self.bond_properties):
                edge_attr[i, j] = prop.encode_property_value(
                    prop.get_bond_property_value(bond)
                )

        edge_index = torch.tensor(
            [
                [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
                [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
            ]
        )
        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def on_finish(self):
        rank_zero_info(f"Failed to read {self.failed_counter} SMILES in total")
        for prop in self.bond_properties:
            prop.on_finish()
        for prop in self.atom_properties:
            prop.on_finish()


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
