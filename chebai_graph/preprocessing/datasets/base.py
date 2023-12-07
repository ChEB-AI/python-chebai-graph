import networkx as nx
import os
import pysmiles as ps

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data.collate import collate as graph_collate
from torch_geometric.utils import from_networkx

from chebai.preprocessing import reader as dr
from chebai.preprocessing import collate

def mol_to_data(smiles):
    try:
        mol = ps.read_smiles(smiles)
    except:
        return None
    d = {}
    for node in mol.nodes:
        el = mol.nodes[node].get("element")
        if el is not None:
            v = atom_index.index(el)
            base = [float(i == v) for i in range(118)]
            wildcard = [0.0]
        else:
            base = [0.0 for i in range(118)]
            wildcard = [1.0]
        d[node] = (
            base
            + [mol.nodes[node].get("charge", 0.0), mol.nodes[node].get("hcount", 0.0)]
            + wildcard
        )

        for attr in list(mol.nodes[node].keys()):
            del mol.nodes[node][attr]
    nx.set_node_attributes(mol, d, "x")
    return from_networkx(mol)


def get_mol_enc(x):
    i, s = x
    return i, mol_to_data(s) if s else None


class MolDatareader(dr.DataReader):
    @classmethod
    def name(cls):
        return "mol"

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.cache = []

    def to_data(self, row):
        return self.get_encoded_mol(
            row[self.SMILES_INDEX], self.cache
        ), self._get_label(row)

    def get_encoded_mol(self, smiles, cache):
        try:
            mol = ps.read_smiles(smiles)
        except ValueError:
            return None
        d = {}
        for node in mol.nodes:
            try:
                m = mol.nodes[node]["element"]
            except KeyError:
                m = "*"
            try:
                x = cache.index(m)
            except ValueError:
                x = len(cache)
                cache.append(m)
            d[node] = torch.tensor(x)
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        nx.set_node_attributes(mol, d, "x")
        return mol

class GraphCollater(collate.Collater):
    def __call__(self, data):
        merged_data = []
        for row in data:
            row["features"].y = row["labels"]
            merged_data.append(row["features"])
        return graph_collate(
            GraphData,
            merged_data,
            follow_batch=["x", "edge_attr", "edge_index", "label"],
        )


class GraphReader(dr.DataReader):
    COLLATER = GraphCollater

    @classmethod
    def name(cls):
        return "graph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dirname = os.path.dirname(dr.__file__)
        with open(os.path.join(dirname, "bin", "tokens.txt"), "r") as pk:
            self.cache = [x.strip() for x in pk]

    def _read_data(self, raw_data):
        try:
            mol = ps.read_smiles(raw_data)
        except ValueError:
            return None
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
            d[node] = self.cache.index(m) + dr.EMBEDDING_OFFSET
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        for edge in mol.edges:
            de[edge] = mol.edges[edge]["order"]
            for attr in list(mol.edges[edge].keys()):
                del mol.edges[edge][attr]
        nx.set_node_attributes(mol, d, "x")
        nx.set_edge_attributes(mol, de, "edge_attr")
        return from_networkx(mol)

    def collate(self, list_of_tuples):
        return self.collater(list_of_tuples)