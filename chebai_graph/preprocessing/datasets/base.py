import networkx as nx
import os
import pysmiles as ps

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data.collate import collate as graph_collate
from torch_geometric.utils import from_networkx

from chebai.preprocessing import reader as dr
from chebai.preprocessing import collate


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