from typing import Tuple, Mapping, Optional

import networkx as nx
import os
import pysmiles as ps

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data.collate import collate as graph_collate
from torch_geometric.utils import from_networkx

from chebai.preprocessing import reader as dr
from chebai.preprocessing import collate
from chebai.preprocessing.structures import XYData


class GraphCollater(collate.RaggedCollater):
    def __call__(self, data):
        _, y, idents = zip(
            *((d["features"], d["labels"], d.get("ident")) for d in data)
        )
        merged_data = []
        for row in data:
            row["features"].y = row["labels"]
            merged_data.append(row["features"])
        # add empty edge_attr to avoid problems during collate (only relevant for molecules without edges)
        for mdata in merged_data:
            for i, store in enumerate(mdata.stores):
                if "edge_attr" not in store:
                    store["edge_attr"] = torch.tensor([])
        for attr in merged_data[0].keys():
            for data in merged_data:
                for store in data.stores:
                    # Im not sure why the following conversion is needed, but it solves this error:
                    # packages/torch_geometric/data/collate.py", line 177, in _collate
                    #     value = torch.cat(values, dim=cat_dim or 0, out=out)
                    # RuntimeError: torch.cat(): input types can't be cast to the desired output type Long
                    if isinstance(torch.tensor, store[attr]):
                        store[attr] = store[attr].to(dtype=torch.float32)
                    else:
                        store[attr] = torch.tensor(store[attr], dtype=torch.float32)
                    # print(f'attr: {attr}, store[attr].dtype: {store[attr].dtype}')

        x = graph_collate(
            GraphData,
            merged_data,
            follow_batch=["x", "edge_attr", "edge_index", "label"],
        )
        y = self.process_label_rows(y)
        x[0].x = x[0].x.to(dtype=torch.int64)
        # x is a Tuple[BaseData, Mapping, Mapping]
        return XYGraphData(
            x,
            y,
            idents=idents,
            model_kwargs={},
            loss_kwargs={},
        )
        # return x[0]


class GraphReader(dr.ChemDataReader):
    COLLATER = GraphCollater

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirname = os.path.dirname(__file__)

    @classmethod
    def name(cls):
        return "graph"

    def _read_data(self, raw_data) -> Optional[GraphData]:
        # raw_data is a SMILES string
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


class XYGraphData(XYData):
    def __len__(self):
        return len(self.y)

    def to_x(self, device):
        if isinstance(self.x, tuple):
            res = []
            for elem in self.x:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = v.to(device) if v is not None else None
                else:
                    elem = elem.to(device)
                res.append(elem)
            return tuple(res)
        return super(self, device)
