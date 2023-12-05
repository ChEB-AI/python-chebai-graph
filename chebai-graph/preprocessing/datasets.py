import fastobo
import glob
import networkx as nx
import os
import pandas as pd
import pickle
import pysmiles as ps
import random
import requests
from itertools import chain
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset as TGDataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data.collate import collate as graph_collate
from torch_geometric.utils import from_networkx

from chebai.preprocessing.datasets.chebi import JCIBase, JCIExtendedBase, atom_index
from chebai.preprocessing.structures import PairData, PrePairData
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
        for d, y, _ in data:
            d.y = y
            merged_data.append(d)
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
        dirname = os.path.dirname(__file__)
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

try:
    from k_gnn import TwoMalkin
except ModuleNotFoundError:
    pass
else:
    from k_gnn.dataloader import collate

    class GraphTwoDataset(GraphDataset):
        @classmethod
        def name(cls):
            return "graph_k2"

        def to_data(self, df: pd.DataFrame):
            for data in super().to_data(df):
                if data.num_nodes >= 6:
                    x = data.x
                    data.x = data.x.unsqueeze(0)
                    data = TwoMalkin()(data)
                    data.x = x
                    yield data

        def collate(self, list_of_tuples):
            return collate(list_of_tuples)


class JCIMolData(JCIBase):
    READER = MolDatareader


class JCIGraphData(JCIBase):
    READER = GraphReader


class JCIExtendedGraphData(JCIExtendedBase):
    READER = GraphReader


class PartOfData(TGDataset):
    def len(self):
        return self.extract_largest_index(self.processed_dir, self.kind)

    def get(self, idx):
        return pickle.load(
            open(os.path.join(self.processed_dir, f"{self.kind}.{idx}.pt"), "rb")
        )

    def __init__(
        self,
        root,
        kind="train",
        batch_size=100,
        train_split=0.95,
        part_split=0.1,
        pre_transform=None,
        **kwargs,
    ):
        self.cache_file = ".part_data.pkl"
        self._ignore = set()
        self.train_split = train_split
        self.part_split = part_split
        self.kind = kind
        self.batch_size = batch_size
        super().__init__(
            root, pre_transform=pre_transform, transform=self.transform, **kwargs
        )
        self.graph = pickle.load(
            open(os.path.join(self.processed_dir, self.processed_cache_names[0]), "rb")
        )

    def transform(self, ppds):
        return [PairData(ppd, self.graph) for ppd in ppds]

    def download(self):
        url = "http://purl.obolibrary.org/obo/chebi.obo"
        r = requests.get(url, allow_redirects=True)
        open(self.raw_paths[0], "wb").write(r.content)

    def process(self):
        doc = fastobo.load(self.raw_paths[0])
        elements = list()
        for clause in doc:
            callback = CALLBACKS.get(type(clause))
            if callback is not None:
                elements.append(callback(clause))

        g = nx.DiGraph()
        for n in elements:
            g.add_node(n["id"], **n)
        g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])

        print("pass parts")
        self.pass_parts(g, 23367, set())
        print("Load data")
        children = frozenset(list(nx.single_source_shortest_path(g, 23367).keys()))
        parts = frozenset({p for c in children for p in g.nodes[c]["has_part"]})

        print("Create molecules")
        nx.set_node_attributes(
            g,
            dict(
                map(
                    get_mol_enc,
                    ((i, g.nodes[i]["smiles"]) for i in (children.union(parts))),
                )
            ),
            "enc",
        )

        print("Filter invalid structures")
        children = [p for p in children if g.nodes[p]["enc"]]
        random.shuffle(children)
        children, children_test_only = train_test_split(
            children, test_size=self.part_split
        )

        parts = [p for p in parts if g.nodes[p]["enc"]]
        random.shuffle(parts)
        parts, parts_test_only = train_test_split(parts, test_size=self.part_split)

        has_parts = {n: g.nodes[n]["has_part"] for n in g.nodes}
        pickle.dump(
            g,
            open(os.path.join(self.processed_dir, self.processed_cache_names[0]), "wb"),
        )
        del g

        print("Transform into torch structure")

        kinds = ("train", "test", "validation")
        batches = {k: list() for k in kinds}
        batch_counts = {k: 0 for k in kinds}
        for l in children:
            pts = has_parts[l]
            for r in parts:
                # If there are no positive labels, move the datapoint to test set (where it has positive labels)
                if pts.intersection(parts):
                    if random.random() < self.train_split:
                        k = "train"
                    elif (
                        random.random() < self.train_split or batch_counts["validation"]
                    ):
                        k = "test"
                    else:
                        k = "validation"
                else:
                    k = "test"
                batches[k].append(PrePairData(l, r, float(r in pts)))
                if len(batches[k]) >= self.batch_size:
                    pickle.dump(
                        batches[k],
                        open(
                            os.path.join(
                                self.processed_dir, f"{k}.{batch_counts[k]}.pt"
                            ),
                            "wb",
                        ),
                    )
                    batch_counts[k] += 1
                    batches[k] = []

        k = k0 = "train"
        b = batches[k]
        if b:
            if not batch_counts["validation"]:
                k = "validation"
            pickle.dump(
                b,
                open(
                    os.path.join(self.processed_dir, f"{k}.{batch_counts[k]}.pt"), "wb"
                ),
            )
            del batches[k0]
            del b

        test_batch = batches["test"]
        batch_count = batch_counts["test"]
        for l, r in chain(
            ((l, r) for l in children for r in parts_test_only),
            ((l, r) for l in children_test_only for r in parts_test_only),
            ((l, r) for l in children_test_only for r in parts),
        ):
            test_batch.append(PrePairData(l, r, float(r in has_parts[l])))
            if len(test_batch) >= self.batch_size:
                pickle.dump(
                    test_batch,
                    open(
                        os.path.join(self.processed_dir, f"test.{batch_count}.pt"), "wb"
                    ),
                )
                batch_count += 1
                test_batch = []
        if test_batch:
            pickle.dump(
                test_batch,
                open(os.path.join(self.processed_dir, f"test.{batch_count}.pt"), "wb"),
            )

    @property
    def raw_file_names(self):
        return ["chebi.obo"]

    @property
    def processed_file_names(self):
        return ["train.0.pt", "test.0.pt", "validation.0.pt"]

    @property
    def processed_cache_names(self):
        return ["cache.pt"]

    def pass_parts(self, d: nx.DiGraph, root, parts=None):
        if parts is None:
            parts = set()
        parts = set(parts.union(d.nodes[root]["has_part"]))
        nx.set_node_attributes(d, {root: parts}, "has_part")
        for child in d.successors(root):
            self.pass_parts(d, child, set(parts))

    def extract_children(self, d: nx.DiGraph, root, part_cache):
        smiles = d.nodes[root]["smiles"]
        if smiles:
            yield root
        for child in d.successors(root):
            for r in self.extract_children(d, child, part_cache):
                yield r

    @staticmethod
    def extract_largest_index(path, kind):
        return (
            max(
                int(n[len(path + kind) + 2 : -len(".pt")])
                for n in glob.glob(os.path.join(path, f"{kind}.*.pt"))
            )
            + 1
        )

