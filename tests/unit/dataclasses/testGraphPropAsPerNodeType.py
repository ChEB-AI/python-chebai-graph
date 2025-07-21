import unittest

import pandas as pd
import torch
from torch_geometric.data.data import Data as GeomData

from chebai_graph.preprocessing.datasets.chebi import ChEBI50_WFGE_WGN_AsPerNodeType
from chebai_graph.preprocessing.properties import (
    AtomNodeLevel,
    AugAtomCharge,
    AugAtomHybridization,
    BondLevel,
    IsFGAlkyl,
    IsHydrogenBondAcceptorFG,
    RDKit2DNormalized,
)


class TestGraphPropAsPerNodeType(unittest.TestCase):
    def test_merge_properties(self):
        num_nodes = 4
        dummy_x = torch.zeros((num_nodes, 0))  # Initial dummy x
        dummy_edge_index = torch.tensor([[0, 1], [1, 0]])
        dummy_edge_attr = torch.zeros((4, 0))  # 4 edges, each with 0 feature

        # Masks
        is_atom_node = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        is_graph_node = torch.tensor([0, 0, 0, 1], dtype=torch.bool)

        # GeomData
        geom_data = GeomData(
            x=dummy_x,
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr,
            is_atom_node=is_atom_node,
            is_graph_node=is_graph_node,
        )

        # Define properties
        # atom props = 5, fg_props = 4, graph_node_props = 6; max = 6
        all_node_prop = AtomNodeLevel(DummyEncoder(2))
        atom_prop = AugAtomCharge(DummyEncoder(1))
        atom_prop_2 = AugAtomHybridization(DummyEncoder(2))
        fg_prop = IsFGAlkyl(DummyEncoder(1))
        fg_prop_2 = IsHydrogenBondAcceptorFG(DummyEncoder(1))
        mol_prop = RDKit2DNormalized(DummyEncoder(4))
        bond_prop = BondLevel(DummyEncoder(2))

        properties = [
            atom_prop,
            atom_prop_2,
            fg_prop,
            fg_prop_2,
            bond_prop,
            all_node_prop,
            mol_prop,
        ]

        merger = ChEBI50_WFGE_WGN_AsPerNodeType(properties)

        # Define encoded property values for the row
        row = pd.Series(
            {
                "features": geom_data,
                "AtomNodeLevel": torch.tensor(
                    [
                        [1.0, 0.0],  # atom
                        [0.0, 1.0],  # fg
                        [0.0, 0.0],  # atom
                        [0.0, 1.0],  # graph
                    ]
                ),
                "AtomCharge": torch.tensor(
                    [
                        [1.0],  # atom
                        [2.0],  # fg
                        [6.0],  # atom
                        [3.0],  # graph
                    ]
                ),
                "AtomHybridization": torch.tensor(
                    [
                        [11.0, 9.0],  # atom
                        [7.0, 3.0],  # fg
                        [3.0, 1.0],  # atom
                        [7.0, 43.0],  # graph
                    ]
                ),
                "IsFGAlkyl": torch.tensor(
                    [
                        [5.0],  # atom
                        [55.0],  # fg
                        [13.0],  # atom
                        [14.0],  # graph
                    ]
                ),  # values for fg at idx 1
                "IsHydrogenBondAcceptorFG": torch.tensor(
                    [
                        [3.0],  # atom
                        [5.0],  # fg
                        [17.0],  # atom
                        [15.0],  # graph
                    ]
                ),
                "RDKit2DNormalized": torch.tensor(
                    [
                        [65.0, 23.0, 6.0, 8.0],  # atom
                        [2.0, 8.0, 55.0, 77.0],  # fg
                        [3.0, 51.0, 55.0, 3.0],  # atom
                        [33.0, 6.0, 10.0, 10.0],  # graph
                    ]
                ),  # only idx 3
                "BondLevel": torch.tensor(
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ]
                ),  # will be duplicated to 4x2
            }
        )
        expected_result = torch.tensor(
            [
                # all node   # ap1   # ap2         # 0 concat
                [1.0, 0.0] + [1.0] + [11.0, 9.0] + [0.0],  # atom node
                # all node   # fg1   # fg2     # 0 concat
                [0.0, 1.0] + [55.0] + [5.0] + [0.0, 0.0],  # fg  node
                # all node   # ap1   # ap2        # 0 concat
                [0.0, 0.0] + [6.0] + [3.0, 1.0] + [0.0],  # atom node
                # all node   # mol props
                [0.0, 1.0] + [33.0, 6.0, 10.0, 10.0],  # graph node
            ]
        )

        result = merger._merge_props_into_base(row, max_len_node_properties=6)
        self.assertTrue(torch.equal(result.x, expected_result))


class DummyEncoder:
    def __init__(self, length):
        self.length = length

    @property
    def name(self):
        return self.__class__.__name__.replace("DummyEncoder", "")

    def get_encoding_length(self):
        return self.length


if __name__ == "__main__":
    unittest.main()
