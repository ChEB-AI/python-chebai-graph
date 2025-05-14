import unittest

import torch
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.reader import GraphPropertyReader
from tests.unit.test_data import MoleculeGraph


class TestGraphPropertyReader(unittest.TestCase):
    """Unit tests for the GraphPropertyReader class, which converts SMILES strings to torch_geometric Data objects."""

    def setUp(self) -> None:
        """Initialize the reader and the reference molecule graph."""
        self.reader: GraphPropertyReader = GraphPropertyReader()
        self.molecule_graph: MoleculeGraph = MoleculeGraph()

    def test_read_data(self) -> None:
        """Test that the reader correctly parses a SMILES string into a graph and matches expected aspirin structure."""
        smiles: str = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

        data: GeomData = self.reader._read_data(smiles)  # noqa

        self.assertIsInstance(
            data,
            GeomData,
            msg="The output should be an instance of torch_geometric.data.Data.",
        )

        assert (
            data.edge_index.shape[0] == 2
        ), f"Expected edge_index to have shape [2, num_edges], but got shape {data.edge_index.shape}"

        assert (
            data.edge_index.shape[1] == data.edge_attr.shape[0]
        ), f"Mismatch between number of edges in edge_index ({data.edge_index.shape[1]}) and edge_attr ({data.edge_attr.shape[0]})"

        assert (
            len(set(data.edge_index[0].tolist())) == data.x.shape[0]
        ), f"Number of unique source nodes in edge_index ({len(set(data.edge_index[0].tolist()))}) does not match number of nodes in x ({data.x.shape[0]})"

        expected_data: GeomData = self.molecule_graph.get_aspirin_graph()
        self.assertTrue(
            torch.equal(data.edge_index, expected_data.edge_index),
            msg=(
                "edge_index tensors do not match.\n"
                f"Differences at indices: {(data.edge_index != expected_data.edge_index).nonzero()}.\n"
                f"Parsed edge_index:\n{data.edge_index}\nExpected edge_index:\n{expected_data.edge_index}"
                f"If fails in future, check if there is change in RDKIT version, the expected graph is generated with RDKIT 2024.9.6"
            ),
        )

        self.assertEqual(
            data.x.shape[0],
            expected_data.x.shape[0],
            msg=(
                "The number of atoms (nodes) in the parsed graph does not match the reference graph.\n"
                f"Parsed: {data.x.shape[0]}, Expected: {expected_data.x.shape[0]}"
            ),
        )

        self.assertEqual(
            data.edge_attr.shape[0],
            expected_data.edge_attr.shape[0],
            msg=(
                "The number of edge attributes does not match the expected value.\n"
                f"Parsed: {data.edge_attr.shape[0]}, Expected: {expected_data.edge_attr.shape[0]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
