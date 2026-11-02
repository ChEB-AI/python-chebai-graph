import unittest

import torch
from torch_geometric.data import Data as GeomData

from chebai_graph.preprocessing.collate import GraphCollator


class TestGraphCollator(unittest.TestCase):
    @staticmethod
    def _graph():
        return GeomData(
            x=torch.tensor([[6]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    def test_mixed_missing_labels_do_not_enter_graph_collate(self):
        batch = GraphCollator()(
            [
                {
                    "features": self._graph(),
                    "labels": [1, None, 0, 0, 1],
                    "ident": "a",
                },
                {"features": self._graph(), "labels": None, "ident": "b"},
                {
                    "features": self._graph(),
                    "labels": [0, 1, None, 0, 1],
                    "ident": "c",
                },
            ]
        )

        expected_labels = torch.tensor(
            [
                [True, False, False, False, True],
                [False, True, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(batch.y, expected_labels))
        self.assertEqual(
            batch.additional_fields["loss_kwargs"]["non_null_labels"], [0, 2]
        )
        self.assertNotIn("y", batch.x[0])

    def test_custom_metadata_survives_repeated_collation(self):
        metadata = {"nodes": {"atom_nodes": [6]}, "edges": {}}
        graph = self._graph()
        graph.custom_metadata = metadata
        row = {"features": graph, "labels": None, "ident": "a"}

        first_batch = GraphCollator()([row])
        second_batch = GraphCollator()([row])

        self.assertEqual(first_batch.additional_fields["custom_metadata"], [[metadata]])
        self.assertEqual(
            second_batch.additional_fields["custom_metadata"], [[metadata]]
        )
        self.assertIn("custom_metadata", graph)


if __name__ == "__main__":
    unittest.main()
