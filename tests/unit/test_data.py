import torch
from torch_geometric.data import Data


class MoleculeGraph:
    """Dummy graph of Aspirin with node and edge features"""

    def get_aspirin_graph(self):
        """
        Aspirin -> CC(=O)OC1=CC=CC=C1C(=O)O ; CHEBI:15365

        Node labels (atom indices):
        O2              C5———C6
          \            /       \
           C1———O3———C4         C7
          /            \       /
        C0              C9———C8
                       /
                     C10
                    /   \
                 O12     O11
        """

        # --- Node features: atomic numbers (C=6, O=8) ---
        # Shape of x : num_nodes x num_of_node_features
        x = torch.tensor(
            [
                [6],  # C0  - This feature belongs to atom with atom `0` in edge_index
                [6],  # C1  - This feature belongs to atom with atom `1` in edge_index
                [8],  # O2  - This feature belongs to atom with atom `2` in edge_index
                [8],  # O3  - This feature belongs to atom with atom `3` in edge_index
                [6],  # C4  - This feature belongs to atom with atom `4` in edge_index
                [6],  # C5  - This feature belongs to atom with atom `5` in edge_index
                [6],  # C6  - This feature belongs to atom with atom `6` in edge_index
                [6],  # C7  - This feature belongs to atom with atom `7` in edge_index
                [6],  # C8  - This feature belongs to atom with atom `8` in edge_index
                [6],  # C9  - This feature belongs to atom with atom `9` in edge_index
                [6],  # C10 - This feature belongs to atom with atom `10` in edge_index
                [8],  # O11 - This feature belongs to atom with atom `11` in edge_index
                [8],  # O12 - This feature belongs to atom with atom `12` in edge_index
            ],
            dtype=torch.float,
        )

        # --- Edge list (bidirectional) ---
        # Shape of edge_index for undirected graph: 2 x num_of_edges
        edge_index = (
            torch.tensor(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                    [1, 3],
                    [3, 1],
                    [3, 4],
                    [4, 3],
                    [4, 5],
                    [5, 4],
                    [5, 6],
                    [6, 5],
                    [6, 7],
                    [7, 6],
                    [7, 8],
                    [8, 7],
                    [8, 9],
                    [9, 8],
                    [4, 9],
                    [9, 4],
                    [9, 10],
                    [10, 9],
                    [10, 11],
                    [11, 10],
                    [10, 12],
                    [12, 10],
                ],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # --- Dummy edge features: bond type (single=1, double=2, ester=3) ---
        # Using all single bonds for simplicity (except C=O as double bonds)
        # Shape of edge_attr: num_of_edges x num_of_edges_features
        edge_attr = torch.tensor(
            [
                [1],
                [1],  # C0 - C1  # This two features to two first bond in
                [2],
                [2],  # C1 = O2 (double bond)
                [1],
                [1],  # C1 - O3
                [1],
                [1],  # O3 - C4
                [1],
                [1],  # C4 - C5
                [1],
                [1],  # C5 - C6
                [1],
                [1],  # C6 - C7
                [1],
                [1],  # C7 - C8
                [1],
                [1],  # C8 - C9
                [1],
                [1],  # C4 - C9 (ring closure)
                [1],
                [1],  # C9 - C10
                [2],
                [2],  # C10 = O11 (carboxylic acid)
                [1],
                [1],  # C10 - O12 (hydroxyl)
            ],
            dtype=torch.float,
        )

        # Create graph data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
