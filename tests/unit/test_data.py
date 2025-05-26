import torch
from torch_geometric.data import Data


class MoleculeGraph:
    """Class representing molecular graph data."""

    def get_aspirin_graph(self):
        """
        Constructs and returns a PyTorch Geometric Data object representing the molecular graph of Aspirin.

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


        Returns:
            torch_geometric.data.Data: A Data object with attributes:
                - x (FloatTensor): Node feature matrix of shape (num_nodes, 1).
                - edge_index (LongTensor): Graph connectivity in COO format of shape (2, num_edges).
                - edge_attr (FloatTensor): Edge feature matrix of shape (num_edges, 1).

        Refer:
            For graph construction: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        """

        # --- Node features: atomic numbers (C=6, O=8) ---
        # Shape of x : num_nodes x num_of_node_features
        # fmt: off
        x = torch.tensor(
            [
                [6],  # C0  - This feature belongs to atom/node with 0 value in edge_index
                [6],  # C1  - This feature belongs to atom/node with 1 value in edge_index
                [8],  # O2  - This feature belongs to atom/node with 2 value in edge_index
                [8],  # O3  - This feature belongs to atom/node with 3 value in edge_index
                [6],  # C4  - This feature belongs to atom/node with 4 value in edge_index
                [6],  # C5  - This feature belongs to atom/node with 5 value in edge_index
                [6],  # C6  - This feature belongs to atom/node with 6 value in edge_index
                [6],  # C7  - This feature belongs to atom/node with 7 value in edge_index
                [6],  # C8  - This feature belongs to atom/node with 8 value in edge_index
                [6],  # C9  - This feature belongs to atom/node with 9 value in edge_index
                [6],  # C10 - This feature belongs to atom/node with 10 value in edge_index
                [8],  # O11 - This feature belongs to atom/node with 11 value in edge_index
                [8],  # O12 - This feature belongs to atom/node with 12 value in edge_index
            ],
            dtype=torch.float,
        )
        # fmt: on

        # --- Edge list (bidirectional) ---
        # Shape of edge_index for undirected graph: 2 x num_of_edges;  (2x26)
        # Generated using RDKIT 2024.9.6
        # fmt: off
        _edge_index = torch.tensor([
            [0, 1, 1, 3, 4, 5, 6, 7, 8, 9,  10, 10, 9],  # Start atoms (u)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 4]   # End atoms (v)
        ], dtype=torch.long)
        # fmt: on

        # Reverse the edges
        reversed_edge_index = _edge_index[[1, 0], :]

        # First all directed edges from source to target are placed,
        # then all directed edges from target to source are placed --- this is needed
        undirected_edge_index = torch.cat([_edge_index, reversed_edge_index], dim=1)

        # --- Dummy edge features ---
        # Shape of undirected_edge_attr: num_of_edges x num_of_edges_features (26 x 1)
        # fmt: off
        _edge_attr = torch.tensor([
            [1],  # C0 - C1, This two features belong to elements at index 0 in `edge_index`
            [2],  # C1 - C2, This two features belong to elements at index 1 in `edge_index`
            [2],  # C1 - O3, This two features belong to elements at index 2 in `edge_index`
            [2],  # O3 - C4, This two features belong to elements at index 3 in `edge_index`
            [1],  # C4 - C5, This two features belong to elements at index 4 in `edge_index`
            [1],  # C5 - C6, This two features belong to elements at index 5 in `edge_index`
            [1],  # C6 - C7, This two features belong to elements at index 6 in `edge_index`
            [1],  # C7 - C8, This two features belong to elements at index 7 in `edge_index`
            [1],  # C8 - C9, This two features belong to elements at index 8 in `edge_index`
            [1],  # C9 - C10, This two features belong to elements at index 9 in `edge_index`
            [1],  # C10 - O11, This two features belong to elements at index 10 in `edge_index`
            [1],  # C10 - O12, This two features belong to elements at index 11 in `edge_index`
            [1],  # C9 - C4, This two features belong to elements at index 12 in `edge_index`
        ], dtype=torch.float)
        # fmt: on

        # Alignement of edge attributes should in same order as of edge_index
        undirected_edge_attr = torch.cat([_edge_attr, _edge_attr], dim=0)

        # Create graph data object
        return Data(
            x=x, edge_index=undirected_edge_index, edge_attr=undirected_edge_attr
        )
