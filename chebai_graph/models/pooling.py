from abc import ABC

import torch
from torch_geometric.data import Data as GraphData
from torch_scatter import scatter_add

from .architectures.base import GraphNetWrapper


class AugmentedNodePoolingNet(GraphNetWrapper, ABC):
    """
    A pooling network that aggregates:
    - Atom node embeddings
    - Augmented node embeddings (FG nodes and graph node)

    The concatenated vector is then passed through a linear sequential block.
    """

    def _get_lin_seq_input_dim(self, gnn_out_dim: int) -> int:
        """
        Compute the input dimension for the final linear sequential block.

        Includes:
        - Atom embeddings
        - Augmented node embeddings

        Args:
            gnn_out_dim (int): Dimension of the GNN output per node.
        Returns:
            int: Total input dimension for the linear sequential block.
        """
        return gnn_out_dim + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for pooling node embeddings.

        Steps:
        1. Identify atom nodes and augmented nodes.
        2. Compute node embeddings with the GNN.
        3. Aggregate embeddings for atoms and augmented nodes separately using scatter add.
        4. Concatenate:
            - Atom nodes vector
            - Augmented nodes vector
        5. Pass the concatenated vector through the linear sequential block.

        Args:
            batch (dict): Input batch containing graph data and features.

        Returns:
            torch.Tensor: Output tensor after pooling and linear transformation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        is_atom_node = graph_data.is_atom_node.bool()
        is_augmented_node = ~is_atom_node

        node_embeddings = self.gnn(batch)

        atoms_embeddings = node_embeddings[is_atom_node]
        atoms_batch = graph_data.batch[is_atom_node]

        augmented_nodes_embeddings = node_embeddings[is_augmented_node]
        augmented_nodes_batch = graph_data.batch[is_augmented_node]

        # Scatter add separately
        atoms_vec = scatter_add(atoms_embeddings, atoms_batch, dim=0)
        aug_nodes_vec = scatter_add(
            augmented_nodes_embeddings, augmented_nodes_batch, dim=0
        )

        # Concatenate all
        graph_vector = torch.cat([atoms_vec, aug_nodes_vec], dim=1)

        return self.lin_sequential(graph_vector)


class GraphNodeFGNodePoolingNet(GraphNetWrapper, ABC):
    """
    A pooling network that pools node embeddings by aggregating:
    - Atom nodes
    - Functional group node embeddings
    - Graph node embeddings

    The concatenated vector is then passed through a linear sequential block.
    """

    def _get_lin_seq_input_dim(self, gnn_out_dim: int) -> int:
        """
        Computes the input dimension for the final linear sequential block.

        Combines:
        - Atom embeddings
        - Functional group node embeddings
        - Graph node embeddings

        Args:
            gnn_out_dim (int): Dimension of the GNN output per node.

        Returns:
            int: Total input dimension for the linear sequential block.
        """
        return gnn_out_dim + gnn_out_dim + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for pooling node embeddings.

        Steps:
        1. Identify graph, atom, and functional group nodes.
        2. Aggregate embeddings for each node type separately.
        3. Concatenate:
            - Atom nodes vector
            - Functional group nodes vector
            - Graph node vector
        4. Pass the concatenated vector through the linear sequential block.

        Args:
            batch (dict): Batch containing graph data and features.

        Returns:
            torch.Tensor: Output tensor after pooling and linear transformation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()
        is_fg_node = (~is_atom_node) & (~is_graph_node)

        node_embeddings = self.gnn(batch)

        graph_node_embedding = node_embeddings[is_graph_node]
        graph_node_batch = graph_data.batch[is_graph_node]

        atoms_embeddings = node_embeddings[is_atom_node]
        atoms_batch = graph_data.batch[is_atom_node]

        fg_nodes_embeddings = node_embeddings[is_fg_node]
        fg_nodes_batch = graph_data.batch[is_fg_node]

        # Scatter add separately
        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        atoms_vec = scatter_add(atoms_embeddings, atoms_batch, dim=0)
        fg_nodes_vec = scatter_add(fg_nodes_embeddings, fg_nodes_batch, dim=0)

        # Concatenate all
        graph_vector = torch.cat([atoms_vec, fg_nodes_vec, graph_node_vec], dim=1)

        return self.lin_sequential(graph_vector)
