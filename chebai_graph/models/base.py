from abc import ABC, abstractmethod
from typing import Optional

import torch
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.structures import XYData
from torch_geometric.data import Data as GraphData
from torch_scatter import scatter_add


class GraphBaseNet(ChebaiBaseNet, ABC):
    """
    Base class for graph-based prediction networks.
    """

    def _get_prediction_and_labels(
        self, data: XYData, labels: torch.Tensor, output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sigmoid activation to outputs and return processed labels.

        Args:
            data (XYData): Input batch data.
            labels (torch.Tensor): Ground-truth labels.
            output (torch.Tensor): Raw model output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (predictions, labels).
        """
        return torch.sigmoid(output), labels.int()

    def _process_labels_in_batch(self, batch: XYData) -> torch.Tensor | None:
        """
        Process labels from XYData batch.

        Returns:
            torch.Tensor | None: Processed labels if present, else None.
        """
        return batch.y.float() if batch.y is not None else None


class GraphModelBase(torch.nn.Module, ABC):
    """
    Abstract base class for graph models with configurable architecture.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Initialize model hyperparameters from configuration.

        Args:
            config (dict): Configuration dictionary with keys:
                - 'num_layers'
                - 'in_channels'
                - 'hidden_channels'
                - 'out_channels'
                - 'edge_dim'
                - 'dropout'
            **kwargs: Additional keyword arguments for torch.nn.Module.
        """
        super().__init__(**kwargs)
        self.num_layers = int(config["num_layers"])
        assert self.num_layers > 1, "Need atleast two convolution layers"
        self.in_channels = int(config["in_channels"])  # number of node/atom properties
        self.hidden_channels = int(config["hidden_channels"])
        self.out_channels = int(config["out_channels"])
        self.edge_dim = int(config["edge_dim"])  # number of bond properties
        self.dropout = float(config["dropout"])


class GraphNetWrapper(GraphBaseNet, ABC):
    """
    Base wrapper class for GNNs with linear layers for property prediction.
    """

    def __init__(
        self,
        config: dict,
        n_linear_layers: int,
        n_molecule_properties: Optional[int] = 0,
        use_batch_norm: bool = False,
        **kwargs,
    ):
        """
        Args:
            config (dict): Model configuration.
            n_linear_layers (int): Number of linear layers.
            n_molecule_properties (int): Number of molecular-level features.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.gnn = self._get_gnn(config)
        gnn_out_dim = int(config["out_channels"])
        self.activation = torch.nn.ELU
        self.lin_input_dim = self._get_lin_seq_input_dim(
            gnn_out_dim=gnn_out_dim,
            n_molecule_properties=(
                n_molecule_properties if n_molecule_properties is not None else 0
            ),
        )
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(self.lin_input_dim)

        lin_hidden_dim = kwargs.get("lin_hidden_dim", gnn_out_dim)
        self.lin_sequential: torch.nn.Sequential = self._get_linear_module_list(
            n_linear_layers=n_linear_layers,
            in_dim=self.lin_input_dim,
            hidden_dim=lin_hidden_dim,
            out_dim=self.out_dim,
        )

    @abstractmethod
    def _get_gnn(self, config: dict) -> torch.nn.Module:
        """
        Create the graph neural network.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            torch.nn.Module: Instantiated GNN module.
        """
        pass

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Compute input dimension for the linear layers.

        Args:
            gnn_out_dim (int): Output dimension of GNN.
            n_molecule_properties (int): Number of molecule-level features.

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties

    def _get_linear_module_list(
        self,
        n_linear_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> torch.nn.Sequential:
        """
        Construct a sequential module of linear layers.

        Args:
            n_linear_layers (int): Number of linear layers.
            in_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            out_dim (int): Output dimension.

        Returns:
            torch.nn.Sequential: Linear layers with activations.
        """
        if n_linear_layers < 1:
            raise ValueError("n_linear_layers must be at least 1")

        layers = []
        if n_linear_layers == 1:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        else:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation())
            for _ in range(n_linear_layers - 2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation())
            layers.append(torch.nn.Linear(hidden_dim, out_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through GNN, pooling and linear layers.

        Args:
            batch (dict): Input batch with graph features.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        graph_data.to(self.device)
        assert isinstance(graph_data, GraphData)
        a = self.gnn(batch)
        a = scatter_add(a, graph_data.batch, dim=0)
        a = torch.cat([a, graph_data.molecule_attr], dim=1)
        if self.use_batch_norm:
            a = self.batch_norm(a)
        return self.lin_sequential(a)


class AugmentedNodePoolingNet(GraphNetWrapper, ABC):
    """
    A pooling network that aggregates:
    - Atom node embeddings
    - Molecular attributes (if provided else skipped)
    - Augmented node embeddings (FG nodes and graph node)

    The concatenated vector is then passed through a linear sequential block.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Compute the input dimension for the final linear sequential block.

        Includes:
        - Atom embeddings
        - Molecular attributes (if any)
        - Augmented node embeddings

        Args:
            gnn_out_dim (int): Dimension of the GNN output per node.
            n_molecule_properties (int): Number of molecule-level attributes.

        Returns:
            int: Total input dimension for the linear sequential block.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for pooling node embeddings.

        Steps:
        1. Identify atom nodes and augmented nodes.
        2. Compute node embeddings with the GNN.
        3. Aggregate embeddings for atoms and augmented nodes separately using scatter add.
        4. Concatenate:
            - Atom nodes vector
            - Molecular attributes
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
        graph_vector = torch.cat(
            [atoms_vec, graph_data.molecule_attr, aug_nodes_vec], dim=1
        )

        return self.lin_sequential(graph_vector)


class FGNodePoolingNet(GraphNetWrapper, ABC):
    """
    A pooling network that pools node embeddings by aggregating:
    - All non-functional-group nodes' embeddings (atom and graph node)
    - Molecular attributes
    - Functional group node embeddings

    The concatenated vector is then passed through a linear sequential block.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Computes the input dimension for the final linear sequential block.

        Combines:
        - All nodes embeddings except functional group nodes
        - Molecular attributes
        - Functional group node embeddings

        Args:
            gnn_out_dim (int): Dimension of the GNN output per node.
            n_molecule_properties (int): Number of molecule-level attributes.

        Returns:
            int: Total input dimension for the linear sequential block.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for pooling node embeddings.

        Steps:
        1. Identify graph, atom, and functional group nodes.
        2. Aggregate embeddings for remaining nodes and functional group nodes separately.
        3. Concatenate:
            - Remaining nodes vector
            - Molecular attributes
            - Functional group nodes vector
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
        is_remaining_node = ~is_fg_node

        node_embeddings = self.gnn(batch)

        remaining_nodes_embedding = node_embeddings[is_remaining_node]
        remaining_nodes_batch = graph_data.batch[is_remaining_node]

        fg_nodes_embeddings = node_embeddings[is_fg_node]
        fg_nodes_batch = graph_data.batch[is_fg_node]

        # Scatter add separately
        remaining_nodes_vec = scatter_add(
            remaining_nodes_embedding, remaining_nodes_batch, dim=0
        )
        fg_nodes_vec = scatter_add(fg_nodes_embeddings, fg_nodes_batch, dim=0)

        # Concatenate all
        graph_vector = torch.cat(
            [remaining_nodes_vec, graph_data.molecule_attr, fg_nodes_vec], dim=1
        )

        return self.lin_sequential(graph_vector)


class GraphNodeFGNodePoolingNet(GraphNetWrapper, ABC):
    """
    A pooling network that pools node embeddings by aggregating:
    - Atom nodes
    - Molecular attributes
    - Functional group node embeddings
    - Graph node embeddings

    The concatenated vector is then passed through a linear sequential block.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Computes the input dimension for the final linear sequential block.

        Combines:
        - Atom embeddings
        - Molecular attributes
        - Functional group node embeddings
        - Graph node embeddings

        Args:
            gnn_out_dim (int): Dimension of the GNN output per node.
            n_molecule_properties (int): Number of molecule-level attributes.

        Returns:
            int: Total input dimension for the linear sequential block.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for pooling node embeddings.

        Steps:
        1. Identify graph, atom, and functional group nodes.
        2. Aggregate embeddings for each node type separately.
        3. Concatenate:
            - Atom nodes vector
            - Molecular attributes
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
        graph_vector = torch.cat(
            [atoms_vec, graph_data.molecule_attr, fg_nodes_vec, graph_node_vec], dim=1
        )

        return self.lin_sequential(graph_vector)


class GraphNodePoolingNet(GraphNetWrapper, ABC):
    """
    Pooling using non-graph nodes and graph node embeddings.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Return input dimension including graph node embeddings.
            - all_nodes_embeddings_except_graph_node + molecule attributes + graph_node_embedding

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass with separate pooling for graph and other nodes.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_not_graph_node = ~is_graph_node

        node_embeddings = self.gnn(batch)
        graph_node_embedding = node_embeddings[is_graph_node]
        graph_node_batch = graph_data.batch[is_graph_node]

        remaining_nodes_embedding = node_embeddings[is_not_graph_node]
        remaining_nodes_batch = graph_data.batch[is_not_graph_node]

        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        remaining_nodes_vec = scatter_add(
            remaining_nodes_embedding, remaining_nodes_batch, dim=0
        )

        graph_vector = torch.cat(
            [remaining_nodes_vec, graph_data.molecule_attr, graph_node_vec], dim=1
        )
        return self.lin_sequential(graph_vector)


class FGNodePoolingNoGraphNodeNet(GraphNetWrapper, ABC):
    """
    Graph Node not considered here in any computation.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Compute input dimension including:
        - atom_embeddings
        - molecule attributes
        - functional_group_node_embeddings

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass pooling atoms and functional group nodes.
        Graph nodes are ignored.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()
        is_fg_node = (~is_atom_node) & (~is_graph_node)

        node_embeddings = self.gnn(batch)

        atoms_embeddings = node_embeddings[is_atom_node]
        atoms_batch = graph_data.batch[is_atom_node]

        fg_nodes_embeddings = node_embeddings[is_fg_node]
        fg_nodes_batch = graph_data.batch[is_fg_node]

        atoms_vec = scatter_add(atoms_embeddings, atoms_batch, dim=0)
        fg_nodes_vec = scatter_add(fg_nodes_embeddings, fg_nodes_batch, dim=0)

        graph_vector = torch.cat(
            [atoms_vec, graph_data.molecule_attr, fg_nodes_vec], dim=1
        )

        return self.lin_sequential(graph_vector)


class GraphNodeNoFGNodePoolingNet(GraphNetWrapper, ABC):
    """
    Functional Group Nodes not considered here in any computation.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Compute input dimension including:
        - atom_embeddings
        - molecule attributes
        - graph_node_embeddings

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass pooling atoms and graph nodes.
        Functional group nodes are ignored.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()

        node_embeddings = self.gnn(batch)

        graph_node_embedding = node_embeddings[is_graph_node]
        graph_node_batch = graph_data.batch[is_graph_node]

        atoms_embeddings = node_embeddings[is_atom_node]
        atoms_batch = graph_data.batch[is_atom_node]

        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        atoms_vec = scatter_add(atoms_embeddings, atoms_batch, dim=0)

        graph_vector = torch.cat(
            [atoms_vec, graph_data.molecule_attr, graph_node_vec], dim=1
        )

        return self.lin_sequential(graph_vector)


class AugmentedOnlyPoolingNet(GraphNetWrapper, ABC):
    """
    Only augmented node embeddings are pooled.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Return input dimension using only augmented node embeddings.

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass pooling only augmented nodes.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        is_atom_node = graph_data.is_atom_node.bool()
        augmented_nodes_embeddings = self.gnn(batch)[~is_atom_node]
        augmented_nodes_batch = graph_data.batch[~is_atom_node]

        aug_nodes_vec = scatter_add(
            augmented_nodes_embeddings, augmented_nodes_batch, dim=0
        )
        graph_vector = torch.cat([aug_nodes_vec, graph_data.molecule_attr], dim=1)

        return self.lin_sequential(graph_vector)


class FGOnlyPoolingNet(GraphNetWrapper, ABC):
    """
    Only functional group node embeddings are pooled.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Return input dimension using only FG node embeddings.

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass pooling only functional group nodes.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()
        is_fg_node = (~is_atom_node) & (~is_graph_node)
        fg_nodes_embeddings = self.gnn(batch)[~is_fg_node]
        fg_nodes_batch = graph_data.batch[~is_fg_node]

        fg_nodes_vec = scatter_add(fg_nodes_embeddings, fg_nodes_batch, dim=0)
        graph_vector = torch.cat([fg_nodes_vec, graph_data.molecule_attr], dim=1)

        return self.lin_sequential(graph_vector)


class GraphNodeOnlyPoolingNet(GraphNetWrapper, ABC):
    """
    Only graph node embeddings are pooled.
    """

    def _get_lin_seq_input_dim(
        self, gnn_out_dim: int, n_molecule_properties: int
    ) -> int:
        """
        Return input dimension using only graph node embeddings.

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim + n_molecule_properties

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass pooling only graph nodes.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        is_graph_node = graph_data.is_graph_node.bool()

        graph_node_embedding = self.gnn(batch)[~is_graph_node]
        graph_node_batch = graph_data.batch[~is_graph_node]

        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        graph_vector = torch.cat([graph_node_vec, graph_data.molecule_attr], dim=1)

        return self.lin_sequential(graph_vector)
