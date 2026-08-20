from abc import ABC, abstractmethod

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
        valid_label_mask = data["loss_kwargs"]["valid_label_mask"]
        predictions = torch.sigmoid(output)
        labels = labels.int()

        if valid_label_mask is not None:
            labels[~valid_label_mask] = -1  # Mark invalid labels as -1
            # https://lightning.ai/docs/torchmetrics/stable/classification/auroc#multilabelauroc
            # -1 as we torchmetrics ignores -1 labels in multilabel metrics
            # metric = MultilabelAUROC(
            #    num_labels=labels.shape[1],
            #    ignore_index=-1,
            # )

        return predictions, labels

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
    Base wrapper class for GNNs with linear layers for graph classification
    with standard pooling .
    """

    def __init__(
        self,
        config: dict,
        n_linear_layers: int,
        use_batch_norm: bool = False,
        **kwargs,
    ):
        """
        Args:
            config (dict): Model configuration.
            n_linear_layers (int): Number of linear layers.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.gnn = self._get_gnn(config)
        gnn_out_dim = int(config["out_channels"])
        self.activation = torch.nn.ELU
        self.lin_input_dim = self._get_lin_seq_input_dim(
            gnn_out_dim=gnn_out_dim,
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

    def _get_lin_seq_input_dim(self, gnn_out_dim: int) -> int:
        """
        Compute input dimension for the linear layers.

        Args:
            gnn_out_dim (int): Output dimension of GNN.

        Returns:
            int: Total input dimension.
        """
        return gnn_out_dim

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
        if self.use_batch_norm:
            a = self.batch_norm(a)
        return self.lin_sequential(a)
