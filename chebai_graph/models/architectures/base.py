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
        self.dropout = float(config.get("dropout", 0.0))
        print("----------Dropout set to: ", self.dropout)


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
        pretrained_checkpoint: str | None = None,
        **kwargs,
    ):
        """
        Args:
            config (dict): Model configuration.
            n_linear_layers (int): Number of linear layers.
            **kwargs: Additional arguments.
        """
        lin_hidden_dim = kwargs.pop("lin_hidden_dim", None)
        super().__init__(**kwargs)
        self.gnn = self._get_gnn(config)
        if pretrained_checkpoint is not None:
            self._load_pretrained_gnn(pretrained_checkpoint)
        gnn_out_dim = int(config["out_channels"])
        self.activation = torch.nn.ELU
        self.lin_input_dim = self._get_lin_seq_input_dim(
            gnn_out_dim=gnn_out_dim,
        )
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(self.lin_input_dim)
        if lin_hidden_dim is None:
            lin_hidden_dim = gnn_out_dim // 2

        self.lin_sequential: torch.nn.Sequential = self._get_linear_module_list(
            n_linear_layers=n_linear_layers,
            in_dim=self.lin_input_dim,
            hidden_dim=lin_hidden_dim,
            out_dim=self.out_dim,
        )

    def _load_pretrained_gnn(self, checkpoint_path: str) -> None:
        """Load encoder weights while leaving the task-specific head untouched."""
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=self.device
        )
        state_dict = checkpoint.get("state_dict", checkpoint)

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} does not contain a state dictionary."
            )

        gnn_state = self.gnn.state_dict()
        gnn_keys = set(gnn_state)
        matching_keys = set(state_dict) & gnn_keys
        if not matching_keys:
            for prefix in ("gnn.", "model.gnn."):
                prefixed_keys = {
                    key.removeprefix(prefix): value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                if set(prefixed_keys) & gnn_keys:
                    state_dict = prefixed_keys
                    matching_keys = set(state_dict) & gnn_keys
                    break

        if not matching_keys:
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} contains no weights matching the GNN."
            )

        pretrained_gnn_state = {
            key: value for key, value in state_dict.items() if key in gnn_keys
        }
        skipped_edge_weights = []
        # for key in list(pretrained_gnn_state):
        #     if (
        #         key.endswith(".lin_edge.weight")
        #         and pretrained_gnn_state[key].shape != gnn_state[key].shape
        #     ):
        #         skipped_edge_weights.append(key)
        #         del pretrained_gnn_state[key]

        load_result = self.gnn.load_state_dict(pretrained_gnn_state, strict=False)
        allowed_missing = set(skipped_edge_weights)
        unexpected_keys = set(load_result.unexpected_keys)
        missing_keys = set(load_result.missing_keys) - allowed_missing
        if unexpected_keys or missing_keys:
            raise RuntimeError(
                "Pretrained GNN checkpoint is incompatible: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(unexpected_keys)}"
            )

        if skipped_edge_weights:
            print(
                "Skipped incompatible pretrained edge weights: "
                + ", ".join(sorted(skipped_edge_weights))
            )
        print(f"Loaded pretrained GNN weights from {checkpoint_path!r} successfully.")

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

    def configure_optimizers(self, **kwargs) -> torch.optim.Optimizer:
        optimizer_kwargs = dict(self.optimizer_kwargs)

        default_lr = optimizer_kwargs.pop("lr", None)
        gnn_lr = optimizer_kwargs.pop("lr_gnn", default_lr)
        linear_lr = optimizer_kwargs.pop("lr_linear", default_lr)

        if gnn_lr is None or linear_lr is None:
            raise ValueError("Set lr or both lr_gnn and lr_linear")

        linear_params = list(self.lin_sequential.parameters())
        if self.use_batch_norm:
            linear_params.extend(self.batch_norm.parameters())

        return torch.optim.Adamax(
            [
                {"params": self.gnn.parameters(), "lr": gnn_lr},
                {"params": linear_params, "lr": linear_lr},
            ],
            **optimizer_kwargs,
        )
