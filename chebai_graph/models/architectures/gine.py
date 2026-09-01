from typing import Any, Final

from torch import Tensor
from torch.nn import ELU
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN

from .base import GraphModelBase, GraphNetWrapper


class GINEModel(BasicGNN):
    """
    A GIN-based GNN model based on PyG's BasicGNN, using GINEConv layers so that
    edge (bond) features are incorporated into the message-passing step.

    See:
        - https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GINEConv.html
        - https://arxiv.org/abs/1810.00826 (GIN)
        - https://arxiv.org/abs/1905.12265 (GINE / edge-feature extension)
        - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py
        - https://github.com/pyg-team/pytorch_geometric/issues/1311

    Attributes:
        supports_edge_weight (bool): Indicates edge weights are not supported.
        supports_edge_attr (bool): Indicates edge attributes are supported.
        supports_norm_batch (bool): Indicates if batch normalization is supported.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(
        self, in_channels: int | tuple[int, int], out_channels: int, **kwargs: Any
    ) -> MessagePassing:
        """
        Initializes a GINEConv layer.

        The inner network is a 2-layer MLP (Linear -> act -> Linear, no
        activation on the last layer), matching both
        `torch_geometric.nn.models.GIN.init_conv` and the GIN paper's
        message-transform MLP. `edge_dim` (passed via **kwargs) lets
        GINEConv linearly project bond features onto the node feature
        space before adding them into the neighbor messages.

        Args:
            in_channels (int or Tuple[int, int]): Number of input channels.
            out_channels (int): Number of output channels.
            **kwargs: Additional keyword arguments for the convolution layer
                (e.g. `edge_dim`, `train_eps`).

        Returns:
            MessagePassing: A GINEConv layer instance.
        """
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return tgnn.GINEConv(mlp, **kwargs)


class GINEConvNetBase(GraphModelBase):
    """
    Base model class for applying GINEConv layers to graph-structured data.

    Based on:
        - Xu et al., "How Powerful are Graph Neural Networks?"
          (https://arxiv.org/abs/1810.00826)
        - Hu et al., "Strategies for Pre-training Graph Neural Networks"
          (https://arxiv.org/abs/1905.12265), reference implementation at
          https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
            Also supports an optional `train_eps` (bool, default True) key,
            which makes GINEConv's epsilon a learnable parameter, as
            recommended in the original GIN paper.
        **kwargs: Additional keyword arguments for parent class.
    """

    def __init__(self, config: dict[str, Any], **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.activation = ELU()  # Instantiate ELU once for reuse.
        self.train_eps = bool(config.get("train_eps", True))

        self.gine: BasicGNN = GINEModel(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            train_eps=self.train_eps,
            act=self.activation,
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """
        Forward pass of the model.

        Args:
            batch (dict): A batch containing graph input features under the key "features".

        Returns:
            Tensor: The output node-level embeddings after the final activation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData), "Expected GraphData instance"

        out = self.gine(
            x=graph_data.x.float(),
            edge_index=graph_data.edge_index.long(),
            edge_attr=graph_data.edge_attr,
        )

        return self.activation(out)


class GINEGraphPred(GraphNetWrapper):
    """
    Wrapper for graph-level prediction using GINEConvNetBase.

    This class instantiates the core GNN model using the provided config.
    Graph-level pooling (scatter-add over nodes) and the final linear
    prediction head are handled by `GraphNetWrapper`, not here.
    """

    NAME = "GINEGraphPred"

    def _get_gnn(self, config: dict[str, Any]) -> GINEConvNetBase:
        """
        Returns the core GINE GNN model.

        Args:
            config (dict): Configuration dictionary for the GNN model.

        Returns:
            GINEConvNetBase: The core graph convolutional network.
        """
        return GINEConvNetBase(config=config)
