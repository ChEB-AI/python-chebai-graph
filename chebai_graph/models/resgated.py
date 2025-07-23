from typing import Any, Final

from torch import Tensor
from torch.nn import ELU
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN

from .base import GraphModelBase, GraphNetWrapper


class ResGatedModel(BasicGNN):
    """
    A residual gated GNN model based on PyG's BasicGNN using ResGatedGraphConv layers.

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
        Initializes a ResGatedGraphConv layer.

        Args:
            in_channels (int or Tuple[int, int]): Number of input channels.
            out_channels (int): Number of output channels.
            **kwargs: Additional keyword arguments for the convolution layer.

        Returns:
            MessagePassing: A ResGatedGraphConv layer instance.
        """
        return tgnn.ResGatedGraphConv(
            in_channels,
            out_channels,
            **kwargs,
        )


class ResGatedGraphConvNetBase(GraphModelBase):
    """
    Base model class for applying ResGatedGraphConv layers to graph-structured data.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
        **kwargs: Additional keyword arguments for parent class.
    """

    def __init__(self, config: dict[str, Any], **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.activation = ELU()  # Instantiate ELU once for reuse.

        self.resgated: BasicGNN = ResGatedModel(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            edge_dim=self.edge_dim,
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

        out = self.resgated(
            x=graph_data.x.float(),
            edge_index=graph_data.edge_index.long(),
            edge_attr=graph_data.edge_attr,
        )

        return self.activation(out)


class ResGatedGraphPred(GraphNetWrapper):
    """
    Wrapper for graph-level prediction using ResGatedGraphConvNetBase.

    This class instantiates the core GNN model using the provided config.
    """

    def _get_gnn(self, config: dict[str, Any]) -> ResGatedGraphConvNetBase:
        """
        Returns the core ResGated GNN model.

        Args:
            config (dict): Configuration dictionary for the GNN model.

        Returns:
            ResGatedGraphConvNetBase: The core graph convolutional network.
        """
        return ResGatedGraphConvNetBase(config=config)
