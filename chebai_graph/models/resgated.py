from typing import Final, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import ELU
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN

from .base import GraphModelBase, GraphNetWrapper


class ResGatedGraphConvNetBase(GraphModelBase):
    """
    Residual Gated Graph Convolutional Network with edge attributes support.

    This model uses a stack of `ResGatedGraphConv` layers from PyTorch Geometric,
    allowing edge attributes as part of message passing. A final projection layer maps
    to the hidden length specified for downstream graph prediction tasks.
    """

    NAME = "ResGatedGraphConvNetBase"

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the ResGatedGraphConvNetBase.

        Args:
            config (dict): Configuration dictionary with keys:
                - 'hidden_length' (int): Intermediate feature length used in GNN layers.
                - Other parameters inherited from GraphModelBase.
            **kwargs: Additional keyword arguments passed to GraphModelBase.
        """
        super().__init__(config=config, **kwargs)

        self.activation = F.elu
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            tgnn.ResGatedGraphConv(
                self.n_node_properties,
                self.hidden_channels,
                # dropout=self.dropout,
                edge_dim=self.n_bond_properties,
            )
        )

        for _ in range(self.num_layers - 2):
            # Intermediate layers
            self.convs.append(
                tgnn.ResGatedGraphConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    edge_dim=self.n_bond_properties,
                )
            )

        # Final projection layer to hidden dimension
        self.final_conv = tgnn.ResGatedGraphConv(
            self.hidden_channels, self.out_channels, edge_dim=self.n_bond_properties
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through residual gated GNN layers.

        Args:
            batch (dict): A batch containing:
                - 'features': A list with a `GraphData` instance as the first element.

        Returns:
            torch.Tensor: Node-level embeddings of shape [num_nodes, hidden_length].
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        x = graph_data.x.float()  # Atom features

        for conv in self.convs:
            assert isinstance(conv, tgnn.ResGatedGraphConv)
            x = self.activation(
                conv(x, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr)
            )

        x = self.activation(
            self.final_conv(
                x, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr
            )
        )

        return x


class ResGatedGraphPred(GraphNetWrapper):
    """
    Residual Gated GNN for Graph Prediction.

    Uses `ResGatedGraphConvNetBase` as the GNN encoder to compute node embeddings.
    """

    NAME = "ResGatedGraphPred"

    def _get_gnn(self, config: dict) -> ResGatedGraphConvNetBase:
        """
        Instantiate the residual gated GNN backbone.

        Args:
            config (dict): Model configuration.

        Returns:
            ResGatedGraphConvNetBase: The GNN encoder.
        """
        return ResGatedGraphConvNetBase(config=config)


class ResGatedModel(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(
        self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs
    ) -> MessagePassing:
        return tgnn.ResGatedGraphConv(
            in_channels,
            out_channels,
            **kwargs,
        )


class ResGatedPyG(GraphModelBase):
    """
    Graph Attention Network (GAT) base module for graph convolution.

    Uses PyTorch Geometric's `GAT` implementation to process atomic node features
    and bond edge attributes through multiple attention heads and layers.
    """

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the GATGraphConvNetBase.

        Args:
            config (dict): Model configuration containing:
                - 'heads' (int): Number of attention heads.
                - 'v2' (bool): Whether to use the GATv2 variant.
                - Other required GraphModelBase parameters.
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(config=config, **kwargs)
        self.activation = ELU()  # Instantiate ELU once for reuse.
        self.gat = ResGatedModel(
            in_channels=self.n_node_properties,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            edge_dim=self.n_bond_properties,
            act=self.activation,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through the GAT network.

        Processes atomic node features and edge attributes, and applies
        an ELU activation to the output.

        Args:
            batch (dict): Input batch containing:
                - 'features': A list with a `GraphData` object as its first element.

        Returns:
            torch.Tensor: Node embeddings after GAT and activation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        out = self.gat(
            x=graph_data.x.float(),
            edge_index=graph_data.edge_index.long(),
            edge_attr=graph_data.edge_attr,
        )

        return self.activation(out)


class ResGatedGraphPredPyG(GraphNetWrapper):
    """
    Residual Gated GNN for Graph Prediction.

    Uses `ResGatedGraphConvNetBase` as the GNN encoder to compute node embeddings.
    """

    NAME = "ResGatedGraphPred"

    def _get_gnn(self, config: dict) -> ResGatedPyG:
        """
        Instantiate the residual gated GNN backbone.

        Args:
            config (dict): Model configuration.

        Returns:
            ResGatedGraphConvNetBase: The GNN encoder.
        """
        return ResGatedPyG(config=config)
