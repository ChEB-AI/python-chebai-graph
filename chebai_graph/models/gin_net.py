import typing

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data as GraphData

from chebai_graph.models.base import GraphModelBase, GraphNetWrapper
from torch_geometric import nn as tgnn


class AggregateMLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(AggregateMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = F.relu
        self.in_layer = torch.nn.Linear(in_channels, hidden_channels)
        self.out_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.activation(self.in_layer(x))
        x = self.activation(self.out_layer(x))
        return x


class GINEConvNet(GraphModelBase):
    """Based on https://arxiv.org/pdf/1810.00826.pdf and https://arxiv.org/abs/1905.12265"""

    NAME = "GINEConvNet"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.activation = F.elu

        self.convs = torch.nn.ModuleList([])
        # self.batch_norms = torch.nn.ModuleList([])
        for i in range(self.num_layers):
            self.convs.append(
                tgnn.GINEConv(
                    AggregateMLP(
                        self.in_channels, self.out_channels, self.hidden_channels
                    ),
                    edge_dim=self.edge_dim,
                )
            )
            # self.batch_norms.append(torch.nn.BatchNorm1d(out_length))

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = graph_data.x

        dropout_used = False  # only apply dropout after first layer
        conv_out = []
        for conv in self.convs:  # , norm in zip(self.convs, self.batch_norms):
            a = self.activation(
                conv(a, graph_data.edge_index.long(), graph_data.edge_attr)
            )
            if not dropout_used:
                a = self.dropout_layer(a)
                dropout_used = True
            # a = norm(a)
            a = scatter_add(a, graph_data.batch, dim=0)
            conv_out.append(a)

        a = torch.cat(conv_out, dim=1)

        return a


class GINEGraphPred(GraphNetWrapper):
    """
    Wrapper for graph-level prediction using GINEConvNet.

    This class instantiates the core GNN model using the provided config.
    """

    def _get_gnn(self, config: dict[str, typing.Any]) -> GINEConvNet:
        """
        Returns the core ResGated GNN model.

        Args:
            config (dict): Configuration dictionary for the GNN model.

        Returns:
            ResGatedGraphConvNetBase: The core graph convolutional network.
        """
        return GINEConvNet(config=config)
