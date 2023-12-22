import logging
import typing

from torch import nn
from torch_geometric import nn as tgnn
from torch_scatter import scatter_add, scatter_mean
import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIGraphNet(ChebaiBaseNet):
    NAME = "GNN"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.in_length = config["in_length"]
        self.hidden_length = config["hidden_length"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 3
        self.n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )

        self.embedding = torch.nn.Embedding(800, self.in_length)

        self.convs = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            if i == 0:
                self.convs.append(
                    tgnn.GraphConv(
                        self.in_length, self.in_length, dropout=self.dropout_rate
                    )
                )
            self.convs.append(tgnn.GraphConv(self.in_length, self.in_length))
        self.final_conv = tgnn.GraphConv(self.in_length, self.hidden_length)

        self.activation = F.elu

        self.linear_layers = torch.nn.ModuleList([])
        for _ in range(self.n_linear_layers - 1):
            self.linear_layers.append(nn.Linear(self.hidden_length, self.hidden_length))
        self.final_layer = nn.Linear(self.hidden_length, self.out_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = graph_data.x
        a = self.embedding(a)

        for conv in self.convs:
            a = self.activation(conv(a, graph_data.edge_index.long()))
        a = self.activation(self.final_conv(a, graph_data.edge_index.long()))
        a = self.dropout(a)
        a = scatter_add(a, graph_data.batch, dim=0)

        for lin in self.linear_layers:
            a = self.activation(lin(a))
        a = self.final_layer(a)
        return a


class ResGatedGraphConvNet(JCIGraphNet):
    """GNN that supports edge attributes"""

    NAME = "ResGatedGraphConvNet"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.in_length = config["in_length"]
        self.hidden_length = config["hidden_length"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 3
        self.n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )
        self.n_atom_properties = config["n_atom_properties"]
        self.n_bond_properties = config["n_bond_properties"]

        self.activation = F.elu
        self.dropout = nn.Dropout(self.dropout_rate)

        self.convs = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            if i == 0:
                self.convs.append(
                    tgnn.ResGatedGraphConv(
                        self.n_atom_properties,
                        self.in_length,
                        dropout=self.dropout_rate,
                        edge_dim=self.n_bond_properties,
                    )
                )
            self.convs.append(
                tgnn.ResGatedGraphConv(
                    self.in_length, self.in_length, edge_dim=self.n_bond_properties
                )
            )
        self.final_conv = tgnn.ResGatedGraphConv(
            self.in_length, self.hidden_length, edge_dim=self.n_bond_properties
        )

        self.linear_layers = torch.nn.ModuleList([])
        for _ in range(self.n_linear_layers - 1):
            self.linear_layers.append(nn.Linear(self.hidden_length, self.hidden_length))
        self.final_layer = nn.Linear(self.hidden_length, self.out_dim)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = graph_data.x.float()
        # a = self.embedding(a)

        for conv in self.convs:
            assert isinstance(conv, tgnn.ResGatedGraphConv)
            a = self.activation(
                conv(a, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr)
            )
        a = self.activation(
            self.final_conv(
                a, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr
            )
        )
        a = self.dropout(a)
        a = scatter_add(a, graph_data.batch, dim=0)

        for lin in self.linear_layers:
            a = self.activation(lin(a))
        a = self.final_layer(a)
        return a


class JCIGraphAttentionNet(ChebaiBaseNet):
    NAME = "AGNN"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        in_length = config["in_length"]
        hidden_length = config["hidden_length"]
        dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 5
        n_heads = config["n_heads"] if "n_heads" in config else 5
        self.n_lin_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )

        self.embedding = torch.nn.Embedding(800, in_length)
        self.edge_embedding = torch.nn.Embedding(4, in_length)
        in_length = in_length + 10

        self.convs = torch.nn.ModuleList([])
        for _ in range(self.n_conv_layers - 1):
            layer = tgnn.GATConv(
                in_length, in_length, n_heads, concat=False, add_self_loops=True
            )
            self.convs.append(layer)
        self.final_conv = tgnn.GATConv(
            in_length, hidden_length, n_heads, concat=False, add_self_loops=True
        )

        self.activation = F.leaky_relu

        self.linear_layers = torch.nn.ModuleList([])
        for _ in range(self.n_lin_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_length, hidden_length))
        self.final_layer = nn.Linear(hidden_length, self.out_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        edge_index = graph_data.edge_index.long().to(self.device)
        a = self.embedding(graph_data.x)
        a = self.dropout(a)
        a = torch.cat([a, torch.rand((*a.shape[:-1], 10)).to(self.device)], dim=1)

        for i, layer in enumerate(self.convs):
            assert isinstance(layer, tgnn.GATConv)
            a = self.activation(layer(a, edge_index))
            if i == 0:
                a = self.dropout(a)
        a = self.activation(self.final_conv(a, edge_index))

        a = self.dropout(a)
        a = scatter_mean(a, graph_data.batch, dim=0)
        for i, layer in enumerate(self.linear_layers):
            a = self.activation(layer(a))
        a = self.final_layer(a)
        return a
