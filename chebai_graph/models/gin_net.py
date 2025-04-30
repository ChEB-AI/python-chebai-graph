import typing

import torch
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_add

from chebai_graph.models.graph import GraphBaseNet


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


class GINEConvNet(GraphBaseNet):
    """Based on https://arxiv.org/pdf/1810.00826.pdf and https://arxiv.org/abs/1905.12265"""

    NAME = "GINEConvNet"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.n_atom_properties = int(config["n_atom_properties"])
        self.n_bond_properties = int(config["n_bond_properties"])
        self.hidden_size = config["hidden_size"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 5
        self.n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )

        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.activation = F.relu

        self.convs = torch.nn.ModuleList([])
        # self.batch_norms = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            in_length = self.n_atom_properties if i == 0 else self.hidden_size
            out_length = self.hidden_size
            self.convs.append(
                torch_geometric.nn.GINEConv(
                    AggregateMLP(in_length, out_length, self.hidden_size),
                    edge_dim=self.n_bond_properties,
                )
            )
            # self.batch_norms.append(torch.nn.BatchNorm1d(out_length))

        self.linear_layers = torch.nn.ModuleList([])
        for i in range(self.n_linear_layers):
            in_length = self.hidden_size
            out_length = (
                self.out_dim if i == self.n_linear_layers - 1 else self.hidden_size
            )
            self.linear_layers.append(torch.nn.Linear(in_length, out_length))

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, torch_geometric.data.Data)
        a = graph_data.x

        dropout_used = False  # only apply dropout after first layer
        conv_out = []
        for conv in self.convs:  # , norm in zip(self.convs, self.batch_norms):
            a = self.activation(
                conv(a, graph_data.edge_index.long(), graph_data.edge_attr)
            )
            if not dropout_used:
                a = self.dropout(a)
                dropout_used = True
            # a = norm(a)
            a = scatter_add(a, graph_data.batch, dim=0)
            conv_out.append(a)

        a = torch.cat(conv_out, dim=1)

        for i in range(self.n_linear_layers):
            if i != self.n_linear_layers - 1:
                a = self.activation(self.linear_layers[i](a))
            else:
                a = self.linear_layers[i](a)
            if i == 0:
                a = self.dropout(a)

        return a
