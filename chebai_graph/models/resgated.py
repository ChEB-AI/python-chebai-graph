import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData

from .base import GraphBaseNet


class ResGatedGraphConvNetBase(GraphBaseNet):
    """GNN that supports edge attributes"""

    NAME = "ResGatedGraphConvNetBase"

    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)

        self.in_length = config["in_length"]
        self.hidden_length = config["hidden_length"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"]
        self.n_atom_properties = int(config["n_atom_properties"])
        self.n_bond_properties = int(config["n_bond_properties"])

        self.activation = F.elu
        self.dropout = nn.Dropout(self.dropout_rate)

        self.convs = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            if i == 0:
                self.convs.append(
                    tgnn.ResGatedGraphConv(
                        self.n_atom_properties,
                        self.in_length,
                        # dropout=self.dropout_rate,
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
        return a
