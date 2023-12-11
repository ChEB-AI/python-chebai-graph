import logging
import sys
import typing

from torch import nn
from torch_geometric import nn as tgnn
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIGraphNet(ChebaiBaseNet):
    NAME = "GNN"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        in_length = config["in_length"]
        hidden_length = config["hidden_length"]
        dropout_rate = config["dropout_rate"]

        self.embedding = torch.nn.Embedding(800, in_length)

        self.conv1 = tgnn.GraphConv(in_length, in_length)
        self.conv2 = tgnn.GraphConv(in_length, in_length)
        self.conv3 = tgnn.GraphConv(in_length, hidden_length)

        self.output_net = nn.Sequential(
            nn.Linear(hidden_length, hidden_length),
            nn.ELU(),
            nn.Linear(hidden_length, hidden_length),
            nn.ELU(),
            nn.Linear(hidden_length, self.out_dim),
        )

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, batch):
        # TODO look into data, use edge attributes
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = self.embedding(graph_data.x)
        a = self.dropout(a)
        a = F.elu(self.conv1(a, graph_data.edge_index.long()))
        a = F.elu(self.conv2(a, graph_data.edge_index.long()))
        a = F.elu(self.conv3(a, graph_data.edge_index.long()))
        a = self.dropout(a)
        a = scatter_add(a, graph_data.batch, dim=0)
        return self.output_net(a)


class JCIGraphAttentionNet(ChebaiBaseNet):
    NAME = "AGNN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_length = kwargs.get("in_length")
        hidden_length = kwargs.get("hidden_length")
        self.embedding = torch.nn.Embedding(800, in_length)
        self.edge_embedding = torch.nn.Embedding(4, in_length)
        in_length = in_length + 10
        self.conv1 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, dropout=0.1, add_self_loops=True
        )
        self.conv2 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv3 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv4 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.conv5 = tgnn.GATConv(
            in_length, in_length, 5, concat=False, add_self_loops=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(in_length, hidden_length),
            nn.LeakyReLU(),
            nn.Linear(hidden_length, hidden_length),
            nn.LeakyReLU(),
            nn.Linear(hidden_length, 500),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        a = self.embedding(batch.x)
        a = self.dropout(a)
        a = torch.cat([a, torch.rand((*a.shape[:-1], 10)).to(self.device)], dim=1)
        a = F.leaky_relu(self.conv1(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv2(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv3(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv4(a, batch.edge_index.long()))
        a = F.leaky_relu(self.conv5(a, batch.edge_index.long()))
        a = self.dropout(a)
        a = scatter_mean(a, batch.batch, dim=0)
        a = self.output_net(a)
        return a
