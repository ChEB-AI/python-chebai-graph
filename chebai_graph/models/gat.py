import torch
from torch.nn import ELU
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.models import GAT

from .base import GraphModelBase, GraphNetWrapper


class GATGraphConvNetBase(GraphModelBase):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.heads = int(config["heads"])
        self.v2 = bool(config["v2"])
        self.activation = ELU()  # instantiate once
        self.gat = GAT(
            in_channels=self.n_atom_properties,
            hidden_channels=self.hidden_length,
            num_layers=self.n_conv_layers,
            dropout=self.dropout_rate,
            edge_dim=self.n_bond_properties,
            heads=self.heads,
            v2=self.v2,
            act=ELU,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        a = self.gat(
            x=graph_data.x.float(),
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
        )

        return self.activation(a)


class GATGraphPred(GraphNetWrapper):
    NAME = "GATGraphPred"

    def _get_gnn(self, config):
        return GATGraphConvNetBase(config=config)
