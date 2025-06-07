import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.models import GAT
from torch_scatter import scatter_add

from .graph import GraphBaseNet


class GATModelWrapper(GraphBaseNet):
    NAME = "GATModel"

    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)

        self._hidden_length = int(config.pop("hidden_length"))
        self._dropout_rate = float(config.pop("dropout_rate", 0.1))
        self._n_conv_layers = int(config.pop("n_conv_layers", 3))
        self._n_linear_layers = int(config.pop("n_linear_layers", 3))
        self._n_atom_properties = int(config.pop("n_atom_properties"))
        self._n_bond_properties = int(config.pop("n_bond_properties"))
        self._n_molecule_properties = int(config.pop("n_molecule_properties"))
        self._gat = GAT(
            in_channels=self._n_atom_properties,
            hidden_channels=self._hidden_length,
            num_layers=self._n_conv_layers,
            dropout=self._dropout_rate,
            edge_dim=self._n_bond_properties,
            **config,
        )

        self._ffn_activation = F.elu

        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self._hidden_length
                    + (self._n_molecule_properties if i == 0 else 0),
                    self._hidden_length,
                )
                for i in range(self._n_linear_layers - 1)
            ]
        )
        self.final_layer = torch.nn.Linear(self._hidden_length, self.out_dim)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        x = graph_data.x.float()
        a = self._gat.forward(
            x=x, edge_index=graph_data.edge_index.long(), edge_attr=graph_data.edge_attr
        )
        a = scatter_add(a, graph_data.batch, dim=0)

        a = torch.cat([a, graph_data.molecule_attr], dim=1)

        for lin in self.linear_layers:
            a = self._ffn_activation(lin(a))
        return self.final_layer(a)
