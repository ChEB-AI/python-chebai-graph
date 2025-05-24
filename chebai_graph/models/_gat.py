import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.models import GAT
from torch_scatter import scatter_add

from .graph import GraphBaseNet


class GATModelWrapper(GraphBaseNet):
    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)

        self._in_length = config["in_length"]
        self._hidden_length = config["hidden_length"]
        self._dropout_rate = config["dropout_rate"]
        self._n_conv_layers = (
            config["n_conv_layers"] if "n_conv_layers" in config else 3
        )
        self._n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )
        self._n_atom_properties = int(config["n_atom_properties"])
        self._n_bond_properties = (
            int(config["n_bond_properties"]) if "n_bond_properties" in config else 7
        )
        self._n_molecule_properties = (
            int(config["n_molecule_properties"])
            if "n_molecule_properties" in config
            else 0
        )
        self._gat = GAT(
            in_channels=self._in_length,
            hidden_channels=self._hidden_length,
            num_layers=self._n_conv_layers,
            dropout=self._dropout_rate,
            **kwargs,
        )

        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.gnn.hidden_length + (i == 0) * self.gnn.n_molecule_properties,
                    self.gnn.hidden_length,
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
            a = self.gnn.activation(lin(a))
        a = self.final_layer(a)
