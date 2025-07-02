from abc import ABC

import torch
from torch_geometric import nn as tgnn

from .base import GraphNetWrapper


class ResGatedModelWrapper(GraphNetWrapper, ABC):
    def _get_gnn(self, config):
        in_length = config["in_length"]
        hidden_length = config["hidden_length"]
        dropout_rate = config["dropout_rate"]
        n_atom_properties = int(config["n_atom_properties"])
        n_bond_properties = int(config["n_bond_properties"])
        n_conv_layers = int(config["n_conv_layers"])

        convs = torch.nn.ModuleList()
        for i in range(n_conv_layers):
            if i == 0:
                convs.append(
                    tgnn.ResGatedGraphConv(
                        n_atom_properties,
                        in_length,
                        # dropout=dropout_rate,
                        edge_dim=n_bond_properties,
                    )
                )
            convs.append(
                tgnn.ResGatedGraphConv(in_length, in_length, edge_dim=n_bond_properties)
            )
        convs.append(
            tgnn.ResGatedGraphConv(in_length, hidden_length, edge_dim=n_bond_properties)
        )

        return convs
