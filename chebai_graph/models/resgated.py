import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData

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
                - 'in_length' (int): Intermediate feature length used in GNN layers.
                - Other parameters inherited from GraphModelBase.
            **kwargs: Additional keyword arguments passed to GraphModelBase.
        """
        super().__init__(config=config, **kwargs)
        self.in_length = int(config["in_length"])

        self.activation = F.elu
        self.dropout = nn.Dropout(self.dropout_rate)

        self.convs = torch.nn.ModuleList()
        for i in range(self.n_conv_layers):
            if i == 0:
                # Initial layer uses atom features as input
                self.convs.append(
                    tgnn.ResGatedGraphConv(
                        self.n_atom_properties,
                        self.in_length,
                        # dropout=self.dropout_rate,
                        edge_dim=self.n_bond_properties,
                    )
                )
            # Intermediate layers
            self.convs.append(
                tgnn.ResGatedGraphConv(
                    self.in_length, self.in_length, edge_dim=self.n_bond_properties
                )
            )

        # Final projection layer to hidden dimension
        self.final_conv = tgnn.ResGatedGraphConv(
            self.in_length, self.hidden_length, edge_dim=self.n_bond_properties
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
