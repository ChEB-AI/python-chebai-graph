import torch
from torch.nn import ELU
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.models import GAT

from .base import GraphModelBase, GraphNetWrapper


class GATGraphConvNetBase(GraphModelBase):
    """
    Graph Attention Network (GAT) base module for graph convolution.

    Uses PyTorch Geometric's `GAT` implementation to process atomic node features
    and bond edge attributes through multiple attention heads and layers.
    """

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the GATGraphConvNetBase.

        Args:
            config (dict): Model configuration containing:
                - 'heads' (int): Number of attention heads.
                - 'v2' (bool): Whether to use the GATv2 variant.
                - Other required GraphModelBase parameters.
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(config=config, **kwargs)
        self.heads = int(config["heads"])
        self.v2 = bool(config["v2"])
        self.activation = ELU()  # Instantiate ELU once for reuse.
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
        """
        Forward pass through the GAT network.

        Processes atomic node features and edge attributes, and applies
        an ELU activation to the output.

        Args:
            batch (dict): Input batch containing:
                - 'features': A list with a `GraphData` object as its first element.

        Returns:
            torch.Tensor: Node embeddings after GAT and activation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)

        out = self.gat(
            x=graph_data.x.float(),
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
        )

        return self.activation(out)


class GATGraphPred(GraphNetWrapper):
    """
    GAT-based graph prediction model.

    Uses a `GATGraphConvNetBase` as the GNN backbone for generating node embeddings,
    which are then pooled by the GraphNetWrapper for final prediction.
    """

    NAME = "GATGraphPred"

    def _get_gnn(self, config: dict) -> GATGraphConvNetBase:
        """
        Instantiate the GAT graph convolutional network base.

        Args:
            config (dict): Model configuration.

        Returns:
            GATGraphConvNetBase: The initialized GNN.
        """
        return GATGraphConvNetBase(config=config)
