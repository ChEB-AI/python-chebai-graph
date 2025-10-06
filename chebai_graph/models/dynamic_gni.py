from typing import Any

import torch
from torch import Tensor
from torch.nn import ELU
from torch_geometric.data import Data as GraphData
from torch_geometric.nn.models.basic_gnn import BasicGNN

from chebai_graph.preprocessing.reader import RandomFeatureInitializationReader

from .base import GraphModelBase, GraphNetWrapper
from .resgated import ResGatedModel


class ResGatedDynamicGNI(GraphModelBase):
    """
    Base model class for applying ResGatedGraphConv layers to graph-structured data
    with dynamic initialization of features for nodes and edges.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
        **kwargs: Additional keyword arguments for parent class.
    """

    def __init__(self, config: dict[str, Any], **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.activation = ELU()  # Instantiate ELU once for reuse.
        distribution = config.get("distribution", "normal")
        assert distribution in ["normal", "uniform", "xavier_normal", "xavier_uniform"]
        self.distribution = distribution

        self.resgated: BasicGNN = ResGatedModel(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            edge_dim=self.edge_dim,
            act=self.activation,
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """
        Forward pass of the model.

        Args:
            batch (dict): A batch containing graph input features under the key "features".

        Returns:
            Tensor: The output node-level embeddings after the final activation.
        """
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData), "Expected GraphData instance"

        random_x = torch.empty(graph_data.x.shape[0], graph_data.x.shape[1])
        RandomFeatureInitializationReader.random_gni(random_x, self.distribution)
        random_edge_attr = torch.empty(
            graph_data.edge_attr.shape[0], graph_data.edge_attr.shape[1]
        )
        RandomFeatureInitializationReader.random_gni(
            random_edge_attr, self.distribution
        )

        out = self.resgated(
            x=random_x.float(),
            edge_index=graph_data.edge_index.long(),
            edge_attr=random_edge_attr.float(),
        )

        return self.activation(out)


class ResGatedDynamicGNIGraphPred(GraphNetWrapper):
    """
    Wrapper for graph-level prediction using ResGatedDynamicGNI.

    This class instantiates the core GNN model using the provided config.
    """

    def _get_gnn(self, config: dict[str, Any]) -> ResGatedDynamicGNI:
        """
        Returns the core ResGated GNN model.

        Args:
            config (dict): Configuration dictionary for the GNN model.

        Returns:
            ResGatedDynamicGNI: The core graph convolutional network.
        """
        return ResGatedDynamicGNI(config=config)
