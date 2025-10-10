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
        assert distribution in RandomFeatureInitializationReader.DISTRIBUTIONS, (
            f"Unsupported distribution: {distribution}. "
            f"Choose from {RandomFeatureInitializationReader.DISTRIBUTIONS}."
        )
        self.distribution = distribution

        self.complete_randomness = (
            str(config.get("complete_randomness", "True")).lower() == "true"
        )

        print("Using complete randomness: ", self.complete_randomness)

        if not self.complete_randomness:
            assert (
                "random_pad_node" in config or "random_pad_edge" in config
            ), "Missing 'random_pad_node' or 'random_pad_edge' in config when complete_randomness is False"
            self.random_pad_node = (
                int(config["random_pad_node"])
                if config.get("random_pad_node") is not None
                else None
            )
            if self.random_pad_node is not None:
                print(
                    f"[Info] Node features will be padded with {self.random_pad_node} "
                    f"new set of random features from distribution {self.distribution} "
                    f"in each forward pass."
                )

            self.random_pad_edge = (
                int(config["random_pad_edge"])
                if config.get("random_pad_edge") is not None
                else None
            )
            if self.random_pad_edge is not None:
                print(
                    f"[Info] Edge features will be padded with {self.random_pad_edge} "
                    f"new set of random features from distribution {self.distribution} "
                    f"in each forward pass."
                )

            assert (
                self.random_pad_node > 0 or self.random_pad_edge > 0
            ), "'random_pad_node' or 'random_pad_edge' must be positive integers"

        self.resgated: BasicGNN = ResGatedModel(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            edge_dim=self.edge_dim,
            act=self.activation,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        new_x = None
        new_edge_attr = None
        if self.complete_randomness:
            new_x = torch.empty(
                graph_data.x.shape[0], graph_data.x.shape[1], device=self.device
            )
            RandomFeatureInitializationReader.random_gni(new_x, self.distribution)

            new_edge_attr = torch.empty(
                graph_data.edge_attr.shape[0],
                graph_data.edge_attr.shape[1],
                device=self.device,
            )
            RandomFeatureInitializationReader.random_gni(
                new_edge_attr, self.distribution
            )
        else:
            if self.random_pad_node is not None:
                pad_node = torch.empty(
                    graph_data.x.shape[0],
                    self.random_pad_node,
                    device=self.device,
                )
                RandomFeatureInitializationReader.random_gni(
                    pad_node, self.distribution
                )
                new_x = torch.cat((graph_data.x, pad_node), dim=1)

            if self.random_pad_edge is not None:
                pad_edge = torch.empty(
                    graph_data.edge_attr.shape[0],
                    self.random_pad_edge,
                    device=self.device,
                )
                RandomFeatureInitializationReader.random_gni(
                    pad_edge, self.distribution
                )
                new_edge_attr = torch.cat((graph_data.edge_attr, pad_edge), dim=1)

        assert (
            new_x is not None and new_edge_attr is not None
        ), "Feature initialization failed"
        out = self.resgated(
            x=new_x.float(),
            edge_index=graph_data.edge_index.long(),
            edge_attr=new_edge_attr.float(),
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
