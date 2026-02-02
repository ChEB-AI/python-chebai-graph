"""
ResGatedDynamicGNIGraphPred
------------------------------------------------

Module providing a ResGated GNN model that applies Random Node Initialization
(RNI) dynamically at each forward pass. This follows the approach from:

Abboud, R., et al. (2020). "The surprising power of graph neural networks with
random node initialization." arXiv preprint arXiv:2010.01179.

The module exposes:
- ResGatedDynamicGNI: a model that can either completely replace node/edge
  features with random tensors each forward pass or pad existing features with
  additional random features.
- ResGatedDynamicGNIGraphPred: a thin wrapper that instantiates the above for
  graph-level prediction pipelines.
"""

__all__ = ["ResGatedDynamicGNIGraphPred"]

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
    ResGated GNN with dynamic Random Node Initialization (RNI).

    This model supports two modes controlled by the `config`:

    - complete_randomness (bool-like): If True, **replace** node and edge
      features entirely with randomly initialized tensors each forward pass.
      If False, the model **pads** existing features with extra randomly
      initialized features on-the-fly.

    - pad_node_features (int, optional): Number of random columns to append
      to each node feature vector when `complete_randomness` is False.

    - pad_edge_features (int, optional): Number of random columns to append
      to each edge feature vector when `complete_randomness` is False.

    - distribution (str): Distribution for random initialization. Must be one
      of RandomFeatureInitializationReader.DISTRIBUTIONS.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing model hyperparameters. Expected keys
        used by this class:
            - distribution (optional, default "normal")
            - complete_randomness (optional, default "True")
            - pad_node_features (optional, int)
            - pad_edge_features (optional, int)
        Keys required by GraphModelBase (e.g., in_channels, hidden_channels,
        out_channels, num_layers, edge_dim) should also be present.
    **kwargs : Any
        Additional keyword arguments forwarded to GraphModelBase.
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
                "pad_node_features" in config or "pad_edge_features" in config
            ), "Missing 'pad_node_features' or 'pad_edge_features' in config when complete_randomness is False"
            self.pad_node_features = (
                int(config["pad_node_features"])
                if config.get("pad_node_features") is not None
                else None
            )
            if self.pad_node_features is not None:
                print(
                    f"[Info] Node features will be padded with {self.pad_node_features} "
                    f"new set of random features from distribution {self.distribution} "
                    f"in each forward pass."
                )

            self.pad_edge_features = (
                int(config["pad_edge_features"])
                if config.get("pad_edge_features") is not None
                else None
            )
            if self.pad_edge_features is not None:
                print(
                    f"[Info] Edge features will be padded with {self.pad_edge_features} "
                    f"new set of random features from distribution {self.distribution} "
                    f"in each forward pass."
                )

            assert (
                self.pad_node_features > 0 or self.pad_edge_features > 0
            ), "'pad_node_features' or 'pad_edge_features' must be positive integers"

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

        # If replacing features entirely with random values
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

        # If padding existing features with additional random columns
        else:
            if self.pad_node_features is not None:
                pad_node = torch.empty(
                    graph_data.x.shape[0],
                    self.pad_node_features,
                    device=self.device,
                )
                RandomFeatureInitializationReader.random_gni(
                    pad_node, self.distribution
                )
                new_x = torch.cat((graph_data.x, pad_node), dim=1)

            if self.pad_edge_features is not None:
                pad_edge = torch.empty(
                    graph_data.edge_attr.shape[0],
                    self.pad_edge_features,
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
