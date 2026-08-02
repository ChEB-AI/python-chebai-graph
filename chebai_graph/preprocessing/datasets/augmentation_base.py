from abc import ABC

import pandas as pd
import torch
from torch_geometric.data.data import Data as GeomData

from .base import GraphPropAsPerNodeType, GraphPropertiesMixIn


class AugGraphPropMixIn_NoGraphNode(GraphPropertiesMixIn, ABC):
    """Mixin for augmented graph data without additional graph nodes."""

    READER = None

    def _merge_props_into_base(self, row: pd.Series) -> GeomData:
        data = super()._merge_props_into_base(row)
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData) and isinstance(data, GeomData)

        is_atom_node = geom_data.is_atom_node
        assert is_atom_node is not None, "is_atom_node must be set in the geom_data"
        data.is_atom_node = is_atom_node
        return data


class AugGraphPropMixIn_WithGraphNode(AugGraphPropMixIn_NoGraphNode, ABC):
    """Mixin for augmented graph data with graph-level nodes."""

    READER = None

    def _merge_props_into_base(self, row: pd.Series) -> GeomData:
        data = super()._merge_props_into_base(row)
        return self._add_graph_node_mask(data, row)

    def _add_graph_node_mask(self, data: GeomData, row: pd.Series) -> GeomData:
        """
        Add a graph node mask to the GeomData object.

        Args:
            data: A GeomData object with features.
            row: A dictionary containing 'features' and other metadata.

        Returns:
            Modified GeomData with graph node mask added.
        """
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData) and isinstance(data, GeomData)
        is_graph_node = geom_data.is_graph_node
        assert is_graph_node is not None, "is_graph_node must be set in the geom_data"
        data.is_graph_node = is_graph_node
        return data


class GraphPropForAtomLevelOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_fg_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_fg_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor

    def _fill_node_tensor_with_molecule_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_graph_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor


class GraphPropForFGLevelOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_atom_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_atom_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor

    def _fill_node_tensor_with_molecule_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_graph_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor


class GraphPropForGraphNodeOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_atom_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_atom_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor

    def _fill_node_tensor_with_fg_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_fg_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor


class GraphPropForAtomAndFGLevelOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_molecule_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_graph_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor


class GraphPropForAtomLevelAndGraphNodeOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_fg_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_fg_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor


class GraphPropForFGLevelAndGraphNodeOnly(GraphPropAsPerNodeType):
    def _fill_node_tensor_with_atom_type_property(
        self,
        node_tensor: torch.Tensor,
        property_values: torch.Tensor,
        offset: int,
        is_atom_node: torch.Tensor,
    ) -> torch.Tensor:
        return node_tensor
