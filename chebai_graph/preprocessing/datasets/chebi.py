from abc import ABC

import pandas as pd
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
    ChEBIOverXPartial,
)
from lightning_utilities.core.rank_zero import rank_zero_info
from torch_geometric.data.data import Data as GeomData

from chebai_graph.preprocessing.reader import (
    AtomFGReader_NoFGEdges_WithGraphNode,
    AtomFGReader_WithFGEdges_NoGraphNode,
    AtomFGReader_WithFGEdges_WithGraphNode,
    AtomReader_WithGraphNodeOnly,
    AtomsFGReader_NoFGEdges_NoGraphNode,
    GN_WithAllNodes_FG_WithAtoms_FGE,
    GN_WithAllNodes_FG_WithAtoms_NoFGE,
    GN_WithAtoms_FG_WithAtoms_FGE,
    GN_WithAtoms_FG_WithAtoms_NoFGE,
    GraphReader,
    RandomFeatureInitializationReader,
)

from .base import DataPropertiesSetter, GraphPropAsPerNodeType, GraphPropertiesMixIn


class ChEBI50GraphData(ChEBIOver50):
    """ChEBI dataset with at least 50 samples per class, using GraphReader."""

    READER = GraphReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ChEBI50_StaticGNI(DataPropertiesSetter, ChEBIOver50):
    READER = RandomFeatureInitializationReader

    def _setup_properties(self): ...

    def load_processed_data_from_file(self, filename):
        base_data = super().load_processed_data_from_file(filename)
        base_df = pd.DataFrame(base_data)

        rank_zero_info(
            f"Use following values for given parameters for model configuration: \n\t"
            f"in_channels: {self.reader.num_node_properties} , "
            f"edge_dim: {self.reader.num_bond_properties}, "
            f"n_molecule_properties: {self.reader.num_molecule_properties}"
        )
        return base_df[base_data[0].keys()].to_dict("records")


class ChEBI50GraphProperties(GraphPropertiesMixIn, ChEBIOver50):
    """ChEBIOver50 dataset with molecular property encodings."""

    pass


class ChEBI100GraphProperties(GraphPropertiesMixIn, ChEBIOver100):
    """ChEBIOver100 dataset with molecular property encodings."""

    pass


class ChEBI50GraphPropertiesPartial(ChEBI50GraphProperties, ChEBIOverXPartial):
    """Partial version of ChEBIOver50 with molecular properties."""

    pass


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


class ChEBI50_WFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with with FG nodes and FG edges and graph node."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_GN_WithAllNodes_FG_WithAtoms_FGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_FGE


class ChEBI50_GN_WithAllNodes_FG_WithAtoms_NoFGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_NoFGE


class ChEBI50_GN_WithAtoms_FG_WithAtoms_FGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_FGE


class ChEBI50_GN_WithAtoms_FG_WithAtoms_NoFGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_NoFGE


class ChEBI50_NFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes but without FG edges, with graph node."""

    READER = AtomFGReader_NoFGEdges_WithGraphNode


class ChEBI50_WFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes and FG edges, no graph node."""

    READER = AtomFGReader_WithFGEdges_NoGraphNode


class ChEBI50_NFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes but without FG edges or graph node."""

    READER = AtomsFGReader_NoFGEdges_NoGraphNode


class ChEBI50_Atom_WGNOnly_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with atom-level nodes and graph node only."""

    READER = AtomReader_WithGraphNodeOnly


class ChEBI50_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI100_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver100):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI25_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOverX):
    READER = AtomFGReader_WithFGEdges_WithGraphNode

    THRESHOLD = 25


if __name__ == "__main__":
    dataset = ChEBI25_WFGE_WGN_AsPerNodeType(chebi_version=248, subset="3_STAR")
    dataset.prepare_data()
    dataset.setup()
