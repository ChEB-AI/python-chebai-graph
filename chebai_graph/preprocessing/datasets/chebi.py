import pandas as pd
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver25,
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
    ChEBIOverXPartial,
)
from lightning_utilities.core.rank_zero import rank_zero_info

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

from .augmentation_base import (
    AugGraphPropMixIn_NoGraphNode,
    AugGraphPropMixIn_WithGraphNode,
    GraphPropForAtomAndFGLevelOnly,
    GraphPropForAtomLevelAndGraphNodeOnly,
    GraphPropForAtomLevelOnly,
    GraphPropForFGLevelAndGraphNodeOnly,
    GraphPropForFGLevelOnly,
    GraphPropForGraphNodeOnly,
    GraphPropNodeLevelPropOnlyForAllNodes,
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
        )
        return base_df[base_data[0].keys()].to_dict("records")


class ChEBI25GraphProperties(GraphPropertiesMixIn, ChEBIOver25):
    """ChEBIOver25 dataset with molecular property encodings."""

    THRESHOLD = 25


class ChEBI50GraphProperties(GraphPropertiesMixIn, ChEBIOver50):
    """ChEBIOver50 dataset with molecular property encodings."""

    pass


class ChEBI100GraphProperties(GraphPropertiesMixIn, ChEBIOver100):
    """ChEBIOver100 dataset with molecular property encodings."""

    pass


class ChEBI50GraphPropertiesPartial(ChEBI50GraphProperties, ChEBIOverXPartial):
    """Partial version of ChEBIOver50 with molecular properties."""

    pass


# ---- Augmentation: Variants with graph Node connected to FG nodes only -------------
class ChEBI50_WFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with with FG nodes and FG edges and graph node."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode


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


# ------- Augmentation: Variants with graph Node connected to all others nodes (FG and atoms) --------------
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


# ------- Augmentation: Variants with graph node connected to atom nodes ONLY -----------
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


# ---------------------- Ablation: Properties ------------------------------
class ChEBI50_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForAtomLevelOnly(GraphPropForAtomLevelOnly, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForFGLevelOnly(GraphPropForFGLevelOnly, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForGraphNodeOnly(GraphPropForGraphNodeOnly, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForAtomAndFGLevelOnly(
    GraphPropForAtomAndFGLevelOnly, ChEBIOver50
):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForAtomLevelAndGraphNodeOnly(
    GraphPropForAtomLevelAndGraphNodeOnly, ChEBIOver50
):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForFGLevelAndGraphNodeOnly(
    GraphPropForFGLevelAndGraphNodeOnly, ChEBIOver50
):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_WFGE_WGN_ForNodeLevelPropOnlyForAllNodes(
    GraphPropNodeLevelPropOnlyForAllNodes, ChEBIOver50
):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


# ---------- Final Augmentation: Different Thresholds ------------------------------
class ChEBI100_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver100):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI25_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOverX):
    READER = AtomFGReader_WithFGEdges_WithGraphNode

    THRESHOLD = 25


if __name__ == "__main__":
    dataset = ChEBI25_WFGE_WGN_AsPerNodeType(chebi_version=248, subset="3_STAR")
    dataset.prepare_data()
    dataset.setup()
