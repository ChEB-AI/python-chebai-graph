from chebai.preprocessing.datasets.molecule_net_classification import (
    BACE,
    BBBP,
    HIV,
    MUV,
    PCBA,
    SIDER,
    ClinTox,
    Tox21,
    ToxCast,
)

from chebai_graph.preprocessing.datasets.base import (
    GraphPropAsPerNodeType,
    GraphPropertiesMixIn,
)
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
)

from .augmentation_base import (
    AugGraphPropMixIn_NoGraphNode,
    AugGraphPropMixIn_WithGraphNode,
)


class PCBA_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, PCBA):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class BACE_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, BACE):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class BBBP_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, BBBP):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ClinTox_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ClinTox):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class HIV_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, HIV):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class SIDER_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, SIDER):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class MUV_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, MUV):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class Tox21_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, Tox21):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ToxCast_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ToxCast):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


# ---- Augmentation: Variants with graph Node connected to FG nodes only -------------
class Tox21_WFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, Tox21):
    """Tox21 with with FG nodes and FG edges and graph node."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ToxCast_WFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ToxCast):
    """ToxCast with with FG nodes and FG edges and graph node."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode


class Tox21_NFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, Tox21):
    """Tox21 with FG nodes but without FG edges, with graph node."""

    READER = AtomFGReader_NoFGEdges_WithGraphNode


class ToxCast_NFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ToxCast):
    """ToxCast with FG nodes but without FG edges, with graph node."""

    READER = AtomFGReader_NoFGEdges_WithGraphNode


class Tox21_WFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, Tox21):
    """Tox21 with FG nodes and FG edges, no graph node."""

    READER = AtomFGReader_WithFGEdges_NoGraphNode


class ToxCast_WFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ToxCast):
    """ToxCast with FG nodes and FG edges, no graph node."""

    READER = AtomFGReader_WithFGEdges_NoGraphNode


class Tox21_NFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, Tox21):
    """Tox21 with FG nodes but without FG edges or graph node."""

    READER = AtomsFGReader_NoFGEdges_NoGraphNode


class ToxCast_NFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ToxCast):
    """ToxCast with FG nodes but without FG edges or graph node."""

    READER = AtomsFGReader_NoFGEdges_NoGraphNode


class Tox21_Atom_WGNOnly_GraphProp(AugGraphPropMixIn_WithGraphNode, Tox21):
    """Tox21 with atom-level nodes and graph node only."""

    READER = AtomReader_WithGraphNodeOnly


class ToxCast_Atom_WGNOnly_GraphProp(AugGraphPropMixIn_WithGraphNode, ToxCast):
    """ToxCast with atom-level nodes and graph node only."""

    READER = AtomReader_WithGraphNodeOnly


# ------- Augmentation: Variants with graph Node connected to all others nodes (FG and atoms) --------------
class Tox21_GN_WithAllNodes_FG_WithAtoms_FGE(AugGraphPropMixIn_WithGraphNode, Tox21):
    """
    Tox21 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_FGE


class ToxCast_GN_WithAllNodes_FG_WithAtoms_FGE(
    AugGraphPropMixIn_WithGraphNode, ToxCast
):
    """
    ToxCast with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_FGE


class Tox21_GN_WithAllNodes_FG_WithAtoms_NoFGE(AugGraphPropMixIn_WithGraphNode, Tox21):
    """
    Tox21 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_NoFGE


class ToxCast_GN_WithAllNodes_FG_WithAtoms_NoFGE(
    AugGraphPropMixIn_WithGraphNode, ToxCast
):
    """
    ToxCast with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_NoFGE


# ------- Augmentation: Variants with graph node connected to atom nodes ONLY -----------
class Tox21_GN_WithAtoms_FG_WithAtoms_FGE(AugGraphPropMixIn_WithGraphNode, Tox21):
    """
    Tox21 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_FGE


class ToxCast_GN_WithAtoms_FG_WithAtoms_FGE(AugGraphPropMixIn_WithGraphNode, ToxCast):
    """
    ToxCast with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_FGE


class Tox21_GN_WithAtoms_FG_WithAtoms_NoFGE(AugGraphPropMixIn_WithGraphNode, Tox21):
    """
    Tox21 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_NoFGE


class ToxCast_GN_WithAtoms_FG_WithAtoms_NoFGE(AugGraphPropMixIn_WithGraphNode, ToxCast):
    """
    ToxCast with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_NoFGE


# ---------------------------Baselines classes for Tox21 and ToxCast datasets
class Tox21GraphProperties(GraphPropertiesMixIn, Tox21):
    """Tox21 dataset with molecular property encodings."""

    pass


class ToxCastGraphProperties(GraphPropertiesMixIn, ToxCast):
    """ToxCast dataset with molecular property encodings."""

    pass


if __name__ == "__main__":
    dataset = BACE_WFGE_WGN_AsPerNodeType()
    dataset.prepare_data()
    dataset.setup()
