from chebai.preprocessing.datasets.molecule_net_classification import (
    BBBP,
    HIV,
    MUV,
    PCBA,
    Bace,
    ClinTox,
    Sider,
    Tox21,
    ToxCast,
)

from chebai_graph.preprocessing.datasets.base import (
    GraphPropAsPerNodeType,
)
from chebai_graph.preprocessing.reader.augmented_reader import (
    AtomFGReader_WithFGEdges_WithGraphNode,
)


class PCBA_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, PCBA):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class Bace_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, Bace):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class BBBP_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, BBBP):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ClinTox_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ClinTox):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class HIV_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, HIV):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class Sider_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, Sider):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class MUV_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, MUV):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class Tox21_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, Tox21):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ToxCast_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ToxCast):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


if __name__ == "__main__":
    dataset = Bace_WFGE_WGN_AsPerNodeType()
    dataset.prepare_data()
    dataset.setup()
