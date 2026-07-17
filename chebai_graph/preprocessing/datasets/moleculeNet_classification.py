from chebai.preprocessing.datasets.molecule_classification import (
    BaceChem,
    BBBPChem,
    ClinToxChem,
    HIVChem,
    MUVChem,
    SiderChem,
)

from chebai_graph.preprocessing.datasets.base import (
    GraphPropAsPerNodeType,
    GraphPropertiesMixIn,
)
from chebai_graph.preprocessing.reader.augmented_reader import (
    AtomFGReader_WithFGEdges_WithGraphNode,
)
from chebai_graph.preprocessing.reader.reader import GraphPropertyReader


class BaceChemDataset(GraphPropertiesMixIn, BaceChem):
    READER = GraphPropertyReader


class BaceChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, BaceChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class BBBPChemDataset(GraphPropertiesMixIn, BBBPChem):
    READER = GraphPropertyReader


class BBBPChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, BBBPChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ClinToxChemDataset(GraphPropertiesMixIn, ClinToxChem):
    READER = GraphPropertyReader


class ClinToxChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ClinToxChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class HIVChemDataset(GraphPropertiesMixIn, HIVChem):
    READER = GraphPropertyReader


class HIVChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, HIVChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class SiderChemDataset(GraphPropertiesMixIn, SiderChem):
    READER = GraphPropertyReader


class SiderChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, SiderChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class MUVChemDataset(GraphPropertiesMixIn, MUVChem):
    READER = GraphPropertyReader


class MUVChem_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, MUVChem):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


if __name__ == "__main__":
    dataset = BaceChemDataset()
    dataset.prepare_data()
    dataset.setup()
