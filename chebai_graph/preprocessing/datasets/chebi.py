from chebai.preprocessing.datasets.chebi import ChEBIOver50

from chebai_graph.preprocessing.datasets.base import GraphReader

class ChEBI50GraphData(ChEBIOver50):
    READER = GraphReader
