from chebai.preprocessing.datasets.chebi import ChEBIOver50

from chebai_graph.preprocessing.reader import GraphReader, GraphPropertyReader


class ChEBI50GraphData(ChEBIOver50):
    READER = GraphReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ChEBI50GraphProperties(ChEBIOver50):
    READER = GraphPropertyReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
