from chebai.preprocessing.datasets.pubchem import PubchemChem

from chebai_graph.preprocessing.datasets.chebi import GraphPropertiesMixIn


class PubChemGraphProperties(GraphPropertiesMixIn, PubchemChem):
    pass
