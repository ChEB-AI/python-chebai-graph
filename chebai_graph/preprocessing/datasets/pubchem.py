from chebai_graph.preprocessing.datasets.chebi import GraphPropertiesMixIn
from chebai.preprocessing.datasets.pubchem import PubchemChem


class PubChemGraphProperties(GraphPropertiesMixIn, PubchemChem):
    pass
