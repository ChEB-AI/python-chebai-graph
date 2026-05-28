from chebai.preprocessing.datasets.pubchem import Pubchem

from chebai_graph.preprocessing.datasets.chebi import GraphPropertiesMixIn


class PubChemGraphProperties(GraphPropertiesMixIn, Pubchem):
    pass
