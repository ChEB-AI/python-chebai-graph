from chebai.preprocessing.datasets.pubchem import PubChem

from chebai_graph.preprocessing.datasets.chebi import GraphPropertiesMixIn


class PubChemGraphProperties(GraphPropertiesMixIn, PubChem):
    pass
