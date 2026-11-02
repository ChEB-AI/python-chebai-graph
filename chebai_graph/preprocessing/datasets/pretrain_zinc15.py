from chebai.preprocessing.datasets.pretrain_zinc15 import Zinc15Pretrain

from chebai_graph.preprocessing.datasets.base import GraphPropAsPerNodeType
from chebai_graph.preprocessing.reader import GN_WithAtoms_FG_WithAtoms_FGE


class Zinc15Pretrain_GNWA_FGWA_WFGE_AsPerNodeType(
    GraphPropAsPerNodeType, Zinc15Pretrain
):
    READER = GN_WithAtoms_FG_WithAtoms_FGE


if __name__ == "__main__":
    dataset = Zinc15Pretrain_GNWA_FGWA_WFGE_AsPerNodeType(
        root="data/Zinc15", name="Zinc15"
    )
    dataset.prepare_data()
    dataset.setup()
