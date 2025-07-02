from .base import (
    AugmentedNodePoolingNet,
    FGNodePoolingNet,
    GraphNodeFGNodePoolingNet,
    GraphNodePoolingNet,
)
from .model_wrappers import ResGatedModelWrapper


class ResGatedAugNodePoolGraphPred(AugmentedNodePoolingNet, ResGatedModelWrapper):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedAugNodePoolGraphPred"


class ResGatedGraphNodePoolGraphPred(GraphNodePoolingNet, ResGatedModelWrapper):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedGraphNodePoolGraphPred"


class ResGatedFGNodePoolGraphPred(FGNodePoolingNet, ResGatedModelWrapper):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedFGNodePoolGraphPred"


class ResGatedGraphNodeFGNodePoolGraphPred(
    GraphNodeFGNodePoolingNet, ResGatedModelWrapper
):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedGraphNodeFGNodePoolGraphPred"
