from .base import (
    AugmentedNodePoolingNet,
    FGNodePoolingNet,
    FGNodePoolingNoGraphNodeNet,
    GraphNodeFGNodePoolingNet,
    GraphNodeNoFGNodePoolingNet,
    GraphNodePoolingNet,
)
from .resgated import ResGatedGraphPred


class ResGatedAugNodePoolGraphPred(AugmentedNodePoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedAugNodePoolGraphPred"


class ResGatedGraphNodePoolGraphPred(GraphNodePoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedGraphNodePoolGraphPred"


class ResGatedFGNodePoolGraphPred(FGNodePoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedFGNodePoolGraphPred"


class ResGatedGraphNodeFGNodePoolGraphPred(
    GraphNodeFGNodePoolingNet, ResGatedGraphPred
):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedGraphNodeFGNodePoolGraphPred"


class ResGatedGraphNodeNoFGNodeGraphPred(
    GraphNodeNoFGNodePoolingNet, ResGatedGraphPred
):
    """GNN for graph-level prediction for augmented graphs without FG nodes"""

    NAME = "ResGatedGraphNodeNoFGNodeGraphPred"


class ResGatedFGNodeNoGraphNodeGraphPred(
    FGNodePoolingNoGraphNodeNet, ResGatedGraphPred
):
    """GNN for graph-level prediction for augmented graphs without FG nodes"""

    NAME = "ResGatedFGNodeNoGraphNodeGraphPred"
