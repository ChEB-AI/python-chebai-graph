from .base import (
    AugmentedNodePoolingNet,
    AugmentedOnlyPoolingNet,
    FGNodePoolingNet,
    FGNodePoolingNoGraphNodeNet,
    FGOnlyPoolingNet,
    GraphNodeFGNodePoolingNet,
    GraphNodeNoFGNodePoolingNet,
    GraphNodeOnlyPoolingNet,
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


class ResGatedAugOnlyPoolGraphPred(AugmentedOnlyPoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedAugOnlyPoolGraphPred"


class ResGatedGraphNodeOnlyPoolGraphPred(GraphNodeOnlyPoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedGraphNodeOnlyPoolGraphPred"


class ResGatedFGOnlyPoolGraphPred(FGOnlyPoolingNet, ResGatedGraphPred):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedFGOnlyPoolGraphPred"
