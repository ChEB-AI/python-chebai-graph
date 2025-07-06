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
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedGraphNodePoolGraphPred(GraphNodePoolingNet, ResGatedGraphPred):
    """
    Combines:
    - GraphNodePoolingNet: Pools atom and graph node embeddings with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedFGNodePoolGraphPred(FGNodePoolingNet, ResGatedGraphPred):
    """
    Combines:
    - FGNodePoolingNet: Pools functional group nodes and other nodes with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedGraphNodeFGNodePoolGraphPred(
    GraphNodeFGNodePoolingNet, ResGatedGraphPred
):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedGraphNodeNoFGNodeGraphPred(
    GraphNodeNoFGNodePoolingNet, ResGatedGraphPred
):
    """
    Combines:
    - GraphNodeNoFGNodePoolingNet: Pools atom and graph nodes, excluding functional groups.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedFGNodeNoGraphNodeGraphPred(
    FGNodePoolingNoGraphNodeNet, ResGatedGraphPred
):
    """
    Combines:
    - FGNodePoolingNoGraphNodeNet: Pools atom and functional group nodes, excluding graph nodes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedAugOnlyPoolGraphPred(AugmentedOnlyPoolingNet, ResGatedGraphPred):
    """
    Combines:
    - AugmentedOnlyPoolingNet: Pools only augmented nodes with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedGraphNodeOnlyPoolGraphPred(GraphNodeOnlyPoolingNet, ResGatedGraphPred):
    """
    Combines:
    - GraphNodeOnlyPoolingNet: Pools only graph nodes with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class ResGatedFGOnlyPoolGraphPred(FGOnlyPoolingNet, ResGatedGraphPred):
    """
    Combines:
    - FGOnlyPoolingNet: Pools only functional group nodes with molecule attributes.
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...
