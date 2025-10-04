from .base import AugmentedNodePoolingNet, GraphNodeFGNodePoolingNet
from .gat import GATGraphPred
from .resgated import ResGatedGraphPred


class ResGatedAugNodePoolGraphPred(AugmentedNodePoolingNet, ResGatedGraphPred):
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings (optionally with molecule attributes).
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class GATAugNodePoolGraphPred(AugmentedNodePoolingNet, GATGraphPred):
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings (optionally with molecule attributes).
    - GATGraphPred: Graph attention network for final graph prediction.
    """

    ...


class ResGatedGraphNodeFGNodePoolGraphPred(
    GraphNodeFGNodePoolingNet, ResGatedGraphPred
):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes (optionally with molecule attributes).
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class GATGraphNodeFGNodePoolGraphPred(GraphNodeFGNodePoolingNet, GATGraphPred):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes (optionally with molecule attributes).
    - GATGraphPred: Graph attention network for final graph prediction.
    """

    ...
