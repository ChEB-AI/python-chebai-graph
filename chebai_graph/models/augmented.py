from .architectures.gat import GATGraphPred
from .architectures.gine import GINEGraphPred
from .architectures.resgated import ResGatedGraphPred
from .pooling import AAPool, AMGPool


class ResGatedAAPoolGraphPred(AAPool, ResGatedGraphPred):
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings (optionally with molecule attributes).
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class GATAAPoolGraphPred(AAPool, GATGraphPred):
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings (optionally with molecule attributes).
    - GATGraphPred: Graph attention network for final graph prediction.
    """

    ...


class GINEAAPoolGraphPred(AAPool, GINEGraphPred):
    """
    Combines:
    - AugmentedNodePoolingNet: Pools atom and augmented node embeddings (optionally with molecule attributes).
    - GINEGraphPred: Graph isomorphism network for final graph prediction.
    """

    ...


class ResGatedAMGPoolGraphPred(AMGPool, ResGatedGraphPred):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes (optionally with molecule attributes).
    - ResGatedGraphPred: Residual gated network for final graph prediction.
    """

    ...


class GATAMGPoolGraphPred(AMGPool, GATGraphPred):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes (optionally with molecule attributes).
    - GATGraphPred: Graph attention network for final graph prediction.
    """

    ...


class GINEAMGPoolGraphPred(AMGPool, GINEGraphPred):
    """
    Combines:
    - GraphNodeFGNodePoolingNet: Pools atom, functional group, and graph nodes (optionally with molecule attributes).
    - GINEGraphPred: Graph isomorphism network for final graph prediction.
    """

    ...
