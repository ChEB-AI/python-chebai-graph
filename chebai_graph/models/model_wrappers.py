from abc import ABC

from .base import GraphNetWrapper
from .resgated import ResGatedGraphConvNetBase


class ResGatedModelWrapper(GraphNetWrapper, ABC):
    def _get_gnn(self, config):
        return ResGatedGraphConvNetBase(config=config)
