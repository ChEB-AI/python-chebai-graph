import torch
from chebai.preprocessing.structures import XYData


class XYGraphData(XYData):
    """
    Extension of XYData supporting `.to(device)` for potentially complex `x` structures.

    `x` can be:
    - a tensor,
    - a tuple of tensors or dicts of tensors,
    and this class recursively sends all tensors to the specified device.

    Args:
        Inherits from XYData.
    """

    def __len__(self) -> int:
        """Return the length of y."""
        return len(self.y)

    def to_x(self, device: torch.device | str) -> object:
        """
        Move the input features `x` to the given device.

        Args:
            device: torch device or device string (e.g. 'cpu' or 'cuda').

        Returns:
            The input `x` moved to the specified device, preserving structure.
        """
        if isinstance(self.x, tuple):
            res = []
            for elem in self.x:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = v.to(device) if v is not None else None
                else:
                    elem = elem.to(device)
                res.append(elem)
            return tuple(res)
        return super(self, device)
