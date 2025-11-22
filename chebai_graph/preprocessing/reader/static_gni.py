"""
RandomFeatureInitializationReader
--------------------------------

Implements random node / edge / molecule feature initialization for graph neural
networks following:

Abboud, R., et al. (2020). "The surprising power of graph neural networks with
random node initialization." arXiv preprint arXiv:2010.01179.

Code reference: https://github.com/ralphabb/GNN-RNI/blob/main/GNNHyb.py

This module provides a reader that replaces node/edge/molecule features with
randomly initialized tensors drawn from a selected distribution.

Notes
-----
- This reader subclasses GraphPropertyReader and is intended to be used where a
  graph object with attributes `x`, `edge_attr`, and optionally `molecule_attr`
  is expected (e.g., `torch_geometric.data.Data`).
- The reader only performs random initialization and does not support reading
  specific properties from the input data.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data as GeomData

from .reader import GraphPropertyReader


class RandomFeatureInitializationReader(GraphPropertyReader):
    """
    Reader that initializes node, bond (edge), and molecule features with
    random values according to a chosen distribution.

    Supported distributions:
        - "normal"         : standard normal (mean=0, std=1)
        - "uniform"        : uniform in [-1, 1]
        - "xavier_normal"  : Xavier normal initialization
        - "xavier_uniform" : Xavier uniform initialization
        - "zeros"          : all zeros

    Parameters
    ----------
    num_node_properties : int
        Number of features to generate per node.
    num_bond_properties : int
        Number of features to generate per edge/bond.
    num_molecule_properties : int
        Number of global molecule-level features to generate.
    distribution : str, optional
        One of the supported distributions (default: "normal").
    *args, **kwargs : Any
        Additional positional and keyword arguments passed to the parent
        GraphPropertyReader.
    """

    DISTRIBUTIONS = [
        "normal",
        "uniform",
        "xavier_normal",
        "xavier_uniform",
        "zeros",
    ]

    def __init__(
        self,
        num_node_properties: int,
        num_bond_properties: int,
        num_molecule_properties: int,
        distribution: str = "normal",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self.DISTRIBUTIONS}, got '{distribution}'"
            )

        self.num_node_properties: int = int(num_node_properties)
        self.num_bond_properties: int = int(num_bond_properties)
        self.num_molecule_properties: int = int(num_molecule_properties)
        self.distribution: str = distribution

    def name(self) -> str:
        """
        Return a human-readable identifier for this reader configuration.

        Returns
        -------
        str
            A name encoding the chosen distribution and generated feature sizes.
        """
        return (
            f"gni-{self.distribution}"
            f"-node{self.num_node_properties}"
            f"-bond{self.num_bond_properties}"
            f"-mol{self.num_molecule_properties}"
        )

    def _read_data(self, raw_data: Any) -> Optional[GeomData]:
        """
        Read and return a `torch_geometric.data.Data` object with randomized
        node/edge/molecule features.

        This method calls the parent's `_read_data` to obtain a graph object,
        then replaces `x`, `edge_attr` and sets `molecule_attr` with new tensors.

        Parameters
        ----------
        raw_data : Any
            Raw input that the parent reader understands.

        Returns
        -------
        Optional[GeomData]
            A `Data` object with randomized attributes or `None` if the parent
            `_read_data` returned `None`.
        """
        data: Optional[GeomData] = super()._read_data(raw_data)
        if data is None:
            return None

        random_x = torch.empty(data.x.shape[0], self.num_node_properties)
        random_edge_attr = torch.empty(
            data.edge_attr.shape[0], self.num_bond_properties
        )
        random_molecule_properties = torch.empty(1, self.num_molecule_properties)

        # Initialize them according to the chosen distribution.
        self.random_gni(random_x, self.distribution)
        self.random_gni(random_edge_attr, self.distribution)
        self.random_gni(random_molecule_properties, self.distribution)

        # Assign randomized attributes back to the data object.
        data.x = random_x
        data.edge_attr = random_edge_attr
        # Use `molecule_attr` as the name in this codebase; if your Data object
        # expects a different name (e.g., `u` or `global_attr`) adapt accordingly.
        data.molecule_attr = random_molecule_properties

        return data

    def read_property(self, *args: Any, **kwargs: Any) -> None:
        """
        This reader does not support reading specific properties from the input.
        It only performs random initialization of features.

        Raises
        ------
        NotImplementedError
            Always raised to indicate unsupported operation.
        """
        raise NotImplementedError(
            "RandomFeatureInitializationReader only performs random initialization."
        )

    @staticmethod
    def random_gni(tensor: Tensor, distribution: str) -> None:
        """
        Fill `tensor` in-place according to the requested initialization.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to initialize in-place.
        distribution : str
            One of the supported distribution identifiers.

        Raises
        ------
        ValueError
            If an unknown distribution string is provided.
        """
        if distribution == "normal":
            torch.nn.init.normal_(tensor)
        elif distribution == "uniform":
            # Uniform in [-1, 1]
            torch.nn.init.uniform_(tensor, a=-1.0, b=1.0)
        elif distribution == "xavier_normal":
            torch.nn.init.xavier_normal_(tensor)
        elif distribution == "xavier_uniform":
            torch.nn.init.xavier_uniform_(tensor)
        elif distribution == "zeros":
            torch.nn.init.zeros_(tensor)
        else:
            raise ValueError(f"Unknown distribution type: '{distribution}'")
