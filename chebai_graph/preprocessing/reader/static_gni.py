"""
Abboud, Ralph, et al.
"The surprising power of graph neural networks with random node initialization."
arXiv preprint arXiv:2010.01179 (2020).

Code Reference: https://github.com/ralphabb/GNN-RNI/blob/main/GNNHyb.py
"""

import torch
from torch_geometric.data import Data as GeomData

from .reader import GraphPropertyReader


class RandomFeatureInitializationReader(GraphPropertyReader):
    DISTRIBUTIONS = ["normal", "uniform", "xavier_normal", "xavier_uniform", "zeros"]

    def __init__(
        self,
        num_node_properties: int,
        num_bond_properties: int,
        num_molecule_properties: int,
        distribution: str = "normal",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_node_properties = num_node_properties
        self.num_bond_properties = num_bond_properties
        self.num_molecule_properties = num_molecule_properties
        assert distribution in self.DISTRIBUTIONS
        self.distribution = distribution

    def name(self) -> str:
        """
        Get the name identifier of the reader.

        Returns:
            str: The name of the reader.
        """
        return f"gni-{self.distribution}-node{self.num_node_properties}-bond{self.num_bond_properties}-mol{self.num_molecule_properties}"

    def _read_data(self, raw_data):
        data: GeomData = super()._read_data(raw_data)
        if data is None:
            return None

        random_x = torch.empty(data.x.shape[0], self.num_node_properties)
        random_edge_attr = torch.empty(
            data.edge_attr.shape[0], self.num_bond_properties
        )
        random_molecule_properties = torch.empty(1, self.num_molecule_properties)

        self.random_gni(random_x, self.distribution)
        self.random_gni(random_edge_attr, self.distribution)
        self.random_gni(random_molecule_properties, self.distribution)

        data.x = random_x
        data.edge_attr = random_edge_attr
        data.molecule_attr = random_molecule_properties
        return data

    def read_property(self, *args, **kwargs) -> Exception:
        """This reader does not support reading specific properties."""
        raise NotImplementedError("This reader only performs random initialization.")

    @staticmethod
    def random_gni(tensor: torch.Tensor, distribution: str) -> None:
        if distribution == "normal":
            torch.nn.init.normal_(tensor)
        elif distribution == "uniform":
            torch.nn.init.uniform_(tensor, a=-1.0, b=1.0)
        elif distribution == "xavier_normal":
            torch.nn.init.xavier_normal_(tensor)
        elif distribution == "xavier_uniform":
            torch.nn.init.xavier_uniform_(tensor)
        elif distribution == "zeros":
            torch.nn.init.zeros_(tensor)
        else:
            raise ValueError("Unknown distribution type")
