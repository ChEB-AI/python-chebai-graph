import os

import chebai.preprocessing.reader as dr
import networkx as nx
import pysmiles as ps
import rdkit.Chem as Chem
import torch
from torch_geometric.data import Data as GeomData
from torch_geometric.utils import from_networkx

from chebai_graph.preprocessing.collate import GraphCollator
from chebai_graph.preprocessing.properties import MolecularProperty


class GraphPropertyReader(dr.DataReader):
    COLLATOR = GraphCollator

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize GraphPropertyReader.

        Args:
            *args: Positional arguments forwarded to the base class.
            **kwargs: Keyword arguments forwarded to the base class.
        """
        super().__init__(*args, **kwargs)
        self.failed_counter = 0
        self.mol_object_buffer: dict[str, Chem.rdchem.Mol | None] = {}

    @classmethod
    def name(cls) -> str:
        """
        Get the name identifier of the reader.

        Returns:
            str: The name of the reader.
        """
        return "graph_properties"

    def _smiles_to_mol(self, smiles: str) -> Chem.rdchem.Mol | None:
        """
        Load SMILES string into an RDKit molecule object and cache it.

        Args:
            smiles (str): The SMILES string to parse.

        Returns:
            Chem.rdchem.Mol | None: Parsed molecule object or None if parsing failed.
        """
        if smiles in self.mol_object_buffer:
            return self.mol_object_buffer[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"RDKit failed to at parsing {smiles} (returned None)")
            self.failed_counter += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Rdkit failed at sanitizing {smiles}, \n Error: {e}")
                self.failed_counter += 1
        self.mol_object_buffer[smiles] = mol
        return mol

    def _read_data(self, raw_data: str) -> GeomData | None:
        """
        Convert raw SMILES string data into a PyTorch Geometric Data object.

        Args:
            raw_data (str): SMILES string.

        Returns:
            GeomData | None: Graph data object or None if molecule parsing failed.
        """
        mol = self._smiles_to_mol(raw_data)
        if mol is None:
            return None

        x = torch.zeros((mol.GetNumAtoms(), 0))

        # First source to target edges, then target to source edges
        src = [bond.GetBeginAtomIdx() for bond in mol.GetBonds()]
        tgt = [bond.GetEndAtomIdx() for bond in mol.GetBonds()]
        edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)

        # edge_index.shape == [2, num_edges]; edge_attr.shape == [num_edges, num_edge_features]
        edge_attr = torch.zeros((edge_index.size(1), 0))

        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def on_finish(self) -> None:
        """
        Called after reading is done to log information and clean up.
        """
        print(f"Failed to read {self.failed_counter} SMILES in total")
        self.mol_object_buffer = {}

    def read_property(self, smiles: str, property: MolecularProperty) -> list | None:
        """
        Read a molecular property for a given SMILES string.

        Args:
            smiles (str): SMILES string of the molecule.
            property (MolecularProperty): Property extractor to apply.

        Returns:
            list | None: Property values or None if molecule parsing failed.
        """
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None
        return property.get_property_value(mol)


class GraphReader(dr.ChemDataReader):
    """Reads each atom as one token (atom symbol + charge), reads bond order as edge attribute.
    Creates nx Graph from SMILES."""

    COLLATOR = GraphCollator

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize GraphReader.

        Args:
            *args: Positional arguments forwarded to the base class.
            **kwargs: Keyword arguments forwarded to the base class.
        """
        super().__init__(*args, **kwargs)
        self.dirname = os.path.dirname(__file__)

    @classmethod
    def name(cls) -> str:
        """
        Get the name identifier of the reader.

        Returns:
            str: The name of the reader.
        """
        return "graph"

    def _read_data(self, raw_data: str) -> GeomData | None:
        """
        Convert a SMILES string into a PyTorch Geometric Data object with atom tokens and bond order attributes.

        Args:
            raw_data (str): SMILES string.

        Returns:
            GeomData | None: Graph data object or None if parsing failed.
        """
        # raw_data is a SMILES string
        try:
            mol = ps.read_smiles(raw_data)
        except ValueError:
            return None
        assert isinstance(mol, nx.Graph)
        d: dict[int, int] = {}
        de: dict[tuple[int, int], int] = {}
        for node in mol.nodes:
            n = mol.nodes[node]
            try:
                m = n["element"]
                charge = n["charge"]
                if charge:
                    if charge > 0:
                        m += "+"
                    else:
                        m += "-"
                        charge *= -1
                    if charge > 1:
                        m += str(charge)
                m = f"[{m}]"
            except KeyError:
                m = "*"
            d[node] = self._get_token_index(m)
            for attr in list(mol.nodes[node].keys()):
                del mol.nodes[node][attr]
        for edge in mol.edges:
            de[edge] = mol.edges[edge]["order"]
            for attr in list(mol.edges[edge].keys()):
                del mol.edges[edge][attr]
        nx.set_node_attributes(mol, d, "x")
        nx.set_edge_attributes(mol, de, "edge_attr")
        data = from_networkx(mol)
        return data

    def collate(self, list_of_tuples: list) -> any:
        """
        Collate a list of samples into a batch.

        Args:
            list_of_tuples (list): List of data tuples to collate.

        Returns:
            Any: Collated batch (type depends on collator).
        """
        return self.collator(list_of_tuples)
