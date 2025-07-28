import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import rdkit.Chem as Chem

from chebai_graph.preprocessing.property_encoder import IndexEncoder, PropertyEncoder

from . import constants as k

# For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
# https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
# https://mail.python.org/pipermail/python-dev/2017-December/151283.html
assert sys.version_info >= (
    3,
    7,
), "This code requires Python 3.7 or higher."
# Order preservation is necessary to to create `prop_list`in Augmented properties


class MolecularProperty(ABC):
    """
    Abstract base class representing a molecular property.

    Properties can be atom-level, bond-level, or molecule-level.
    Each property is associated with a PropertyEncoder that encodes
    the raw property values into suitable feature representations.

    Args:
        encoder: Optional encoder instance to encode property values.
                 Defaults to IndexEncoder if not provided.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        if encoder is None:
            encoder = IndexEncoder(self)
        self.encoder: PropertyEncoder = encoder

    @property
    def name(self) -> str:
        """
        Unique identifier for this property, typically the class name.

        Returns:
            The class name as the property's unique name.
        """
        return self.__class__.__name__

    def on_finish(self) -> None:
        """
        Called after dataset processing is complete.

        Typically used to finalize encoder states, e.g., saving cache.
        """
        self.encoder.on_finish()

    def __str__(self) -> str:
        """
        String representation of the property.

        Returns:
            The property name.
        """
        return self.name

    @abstractmethod
    def get_property_value(self, mol: Chem.rdchem.Mol | dict) -> list:
        """
        Abstract method to extract the raw property value(s) from a molecule.

        Args:
            mol: RDKit molecule object or a dictionary representation.

        Returns:
            A list of raw property values for the molecule.
        """
        ...


class AtomProperty(MolecularProperty, ABC):
    """
    Abstract base class representing an atom-level molecular property.

    Subclasses must implement get_atom_value to extract property per atom.
    """

    def get_property_value(self, mol: Chem.rdchem.Mol) -> list:
        """
        Extract the property value for each atom in the molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            List of property values, one per atom.
        """
        return [self.get_atom_value(atom) for atom in mol.GetAtoms()]

    @abstractmethod
    def get_atom_value(self, atom: Chem.rdchem.Atom) -> object:
        """
        Abstract method to extract the property value of a single atom.

        Args:
            atom: RDKit atom object.

        Returns:
            The property value for the atom.
        """
        pass


class BondProperty(MolecularProperty, ABC):
    """
    Abstract base class representing a bond-level molecular property.

    Subclasses must implement get_bond_value to extract property per bond.
    """

    def get_property_value(self, mol: Chem.rdchem.Mol) -> list:
        """
        Extract the property value for each bond in the molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            List of property values, one per bond.
        """
        return [self.get_bond_value(bond) for bond in mol.GetBonds()]

    @abstractmethod
    def get_bond_value(self, bond: Chem.rdchem.Bond) -> object:
        """
        Abstract method to extract the property value of a single bond.

        Args:
            bond: RDKit bond object.

        Returns:
            The property value for the bond.
        """
        pass


class MoleculeProperty(MolecularProperty, ABC):
    """
    Class representing a global (molecule-level) property.

    Subclasses should override get_property_value for molecule-wide values.
    """

    pass


class FrozenPropertyAlias(MolecularProperty, ABC):
    """
    Wrapper base class for augmented graph properties that reuse existing molecular properties.

    This allows an augmented property class (with an 'Aug' prefix in its name) to:
    - Reuse the encoder and index files of the base property by removing the 'Aug' prefix from its name.
    - Prevent adding new tokens to the encoder cache by freezing it (using MappingProxyType).

    Usage:
        Inherit from FrozenPropertyAlias and the desired base molecular property class,
        and name the class with an 'Aug' prefix (e.g., 'AugAtomType').

    Example:
        ```python
        class AugAtomType(FrozenPropertyAlias, AtomType): ...
        ```

    Raises:
        ValueError: If new tokens are added to the frozen encoder during processing.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder)
        # Lock the encoder's cache to prevent adding new tokens
        if hasattr(self.encoder, "cache") and isinstance(self.encoder.cache, dict):
            self.encoder.cache = MappingProxyType(self.encoder.cache)

    @property
    def name(self) -> str:
        """
        Unique identifier for this property.

        Returns:
            The class name with the 'Aug' prefix removed if present,
            allowing reuse of the base property encoder/index files.
        """
        class_name = self.__class__.__name__
        return class_name[3:] if class_name.startswith("Aug") else class_name

    def on_finish(self) -> None:
        """
        Called after dataset processing.

        Ensures no new tokens were added to the frozen encoder cache.
        Raises an error if this condition is violated.
        """
        if (
            hasattr(self.encoder, "cache")
            and len(self.encoder.cache) > self.encoder.index_length_start
        ):
            raise ValueError(
                f"{self.__class__.__name__} attempted to add new tokens "
                f"to a frozen encoder at {self.encoder.index_path}"
            )
        super().on_finish()


class AugmentedAtomProperty(AtomProperty, ABC):
    MAIN_KEY = "nodes"

    def get_property_value(self, augmented_mol: dict) -> list:
        """
        Extract property values for atoms from the augmented molecule dictionary.

        Args:
            augmented_mol (dict): Dictionary representing the augmented molecule.

        Raises:
            KeyError: If required keys are missing in the dictionary.
            TypeError: If types of contained objects are incorrect.
            AssertionError: If the number of property values does not match number of nodes.

        Returns:
            list: List of property values for all atoms, functional groups, and graph nodes.
        """
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = {"atom_nodes"} - augmented_mol[self.MAIN_KEY].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Chem.Mol = augmented_mol[self.MAIN_KEY]["atom_nodes"]
        if not isinstance(atom_molecule, Chem.Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["atom_nodes"] must be an instance of rdkit.Chem.Mol'
            )
        prop_list = [self.get_atom_value(atom) for atom in atom_molecule.GetAtoms()]

        if "fg_nodes" in augmented_mol[self.MAIN_KEY]:
            fg_nodes = augmented_mol[self.MAIN_KEY]["fg_nodes"]
            if not isinstance(fg_nodes, dict):
                raise TypeError(
                    f'augmented_mol["{self.MAIN_KEY}"](["fg_nodes"]) must be an instance of dict '
                    f"containing its properties"
                )
            prop_list.extend([self.get_atom_value(atom) for atom in fg_nodes.values()])

        if "graph_node" in augmented_mol[self.MAIN_KEY]:
            graph_node = augmented_mol[self.MAIN_KEY]["graph_node"]
            if not isinstance(graph_node, dict):
                raise TypeError(
                    f'augmented_mol["{self.MAIN_KEY}"](["graph_node"]) must be an instance of dict '
                    f"containing its properties"
                )
            prop_list.append(self.get_atom_value(graph_node))

        assert (
            len(prop_list) == augmented_mol[self.MAIN_KEY]["num_nodes"]
        ), "Number of property values should be equal to number of nodes"
        return prop_list

    def _check_modify_atom_prop_value(
        self, atom: Chem.rdchem.Atom | dict, prop: str
    ) -> str | int | bool:
        """
        Check that the property value for the atom/node exists and is not empty.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node representation.
            prop (str): Property name.

        Raises:
            ValueError: If the property is empty.

        Returns:
            str | int | bool: The property value.
        """
        value = self._get_atom_prop_value(atom, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    def _get_atom_prop_value(
        self, atom: Chem.rdchem.Atom | dict, prop: str
    ) -> str | int | bool:
        """
        Retrieve a property value from an atom or dict node.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.
            prop (str): Property name.

        Raises:
            TypeError: If atom is not an expected type.

        Returns:
            str | int | bool: The property value.
        """
        if isinstance(atom, Chem.rdchem.Atom):
            return atom.GetProp(prop)
        elif isinstance(atom, dict):
            return atom[prop]
        else:
            raise TypeError(
                f"Atom/Node in key `{self.MAIN_KEY}` should be of type `Chem.rdchem.Atom` or `dict`."
            )


class AtomNodeTypeProperty(AugmentedAtomProperty, ABC): ...


class FGNodeTypeProperty(AugmentedAtomProperty, ABC): ...


class AllNodeTypeProperty(AugmentedAtomProperty, ABC): ...


class AugmentedBondProperty(BondProperty, ABC):
    MAIN_KEY = "edges"

    def get_property_value(self, augmented_mol: dict) -> list:
        """
        Get bond property values from augmented molecule dict.

        Args:
            augmented_mol (dict): Augmented molecule dictionary containing edges.

        Returns:
            list: List of property values for bonds in the augmented molecule.

        Raises:
            KeyError: If required keys are missing in augmented_mol.
            TypeError: If the expected objects are not of correct types.
            AssertionError: If number of property values does not match expected edge count.
        """
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = {k.WITHIN_ATOMS_EDGE} - augmented_mol[self.MAIN_KEY].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Chem.Mol = augmented_mol[self.MAIN_KEY][k.WITHIN_ATOMS_EDGE]
        if not isinstance(atom_molecule, Chem.Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["{k.WITHIN_ATOMS_EDGE}"] must be an instance of rdkit.Chem.Mol'
            )
        prop_list = [self.get_bond_value(bond) for bond in atom_molecule.GetBonds()]

        if k.ATOM_FG_EDGE in augmented_mol[self.MAIN_KEY]:
            fg_atom_edges = augmented_mol[self.MAIN_KEY][k.ATOM_FG_EDGE]
            if not isinstance(fg_atom_edges, dict):
                raise TypeError(
                    f"augmented_mol['{self.MAIN_KEY}'](['{k.ATOM_FG_EDGE}'])"
                    f"must be an instance of dict containing its properties"
                )
            prop_list.extend(
                [self.get_bond_value(bond) for bond in fg_atom_edges.values()]
            )

        if k.WITHIN_FG_EDGE in augmented_mol[self.MAIN_KEY]:
            fg_edges = augmented_mol[self.MAIN_KEY][k.WITHIN_FG_EDGE]
            if not isinstance(fg_edges, dict):
                raise TypeError(
                    f"augmented_mol['{self.MAIN_KEY}'](['{k.WITHIN_FG_EDGE}'])"
                    f"must be an instance of dict containing its properties"
                )
            prop_list.extend([self.get_bond_value(bond) for bond in fg_edges.values()])

        if k.TO_GRAPHNODE_EDGE in augmented_mol[self.MAIN_KEY]:
            fg_graph_node_edges = augmented_mol[self.MAIN_KEY][k.TO_GRAPHNODE_EDGE]
            if not isinstance(fg_graph_node_edges, dict):
                raise TypeError(
                    f"augmented_mol['{self.MAIN_KEY}'](['{k.TO_GRAPHNODE_EDGE}'])"
                    f"must be an instance of dict containing its properties"
                )
            prop_list.extend(
                [self.get_bond_value(bond) for bond in fg_graph_node_edges.values()]
            )

        num_directed_edges = augmented_mol[self.MAIN_KEY][k.NUM_EDGES] // 2
        assert (
            len(prop_list) == num_directed_edges
        ), f"Number of property values ({len(prop_list)}) should be equal to number of half the number of undirected edges i.e. must be equal to {num_directed_edges} "

        return prop_list

    def _check_modify_bond_prop_value(
        self, bond: Chem.rdchem.Bond | dict, prop: str
    ) -> str:
        """
        Helper to check and get bond property value.

        Args:
            bond (Chem.rdchem.Bond | dict): Bond object or bond property dict.
            prop (str): Property key to get.

        Returns:
            str: Property value.

        Raises:
            ValueError: If value is empty or falsy.
        """
        value = self._get_bond_prop_value(bond, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    @staticmethod
    def _get_bond_prop_value(bond: Chem.rdchem.Bond | dict, prop: str) -> str:
        """
        Extract bond property value from bond or dict.

        Args:
            bond (Chem.rdchem.Bond | dict): Bond object or dict.
            prop (str): Property key.

        Returns:
            str: Property value.

        Raises:
            TypeError: If bond is not the expected type.
        """
        if isinstance(bond, Chem.rdchem.Bond):
            return bond.GetProp(prop)
        elif isinstance(bond, dict):
            return bond[prop]
        else:
            raise TypeError("Bond/Edge should be of type `Chem.rdchem.Bond` or `dict`.")


class AugmentedMoleculeProperty(MoleculeProperty, ABC):
    def get_property_value(self, augmented_mol: dict) -> list:
        """
        Get molecular property values from augmented molecule dict.
        Args:
            augmented_mol (dict): Augmented molecule dict.
        Returns:
            list: Property values of molecule.
        """
        mol: Chem.Mol = augmented_mol[AugmentedAtomProperty.MAIN_KEY]["atom_nodes"]
        assert isinstance(mol, Chem.Mol), "Molecule should be instance of `Chem.Mol`"
        return super().get_property_value(mol)
