from abc import ABC, abstractmethod
from types import MappingProxyType

import rdkit.Chem as Chem

from chebai_graph.preprocessing.property_encoder import IndexEncoder, PropertyEncoder


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


class MoleculeProperty(MolecularProperty):
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
        class AugAtomType(FrozenPropertyAlias, AtomType):
            ...
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
