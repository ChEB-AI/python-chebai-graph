from abc import ABC, abstractmethod
from types import MappingProxyType

import rdkit.Chem as Chem

from chebai_graph.preprocessing.property_encoder import IndexEncoder, PropertyEncoder


class MolecularProperty(ABC):
    def __init__(self, encoder: PropertyEncoder | None = None):
        if encoder is None:
            encoder = IndexEncoder(self)
        self.encoder = encoder

    @property
    def name(self):
        """Unique identifier for this property."""
        return self.__class__.__name__

    def on_finish(self):
        """Called after dataset processing is done."""
        self.encoder.on_finish()

    def __str__(self):
        return self.name

    @abstractmethod
    def get_property_value(self, mol: Chem.rdchem.Mol | dict): ...


class AtomProperty(MolecularProperty, ABC):
    """Property of an atom."""

    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_atom_value(atom) for atom in mol.GetAtoms()]

    @abstractmethod
    def get_atom_value(self, atom: Chem.rdchem.Atom):
        pass


class BondProperty(MolecularProperty, ABC):
    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_bond_value(bond) for bond in mol.GetBonds()]

    @abstractmethod
    def get_bond_value(self, bond: Chem.rdchem.Bond):
        pass


class MoleculeProperty(MolecularProperty):
    """Global property of a molecule."""


class FrozenPropertyAlias(MolecularProperty, ABC):
    """
    Wrapper base class for augmented graph properties that want to reuse existing molecular properties.

    This class allows augmented graph property classes to inherit both from this wrapper and a standard
    molecular property (from `.properties`), enabling reuse of their encoders and index files without
    modifying them.

    Key Features:
    - Prevents new tokens from being added to the encoder cache by freezing it.
    - Automatically aligns the property name (used for encoder/index resolution) with the inherited
      base property by removing the "Aug" prefix from the class name.

    Usage:
        The derived class should:
        - Inherit from `FrozenPropertyAlias` **and** a valid base molecular property class.
        - Have a name starting with "Aug" (e.g., `AugAtomType`), which will be resolved to `AtomType`.

    Example:
        ```python
        class AugAtomType(FrozenPropertyAlias, AtomType):
            ...
        ```
    Note:
        Subclass name of this class should with prefix "Aug" for above effect to take place.

    This allows `AugAtomType` to reuse the encoder, index files, and logic of `AtomType` while
    integrating into augmented graph pipelines.
    """

    def __init__(self, encoder: PropertyEncoder | None = None):
        super().__init__(encoder)
        # Lock the encoder's cache to prevent adding new tokens
        if hasattr(self.encoder, "cache") and isinstance(self.encoder.cache, dict):
            self.encoder.cache = MappingProxyType(self.encoder.cache)

    @property
    def name(self):
        """
        Unique identifier for this property, with 'Aug' prefix removed if present.
        This allows the encoder to reuse index files of the corresponding base property.
        """
        class_name = self.__class__.__name__
        return class_name[3:] if class_name.startswith("Aug") else class_name

    def on_finish(self):
        if (
            hasattr(self.encoder, "cache")
            and len(self.encoder.cache) > self.encoder.index_length_start
        ):
            raise ValueError(
                f"{self.__class__.__name__} attempted to add new tokens to a {self.encoder.index_path}"
            )
        super().on_finish()
