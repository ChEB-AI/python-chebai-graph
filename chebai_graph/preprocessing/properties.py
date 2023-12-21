import abc
from typing import Optional

import rdkit.Chem as Chem

from chebai_graph.preprocessing.property_encoder import (
    PropertyEncoder,
    IndexEncoder,
    OneHotEncoder,
    AsIsEncoder,
    BoolEncoder,
)


class MolecularProperty(abc.ABC):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
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


class AtomProperty(MolecularProperty):
    """Property of an atom."""

    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        """Calculate the value of this property for a given atom."""
        raise NotImplementedError


class BondProperty(MolecularProperty):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        """Calculate the value of this property for a given bond."""
        raise NotImplementedError


class AtomType(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_property_value(self, atom):
        return atom.GetAtomicNum()


class NumAtomBonds(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or AsIsEncoder(self))

    def get_atom_property_value(self, atom):
        return atom.GetDegree()


class FormalCharge(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetFormalCharge()


class Chirality(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetChiralTag()


class Aromaticity(AtomProperty, BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or BoolEncoder(self))

    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetIsAromatic()

    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetIsAromatic()


class BondType(BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetBondType()


class BondInRing(BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or BoolEncoder(self))

    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.IsInRing()
