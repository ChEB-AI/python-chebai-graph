import abc
from chebai_graph.preprocessing.property_encoder import PropertyEncoder, IndexEncoder
import rdkit.Chem as Chem
from typing import Optional


class MolecularProperty(abc.ABC):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        self._encoder = encoder

    @property
    def encoder(self) -> PropertyEncoder:
        if self._encoder is None:
            self._encoder = IndexEncoder(self)
        return self._encoder

    @property
    def name(self):
        """Unique identifier for this property."""
        return self.__class__.__name__

    def on_finish(self):
        """Called after dataset processing is done."""
        self.encoder.on_finish()


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
    def get_atom_property_value(self, atom):
        return atom.GetAtomicNum()


class NumAtomBonds(AtomProperty):
    def get_atom_property_value(self, atom):
        return atom.GetDegree()


class FormalCharge(AtomProperty):
    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetFormalCharge()


class Chirality(AtomProperty):
    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetChiralTag()


class Aromaticity(AtomProperty, BondProperty):
    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetIsAromatic()

    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetIsAromatic()


class BondType(BondProperty):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetBondType()


class BondInRing(BondProperty):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.IsInRing()
