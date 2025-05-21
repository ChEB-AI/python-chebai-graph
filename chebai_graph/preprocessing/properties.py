import abc
from typing import Optional

import numpy as np
import rdkit.Chem as Chem
from descriptastorus.descriptors import rdNormalizedDescriptors

from chebai_graph.preprocessing.property_encoder import (
    AsIsEncoder,
    BoolEncoder,
    IndexEncoder,
    OneHotEncoder,
    PropertyEncoder,
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

    def get_property_value(self, mol: Chem.rdchem.Mol):
        raise NotImplementedError


class AtomProperty(MolecularProperty):
    """Property of an atom."""

    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_atom_value(atom) for atom in mol.GetAtoms()]

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return NotImplementedError


class BondProperty(MolecularProperty):
    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_bond_value(bond) for bond in mol.GetBonds()]

    def get_bond_value(self, bond: Chem.rdchem.Bond):
        return NotImplementedError


class MoleculeProperty(MolecularProperty):
    """Global property of a molecule."""


class AtomType(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetAtomicNum()


class NumAtomBonds(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetDegree()


class AtomCharge(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetFormalCharge()


class AtomChirality(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetChiralTag()


class AtomHybridization(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetHybridization()


class AtomNumHs(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetTotalNumHs()


class AtomAromaticity(AtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or BoolEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return atom.GetIsAromatic()


class BondAromaticity(BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or BoolEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond):
        return bond.GetIsAromatic()


class BondType(BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond):
        return bond.GetBondType()


class BondInRing(BondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or BoolEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond):
        return bond.IsInRing()


class MoleculeNumRings(MolecularProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [mol.GetRingInfo().NumRings()]


class RDKit2DNormalized(MolecularProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or AsIsEncoder(self))

    def get_property_value(self, mol: Chem.rdchem.Mol):
        generator_normalized = rdNormalizedDescriptors.RDKit2DNormalized()
        features_normalized = generator_normalized.processMol(
            mol, Chem.MolToSmiles(mol)
        )
        np.nan_to_num(features_normalized, copy=False)
        return [features_normalized[1:]]
