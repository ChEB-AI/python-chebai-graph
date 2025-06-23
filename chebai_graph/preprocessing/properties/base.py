from abc import ABC, abstractmethod

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
