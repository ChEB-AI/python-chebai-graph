import abc
import networkx as nx
import rdkit.Chem as Chem
import os


class MolecularProperty(abc.ABC):
    @property
    def name(self):
        """Unique identifier for this property."""
        return self.__class__.__name__

    def encode_property_value(self):
        """Encode the property value (e.g., one-hot, index)."""
        raise NotImplementedError

    def on_finish(self):
        """Called after dataset processing is done."""
        return


class IndexEncodingMixIn(MolecularProperty):
    """Enocde property values as indices."""

    def __init__(self):
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        with open(self.index_path, "r") as pk:
            self.cache = [x.strip() for x in pk]
        self.index_length_start = len(self.cache)
        self.offset = 0

    @property
    def index_path(self):
        """Get path to store indices of property values, create file if it does not exist yet"""
        dirname = os.path.dirname(os.path.realpath(__file__))
        index_path = os.path.join(dirname, "bin", self.name, "indices.txt")
        os.makedirs(os.path.join(dirname, "bin", self.name), exist_ok=True)
        if not os.path.exists(index_path):
            with open(index_path, "x"):
                pass
        return index_path

    def on_finish(self):
        """Save cache"""
        with open(self.index_path, "w") as pk:
            new_length = len(self.cache) - self.index_length_start
            print(
                f"saving index of property {self.name} to {self.index_path}, "
                f"index length: {len(self.cache)} (new: {new_length})"
            )
            pk.writelines([f"{c}\n" for c in self.cache])

    def encode_property_value(self, token):
        """Returns a unique number for each token, automatically adds new tokens"""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return self.cache.index(str(token)) + self.offset


class AtomProperty(MolecularProperty):
    """Property of an atom."""

    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        """Calculate the value of this property for a given atom."""
        raise NotImplementedError


class AtomPropertyIndexEncoded(AtomProperty, IndexEncodingMixIn):
    pass


class BondProperty(MolecularProperty):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        """Calculate the value of this property for a given bond."""
        raise NotImplementedError


class BondPropertyIndexEncoded(BondProperty, IndexEncodingMixIn):
    pass


class AtomType(AtomPropertyIndexEncoded):
    def get_atom_property_value(self, atom):
        return atom.GetAtomicNum()


class NumAtomBonds(AtomPropertyIndexEncoded):
    def get_atom_property_value(self, atom):
        return atom.GetDegree()


class FormalCharge(AtomPropertyIndexEncoded):
    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetFormalCharge()


class Aromaticity(AtomPropertyIndexEncoded, BondPropertyIndexEncoded):
    def get_atom_property_value(self, atom: Chem.rdchem.Atom):
        return atom.GetIsAromatic()

    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetIsAromatic()


class BondType(BondPropertyIndexEncoded):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.GetBondType()


class BondInRing(BondPropertyIndexEncoded):
    def get_bond_property_value(self, bond: Chem.rdchem.Bond):
        return bond.IsInRing()
