import numpy as np
import rdkit.Chem as Chem
from descriptastorus.descriptors import rdNormalizedDescriptors

from chebai_graph.preprocessing.property_encoder import (
    AsIsEncoder,
    BoolEncoder,
    OneHotEncoder,
    PropertyEncoder,
)

from .base import AtomProperty, BondProperty, MoleculeProperty


class AtomType(AtomProperty):
    """
    Atom property representing the atomic number (type) of an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> int:
        """
        Get the atomic number of the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            int: Atomic number of the atom.
        """
        return atom.GetAtomicNum()


class NumAtomBonds(AtomProperty):
    """
    Atom property representing the number of bonds (degree) of an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> int:
        """
        Get the number of bonds for the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            int: Number of bonds (degree).
        """
        return atom.GetDegree()


class AtomCharge(AtomProperty):
    """
    Atom property representing the formal charge of an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> int:
        """
        Get the formal charge of the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            int: Formal charge.
        """
        return atom.GetFormalCharge()


class AtomChirality(AtomProperty):
    """
    Atom property representing the chirality tag of an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> Chem.rdchem.ChiralType:
        """
        Get the chirality tag of the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            Chem.rdchem.ChiralType: Chirality tag.
        """
        return atom.GetChiralTag()


class AtomHybridization(AtomProperty):
    """
    Atom property representing the hybridization state of an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> Chem.rdchem.HybridizationType:
        """
        Get the hybridization state of the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            Chem.rdchem.HybridizationType: Hybridization state.
        """
        return atom.GetHybridization()


class AtomNumHs(AtomProperty):
    """
    Atom property representing the total number of hydrogens bonded to an atom.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> int:
        """
        Get the total number of hydrogens attached to the atom.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            int: Number of attached hydrogens.
        """
        return atom.GetTotalNumHs()


class AtomAromaticity(AtomProperty):
    """
    Atom property representing whether an atom is aromatic.

    Uses a boolean encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or BoolEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom) -> bool:
        """
        Check if the atom is aromatic.

        Args:
            atom (Chem.rdchem.Atom): RDKit atom object.

        Returns:
            bool: True if aromatic, else False.
        """
        return atom.GetIsAromatic()


class BondAromaticity(BondProperty):
    """
    Bond property representing whether a bond is aromatic.

    Uses a boolean encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or BoolEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond) -> bool:
        """
        Check if the bond is aromatic.

        Args:
            bond (Chem.rdchem.Bond): RDKit bond object.

        Returns:
            bool: True if aromatic, else False.
        """
        return bond.GetIsAromatic()


class BondType(BondProperty):
    """
    Bond property representing the bond type (single, double, etc.).

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond) -> Chem.rdchem.BondType:
        """
        Get the bond type.

        Args:
            bond (Chem.rdchem.Bond): RDKit bond object.

        Returns:
            Chem.rdchem.BondType: Type of bond.
        """
        return bond.GetBondType()


class BondInRing(BondProperty):
    """
    Bond property indicating whether a bond is in a ring.

    Uses a boolean encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or BoolEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond) -> bool:
        """
        Check if the bond is part of a ring.

        Args:
            bond (Chem.rdchem.Bond): RDKit bond object.

        Returns:
            bool: True if in a ring, else False.
        """
        return bond.IsInRing()


class MoleculeNumRings(MoleculeProperty):
    """
    Molecule-level property representing the number of rings in the molecule.

    Uses a one-hot encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or OneHotEncoder(self))

    def get_property_value(self, mol: Chem.rdchem.Mol) -> list[int]:
        """
        Get the number of rings in the molecule.

        Args:
            mol (Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            list[int]: List with single integer representing number of rings.
        """
        return [mol.GetRingInfo().NumRings()]


class RDKit2DNormalized(MoleculeProperty):
    """
    Molecule-level property representing normalized 2D descriptors from RDKit.

    Uses an identity encoder by default.
    """

    def __init__(self, encoder: PropertyEncoder | None = None) -> None:
        super().__init__(encoder or AsIsEncoder(self))
        self.generator_normalized = rdNormalizedDescriptors.RDKit2DNormalized()
        # Create a dummy molecule (e.g., methane) to extract the length of descriptor vector
        dummy_mol = Chem.MolFromSmiles("C")
        descr_values = self.generator_normalized.processMol(
            dummy_mol, Chem.MolToSmiles(dummy_mol)
        )
        self.encoder.set_encoding_length(len(descr_values) - 1)

    def get_property_value(self, mol: Chem.rdchem.Mol) -> list[np.ndarray]:
        """
        Compute normalized RDKit 2D descriptors for the molecule.

        Args:
            mol (Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            list[np.ndarray]: List containing the descriptor numpy array (excluding first element).
        """
        features_normalized = self.generator_normalized.processMol(
            mol, Chem.MolToSmiles(mol)
        )
        features_normalized = np.nan_to_num(features_normalized)
        return [features_normalized[1:]]
