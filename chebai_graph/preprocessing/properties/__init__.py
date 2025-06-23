# Formating is turned off here, because isort sorts the augmented properties imports in first order,
# but it has to be imported after properties module, to avoid circular imports
# This is because augmented properties module imports from properties module
# isort: off

from .base import MolecularProperty, AtomProperty, BondProperty

from .properties import (
    AtomType,
    NumAtomBonds,
    AtomCharge,
    AtomChirality,
    AtomHybridization,
    AtomNumHs,
    AtomAromaticity,
    BondAromaticity,
    BondType,
    BondInRing,
    MoleculeNumRings,
    RDKit2DNormalized,
)

from .augmented_properties import (
    AugAtomNodeLevel,
    AugAtomFunctionalGroup,
    AugAtomRingSize,
    AugBondLevel,
    AugAtomType,
    AugNumAtomBonds,
    AugAtomCharge,
    AugAtomHybridization,
    AugAtomNumHs,
    AugAtomAromaticity,
    AugBondAromaticity,
    AugBondType,
    AugBondInRing,
    AugRDKit2DNormalized,
)

# isort: on

__all__ = [
    "MolecularProperty",
    "AtomProperty",
    "BondProperty",
    "AtomType",
    "NumAtomBonds",
    "AtomCharge",
    "AtomChirality",
    "AtomHybridization",
    "AtomNumHs",
    "AtomAromaticity",
    "BondAromaticity",
    "BondType",
    "BondInRing",
    "MoleculeNumRings",
    "RDKit2DNormalized",
    # -------- Augmented Molecular Properties --------
    "AugAtomNodeLevel",
    "AugAtomFunctionalGroup",
    "AugAtomRingSize",
    "AugBondLevel",
    "AugAtomType",
    "AugNumAtomBonds",
    "AugAtomCharge",
    "AugAtomHybridization",
    "AugAtomNumHs",
    "AugAtomAromaticity",
    "AugBondAromaticity",
    "AugBondType",
    "AugBondInRing",
    "AugRDKit2DNormalized",
]
