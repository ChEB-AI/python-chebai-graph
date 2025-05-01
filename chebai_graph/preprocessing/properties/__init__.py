# Formating is turned off here, because isort sorts the augmented properties imports in first order,
# but it has to be imported after properties module, to avoid circular imports
# This is because augmented properties module imports from properties module
# isort: off
from .properties import (
    MolecularProperty,
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
    AtomNodeLevel,
    AtomFunctionalGroup,
    AtomRingSize,
    BondLevel,
)

# isort: on

__all__ = [
    "MolecularProperty",
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
    "AtomNodeLevel",
    "AtomFunctionalGroup",
    "AtomRingSize",
    "BondLevel",
]
