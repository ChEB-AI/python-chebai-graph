# Formating is turned off here, because isort sorts the augmented properties imports in first order,
# but it has to be imported after properties module, to avoid circular imports
# This is because augmented properties module imports from properties module
# isort: off

from .base import (
    MolecularProperty,
    AtomProperty,
    BondProperty,
    MoleculeProperty,
    AllNodeTypeProperty,
    AtomNodeTypeProperty,
    FGNodeTypeProperty,
)

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
    RDKit2DNormalized,
)

from .augmented_properties import (
    AtomNodeLevel,
    AtomFunctionalGroup,
    IsHydrogenBondDonorFG,
    IsHydrogenBondAcceptorFG,
    IsFGAlkyl,
    BondLevel,
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
    # -------------- Properties Base classes --------------
    "MolecularProperty",
    "MoleculeProperty",
    "AtomProperty",
    "BondProperty",
    "AllNodeTypeProperty",
    "AtomNodeTypeProperty",
    "FGNodeTypeProperty",
    # -------------- Regular Properties -----------------
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
    "RDKit2DNormalized",
    # -------- Augmented Molecular Properties ----------
    "AtomNodeLevel",
    "AtomFunctionalGroup",
    "IsHydrogenBondDonorFG",
    "IsHydrogenBondAcceptorFG",
    "IsFGAlkyl",
    "BondLevel",
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
