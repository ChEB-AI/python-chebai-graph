from abc import ABC

from rdkit import Chem

from chebai_graph.preprocessing.property_encoder import (
    BoolEncoder,
    OneHotEncoder,
    PropertyEncoder,
)

from . import constants as k
from . import properties as pr
from .base import (
    AllNodeTypeProperty,
    AtomNodeTypeProperty,
    AugmentedBondProperty,
    AugmentedMoleculeProperty,
    FGNodeTypeProperty,
    FrozenPropertyAlias,
)

# --------------------- Atom Properties -----------------------------


class AtomNodeLevel(AllNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Initialize AtomNodeLevel with an optional encoder.

        Args:
            encoder (PropertyEncoder | None): Property encoder to use. Defaults to OneHotEncoder.
        """
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> str | int | bool:
        """
        Get the node level property for a given atom/node.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.

        Returns:
            str | int | bool: Property value.
        """
        return self._check_modify_atom_prop_value(atom, k.NODE_LEVEL)


class AtomFunctionalGroup(FGNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Initialize AtomFunctionalGroup with an optional encoder.

        Args:
            encoder (PropertyEncoder | None): Property encoder to use. Defaults to OneHotEncoder.
        """
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> str | int | bool:
        """
        Get the functional group property for a given atom/node.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.

        Returns:
            str | int | bool: Property value.
        """
        return self._check_modify_atom_prop_value(atom, "FG")


class AtomRingSize(FGNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Initialize AtomRingSize with an optional encoder.

        Args:
            encoder (PropertyEncoder | None): Property encoder to use. Defaults to OneHotEncoder.
        """
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> int:
        """
        Get the ring size for a given atom/node.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.

        Returns:
            int: Maximum ring size the atom belongs to, or 0 if none.
        """
        return self._check_modify_atom_prop_value(atom, "RING")

    def _check_modify_atom_prop_value(
        self, atom: Chem.rdchem.Atom | dict, prop: str
    ) -> int:
        """
        Override to parse and return maximum ring size from a property string.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.
            prop (str): Property name.

        Returns:
            int: Maximum ring size or 0.
        """
        ring_size_str = self._get_atom_prop_value(atom, prop)
        if ring_size_str:
            ring_sizes = list(map(int, str(ring_size_str).split("-")))
            # TODO: Decide ring size for atoms belongs to fused rings, rn only max ring size taken
            return max(ring_sizes)
        else:
            return 0


class IsHydrogenBondDonorFG(FGNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Initialize IsHydrogenBondDonorFG with an optional encoder.

        Args:
            encoder (PropertyEncoder | None): Property encoder to use. Defaults to BoolEncoder.
        """
        super().__init__(encoder or BoolEncoder(self))
        # fmt: off
        # https://github.com/thaonguyen217/farm_molecular_representation/blob/main/src/(6)gen_FG_KG.py#L26-L31
        self._hydrogen_bond_donor: set[str] = {
            'hydroxyl', 'hydroperoxy', 'primary_amine', 'secondary_amine',
            'hydrazone', 'primary_ketimine', 'secondary_ketimine', 'primary_aldimine',
            'amide', 'sulfhydryl', 'sulfonic_acid', 'thiolester', 'hemiacetal',
            'hemiketal', 'carboxyl', 'aldoxime', 'ketoxime'
        }
        # fmt: on

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> bool:
        """
        Check if the atom's functional group is a hydrogen bond donor.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom or node.

        Returns:
            bool: True if hydrogen bond donor, else False.
        """
        fg = self._check_modify_atom_prop_value(atom, "FG")
        return fg in self._hydrogen_bond_donor


class IsHydrogenBondAcceptorFG(FGNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Initialize IsHydrogenBondAcceptorFG with an optional encoder.

        Args:
            encoder (PropertyEncoder | None): Property encoder to use. Defaults to BoolEncoder.
        """
        super().__init__(encoder or BoolEncoder(self))
        # fmt: off
        # https://github.com/thaonguyen217/farm_molecular_representation/blob/main/src/(6)gen_FG_KG.py#L33-L39
        self._hydrogen_bond_acceptor: set[str] = {
            'ether', 'peroxy', 'haloformyl', 'ketone', 'aldehyde', 'carboxylate',
            'carboxyl', 'ester', 'ketal', 'carbonate_ester', 'carboxylic_anhydride',
            'primary_amine', 'secondary_amine', 'tertiary_amine', '4_ammonium_ion',
            'hydrazone', 'primary_ketimine', 'secondary_ketimine', 'primary_aldimine',
            'amide', 'sulfhydryl', 'sulfonic_acid', 'thiolester', 'aldoxime', 'ketoxime'
        }
        # fmt: on

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> bool:
        """
        Determine if the atom is a hydrogen bond acceptor.

        Args:
            atom (Chem.rdchem.Atom | dict): The atom object or a dictionary of atom properties.

        Returns:
            bool: True if the atom is a hydrogen bond acceptor, False otherwise.
        """
        fg = self._check_modify_atom_prop_value(atom, "FG")
        return fg in self._hydrogen_bond_acceptor


class IsFGAlkyl(FGNodeTypeProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Args:
            encoder (PropertyEncoder | None): Optional encoder to use for this property.
                Defaults to BoolEncoder if not provided.
        """
        super().__init__(encoder or BoolEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> int:
        """
        Get the alkyl group status of the given atom.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom object or atom property dictionary.

        Returns:
            int: 1 if alkyl, 0 otherwise.
        """
        return int(self._check_modify_atom_prop_value(atom, "is_alkyl"))


class AugNodeValueDefaulter(AtomNodeTypeProperty, FrozenPropertyAlias, ABC):
    def get_atom_value(self, atom: Chem.rdchem.Atom | dict) -> int | None:
        """
        Get the property value for an atom or dict node.

        Args:
            atom (Chem.rdchem.Atom | dict): Atom object or dict representing node properties.

        Returns:
            int | None: Property value or None for dict nodes.

        Raises:
            TypeError: If input is neither Chem.rdchem.Atom nor dict.
        """
        if isinstance(atom, Chem.rdchem.Atom):
            # Delegate to superclass method for atom
            return super().get_atom_value(atom)
        elif isinstance(atom, dict):
            return None
        else:
            raise TypeError(
                f"Expected Chem.rdchem.Atom or dict, got {type(atom).__name__}"
            )


class AugAtomType(AugNodeValueDefaulter, pr.AtomType):
    """
    This property uses OneHotEncoder as default encoder

    TODO: Can we return 0 for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    Currently, we return None which leads to zero-tensor for augmented nodes

    RDKit uses 0 as the atomic number for a "dummy atom", which usually means:
    - A placeholder atom (e.g. [*], R#, or attachment points in SMARTS/SMILES).
    - An undefined or wildcard atom.
    - A pseudoatom (e.g., for certain fragments or placeholders in reaction centers).
    """

    ...


class AugNumAtomBonds(AugNodeValueDefaulter, pr.NumAtomBonds):
    """
    This property uses OneHotEncoder as default encoder

    Default return value for this property can't be zero, 0 is used for isolated atoms in molecule.
    It has to be None or actual node degree.

    TODO: Can return actual node degree/num of connections for augmented Nodes for this property?
    which will lead to use of one hot tensor for augmented nodes

    Currently, we return None which leads to zero-tensor for augmented nodes

    But then the question aries shall we count only the atoms connected to a fg node, or all nodes including atoms.
    Consider graph node too.
    """

    ...


class AugAtomCharge(AugNodeValueDefaulter, pr.AtomCharge):
    """
    This property uses OneHotEncoder as default encoder

    Default return value for this property can't be zero, as atoms can have 0 charge.

    TODO: Can return some `unk` value for augmented Nodes for this property?
    which will lead to use of one hot tensor for augmented nodes

    Currently, we return None which leads to zero-tensor for augmented nodes
    """

    ...


class AugAtomHybridization(AugNodeValueDefaulter, pr.AtomHybridization):
    """
    This property uses OneHotEncoder as default encoder

    TODO: Can return some `HybridizationType.UNSPECIFIED` value which is 0 for augmented Nodes for this property?
    which will lead to use of one hot tensor for augmented nodes

    Check: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.HybridizationType

    Currently, we return None which leads to zero-tensor for augmented nodes
    """

    ...


class AugAtomNumHs(AugNodeValueDefaulter, pr.AtomNumHs):
    """
    This property uses OneHotEncoder as default encoder

    Default return value for this property can't be zero, as atoms can have 0 Hydrogen atoms attached
    which mean atoms is full balanced by bonding with other non-hydrogen atoms.

    TODO: Can return some `unk` value for augmented Nodes for this property?
    which will lead to use of one hot tensor for augmented nodes

    Currently, we return None which leads to zero-tensor for augmented nodes
    """

    ...


class AugAtomAromaticity(AugNodeValueDefaulter, pr.AtomAromaticity):
    """
    This property uses BoolEncoder as default encoder

    Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.

    This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    """

    ...


# --------------------- Bond Properties ------------------------------


class BondLevel(AugmentedBondProperty):
    def __init__(self, encoder: PropertyEncoder | None = None):
        """
        Args:
            encoder (PropertyEncoder | None): Optional encoder to use. Defaults to OneHotEncoder.
        """
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond | dict) -> str:
        """
        Get the bond level property value.

        Args:
            bond (Chem.rdchem.Bond | dict): Bond or bond dict.

        Returns:
            str: Bond level property.
        """
        return self._check_modify_bond_prop_value(bond, k.EDGE_LEVEL)


class AugBondValueDefaulter(AugmentedBondProperty, FrozenPropertyAlias, ABC):
    def get_bond_value(self, bond: Chem.rdchem.Bond | dict) -> str | None:
        """
        Get bond property value or None for dict bonds.

        Args:
            bond (Chem.rdchem.Bond | dict): Bond or bond dict.

        Returns:
            str | None: Property value or None for dict.

        Raises:
            TypeError: If input type is invalid.
        """
        if isinstance(bond, Chem.rdchem.Bond):
            # Delegate to superclass method for bond
            return super().get_bond_value(bond)
        elif isinstance(bond, dict):
            return None
        else:
            raise TypeError("Bond/Edge should be of type `Chem.rdchem.Bond` or `dict`.")


class AugBondAromaticity(AugBondValueDefaulter, pr.BondAromaticity):
    """
    This property uses BoolEncoder as default encoder

    Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.

    This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    """

    ...


class AugBondType(AugBondValueDefaulter, pr.BondType):
    """
    This property uses OneHotEncoder as default encoder

    TODO: Can return some `BondType.UNSPECIFIED` value which is 0 for augmented Nodes for this property?
    which will lead to use of one hot tensor for augmented nodes

    Check: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType

    Currently, we return None which leads to zero-tensor for augmented nodes
    """

    ...


class AugBondInRing(AugBondValueDefaulter, pr.BondInRing):
    """
    This property uses BoolEncoder as default encoder

    Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.

    This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    """

    ...


# --------------------- Molecule Properties ------------------------------


class AugRDKit2DNormalized(AugmentedMoleculeProperty, pr.RDKit2DNormalized): ...
