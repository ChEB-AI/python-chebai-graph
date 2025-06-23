from abc import ABC
from typing import Dict, List, Optional

from rdkit import Chem

from chebai_graph.preprocessing.property_encoder import OneHotEncoder, PropertyEncoder

from . import constants as k
from . import properties as pr
from .base import AtomProperty, BondProperty, FrozenPropertyAlias


# --------------------- Atom Properties -----------------------------
class AugmentedAtomProperty(AtomProperty, ABC):
    MAIN_KEY = "nodes"

    def get_property_value(self, augmented_mol: Dict):
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = {"atom_nodes", "fg_nodes", "graph_node"} - augmented_mol[
            self.MAIN_KEY
        ].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Chem.Mol = augmented_mol[self.MAIN_KEY]["atom_nodes"]
        if not isinstance(atom_molecule, Chem.Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["atom_nodes"] must be an instance of rdkit.Chem.Mol'
            )

        prop_list = [self.get_atom_value(atom) for atom in atom_molecule.GetAtoms()]

        fg_nodes = augmented_mol[self.MAIN_KEY]["fg_nodes"]
        graph_node = augmented_mol[self.MAIN_KEY]["graph_node"]
        if not isinstance(fg_nodes, dict) or not isinstance(graph_node, dict):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"](["fg_nodes"]/["graph_node"]) must be an instance of dict '
                f"containing its properties"
            )

        # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
        # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
        # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        prop_list.extend([self.get_atom_value(atom) for atom in fg_nodes.values()])
        prop_list.append(self.get_atom_value(graph_node))
        assert (
            len(prop_list) == augmented_mol[self.MAIN_KEY]["num_nodes"]
        ), "Number of property values should be equal to number of nodes"

        return prop_list

    def _check_modify_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        value = self._get_atom_prop_value(atom, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    def _get_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        if isinstance(atom, Chem.rdchem.Atom):
            return atom.GetProp(prop)
        elif isinstance(atom, dict):
            return atom[prop]
        else:
            raise TypeError(
                f"Atom/Node in key `{self.MAIN_KEY}` should be of type `Chem.rdchem.Atom` or `dict`."
            )


class AtomNodeLevel(AugmentedAtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        return self._check_modify_atom_prop_value(atom, k.NODE_LEVEL)


class AtomFunctionalGroup(AugmentedAtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        return self._check_modify_atom_prop_value(atom, "FG")


class AtomRingSize(AugmentedAtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        return self._check_modify_atom_prop_value(atom, "RING")

    def _check_modify_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        ring_size_str = self._get_atom_prop_value(atom, prop)
        if ring_size_str:
            ring_sizes = list(map(int, ring_size_str.split("-")))
            # TODO: Decide ring size for atoms belongs to fused rings, rn only max ring size taken
            return max(ring_sizes)
        else:
            return 0


class AugNodeValueDefaulter(AugmentedAtomProperty, FrozenPropertyAlias, ABC):
    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
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
    # This property uses OneHotEncoder as default encoder
    # TODO: Can we return 0 for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Currently, we return None which leads to zero-tensor for augmented nodes

    # RDKit uses 0 as the atomic number for a "dummy atom", which usually means:
    # A placeholder atom (e.g. [*], R#, or attachment points in SMARTS/SMILES).
    # An undefined or wildcard atom.
    # A pseudoatom (e.g., for certain fragments or placeholders in reaction centers).
    ...


class AugNumAtomBonds(AugNodeValueDefaulter, pr.NumAtomBonds):
    # This property uses OneHotEncoder as default encoder
    # Default return value for this property can't be zero, 0 is used for isolated atoms in molecule. It has to be None or actual node degree.
    # TODO: Can return actual node degree/num of connections for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Currently, we return None which leads to zero-tensor for augmented nodes
    # But then the question aries shall we count only the atoms connected to a fg node, or all nodes including atoms. Consider graph node too.
    ...


class AugAtomCharge(AugNodeValueDefaulter, pr.AtomCharge):
    # This property uses OneHotEncoder as default encoder
    # Default return value for this property can't be zero, as atoms can have 0 charge.
    # TODO: Can return some `unk` value for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Currently, we return None which leads to zero-tensor for augmented nodes
    ...


class AugAtomHybridization(AugNodeValueDefaulter, pr.AtomHybridization):
    # This property uses OneHotEncoder as default encoder
    # TODO: Can return some `HybridizationType.UNSPECIFIED` value which is 0 for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Check: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.HybridizationType
    # Currently, we return None which leads to zero-tensor for augmented nodes
    ...


class AugAtomNumHs(AugNodeValueDefaulter, pr.AtomNumHs):
    # This property uses OneHotEncoder as default encoder
    # Default return value for this property can't be zero, as atoms can have 0 Hydrogen atoms attached which mean atoms is full balanced by bonding with other non-hydrogen atoms.
    # TODO: Can return some `unk` value for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Currently, we return None which leads to zero-tensor for augmented nodes
    ...


class AugAtomAromaticity(AugNodeValueDefaulter, pr.AtomAromaticity):
    # This property uses BoolEncoder as default encoder
    # Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.
    # This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    ...


# --------------------- Bond Properties ------------------------------
class AugmentedBondProperty(BondProperty, ABC):
    MAIN_KEY = "edges"

    def get_property_value(self, augmented_mol: Dict) -> List:
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = k.EDGE_LEVELS - augmented_mol[self.MAIN_KEY].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Chem.Mol = augmented_mol[self.MAIN_KEY][k.WITHIN_ATOMS_EDGE]
        if not isinstance(atom_molecule, Chem.Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["{k.WITHIN_ATOMS_EDGE}"] must be an instance of rdkit.Chem.Mol'
            )

        prop_list = [self.get_bond_value(bond) for bond in atom_molecule.GetBonds()]

        fg_atom_edges = augmented_mol[self.MAIN_KEY][k.ATOM_FG_EDGE]
        fg_edges = augmented_mol[self.MAIN_KEY][k.WITHIN_FG_EDGE]
        fg_graph_node_edges = augmented_mol[self.MAIN_KEY][k.FG_GRAPHNODE_EDGE]

        if (
            not isinstance(fg_atom_edges, dict)
            or not isinstance(fg_edges, dict)
            or not isinstance(fg_graph_node_edges, dict)
        ):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"](["{k.ATOM_FG_EDGE}"]/["{k.WITHIN_FG_EDGE}"]/["{k.FG_GRAPHNODE_EDGE}"]) '
                f"must be an instance of dict containing its properties"
            )

        # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
        # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
        # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        prop_list.extend([self.get_bond_value(bond) for bond in fg_atom_edges.values()])
        prop_list.extend([self.get_bond_value(bond) for bond in fg_edges.values()])
        prop_list.extend(
            [self.get_bond_value(bond) for bond in fg_graph_node_edges.values()]
        )

        num_directed_edges = augmented_mol[self.MAIN_KEY][k.NUM_EDGES] // 2
        assert (
            len(prop_list) == num_directed_edges
        ), f"Number of property values ({len(prop_list)}) should be equal to number of half the number of undirected edges i.e. must be equal to {num_directed_edges} "

        return prop_list

    def _check_modify_bond_prop_value(self, bond: Chem.rdchem.Bond | Dict, prop: str):
        value = self._get_bond_prop_value(bond, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    @staticmethod
    def _get_bond_prop_value(bond: Chem.rdchem.Bond | Dict, prop: str):
        if isinstance(bond, Chem.rdchem.Bond):
            return bond.GetProp(prop)
        elif isinstance(bond, dict):
            return bond[prop]
        else:
            raise TypeError("Bond/Edge should be of type `Chem.rdchem.Bond` or `dict`.")


class BondLevel(AugmentedBondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond | Dict):
        return self._check_modify_bond_prop_value(bond, k.EDGE_LEVEL)


class AugBondValueDefaulter(AugmentedBondProperty, FrozenPropertyAlias, ABC):
    def get_bond_value(self, bond: Chem.rdchem.Bond | Dict):
        if isinstance(bond, Chem.rdchem.Bond):
            # Delegate to superclass method for bond
            return super().get_bond_value(bond)
        elif isinstance(bond, dict):
            return None
        else:
            raise TypeError("Bond/Edge should be of type `Chem.rdchem.Bond` or `dict`.")


class AugBondAromaticity(AugBondValueDefaulter, pr.BondAromaticity):
    # This property uses BoolEncoder as default encoder
    # Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.
    # This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    ...


class AugBondType(AugBondValueDefaulter, pr.BondType):
    # This property uses OneHotEncoder as default encoder
    # TODO: Can return some `BondType.UNSPECIFIED` value which is 0 for augmented Nodes for this property? which will lead to use of one hot tensor for augmented nodes
    # Check: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType
    # Currently, we return None which leads to zero-tensor for augmented nodes
    ...


class AugBondInRing(AugBondValueDefaulter, pr.BondInRing):
    # This property uses BoolEncoder as default encoder
    # Currently, we return None for augmented nodes which leads to BoolEncoder setting 0 internally.
    # This is None is right value for augmented nodes its not part of any kind of aromatic ring.
    ...


# --------------------- Molecular Properties ------------------------------
class AugmentedMolecularProperty(pr.MolecularProperty, ABC):
    def get_property_value(self, augmented_mol: Dict) -> list:
        mol: Chem.Mol = augmented_mol[AugmentedAtomProperty.MAIN_KEY]["atom_nodes"]
        assert isinstance(mol, Chem.Mol), "Molecule should be instance of `Chem.Mol`"
        return super().get_property_value(mol)


class AugRDKit2DNormalized(AugmentedMolecularProperty, pr.RDKit2DNormalized): ...
