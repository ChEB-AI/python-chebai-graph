import abc
from typing import Dict, Optional

import numpy as np
import rdkit.Chem as Chem
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit.Chem import Mol

from chebai_graph.preprocessing.fg_detection.fg_constants import (
    EDGE_LEVELS,
    NODE_LEVEL,
    WITHIN_ATOMS_EDGE,
)
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


class AtomProperty(MolecularProperty, abc):
    """Property of an atom."""

    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_atom_value(atom) for atom in mol.GetAtoms()]

    def get_atom_value(self, atom: Chem.rdchem.Atom):
        return NotImplementedError


class AugmentedAtomProperty(MolecularProperty, abc):
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

        atom_molecule: Mol = augmented_mol[self.MAIN_KEY]["atom_nodes"]
        if not isinstance(atom_molecule, Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["atom_nodes"] must be an instance of rdkit.Chem.Mol'
            )

        prop_list = [self.get_atom_value(atom) for atom in atom_molecule.GetAtoms()]

        fg_nodes = augmented_mol[self.MAIN_KEY]["fg_nodes"]
        graph_node = atom_molecule[self.MAIN_KEY]["graph_node"]
        if not isinstance(fg_nodes, dict) or not isinstance(graph_node, dict):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"](["fg_nodes"]/["graph_node"]) must be an instance of dict '
                f"containing its properties"
            )

        # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
        # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
        # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        prop_list.extend([self.get_atom_value(atom) for atom in fg_nodes])
        prop_list.extend([self.get_atom_value(atom) for atom in graph_node])

        return prop_list

    @abc.abstractmethod
    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        pass

    def _check_modify_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        value = self._get_atom_prop_value(atom, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    @staticmethod
    def _get_atom_prop_value(atom: Chem.rdchem.Atom | Dict, prop: str):
        if isinstance(atom, Chem.rdchem.Atom):
            return atom.GetProp(prop)
        elif isinstance(atom, dict):
            return atom[prop]
        else:
            raise TypeError("Atom/Node should be of type `Chem.rdchem.Atom` or `dict`.")


class BondProperty(MolecularProperty):
    def get_property_value(self, mol: Chem.rdchem.Mol):
        return [self.get_bond_value(bond) for bond in mol.GetBonds()]

    def get_bond_value(self, bond: Chem.rdchem.Bond):
        return NotImplementedError


class AugmentedBondProperty(MolecularProperty, abc):
    MAIN_KEY = "edges"

    def get_property_value(self, augmented_mol: Dict):
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = EDGE_LEVELS - augmented_mol[self.MAIN_KEY].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Mol = augmented_mol[self.MAIN_KEY][WITHIN_ATOMS_EDGE]
        if not isinstance(atom_molecule, Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["atom_nodes"] must be an instance of rdkit.Chem.Mol'
            )

        prop_list = [self.get_atom_value(atom) for atom in atom_molecule.GetAtoms()]

        fg_nodes = augmented_mol[self.MAIN_KEY]["fg_nodes"]
        graph_node = atom_molecule[self.MAIN_KEY]["graph_node"]
        if not isinstance(fg_nodes, dict) or not isinstance(graph_node, dict):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"](["fg_nodes"]/["graph_node"]) must be an instance of dict '
                f"containing its properties"
            )

        # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
        # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
        # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        prop_list.extend([self.get_atom_value(atom) for atom in fg_nodes])
        prop_list.extend([self.get_atom_value(atom) for atom in graph_node])

        return prop_list

    @abc.abstractmethod
    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        pass

    def _check_modify_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        value = self._get_atom_prop_value(atom, prop)
        if not value:
            # Every atom/node should have given value
            raise ValueError(f"'{prop}' is set but empty.")
        return value

    @staticmethod
    def _get_atom_prop_value(atom: Chem.rdchem.Atom | Dict, prop: str):
        if isinstance(atom, Chem.rdchem.Atom):
            return atom.GetProp(prop)
        elif isinstance(atom, dict):
            return atom[prop]
        else:
            raise TypeError("Atom/Node should be of type `Chem.rdchem.Atom` or `dict`.")


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


class AtomNodeLevel(AugmentedAtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        return self._check_modify_atom_prop_value(atom, NODE_LEVEL)


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
        np.nan_to_num(features_normalized)
        return [features_normalized[1:]]
