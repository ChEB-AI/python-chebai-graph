from abc import ABC, abstractmethod
from typing import Dict, Optional

from rdkit import Chem

from chebai_graph.preprocessing.properties import MolecularProperty
from chebai_graph.preprocessing.properties.constants import *
from chebai_graph.preprocessing.property_encoder import OneHotEncoder, PropertyEncoder


class AugmentedBondProperty(MolecularProperty, ABC):
    MAIN_KEY = "edges"

    def get_property_value(self, augmented_mol: Dict):
        if self.MAIN_KEY not in augmented_mol:
            raise KeyError(
                f"Key `{self.MAIN_KEY}` should be present in augmented molecule dict"
            )

        missing_keys = EDGE_LEVELS - augmented_mol[self.MAIN_KEY].keys()
        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} in augmented molecule nodes")

        atom_molecule: Chem.Mol = augmented_mol[self.MAIN_KEY][WITHIN_ATOMS_EDGE]
        if not isinstance(atom_molecule, Chem.Mol):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"]["{WITHIN_ATOMS_EDGE}"] must be an instance of rdkit.Chem.Mol'
            )

        prop_list = [self.get_bond_value(bond) for bond in atom_molecule.GetBonds()]

        fg_atom_edges = augmented_mol[self.MAIN_KEY][ATOM_FG_EDGE]
        fg_edges = augmented_mol[self.MAIN_KEY][WITHIN_FG_EDGE]
        fg_graph_node_edges = augmented_mol[self.MAIN_KEY][FG_GRAPHNODE_EDGE]

        if (
            not isinstance(fg_atom_edges, dict)
            or not isinstance(fg_edges, dict)
            or not isinstance(fg_graph_node_edges, dict)
        ):
            raise TypeError(
                f'augmented_mol["{self.MAIN_KEY}"](["{ATOM_FG_EDGE}"]/["{WITHIN_FG_EDGE}"]/["{FG_GRAPHNODE_EDGE}"]) '
                f"must be an instance of dict containing its properties"
            )

        # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
        # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
        # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        prop_list.extend([self.get_bond_value(bond) for bond in fg_atom_edges])
        prop_list.extend([self.get_bond_value(bond) for bond in fg_edges])
        prop_list.extend([self.get_bond_value(bond) for bond in fg_graph_node_edges])

        return prop_list

    @abstractmethod
    def get_bond_value(self, bond: Chem.rdchem.Bond | Dict):
        pass

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


class AugmentedAtomProperty(MolecularProperty, ABC):
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

        return prop_list

    @abstractmethod
    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        pass

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
        return self._check_modify_atom_prop_value(atom, NODE_LEVEL)


class AtomFunctionalGroup(AugmentedAtomProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

        # To avoid circular imports
        from chebai_graph.preprocessing.reader import RuleBasedFGReader

        self.fg_reader = RuleBasedFGReader()

    def get_atom_value(self, atom: Chem.rdchem.Atom | Dict):
        return self._check_modify_atom_prop_value(atom, "FG")

    def _get_atom_prop_value(self, atom: Chem.rdchem.Atom | Dict, prop: str):
        if isinstance(atom, Chem.rdchem.Atom):
            return self.fg_reader._read_data(atom.GetProp(prop))  # noqa
        elif isinstance(atom, dict):
            return self.fg_reader._read_data(atom[prop])  # noqa
        else:
            raise TypeError("Atom/Node should be of type `Chem.rdchem.Atom` or `dict`.")


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


class BondLevel(AugmentedBondProperty):
    def __init__(self, encoder: Optional[PropertyEncoder] = None):
        super().__init__(encoder or OneHotEncoder(self))

    def get_bond_value(self, bond: Chem.rdchem.Bond | Dict):
        return self._check_modify_bond_prop_value(bond, EDGE_LEVEL)
