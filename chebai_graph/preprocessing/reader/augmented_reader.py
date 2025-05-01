from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from chebai.preprocessing.reader import ChemDataReader
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from rdkit import Chem
from torch_geometric.data import Data as GeomData
from wandb.integration.torch.wandb_torch import torch

from chebai_graph.preprocessing.collate import GraphCollator
from chebai_graph.preprocessing.fg_detection.rule_based import (
    detect_functional_group,
    get_structure,
    set_atom_map_num,
)
from chebai_graph.preprocessing.properties import MolecularProperty
from chebai_graph.preprocessing.properties.constants import *


class _AugmentorReader(ChemDataReader, ABC):
    COLLATOR = GraphCollator

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.failed_counter = 0
        self.mol_object_buffer = {}

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @abstractmethod
    def _get_augmented_molecule(self, smile: str) -> Tuple[Dict, torch.Tensor]:
        pass

    @abstractmethod
    def _read_data(self, raw_data: str) -> List[int]:
        pass

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.rdchem.Mol]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rank_zero_warn(f"RDKit failed to at parsing {smiles} (returned None)")
            self.failed_counter += 1
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                rank_zero_warn(f"Rdkit failed at sanitizing {smiles}, Error {e}")
                self.failed_counter += 1
        return mol

    def on_finish(self):
        rank_zero_info(f"Failed to read {self.failed_counter} SMILES in total")
        self.mol_object_buffer = {}

    def read_property(self, smiles: str, property: MolecularProperty) -> Optional[List]:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None

        if smiles in self.mol_object_buffer:
            return property.get_property_value(self.mol_object_buffer[smiles])

        augmented_mol, _ = self._get_augmented_molecule(smiles)
        return property.get_property_value(mol)


class GraphFGAugmentorReader(_AugmentorReader):

    @classmethod
    def name(cls):
        return "graph_fg_augmentor"

    def _read_data(self, raw_data):
        augmented_mol, edge_index = self._get_augmented_molecule(raw_data)

        x = torch.zeros((augmented_mol["nodes"]["num_nodes"], 0))
        edge_attr = torch.zeros((augmented_mol["nodes"]["num_edges"], 0))

        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _get_augmented_molecule(self, smiles):
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return None

        edge_index, augmented_graph_nodes, augmented_graph_edges = self._augment_graph(
            mol
        )

        augmented_mol = {"nodes": augmented_graph_nodes, "edges": augmented_graph_edges}
        self.mol_object_buffer[smiles] = augmented_mol

        return augmented_mol, edge_index

    def _augment_graph(self, mol: Chem.Mol):
        edge_index = torch.tensor(
            [
                [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
                [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
            ]
        )
        within_atoms_edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

        num_of_nodes = mol.GetNumAtoms()
        num_of_edges = mol.GetNumBonds()

        set_atom_map_num(mol)
        detect_functional_group(mol)

        for atom in mol.GetAtoms():
            atom.SetProp(NODE_LEVEL, ATOM_NODE_LEVEL)

        for edge in mol.GetBonds():
            edge.SetProp(EDGE_LEVEL, WITHIN_ATOMS_EDGE)

        structure, bonds = get_structure(mol)

        if not structure:
            raise ValueError("")

        # Preprocess the molecular structure to match feature dictionary keys
        fg_to_atoms_edge_index = [[], []]
        fg_nodes, fg_atom_edges = {}, {}
        new_structure = {}
        for idx, fg in enumerate(structure):
            # new_sm = preprocess_smiles(sm)  # Preprocess SMILES to match the feature dictionary
            new_structure[num_of_nodes] = {
                "atom": structure[fg]["atom"]  # Get atom list for fragment
            }
            for atom in structure[fg]["atom"]:
                fg_to_atoms_edge_index[0].extend([num_of_nodes, atom])
                fg_to_atoms_edge_index[1].extend([atom, num_of_nodes])
                fg_atom_edges[f"{num_of_nodes}_{atom}"] = {EDGE_LEVEL: ATOM_FG_EDGE}
                num_of_edges += 1

            fg_set = {
                mol.GetAtomWithIdx(atom_idx).GetProp("FG")
                for atom_idx in structure[fg]["atom"]
                if mol.GetAtomWithIdx(atom_idx).GetProp("FG")
            }

            if len(fg_set) > 1:
                raise Exception("connected atoms should belong to only one fg")

            elif len(fg_set) == 0:
                ring_sizes = set()
                for atom_idx in structure[fg]["atom"]:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    ring_size_prop = atom.GetProp("RING")
                    if not ring_size_prop:
                        raise Exception("All atoms should have ring size")
                    ring_sizes.add(int(ring_size_prop))
                    atom.SetProp("FG", f"RING_{ring_size_prop}")

                # TODO: Incase error is raised check logic for fused rings
                assert len(ring_sizes) == 1, "all atoms should have one ring size"
                ring_size = list(ring_sizes)[0]

                fg_nodes[num_of_nodes] = {
                    NODE_LEVEL: FG_NODE_LEVEL,
                    "FG": f"RING_{ring_size}",
                    "RING": ring_size,
                }

            else:
                any_atom = None
                for atom_idx in structure[fg]["atom"]:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetProp("FG"):
                        any_atom = atom

                fg_nodes[num_of_nodes] = {
                    NODE_LEVEL: FG_NODE_LEVEL,
                    "FG": any_atom.GetProp("FG"),
                    "RING": any_atom.GetProp("RING"),
                }

            num_of_nodes += 1

        fg_edges = {}
        within_fg_edge_index = [[], []]
        # TODO: Can we optimize this ?
        for bond in bonds:
            start_idx, end_idx = bond[:2]
            for key, value in new_structure.items():
                if start_idx in value["atom"]:
                    source_fg = key
                if end_idx in value["atom"]:
                    target_fg = key
            within_fg_edge_index[0].extend([source_fg, target_fg])
            within_fg_edge_index[1].extend([target_fg, source_fg])
            fg_edges[f"{source_fg}_{target_fg}"] = {EDGE_LEVEL: WITHIN_FG_EDGE}
            num_of_edges += 1

        graph_node = {
            NODE_LEVEL: GRAPH_NODE_LEVEL,
            "FG": "graph_fg",
            "RING": "0",
        }

        fg_graphNode_edges = {}
        global_node_edge_index = [[], []]
        for fg in new_structure.keys():
            global_node_edge_index[0].extend([num_of_nodes, fg])
            global_node_edge_index[1].extend([fg, num_of_nodes])
            fg_graphNode_edges[f"{num_of_nodes}_{fg}"] = {NODE_LEVEL: FG_GRAPHNODE_EDGE}
            num_of_edges += 1

        all_edges = torch.cat(
            [
                within_atoms_edge_index,
                torch.tensor(fg_to_atoms_edge_index, dtype=torch.long),
                torch.tensor(within_fg_edge_index, dtype=torch.long),
                torch.tensor(global_node_edge_index, dtype=torch.long),
            ],
            dim=1,
        )

        augmented_graph_nodes = {
            "atom_nodes": mol,
            "fg_nodes": fg_nodes,
            "graph_node": graph_node,
            "num_nodes": num_of_nodes,
        }
        augmented_graph_edges = {
            WITHIN_ATOMS_EDGE: mol,
            ATOM_FG_EDGE: fg_atom_edges,
            WITHIN_FG_EDGE: fg_edges,
            FG_GRAPHNODE_EDGE: fg_graphNode_edges,
            "num_edges": num_of_edges,
        }

        return all_edges, augmented_graph_nodes, augmented_graph_edges

    def _get_fg_index(self, atom):
        fg_group = atom.GetProp("FG")
        if fg_group:
            fg_index = self._get_token_index(fg_group)
            return fg_index
        else:
            raise Exception("")

    def _get_ring_size(self, atom):
        ring_size_str = atom.GetProp("RING")
        if ring_size_str:
            ring_sizes = list(map(int, ring_size_str.split("-")))
            # TODO: Decide ring size for atoms belongs to fused rings, rn only max ring size taken
            return max(ring_sizes)
        else:
            return 0


class RuleBasedFGReader(ChemDataReader):

    @classmethod
    def name(cls) -> str:
        return "rule_based_fg"

    def _read_data(self, fg: str) -> int | None:
        return self._get_token_index(fg)
