"""
Multi-level Self-supervised Pre-training from
Zang, X., Zhao, X. & Tang, B. Hierarchical Molecular Graph Self-Supervised Learning for property prediction. Commun Chem 6, 34 (2023). https://doi.org/10.1038/s42004-023-00825-5

This file contains modified code from:
    - https://github.com/ZangXuan/HiMol/blob/main/pretrain.py
"""

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from chebai_graph.preprocessing.properties.augmented_properties import (
    AugAtomType,
    AugBondType,
)
from chebai_graph.preprocessing.property_encoder import IndexEncoder


class MSPPretrainingLoss(torch.nn.Module):
    """HiMol-style molecular structure pretraining loss.

    The model output must be ``(graph_data, node_embeddings)``. The graph data
    must contain ``is_atom_node`` and a PyG ``batch`` vector. If present,
    ``is_graph_node`` identifies the molecule-level representation.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_type_encoder = IndexEncoder(
            property=AugAtomType(),
            use_one_hot_cache=True,
            eval=True,
            tokens_directory="_shared",
        )
        self.bond_type_encoder = IndexEncoder(
            property=AugBondType(),
            use_one_hot_cache=True,
            eval=True,
            tokens_directory="_shared",
        )

        # Bond existence projection
        self.bond_if_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        # Bond existence classifier
        self.bond_if_s = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

        # Bond type classifier
        self.bond_type_s = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.bond_type_encoder.cache)),
        )
        # Atom type classifier
        self.atom_type_s = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(self.atom_type_encoder.cache)),
        )

        count_hidden = max(hidden_size // 4, 1)
        # Atom count regressor
        self.atom_num_s = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, count_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(count_hidden, 1),
        )
        # Bond count regressor
        self.bond_num_s = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, count_hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(count_hidden, 1),
        )
        self.feat_drop = (
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        )
        self.bond_if_loss = torch.nn.BCEWithLogitsLoss()
        self.bond_type_loss = torch.nn.CrossEntropyLoss()
        self.atom_type_loss = torch.nn.CrossEntropyLoss()
        self.count_loss = torch.nn.SmoothL1Loss()

        self.loss_weight = torch.nn.Parameter(torch.rand(5), requires_grad=True)

    @staticmethod
    def _bond_count(edge_index: torch.Tensor) -> int:
        if edge_index.numel() == 0:
            return 0
        return int(edge_index.t().sort(dim=1).values.unique(dim=0).shape[0])

    def _super(
        self, data: Data, embeddings: torch.Tensor, graphs: list[torch.Tensor]
    ) -> torch.Tensor:
        if not hasattr(data, "is_graph_node") or data.is_graph_node is None:
            raise ValueError(
                "MSP pretraining requires a boolean is_graph_node attribute in the graph data"
            )

        graph_mask = data.is_graph_node.bool()
        graph_representations = []
        for graph_index in range(len(graphs)):
            representations = embeddings[graph_mask & (data.batch == graph_index)]
            if representations.numel() == 0:
                raise ValueError(
                    f"Graph {graph_index} has no node marked is_graph_node"
                )
            if len(representations) > 1:
                raise ValueError(
                    f"Graph {graph_index} has multiple nodes marked is_graph_node"
                )
            graph_representations.append(representations[0])
        return torch.stack(graph_representations)

    @staticmethod
    def _molecule_metadata(custom_metadata, graph_index: int) -> dict:
        metadata = custom_metadata[graph_index]
        if isinstance(metadata, list):
            metadata = next(
                (
                    item
                    for item in metadata
                    if isinstance(item, dict) and "nodes" in item
                ),
                None,
            )
        if (
            not isinstance(metadata, dict)
            or "nodes" not in metadata
            or "edges" not in metadata
        ):
            raise ValueError(
                f"Missing molecular metadata for graph {graph_index}; expected nodes and edges"
            )
        return metadata

    def _metadata_targets(self, metadata: dict, device: torch.device):
        # atom_vocab = list(range(1, 119))  # Atomic numbers from 1 to 118
        # bond_vocab = ["SINGLE", "DOUBLE", "AROMATIC", "TRIPLE", "DATIVE", "UNSPECIFIED"]
        atom_mol = metadata["nodes"]["atom_nodes"]

        atom_type_targets = torch.tensor(
            [
                self.atom_type_encoder.encode(str(atom.GetAtomicNum()))
                for atom in atom_mol.GetAtoms()
            ],
            dtype=torch.long,
            device=device,
        )
        within_atoms = metadata["edges"]["within_atoms_lvl"]
        bond_indices = []
        bond_type_targets = []
        for bond in within_atoms.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_class = self.bond_type_encoder.encode(str(bond.GetBondType()))
            bond_indices.extend(((begin, end), (end, begin)))
            bond_type_targets.extend((bond_class, bond_class))
        edge_index = (
            torch.tensor(bond_indices, dtype=torch.long, device=device).t()
            if bond_indices
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
        return (
            atom_type_targets,
            torch.tensor(bond_type_targets, dtype=torch.long, device=device),
            edge_index,
        )

    def forward(
        self, output, target, features, custom_metadata, **loss_kwargs
    ) -> torch.Tensor:
        node_embeddings = output
        graph_data = features[0]
        if not isinstance(graph_data, Data):
            raise TypeError("features[0] must be a torch_geometric.data.Data object")
        if graph_data.batch is None or not hasattr(graph_data, "is_atom_node"):
            raise ValueError("MSP pretraining requires batch and is_atom_node")
        is_atom_node = graph_data.is_atom_node.bool()
        batch_index = graph_data.batch.long()
        graph_nodes = [
            torch.where(batch_index == graph_index)[0]
            for graph_index in range(int(batch_index.max().item()) + 1)
        ]
        graph_node_embeddings = self._super(graph_data, node_embeddings, graph_nodes)

        # Number of atoms prediction loss
        atom_counts = torch.stack(
            [
                (is_atom_node & (batch_index == graph_index)).sum()
                for graph_index in range(len(graph_nodes))
            ]
        ).to(node_embeddings.dtype)
        atom_count_pred = self.atom_num_s(graph_node_embeddings).squeeze(-1)
        atom_count_loss = self.count_loss(atom_count_pred, atom_counts)

        bond_if_loss = node_embeddings.new_zeros(())
        bond_type_loss = node_embeddings.new_zeros(())
        atom_type_loss = node_embeddings.new_zeros(())
        bond_counts = []
        atom_correct = bond_correct = 0
        atom_total = bond_total = 0
        auc_values, ap_values = [], []
        for graph_index, nodes in enumerate(graph_nodes):
            metadata = self._molecule_metadata(custom_metadata, graph_index)
            mol = metadata["nodes"]["atom_nodes"]
            atom_type_targets, bond_type_targets, local_edges = self._metadata_targets(
                metadata, node_embeddings.device
            )

            # Bond existence prediction loss
            atoms = nodes[is_atom_node[nodes]]
            if atoms.numel() != atom_type_targets.numel():
                raise ValueError(
                    f"Graph {graph_index} has {atoms.numel()} atom nodes but metadata has "
                    f"{atom_type_targets.numel()} atoms"
                )
            atom_embeddings = self.feat_drop(self.bond_if_proj(node_embeddings[atoms]))
            bond_counts.append(mol.GetNumBonds())
            n_atoms = atoms.numel()
            pairs = torch.cat(
                [
                    atom_embeddings[:, None, :].expand(-1, n_atoms, -1),
                    atom_embeddings[None, :, :].expand(n_atoms, -1, -1),
                ],
                dim=-1,
            )
            bond_pred = self.bond_if_s(pairs).squeeze(-1)
            bond_target = torch.zeros_like(bond_pred)
            if local_edges.numel():
                bond_target[local_edges[0], local_edges[1]] = 1
            bond_if_loss = bond_if_loss + self.bond_if_loss(bond_pred, bond_target)

            if loss_kwargs.get("compute_metrics", True) and local_edges.numel():
                target_cpu = bond_target.flatten().detach().cpu()
                pred_cpu = bond_pred.flatten().detach().sigmoid().cpu()
                if target_cpu.unique().numel() > 1:
                    auc_values.append(roc_auc_score(target_cpu, pred_cpu))
                ap_values.append(average_precision_score(target_cpu, pred_cpu))

            # Atom type prediction loss
            atom_pred = self.atom_type_s(node_embeddings[atoms])
            atom_type_loss = atom_type_loss + self.atom_type_loss(
                atom_pred, atom_type_targets
            )
            atom_correct += int((atom_pred.argmax(1) == atom_type_targets).sum())
            atom_total += atom_type_targets.numel()

            # Bond type prediction loss
            if local_edges.numel():
                edge_rep = torch.cat(
                    [atom_embeddings[local_edges[0]], atom_embeddings[local_edges[1]]],
                    dim=1,
                )
                bond_pred_type = self.bond_type_s(edge_rep)
                bond_type_loss = bond_type_loss + self.bond_type_loss(
                    bond_pred_type, bond_type_targets
                )
                bond_correct += int(
                    (bond_pred_type.argmax(1) == bond_type_targets).sum()
                )
                bond_total += bond_type_targets.numel()

        # Number of bonds prediction loss
        bond_counts = torch.tensor(
            bond_counts,
            dtype=graph_node_embeddings.dtype,
            device=graph_node_embeddings.device,
        )
        bond_count_pred = self.bond_num_s(graph_node_embeddings).squeeze(-1)
        bond_count_loss = self.count_loss(bond_count_pred, bond_counts)

        divisor = max(len(graph_nodes), 1)
        losses = torch.stack(
            [
                bond_if_loss / divisor,
                bond_type_loss / divisor,
                atom_type_loss / divisor,
                atom_count_loss,
                bond_count_loss,
            ]
        )
        if self.loss_weight.numel() != losses.numel():
            raise RuntimeError(
                f"Expected {losses.numel()} loss weights, got {self.loss_weight.numel()}"
            )
        loss = (losses * torch.softmax(self.loss_weight, dim=0)).sum()
        return loss
