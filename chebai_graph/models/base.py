from abc import ABC, abstractmethod

import torch
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.structures import XYData
from torch_geometric.data import Data as GraphData
from torch_scatter import scatter_add


class GraphBaseNet(ChebaiBaseNet, ABC):
    def _get_prediction_and_labels(self, data, labels, output):
        return torch.sigmoid(output), labels.int()

    def _process_labels_in_batch(self, batch: XYData) -> torch.Tensor:
        return batch.y.float() if batch.y is not None else None


class GraphNetWrapper(GraphBaseNet, ABC):
    def __init__(self, config: dict, n_linear_layers, n_molecule_properties, **kwargs):
        super().__init__(**kwargs)
        self.gnn = self._get_gnn(config)
        gnn_out_dim = config["out_dim"] if "out_dim" in config else config["hidden_dim"]
        self.activation = torch.nn.ELU
        self.lin_input_dim = self._get_lin_seq_input_dim(
            gnn_out_dim=gnn_out_dim,
            n_molecule_properties=n_molecule_properties,
        )

        lin_hidden_dim = kwargs.get("lin_hidden_dim", gnn_out_dim)
        self.lin_sequential: torch.nn.Sequential = self._get_linear_module_list(
            n_linear_layers=n_linear_layers,
            in_dim=self.lin_input_dim,
            hidden_dim=lin_hidden_dim,
            out_dim=self.out_dim,
        )

    @abstractmethod
    def _get_gnn(self, config):
        pass

    def _get_lin_seq_input_dim(self, gnn_out_dim, n_molecule_properties):
        return gnn_out_dim + n_molecule_properties

    def _get_linear_module_list(self, n_linear_layers, in_dim, hidden_dim, out_dim):
        if n_linear_layers < 1:
            raise ValueError("n_linear_layers must be at least 1")

        layers = []
        if n_linear_layers == 1:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        else:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation())
            for _ in range(n_linear_layers - 2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation())
            layers.append(torch.nn.Linear(hidden_dim, out_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = self.gnn(batch)
        a = scatter_add(a, graph_data.batch, dim=0)
        a = torch.cat([a, graph_data.molecule_attr], dim=1)

        return self.lin_sequential(a)


class AugmentedNodePoolingNet(GraphNetWrapper, ABC):
    def _get_lin_seq_input_dim(self, gnn_out_dim, n_molecule_properties):
        # atom_embeddings + molecule attributes + augmented_node_embeddings
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_atom_node = graph_data.is_atom_node.bool()  # Boolean mask: shape [num_nodes]
        is_augmented_node = ~is_atom_node

        node_embeddings = self.gnn(batch)

        atom_embeddings = node_embeddings[is_atom_node]
        atom_batch = graph_data.batch[is_atom_node]

        augmented_node_embeddings = node_embeddings[is_augmented_node]
        augmented_node_batch = graph_data.batch[is_augmented_node]

        # Scatter add separately
        graph_vec_atoms = scatter_add(atom_embeddings, atom_batch, dim=0)
        graph_vec_augmented_nodes = scatter_add(
            augmented_node_embeddings, augmented_node_batch, dim=0
        )

        # Concatenate all
        graph_vector = torch.cat(
            [
                graph_vec_atoms,
                graph_data.molecule_attr,
                graph_vec_augmented_nodes,
            ],
            dim=1,
        )

        return self.lin_sequential(graph_vector)


class GraphNodePoolingNet(GraphNetWrapper, ABC):
    def _get_lin_seq_input_dim(self, gnn_out_dim, n_molecule_properties):
        # all_nodes_embeddings_except_graph_node + molecule attributes + graph_node_embedding
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_not_graph_node = ~is_graph_node

        node_embeddings = self.gnn(batch)

        graph_node_embedding = node_embeddings[is_graph_node]
        graph_node_batch = graph_data.batch[is_graph_node]

        remaining_node_embedding = node_embeddings[is_not_graph_node]
        remaining_node_batch = graph_data.batch[is_not_graph_node]

        # Scatter add separately
        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        remaining_nodes_vec = scatter_add(
            remaining_node_embedding, remaining_node_batch, dim=0
        )

        # Concatenate all
        graph_vector = torch.cat(
            [
                remaining_nodes_vec,
                graph_data.molecule_attr,
                graph_node_vec,
            ],
            dim=1,
        )

        return self.lin_sequential(graph_vector)


class FGNodePoolingNet(GraphNetWrapper, ABC):
    def _get_lin_seq_input_dim(self, gnn_out_dim, n_molecule_properties):
        # all_nodes_embeddings_except_fg_nodes + molecule attributes + fg_node_embedding
        return gnn_out_dim + n_molecule_properties + gnn_out_dim

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()
        is_fg_node = (~is_atom_node) & (~is_graph_node)
        is_remaining_node = ~is_fg_node

        node_embeddings = self.gnn(batch)

        remaining_node_embedding = node_embeddings[is_remaining_node]
        remaining_node_batch = graph_data.batch[is_remaining_node]

        fg_node_embeddings = node_embeddings[is_fg_node]
        fg_node_batch = graph_data.batch[is_fg_node]

        # Scatter add separately
        remaining_node_vec = scatter_add(
            remaining_node_embedding, remaining_node_batch, dim=0
        )
        fg_node_vec = scatter_add(fg_node_embeddings, fg_node_batch, dim=0)

        # Concatenate all
        graph_vector = torch.cat(
            [
                remaining_node_vec,
                graph_data.molecule_attr,
                fg_node_vec,
            ],
            dim=1,
        )

        return self.lin_sequential(graph_vector)


class GraphNodeFGNodePoolingNet(GraphNetWrapper, ABC):
    def _get_lin_seq_input_dim(self, gnn_out_dim, n_molecule_properties):
        # atom_embeddings + molecule attributes + functional_group_node_embeddings + graph_node_embeddings
        return gnn_out_dim + n_molecule_properties + gnn_out_dim + gnn_out_dim

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        is_graph_node = graph_data.is_graph_node.bool()
        is_atom_node = graph_data.is_atom_node.bool()
        is_fg_node = (~is_atom_node) & (~is_graph_node)

        node_embeddings = self.gnn(batch)

        graph_node_embedding = node_embeddings[is_graph_node]
        graph_node_batch = graph_data.batch[is_graph_node]

        atom_embeddings = node_embeddings[is_atom_node]
        atom_batch = graph_data.batch[is_atom_node]

        fg_node_embeddings = node_embeddings[is_fg_node]
        fg_node_batch = graph_data.batch[is_fg_node]

        # Scatter add separately
        graph_node_vec = scatter_add(graph_node_embedding, graph_node_batch, dim=0)
        atom_vec = scatter_add(atom_embeddings, atom_batch, dim=0)
        fg_node_vec = scatter_add(fg_node_embeddings, fg_node_batch, dim=0)

        # Concatenate all
        graph_vector = torch.cat(
            [
                atom_vec,
                graph_data.molecule_attr,
                fg_node_vec,
                graph_node_vec,
            ],
            dim=1,
        )

        return self.lin_sequential(graph_vector)
