import logging
import typing

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as GraphData
from torch_scatter import scatter_add, scatter_mean

from chebai_graph.loss.pretraining import MaskPretrainingLoss

from .base import GraphBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIGraphNet(GraphBaseNet):
    NAME = "GNN"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.in_length = config["in_length"]
        self.hidden_length = config["hidden_length"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 3
        self.n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )

        self.embedding = torch.nn.Embedding(800, self.in_length)

        self.convs = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            if i == 0:
                self.convs.append(
                    tgnn.GraphConv(
                        self.in_length, self.in_length, dropout=self.dropout_rate
                    )
                )
            self.convs.append(tgnn.GraphConv(self.in_length, self.in_length))
        self.final_conv = tgnn.GraphConv(self.in_length, self.hidden_length)

        self.activation = F.elu

        self.linear_layers = torch.nn.ModuleList([])
        for _ in range(self.n_linear_layers - 1):
            self.linear_layers.append(nn.Linear(self.hidden_length, self.hidden_length))
        self.final_layer = nn.Linear(self.hidden_length, self.out_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = graph_data.x
        a = self.embedding(a)

        for conv in self.convs:
            a = self.activation(conv(a, graph_data.edge_index.long()))
        a = self.activation(self.final_conv(a, graph_data.edge_index.long()))
        a = self.dropout(a)
        a = scatter_add(a, graph_data.batch, dim=0)

        for lin in self.linear_layers:
            a = self.activation(lin(a))
        a = self.final_layer(a)
        return a


class ResGatedGraphConvNetBase(GraphBaseNet):
    """GNN that supports edge attributes"""

    NAME = "ResGatedGraphConvNetBase"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        self.in_length = config["in_length"]
        self.hidden_length = config["hidden_length"]
        self.dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 3
        self.n_linear_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )
        self.n_atom_properties = int(config["n_atom_properties"])
        self.n_bond_properties = (
            int(config["n_bond_properties"]) if "n_bond_properties" in config else 7
        )
        self.n_molecule_properties = (
            int(config["n_molecule_properties"])
            if "n_molecule_properties" in config
            else 0
        )

        self.activation = F.elu
        self.dropout = nn.Dropout(self.dropout_rate)

        self.convs = torch.nn.ModuleList([])
        for i in range(self.n_conv_layers):
            if i == 0:
                self.convs.append(
                    tgnn.ResGatedGraphConv(
                        self.n_atom_properties,
                        self.in_length,
                        # dropout=self.dropout_rate,
                        edge_dim=self.n_bond_properties,
                    )
                )
            self.convs.append(
                tgnn.ResGatedGraphConv(
                    self.in_length, self.in_length, edge_dim=self.n_bond_properties
                )
            )
        self.final_conv = tgnn.ResGatedGraphConv(
            self.in_length, self.hidden_length, edge_dim=self.n_bond_properties
        )

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = graph_data.x.float()
        # a = self.embedding(a)

        for conv in self.convs:
            assert isinstance(conv, tgnn.ResGatedGraphConv)
            a = self.activation(
                conv(a, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr)
            )
        a = self.activation(
            self.final_conv(
                a, graph_data.edge_index.long(), edge_attr=graph_data.edge_attr
            )
        )
        return a


class ResGatedGraphConvNetGraphPred(GraphBaseNet):
    """GNN for graph-level prediction"""

    NAME = "ResGatedGraphConvNetPred"

    def __init__(
        self,
        config: typing.Dict,
        n_linear_layers=2,
        pretrained_checkpoint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if pretrained_checkpoint:
            self.gnn = ResGatedGraphConvNetPretrain.load_from_checkpoint(
                pretrained_checkpoint, map_location=self.device
            ).as_pretrained
        else:
            self.gnn = ResGatedGraphConvNetBase(config, **kwargs)
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.gnn.hidden_length + (i == 0) * self.gnn.n_molecule_properties,
                    self.gnn.hidden_length,
                )
                for i in range(n_linear_layers - 1)
            ]
        )
        self.final_layer = nn.Linear(self.gnn.hidden_length, self.out_dim)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        a = self.gnn(batch)
        a = scatter_add(a, graph_data.batch, dim=0)

        a = torch.cat([a, graph_data.molecule_attr], dim=1)

        for lin in self.linear_layers:
            a = self.gnn.activation(lin(a))
        a = self.final_layer(a)
        return a


class ResGatedAugmentedGraphPred(GraphBaseNet):
    """GNN for graph-level prediction for augmented graphs"""

    NAME = "ResGatedAugmentedGraphPred"

    def __init__(
        self,
        config: typing.Dict,
        n_linear_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn = ResGatedGraphConvNetBase(config, **kwargs)
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.gnn.hidden_length
                    + (i == 0) * self.gnn.n_molecule_properties
                    + (i == 0) * self.gnn.hidden_length,
                    self.gnn.hidden_length,
                )
                for i in range(n_linear_layers - 1)
            ]
        )
        self.final_layer = nn.Linear(self.gnn.hidden_length, self.out_dim)

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

        # Concatenate both
        graph_vector = torch.cat(
            [
                graph_vec_atoms,
                graph_data.molecule_attr,
                graph_vec_augmented_nodes,
            ],
            dim=1,
        )

        for lin in self.linear_layers:
            a = self.gnn.activation(lin(graph_vector))
        return self.final_layer(a)


class ResGatedGraphConvNetPretrain(GraphBaseNet):
    """For pretraining. BaseNet with an additional output layer for predicting atom properties"""

    NAME = "ResGatedGraphConvNetPre"

    def __init__(self, config: typing.Dict, **kwargs):
        if "criterion" not in kwargs or kwargs["criterion"] is None:
            kwargs["criterion"] = MaskPretrainingLoss()
        print(f"Initing ResGatedGraphConvNetPre with criterion: {kwargs['criterion']}")
        super().__init__(**kwargs)
        self.gnn = ResGatedGraphConvNetBase(config, **kwargs)
        self.atom_prediction = nn.Linear(
            self.gnn.hidden_length, self.gnn.n_atom_properties
        )

    def forward(self, batch):
        data = batch["features"][0]
        embedding = self.gnn(batch)
        node_rep = embedding[data.masked_atom_indices.int()]
        atom_pred = torch.gather(
            self.atom_prediction(node_rep),
            1,
            data.masked_property_indices.to(torch.int64),
        )
        return atom_pred

    @property
    def as_pretrained(self):
        return self.gnn

    def _process_labels_in_batch(self, batch):
        return batch.x[0].mask_node_label


class ResGatedGraphConvNetPretrainBonds(GraphBaseNet):
    """For pretraining. BaseNet with two output layers for predicting atom and bond properties"""

    NAME = "ResGatedGraphConvNetPreBonds"

    def __init__(self, config: typing.Dict, **kwargs):
        if "criterion" not in kwargs or kwargs["criterion"] is None:
            kwargs["criterion"] = MaskPretrainingLoss()
        print(f"Initing ResGatedGraphConvNetPre with criterion: {kwargs['criterion']}")
        super().__init__(config, **kwargs)
        self.bond_prediction = nn.Linear(
            self.gnn.hidden_length, self.gnn.n_bond_properties
        )

    def forward(self, batch):
        data = batch["features"][0]
        embedding = self.gnn(batch)
        node_rep = embedding[data.masked_atom_indices.int()]
        atom_pred_all_properties = self.atom_prediction(node_rep)
        atom_pred = torch.gather(
            atom_pred_all_properties, 1, data.masked_property_indices.to(torch.int64)
        )

        masked_edge_index = data.edge_index[:, data.connected_edge_indices.int()].int()
        edge_rep = embedding[masked_edge_index[0]] + embedding[masked_edge_index[1]]
        bond_pred = self.bond_prediction(edge_rep)
        return atom_pred, bond_pred

    def _get_prediction_and_labels(self, data, labels, output):
        if isinstance(labels, tuple):
            labels = tuple(label.int() for label in labels)
        return tuple(torch.sigmoid(out) for out in output), labels

    def _process_labels_in_batch(self, batch):
        return batch.x[0].mask_node_label, batch.x[0].mask_edge_label


class JCIGraphAttentionNet(GraphBaseNet):
    NAME = "AGNN"

    def __init__(self, config: typing.Dict, **kwargs):
        super().__init__(**kwargs)

        in_length = config["in_length"]
        hidden_length = config["hidden_length"]
        dropout_rate = config["dropout_rate"]
        self.n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 5
        n_heads = config["n_heads"] if "n_heads" in config else 5
        self.n_lin_layers = (
            config["n_linear_layers"] if "n_linear_layers" in config else 3
        )

        self.embedding = torch.nn.Embedding(800, in_length)
        self.edge_embedding = torch.nn.Embedding(4, in_length)
        in_length = in_length + 10

        self.convs = torch.nn.ModuleList([])
        for _ in range(self.n_conv_layers - 1):
            layer = tgnn.GATConv(
                in_length, in_length, n_heads, concat=False, add_self_loops=True
            )
            self.convs.append(layer)
        self.final_conv = tgnn.GATConv(
            in_length, hidden_length, n_heads, concat=False, add_self_loops=True
        )

        self.activation = F.leaky_relu

        self.linear_layers = torch.nn.ModuleList([])
        for _ in range(self.n_lin_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_length, hidden_length))
        self.final_layer = nn.Linear(hidden_length, self.out_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch):
        graph_data = batch["features"][0]
        assert isinstance(graph_data, GraphData)
        edge_index = graph_data.edge_index.long().to(self.device)
        a = self.embedding(graph_data.x)
        a = self.dropout(a)
        a = torch.cat([a, torch.rand((*a.shape[:-1], 10)).to(self.device)], dim=1)

        for i, layer in enumerate(self.convs):
            assert isinstance(layer, tgnn.GATConv)
            a = self.activation(layer(a, edge_index))
            if i == 0:
                a = self.dropout(a)
        a = self.activation(self.final_conv(a, edge_index))

        a = self.dropout(a)
        a = scatter_mean(a, graph_data.batch, dim=0)
        for i, layer in enumerate(self.linear_layers):
            a = self.activation(layer(a))
        a = self.final_layer(a)
        return a
