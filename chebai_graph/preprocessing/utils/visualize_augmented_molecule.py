import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from jsonargparse import CLI
from rdkit.Chem import AllChem

from chebai_graph.preprocessing.properties.constants import *
from chebai_graph.preprocessing.reader import GraphFGAugmentorReader

matplotlib.use("TkAgg")


def plot_augmented_graph(
    edge_index, augmented_graph_nodes, augmented_graph_edges, mol, plot_type
):
    G = nx.Graph()

    node_labels = {}
    node_colors = []
    node_type_map = {}

    atom_nodes = augmented_graph_nodes["atom_nodes"]
    atom_ids = []
    for atom in atom_nodes.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(idx)
        node_labels[idx] = atom.GetSymbol()
        node_colors.append("#9ecae1")
        node_type_map[idx] = "atom"
        atom_ids.append(idx)

    fg_nodes = augmented_graph_nodes["fg_nodes"]
    fg_ids = []
    for fg_idx, fg_props in fg_nodes.items():
        G.add_node(fg_idx)
        node_labels[fg_idx] = f"FG:{fg_props['FG']}"
        node_colors.append("#fdae6b")
        node_type_map[fg_idx] = "fg"
        fg_ids.append(fg_idx)

    graph_node_idx = augmented_graph_nodes["num_nodes"]
    G.add_node(graph_node_idx)
    node_labels[graph_node_idx] = "Graph Node"
    node_colors.append("#d62728")
    node_type_map[graph_node_idx] = "graph"
    graph_ids = [graph_node_idx]

    src_nodes, tgt_nodes = edge_index.tolist()
    with_atom_edges = {
        f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}"
        for bond in augmented_graph_edges[WITHIN_ATOMS_EDGE].GetBonds()
    }
    atom_fg_edges = set(augmented_graph_edges[ATOM_FG_EDGE])
    within_fg_edges = set(augmented_graph_edges[WITHIN_FG_EDGE])
    fg_graph_edges = set(augmented_graph_edges[FG_GRAPHNODE_EDGE])

    edge_color_map = {
        WITHIN_ATOMS_EDGE: "#1f77b4",
        ATOM_FG_EDGE: "#9467bd",
        WITHIN_FG_EDGE: "#ff7f0e",
        FG_GRAPHNODE_EDGE: "#2ca02c",
    }

    edges = []
    edge_colors = []
    for src, tgt in zip(src_nodes, tgt_nodes):
        undirected_edge_set = {f"{src}_{tgt}", f"{tgt}_{src}"}
        if undirected_edge_set & with_atom_edges:
            edge_type = WITHIN_ATOMS_EDGE
        elif undirected_edge_set & atom_fg_edges:
            edge_type = ATOM_FG_EDGE
        elif undirected_edge_set & within_fg_edges:
            edge_type = WITHIN_FG_EDGE
        elif undirected_edge_set & fg_graph_edges:
            edge_type = FG_GRAPHNODE_EDGE
        else:
            raise Exception("Unexpected edge type")
        edges.append((src, tgt, {"type": edge_type}))
        edge_colors.append(edge_color_map[edge_type])
    G.add_edges_from(edges)

    if plot_type == "h":  # hierarchy
        # 1. Get atom positions from RDKit
        AllChem.Compute2DCoords(mol)
        atom_pos, max_atom_pos_y = {}, 0
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = mol.GetConformer().GetAtomPosition(idx)
            atom_pos[idx] = (pos.x, pos.y)  # Flip y-axis so graph node is on top
            if pos.y > max_atom_pos_y:
                max_atom_pos_y = pos.y

        # 2. Layout for FG and Graph nodes
        fg_subgraph = G.subgraph(fg_ids)
        fg_pos = nx.spring_layout(fg_subgraph, seed=42)
        fg_pos = {
            node: (x, y + max_atom_pos_y + 2) for node, (x, y) in fg_pos.items()
        }  # Below atoms

        graph_node_subgraph = G.subgraph(graph_ids)
        graph_pos = nx.spring_layout(graph_node_subgraph, seed=123)
        graph_pos = {
            node: (x, y + max_atom_pos_y + 3) for node, (x, y) in graph_pos.items()
        }  # Above atoms

        # Combine all positions
        pos = {**atom_pos, **fg_pos, **graph_pos}

        # Final node color mapping
        node_colors_final = [
            {"atom": "#9ecae1", "fg": "#fdae6b", "graph": "#d62728"}[node_type_map[n]]
            for n in G.nodes
        ]

        # Draw
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_final, node_size=600)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color=edge_colors)

        plt.title("Augmented Graph with RDKit Atom Layout + FG/Graph Clusters")
        plt.axis("off")
        plt.show()

    elif plot_type == "simple":
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color=node_colors,
            node_size=600,
            edge_color=[
                edge_color_map[data["type"]] for _, _, data in G.edges(data=True)
            ],
            width=2,
        )
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        plt.title("Augmented Graph with simple layout")
        plt.axis("off")
        plt.show()
    else:
        raise Exception("Unknown plot type")


class Main:
    def __init__(self):
        self._fg_reader = GraphFGAugmentorReader()

    def plot(self, smiles: str = "OC(=O)c1ccccc1O", plot_type: str = "simple"):
        mol = self._fg_reader._smiles_to_mol(smiles)  # noqa
        edge_index, augmented_nodes, augmented_edges = self._fg_reader._augment_graph(
            mol
        )  # noqa
        plot_augmented_graph(
            edge_index, augmented_nodes, augmented_edges, mol, plot_type
        )


if __name__ == "__main__":
    # use:- visualize_augmented_molecule.py plot --smiles="OC(=O)c1ccccc1O"
    CLI(Main)
