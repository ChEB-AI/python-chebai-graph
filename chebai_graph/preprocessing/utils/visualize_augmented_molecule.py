import matplotlib
import networkx as nx

from chebai_graph.preprocessing.properties.constants import *
from chebai_graph.preprocessing.reader import GraphFGAugmentorReader

matplotlib.use("TkAgg")  # or "QtAgg", if you have PyQt/PySide installed
import matplotlib.pyplot as plt  # noqa


def plot_augmented_graph(edge_index, augmented_graph_nodes, augmented_graph_edges):
    G = nx.Graph()

    # Node labels and types for visualization
    node_labels = {}
    node_colors = []

    # Add atom nodes
    atom_nodes = augmented_graph_nodes["atom_nodes"]
    for atom in atom_nodes.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(idx)
        node_labels[idx] = atom.GetSymbol()
        node_colors.append("#9ecae1")  # soft blue

    # Add functional group nodes
    fg_nodes = augmented_graph_nodes["fg_nodes"]
    for fg_idx, fg_props in fg_nodes.items():
        G.add_node(fg_idx)
        node_labels[fg_idx] = f"FG:{fg_props['FG']}"
        node_colors.append("#fdae6b")  # orange

    # Add graph-level node
    graph_node_idx = augmented_graph_nodes["num_nodes"]
    G.add_node(graph_node_idx)
    node_labels[graph_node_idx] = "Graph Node"
    node_colors.append("#d62728")  # red

    # Ensure edge_index is undirected by converting it to undirected
    edge_index = edge_index

    # Add edges
    src_nodes, tgt_nodes = edge_index.tolist()

    with_atom_edges = {
        f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}"
        for bond in augmented_graph_edges[WITHIN_ATOMS_EDGE].GetBonds()
    }
    atom_fg_edges = set(augmented_graph_edges[ATOM_FG_EDGE])
    within_fg_edges = set(augmented_graph_edges[WITHIN_FG_EDGE])
    fg_graph_edges = set(augmented_graph_edges[FG_GRAPHNODE_EDGE])

    edge_color_map = {
        WITHIN_ATOMS_EDGE: "#1f77b4",  # blue
        ATOM_FG_EDGE: "#9467bd",  # purple
        WITHIN_FG_EDGE: "#ff7f0e",  # orange
        FG_GRAPHNODE_EDGE: "#2ca02c",  # green
    }
    augmented_edges = []
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

        augmented_edges.append((src, tgt, {"type": edge_type}))

    G.add_edges_from(augmented_edges)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=600,
        edge_color=[edge_color_map[data["type"]] for _, _, data in G.edges(data=True)],
        width=2,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    plt.title("Augmented Molecular Graph")
    plt.axis("off")
    plt.show()


def main(smiles: str):
    reader = GraphFGAugmentorReader()
    mol = reader._smiles_to_mol(smiles)
    edge_index, augmented_nodes, augmented_edges = reader._augment_graph(mol)
    plot_augmented_graph(edge_index, augmented_nodes, augmented_edges)


if __name__ == "__main__":
    smiles = "OC(=O)c1ccccc1O"
    main(smiles)
