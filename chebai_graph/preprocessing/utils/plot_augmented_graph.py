import matplotlib
import networkx as nx

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
        label = atom.GetSymbol()
        G.add_node(idx)
        node_labels[idx] = label
        node_colors.append("lightblue")

    # Add functional group nodes
    fg_nodes = augmented_graph_nodes["fg_nodes"]
    for fg_idx, fg_props in fg_nodes.items():
        label = f"FG:{fg_props['FG']}"
        G.add_node(fg_idx)
        node_labels[fg_idx] = label
        node_colors.append("orange")

    # Add graph-level node
    graph_node_idx = augmented_graph_nodes["num_nodes"]
    G.add_node(graph_node_idx)
    node_labels[graph_node_idx] = "Graph Node"
    node_colors.append("red")

    # Add edges
    src_nodes, tgt_nodes = edge_index.tolist()
    for src, tgt in zip(src_nodes, tgt_nodes):
        G.add_edge(src, tgt)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=600,
        edge_color="gray",
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
