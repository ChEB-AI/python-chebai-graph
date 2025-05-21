import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from jsonargparse import CLI
from rdkit.Chem import AllChem, Mol
from torch import Tensor

from chebai_graph.preprocessing.properties.constants import *
from chebai_graph.preprocessing.reader import GraphFGAugmentorReader

matplotlib.use("TkAgg")

EDGE_COLOR_MAP = {
    WITHIN_ATOMS_EDGE: "#1f77b4",
    ATOM_FG_EDGE: "#9467bd",
    WITHIN_FG_EDGE: "#ff7f0e",
    FG_GRAPHNODE_EDGE: "#2ca02c",
}

NODE_COLOR_MAP = {
    "atom": "#9ecae1",
    "fg": "#fdae6b",
    "graph": "#d62728",
}


def _create_graph(
    edge_index: Tensor, augmented_graph_nodes: dict, augmented_graph_edges: dict
) -> nx.Graph:
    """
    Create a NetworkX graph from augmented molecular information.

    Args:
        edge_index (torch.Tensor): Tensor of shape (2, num_edges) with source and target indices.
        augmented_graph_nodes (dict): Dictionary of node attributes grouped by type ('atom_nodes', 'fg_nodes', etc.).
        augmented_graph_edges (dict): Dictionary of edges grouped by predefined edge type.

    Returns:
        nx.Graph: Constructed NetworkX graph with annotated nodes and edges.
    """
    G = nx.Graph()

    # Add atom nodes
    atom_nodes = augmented_graph_nodes["atom_nodes"]
    for atom in atom_nodes.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(
            idx,
            node_name=atom.GetSymbol(),
            node_type="atom",
            node_color=NODE_COLOR_MAP["atom"],
        )

    # Add functional group (FG) nodes
    fg_nodes = augmented_graph_nodes["fg_nodes"]
    for fg_idx, fg_props in fg_nodes.items():
        G.add_node(
            fg_idx,
            node_name=f"FG:{fg_props['FG']}",
            node_type="fg",
            node_color=NODE_COLOR_MAP["fg"],
        )

    # Add special graph node
    graph_node_idx = augmented_graph_nodes["num_nodes"]
    G.add_node(
        graph_node_idx,
        node_name="Graph Node",
        node_type="graph",
        node_color=NODE_COLOR_MAP["graph"],
    )

    # Decode edge types and add edges with proper color and type
    src_nodes, tgt_nodes = edge_index.tolist()
    with_atom_edges = {
        f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}"
        for bond in augmented_graph_edges[WITHIN_ATOMS_EDGE].GetBonds()
    }
    atom_fg_edges = set(augmented_graph_edges[ATOM_FG_EDGE])
    within_fg_edges = set(augmented_graph_edges[WITHIN_FG_EDGE])
    fg_graph_edges = set(augmented_graph_edges[FG_GRAPHNODE_EDGE])

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
        G.add_edge(src, tgt, edge_type=edge_type, edge_color=EDGE_COLOR_MAP[edge_type])

    return G


def _get_subgraph_by_node_type(G: nx.Graph, node_type: str) -> nx.Graph:
    """
    Extract a subgraph containing only nodes of the given type.

    Args:
        G (nx.Graph): Full graph with node_type attributes.
        node_type (str): Type of node to extract ('atom', 'fg', or 'graph').

    Returns:
        nx.Graph: Subgraph with selected node type.
    """
    selected_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("node_type") == node_type
    ]
    return G.subgraph(selected_nodes).copy()


def _draw_hierarchy(G: nx.Graph, mol: Mol) -> None:
    """
    Draw a hierarchical layout combining RDKit 2D coordinates for atoms and spring layout for FG/graph nodes.

    Args:
        G (nx.Graph): Augmented molecular graph.
        mol (Chem.Mol): RDKit molecule object with atom layout.
    """
    AllChem.Compute2DCoords(mol)

    # Get 2D positions from RDKit
    atom_pos = {}
    max_atom_pos_y = 0
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = mol.GetConformer().GetAtomPosition(idx)
        atom_pos[idx] = (pos.x, pos.y)
        max_atom_pos_y = max(max_atom_pos_y, pos.y)

    # Position FG nodes above atoms
    fg_graph = _get_subgraph_by_node_type(G, "fg")
    fg_pos = {
        node: (x, y + max_atom_pos_y + 2)
        for node, (x, y) in nx.spring_layout(fg_graph, seed=42).items()
    }

    # Position the graph node further above
    graph_node_graph = _get_subgraph_by_node_type(G, "graph")
    graph_pos = {
        node: (x, y + max_atom_pos_y + 3)
        for node, (x, y) in nx.spring_layout(graph_node_graph, seed=123).items()
    }

    # Merge all positions
    pos = {**atom_pos, **fg_pos, **graph_pos}
    node_colors = [G.nodes[n]["node_color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["node_name"] for n in G.nodes}
    edge_colors = [G.edges[e]["edge_color"] for e in G.edges]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    plt.title("Augmented Graph with RDKit Atom Layout + FG/Graph Clusters")
    plt.axis("off")
    plt.show()


def _draw_simple(G: nx.Graph) -> None:
    """
    Draw the graph using a simple spring layout.

    Args:
        G (nx.Graph): Augmented molecular graph.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["node_color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["node_name"] for n in G.nodes}
    edge_colors = [G.edges[e]["edge_color"] for e in G.edges]
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=600,
        edge_color=edge_colors,
        width=2,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    plt.title("Augmented Graph with simple layout")
    plt.axis("off")
    plt.show()


def _draw_3d(G: nx.Graph, mol: Mol) -> None:
    """
    Visualize the graph in 3D using Plotly.

    Args:
        G (nx.Graph): Augmented molecular graph.
        mol (Chem.Mol): RDKit molecule object for 3D coordinates.

    Raises:
        ImportError: If Plotly is not installed.
    """
    try:
        from plotly import graph_objects as go
    except ImportError:
        raise ImportError(
            "Plotly is required for 3D plotting. Install it with `pip install plotly`."
        )

    # Generate 3D coordinates for atoms
    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()

    atom_pos = {
        atom.GetIdx(): (pos.x, pos.y, 0)
        for atom in mol.GetAtoms()
        for pos in [conf.GetAtomPosition(atom.GetIdx())]
    }

    # Generate 3D layout for FG and graph nodes
    fg_graph = _get_subgraph_by_node_type(G, "fg")
    fg_pos_3d = nx.spring_layout(fg_graph, seed=42, dim=3)
    fg_pos = {k: (x, y, z + 2) for k, (x, y, z) in fg_pos_3d.items()}

    graph_node_graph = _get_subgraph_by_node_type(G, "graph")
    graph_pos_3d = nx.spring_layout(graph_node_graph, seed=123, dim=3)
    graph_pos = {k: (x, y, z + 4) for k, (x, y, z) in graph_pos_3d.items()}

    pos = {**atom_pos, **fg_pos, **graph_pos}

    # Collect edges by type
    edge_type_to_edges = {
        WITHIN_ATOMS_EDGE: [],
        ATOM_FG_EDGE: [],
        WITHIN_FG_EDGE: [],
        FG_GRAPHNODE_EDGE: [],
    }
    for src, tgt, data in G.edges(data=True):
        edge_type_to_edges[data["edge_type"]].append((src, tgt))

    edge_traces = []
    for edge_type, edge_list in edge_type_to_edges.items():
        xs, ys, zs = [], [], []
        for src, tgt in edge_list:
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[tgt]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]

        trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=EDGE_COLOR_MAP[edge_type], width=4),
            name=edge_type,
            hoverinfo="none",
        )
        edge_traces.append(trace)

    # Collect node attributes for visualization
    pos_x, pos_y, pos_z, node_colors, node_names, node_ids = zip(
        *[
            (pos[n][0], pos[n][1], pos[n][2], attr["node_color"], attr["node_name"], n)
            for n, attr in G.nodes(data=True)
        ]
    )

    node_trace = go.Scatter3d(
        x=pos_x,
        y=pos_y,
        z=pos_z,
        mode="markers+text",
        marker=dict(size=8, color=node_colors, opacity=0.9),
        text=node_names,
        textposition="top center",
        hovertext=node_ids,
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="3D Augmented Molecule Graph",
        showlegend=True,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


def plot_augmented_graph(
    edge_index: Tensor,
    augmented_molecule: dict,
    mol: Mol,
    plot_type: str,
) -> None:
    """
    Main plotting function to visualize the augmented graph.

    Args:
        edge_index (torch.Tensor): Edge indices tensor (2, num_edges).
        augmented_molecule (dict): Augmented Molecule.
        mol (Chem.Mol): RDKit molecule object.
        plot_type (str): One of ["simple", "h", "3d"].
    """
    G = _create_graph(
        edge_index, augmented_molecule["nodes"], augmented_molecule["edges"]
    )

    if plot_type == "h":
        _draw_hierarchy(G, mol)
    elif plot_type == "simple":
        _draw_simple(G)
    elif plot_type == "3d":
        _draw_3d(G, mol)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


class Main:
    """
    Command-line wrapper class for plotting augmented molecular graphs.
    """

    def __init__(self):
        self._fg_reader = GraphFGAugmentorReader()

    def plot(self, smiles: str = "OC(=O)c1ccccc1O", plot_type: str = "simple") -> None:
        """
        Plot an augmented molecular graph from SMILES.

        Args:
            smiles (str): SMILES string to parse.
            plot_type (str): Type of plot. One of ['simple', 'h', '3d'].
                - simple : 2D graph with all nodes on same plane
                - h: Hierarchical 2D-graph with separate plane for each node type
                - 3d: Hierarchical 3D-graph
        """
        mol = self._fg_reader._smiles_to_mol(smiles)  # noqa
        edge_index, augmented_molecule = self._fg_reader._create_augmented_graph(
            mol
        )  # noqa
        plot_augmented_graph(edge_index, augmented_molecule, mol, plot_type)


if __name__ == "__main__":
    # Example: python visualize_augmented_molecule.py plot --smiles="OC(=O)c1ccccc1O" --plot_type="h"
    # Aspirin ->  CC(=O)OC1=CC=CC=C1C(=O)O ; CHEBI:15365, acetylsalicylic acid
    # Salicylic acid -> OC(=O)c1ccccc1O ; CHEBI:16914
    # 1-hydroxy-2-naphthoic acid -> OC(=O)c1ccc2ccccc2c1O ; CHEBI:36108 ; Fused Rings
    #  3-nitrobenzoic acid -> OC(=O)C1=CC(=CC=C1)[N+]([O-])=O ; CHEBI:231494 ; Ring + Novel atom (Nitrogen)
    # nile blue A -> [Cl-].CCN(CC)c1ccc2nc3c(cc(N)c4ccccc34)[o+]c2c1 ; CHEBI:52163 ; Fused rings + Novel atoms
    CLI(Main)
