"""
Graph visualization module.
Generates .dot files for visualization with Graphviz.
"""

from pathlib import Path


def draw_simple_graph(g, filename, title="Graph"):
    """
    Generates a .dot file for a simple (unweighted) graph.

    Args:
        g: Adjacency dictionary {node: [neighbors]}
        filename: Output .dot file path
        title: Graph title

    Returns:
        Path of generated file
    """
    lines = []
    lines.append("graph {")
    lines.append(f'  label="{title}";')
    lines.append('  node [shape=circle];')
    lines.append('')

    # Add edges (avoid duplicates)
    added_edges = set()
    for u in sorted(g.keys()):
        for v in g[u]:
            edge = tuple(sorted([u, v]))
            if edge not in added_edges:
                lines.append(f'  {u} -- {v};')
                added_edges.add(edge)

    lines.append('}')

    # Write file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

    return Path(filename)


def draw_weighted_graph(g, filename, title="Weighted Graph"):
    """
    Generates a .dot file for a weighted graph.

    Args:
        g: Adjacency dictionary {node: [(neighbor, weight)]}
        filename: Output .dot file path
        title: Graph title

    Returns:
        Path of generated file
    """
    lines = []
    lines.append("graph {")
    lines.append(f'  label="{title}";')
    lines.append('  node [shape=circle];')
    lines.append('')

    # Add weighted edges (avoid duplicates)
    added_edges = set()
    for u in sorted(g.keys()):
        for v, weight in g[u]:
            edge = tuple(sorted([u, v]))
            if edge not in added_edges:
                weight_str = str(int(weight) if weight == int(weight) else weight)
                lines.append(f'  {u} -- {v} [label="{weight_str}"];')
                added_edges.add(edge)

    lines.append('}')

    # Write file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

    return Path(filename)


def draw_connected_components(g, components, filename, title="Connected Components"):
    """
    Generates a .dot file for a graph with connected components.

    Args:
        g: Adjacency dictionary
        components: List of lists with nodes of each component
        filename: Output .dot file path
        title: Graph title

    Returns:
        Path of generated file
    """
    return draw_simple_graph(g, filename, title)


def visualize_graphs(electric_graph, road_graph, water_graph, output_dir="resources"):
    """
    Generates .dot files for the three graph types.

    Args:
        electric_graph: Electric graph (simple)
        road_graph: Road graph (weighted)
        water_graph: Water graph (simple)
        output_dir: Directory to save .dot files

    Returns:
        Dict with paths of generated files
    """
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    print("📊 Generating graph .dot files...")

    # Electric graph
    print("  → Generating electrico.dot...")
    path = draw_simple_graph(
        electric_graph,
        f"{output_dir}/electrico.dot",
        "Red Eléctrica"
    )
    results['electric'] = path
    print(f"    ✓ Saved: {path}")

    # Road graph
    print("  → Generating vial.dot...")
    path = draw_weighted_graph(
        road_graph,
        f"{output_dir}/vial.dot",
        "Red Vial"
    )
    results['road'] = path
    print(f"    ✓ Saved: {path}")

    # Water graph
    print("  → Generating hidrico.dot...")
    path = draw_simple_graph(
        water_graph,
        f"{output_dir}/hidrico.dot",
        "Red Hídrica"
    )
    results['water'] = path
    print(f"    ✓ Saved: {path}")

    print(f"✓ .dot files generated ({len(results)} graphs)")
    print("  To visualize: dot -Tpng electrico.dot -o electrico.png\n")
    return results
