#!/usr/bin/env python3
"""
Entry point for the graph analyzer.
Facilitates running the program from the project root.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import *
from src.visualizer import visualize_graphs

if __name__ == "__main__":
    # Validate required arguments
    if len(sys.argv) not in [6, 7]:
        print("Usage: python run.py <electric_file> <road_file> <water_file> <queries_file> <output_file> [--no-draw]")
        print("\nExamples:")
        print("  python run.py resources/ejemplo/ejemplo_electrico.txt resources/ejemplo/ejemplo_vial.txt resources/ejemplo/ejemplo_hidrico.txt resources/ejemplo/ejemplo_consultas.txt resources/ejemplo/ejemplo_respuestas.txt")
        print("  python run.py resources/ejemplo-48/grafo_electrico_48.txt resources/ejemplo-48/grafo_vial_48.txt resources/ejemplo-48/grafo_hidrico_48.txt resources/ejemplo-48/consultas.txt resources/ejemplo-48/respuestas.txt")
        print("\nOptions:")
        print("  --no-draw    Do not generate graph visualizations")
        sys.exit(1)

    electric_file = sys.argv[1]
    road_file = sys.argv[2]
    water_file = sys.argv[3]
    queries_file = sys.argv[4]
    output_file = sys.argv[5]

    # Check if drawing should be disabled
    no_draw = len(sys.argv) == 7 and sys.argv[6] == "--no-draw"

    # Load graphs
    electric_graph = load_graph(electric_file)
    road_graph = load_weighted_graph(road_file)
    water_graph = load_graph(water_file)

    # Visualize graphs (if not disabled)
    if not no_draw:
        output_dir = str(Path(output_file).parent)
        visualize_graphs(electric_graph, road_graph, water_graph, output_dir)

    # Process queries
    process_queries(queries_file, output_file, electric_graph, road_graph, water_graph)

    print(f"✓ Analysis completed. Results saved to: {output_file}")
