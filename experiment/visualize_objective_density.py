import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_objective_reality(analyzed_file, output_dir="experiment/thesis_img"):
    os.makedirs(output_dir, exist_ok=True)
    with open(analyzed_file, "r") as f:
        data = json.load(f)

    agents = data["original_metadata"]["agents"]
    objective = data.get("objective_network", {})

    # Create a Matrix for a Heatmap
    size = len(agents)
    matrix = np.zeros((size, size))

    for i, name_i in enumerate(agents):
        for j, name_j in enumerate(agents):
            if i == j:
                matrix[i, j] = 0  # Diagonal is zero
            else:
                matrix[i, j] = objective.get(name_i, {}).get(name_j, 0)

    # Plot 1: The Heatmap (Shows intensity/volume)
    plt.figure(figsize=(10, 8), dpi=300)
    im = plt.imshow(matrix, cmap="YlGnBu")

    # Add labels
    plt.xticks(range(size), agents, rotation=45, ha="right")
    plt.yticks(range(size), agents)

    # Add values in the boxes
    for i in range(size):
        for j in range(size):
            if i != j:
                plt.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > (matrix.max() / 2) else "black",
                )

    plt.colorbar(im, label="Number of Shared Simulation Steps")
    # plt.title("Objective Interaction Matrix (The 'Actual' Reality)", fontsize=14, pad=20)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "objective_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

    # Plot 2: Circular "Messy" Graph (Shows the 100% density visually)
    G = nx.complete_graph(agents)
    plt.figure(figsize=(8, 8), dpi=300)
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#6c8cff", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color="grey", alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title("Objective Interaction Density (100%)", fontsize=14)
    plt.axis("off")

    density_path = os.path.join(output_dir, "objective_density_graph.png")
    plt.savefig(density_path)
    plt.close()
    print(f"Density graph saved to: {density_path}")


if __name__ == "__main__":
    results_path = "small_town_extended_20260504_103514_results_analyzed.json"
    if os.path.exists(results_path):
        visualize_objective_reality(results_path)
    else:
        print(f"File not found: {results_path}")
