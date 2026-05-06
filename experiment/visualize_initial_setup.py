import json
import os
import matplotlib.pyplot as plt
import networkx as nx


def visualize_initial_setup(setup_file, output_dir="experiment/thesis_img"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(setup_file):
        print(f"Error: {setup_file} not found.")
        return

    with open(setup_file, "r") as f:
        data = json.load(f)

    gt_network = data.get("ground_truth_network", {})
    if not gt_network:
        print("No ground_truth_network found in setup file.")
        return

    # Create Directed Graph
    G = nx.DiGraph()
    for person, friends in gt_network.items():
        for friend in friends:
            G.add_edge(person, friend)

    # Visualization settings
    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.spring_layout(G, seed=42, k=1.0)  # Seed for consistent layout

    # Draw nodes
    # Use a constant size for all nodes as requested
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#6c8cff", alpha=0.9)

    # Draw edges with arrows - significantly increased margins and switched to a thinner arrow style
    nx.draw_networkx_edges(
        G,
        pos,
        width=1.5,
        edge_color="#2a2d3a",
        alpha=0.5,
        arrows=True,
        arrowsize=12,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20,
        min_target_margin=20,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#0f1117",
    )

    # plt.title(f"Initial Social Network: {data.get('name', 'Willowbrook')}", fontsize=16, pad=20)
    plt.axis("off")

    # Save
    output_path = os.path.join(output_dir, "initial_network.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Visualization saved to: {output_path}")


def visualize_final_consensus(
    analyzed_file, output_dir="experiment/thesis_img", step="30"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(analyzed_file, "r") as f:
        data = json.load(f)

    if step not in data["steps"]:
        step = sorted(data["steps"].keys(), key=int)[-1]

    consensus = data["steps"][step].get("consensus_network", {})

    # Create Directed Graph
    G = nx.DiGraph()
    # Lower threshold to 1 to ensure data is visible, but use weight for styling
    for src, targets in consensus.items():
        for tgt, weight in targets.items():
            if weight >= 1:
                G.add_edge(src, tgt, weight=weight)

    if G.number_of_nodes() == 0:
        # Add nodes anyway so the picture isn't blank
        for agent in data["original_metadata"]["agents"]:
            G.add_node(agent)

    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Styling based on consensus
    if G.number_of_edges() > 0:
        edge_weights = [d["weight"] for u, v, d in G.edges(data=True)]
        # Scale widths for visibility
        widths = [max(1.5, w * 1.5) for w in edge_weights]

        # Draw edges with arrows - increased margins and thinner arrow style
        nx.draw_networkx_edges(
            G,
            pos,
            width=widths,
            edge_color="#4a4a4a",
            alpha=0.6,
            arrows=True,
            arrowsize=12,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=20,
            min_target_margin=20,
        )

    # Draw nodes
    # Use a constant size for all nodes as requested
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#ff6c8c", alpha=0.9)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    # plt.title(f"Final Consensus Network (Step {step})", fontsize=16, pad=20)

    plt.axis("off")

    output_path = os.path.join(output_dir, f"final_consensus_step_{step}.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Final consensus saved to: {output_path}")


if __name__ == "__main__":
    setup_path = "experiment/setups/small_town_extended.json"
    results_path = "experiment/outputs/validsim_2_results_analyzed.json"

    visualize_initial_setup(setup_path)
    if os.path.exists(results_path):
        visualize_final_consensus(results_path)
