import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


def visualize_perceived_centrality(
    analyzed_file, output_dir="experiment/thesis_img", step="30"
):
    os.makedirs(output_dir, exist_ok=True)

    with open(analyzed_file, "r") as f:
        data = json.load(f)

    agents = data["original_metadata"]["agents"]
    # Filter out any non-agent names like "the young activist"
    valid_agents = set(agents)

    if step not in data["steps"]:
        step = sorted(data["steps"].keys(), key=int)[-1]

    cognitive = data["steps"][step].get("cognitive_networks", {})

    votes_received = defaultdict(int)
    nomination_edges = []

    for voter, agent_data in cognitive.items():
        nominees = agent_data.get("centrality", [])
        for nominee in nominees:
            # Clean name and check if valid
            clean_nominee = nominee.strip()
            if clean_nominee in valid_agents:
                votes_received[clean_nominee] += 1
                nomination_edges.append((voter, clean_nominee))

    # --- 1. Bar Chart of Perceived Centrality ---
    plt.figure(figsize=(10, 6), dpi=300)
    sorted_votes = sorted(
        [(name, votes_received[name]) for name in agents],
        key=lambda x: x[1],
        reverse=True,
    )
    names = [x[0] for x in sorted_votes]
    counts = [x[1] for x in sorted_votes]

    bars = plt.bar(names, counts, color="#ff6c8c", alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Nominations")
    # plt.title("Perceived Centrality (Top 3 Influential People)") # No title as requested

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "perceived_centrality_bar.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Bar chart saved to: {bar_path}")

    # --- 2. Directed Nomination Graph ---
    G = nx.DiGraph()
    G.add_nodes_from(agents)
    G.add_edges_from(nomination_edges)

    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#ff6c8c", alpha=0.9)

    # Draw edges with arrows and margins
    nx.draw_networkx_edges(
        G,
        pos,
        width=1.2,
        edge_color="#4a4a4a",
        alpha=0.5,
        arrows=True,
        arrowsize=12,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.15",
        min_source_margin=20,
        min_target_margin=20,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.axis("off")
    graph_path = os.path.join(output_dir, "perceived_centrality_graph.png")
    plt.savefig(graph_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Directed nomination graph saved to: {graph_path}")


if __name__ == "__main__":
    results_path = "small_town_extended_20260504_103514_results_analyzed.json"
    if os.path.exists(results_path):
        visualize_perceived_centrality(results_path)
