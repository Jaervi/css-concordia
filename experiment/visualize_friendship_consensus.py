import json
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


def visualize_friendship_consensus(
    analyzed_file, output_dir="experiment/thesis_img", step=30
):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(analyzed_file):
        print(f"Error: {analyzed_file} not found.")
        return

    with open(analyzed_file, "r") as f:
        data = json.load(f)

    agents = data["original_metadata"]["agents"]
    raw_results = data.get("raw_questionnaire_results", [])

    # Extract votes
    friendship_votes = defaultdict(int)
    agent_pattern = r"\b(" + "|".join(re.escape(name) for name in agents) + r")\b"

    for entry in raw_results:
        if entry.get("step") == step and entry.get("dimension") == "css_friendship":
            text = entry.get("answer_text", "")
            matches = re.findall(agent_pattern, text, re.IGNORECASE)
            name_map = {n.lower(): n for n in agents}
            found_names = sorted(
                list(
                    set([name_map[m.lower()] for m in matches if m.lower() in name_map])
                )
            )

            if len(found_names) >= 2:
                for i in range(len(found_names)):
                    for j in range(i + 1, len(found_names)):
                        u, v = sorted([found_names[i], found_names[j]])
                        friendship_votes[(u, v)] += 1

    def save_graph(threshold, filename):
        G = nx.Graph()
        for agent in agents:
            G.add_node(agent)

        for (u, v), weight in friendship_votes.items():
            if weight >= threshold:
                G.add_edge(u, v, weight=weight)

        plt.figure(figsize=(10, 8), dpi=300)
        # Circular layout is often better for showing sparsity/unconnectedness in thesis
        pos = nx.circular_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#4caf84", alpha=0.8)

        # Draw edges
        if G.number_of_edges() > 0:
            edge_weights = [d["weight"] for u, v, d in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, width=1.0, edge_color="#2a2d3a", alpha=0.4)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

        plt.axis("off")
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {output_path} (Edges: {G.number_of_edges()})")

    # Save both versions
    save_graph(threshold=1, filename="friendship_consensus_t1.png")
    save_graph(threshold=2, filename="friendship_consensus_t2.png")


if __name__ == "__main__":
    results_path = "experiment/outputs/validsim_2_results_analyzed.json"
    visualize_friendship_consensus(results_path)
