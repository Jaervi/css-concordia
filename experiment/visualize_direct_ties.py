import json
import os
import matplotlib.pyplot as plt
import networkx as nx


def visualize_direct_ties(analyzed_file, output_dir="experiment/thesis_img", step="30"):
    os.makedirs(output_dir, exist_ok=True)

    with open(analyzed_file, "r") as f:
        data = json.load(f)

    agents = data["original_metadata"]["agents"]
    if step not in data["steps"]:
        step = sorted(data["steps"].keys(), key=int)[-1]

    cognitive = data["steps"][step].get("cognitive_networks", {})

    # 1. Combined Graph of all direct ties
    G_combined = nx.DiGraph()
    G_combined.add_nodes_from(agents)

    for ego, agent_data in cognitive.items():
        # Switching to 'ego_network' (Close friends/allies) for higher quality ties
        ties = agent_data.get("ego_network", [])
        for target in ties:
            if target in agents and target != ego:
                G_combined.add_edge(ego, target)

    def save_directed_graph(G, filename, highlight_node=None, is_ego_map=False):
        # Always keep all nodes as requested

        # Adjust size: Combined is large, Ego is small/condensed
        figsize = (10, 8) if not is_ego_map else (5, 5)
        plt.figure(figsize=figsize, dpi=300)

        # Use circular layout for everything to maintain consistency
        pos = nx.circular_layout(G)

        if is_ego_map:
            node_size = 600
            font_size = 5
            margin = 12
            arrow_size = 8
        else:
            node_size = 2000
            font_size = 9
            margin = 20
            arrow_size = 12

        # Color nodes
        node_colors = []
        for n in G.nodes():
            if n == highlight_node:
                node_colors.append("#ff6c8c")  # Highlight ego
            else:
                node_colors.append("#6c8cff")  # Standard

        nx.draw_networkx_nodes(
            G, pos, node_size=node_size, node_color=node_colors, alpha=0.9
        )

        # Draw edges with arrows and margins
        nx.draw_networkx_edges(
            G,
            pos,
            width=1.0,
            edge_color="#4a4a4a",
            alpha=0.5,
            arrows=True,
            arrowsize=arrow_size,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=margin,
            min_target_margin=margin,
        )

        nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold")

        plt.axis("off")
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path}")

    # Save the combined graph (Full size, all nodes)
    save_directed_graph(G_combined, "direct_ties_combined.png", is_ego_map=False)

    # 2. Individual ego graphs (Condensed size, only friends)
    for ego in agents:
        G_ego = nx.DiGraph()
        G_ego.add_nodes_from(agents)  # Start with all to ensure context if needed

        ties = cognitive.get(ego, {}).get("ego_network", [])
        for target in ties:
            if target in agents and target != ego:
                G_ego.add_edge(ego, target)

        save_directed_graph(
            G_ego, f"direct_ties_{ego.lower()}.png", highlight_node=ego, is_ego_map=True
        )


if __name__ == "__main__":
    # results_path = "experiment/outputs/validsim_2_results_analyzed.json"
    results_path = "small_town_extended_20260504_103514_results_analyzed.json"
    if os.path.exists(results_path):
        visualize_direct_ties(results_path)
