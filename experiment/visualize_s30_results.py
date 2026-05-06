import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import re


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_directories(base_path, keys):
    for key in keys:
        path = os.path.join(base_path, key)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_network(data, key, output_dir, title):
    G = nx.DiGraph()

    names = set()
    for item in data["answers"]:
        names.add(item["name"])
        for target in item["answer"]:
            if isinstance(target, list):
                for t in target:
                    names.add(t)
            else:
                names.add(target)

    G.add_nodes_from(names)

    for item in data["answers"]:
        source = item["name"]
        targets = item["answer"]
        for target in targets:
            if isinstance(target, list):
                if len(target) == 2:
                    G.add_edge(target[0], target[1])
            else:
                G.add_edge(source, target)

    plt.figure(figsize=(14, 12))
    # Kamada-Kawai is often clearer for small networks as it minimizes edge distance variance
    try:
        pos = nx.kamada_kawai_layout(G, scale=2)
    except:
        pos = nx.spring_layout(G, k=1.5, iterations=100, scale=2)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3000,
        node_color="#A0CBE2",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_family="sans-serif", font_weight="bold"
    )

    # Draw edges with curvature to avoid overlapping bidirectional edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=2,
        edge_color="#666666",
        arrowsize=25,
        alpha=0.6,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20,
        min_target_margin=20,
    )

    plt.title(title, fontsize=20, fontweight="bold", pad=30)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{key}_network.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_bar_chart(data, key, output_dir, title, ylabel):
    counts = {}
    for item in data["answers"]:
        for target in item["answer"]:
            counts[target] = counts.get(target, 0) + 1

    if not counts:
        return

    df = pd.DataFrame(list(counts.items()), columns=["Name", "Count"]).sort_values(
        by="Count", ascending=False
    )

    plt.figure(figsize=(12, 8))
    plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    bars = plt.bar(
        df["Name"],
        df["Count"],
        color=plt.cm.viridis(np.linspace(0.3, 0.8, len(df))),
        edgecolor="black",
        zorder=3,
    )

    plt.title(title, fontsize=18, fontweight="bold", pad=25)
    plt.xlabel("Resident", fontsize=14, labelpad=15)
    plt.ylabel(ylabel, fontsize=14, labelpad=15)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{key}_bar.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_affective_heatmap(data, key, output_dir, title):
    all_names = sorted(list(set(item["name"] for item in data["answers"])))
    matrix = pd.DataFrame(np.nan, index=all_names, columns=all_names)

    for item in data["answers"]:
        rater = item["name"]
        for val_str in item["answer"]:
            match = re.search(r"(.*?)\s*\(([-+]?\d+)\)", val_str)
            if match:
                target = match.group(1).strip()
                score = int(match.group(2))
                if target in matrix.columns:
                    matrix.loc[rater, target] = score

    plt.figure(figsize=(14, 11))

    # Custom divergent colormap for valence
    cmap = plt.cm.RdYlGn

    # Use matshow-like approach with text annotations
    im = plt.imshow(matrix.values, cmap=cmap, vmin=-5, vmax=5)
    plt.colorbar(im, label="Valence Score (-5 to +5)", shrink=0.8)

    plt.xticks(range(len(all_names)), all_names, rotation=45, ha="right", fontsize=11)
    plt.yticks(range(len(all_names)), all_names, fontsize=11)

    # Add numbers in the cells
    for i in range(len(all_names)):
        for j in range(len(all_names)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                color = "black" if abs(val) < 2 else "white"
                plt.text(
                    j,
                    i,
                    f"{int(val):+d}",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

    plt.title(title, fontsize=18, fontweight="bold", pad=30)
    plt.xlabel("Target Resident", fontsize=14, labelpad=15)
    plt.ylabel("Rater Resident", fontsize=14, labelpad=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{key}_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    input_file = "experiment/results/interview_results_s30.json"
    output_base = "experiment/visualizations/s30_results"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    data_json = load_data(input_file)
    keys = [q["key"] for q in data_json["data"]]
    setup_directories(output_base, keys)

    for q_data in data_json["data"]:
        key = q_data["key"]
        output_dir = os.path.join(output_base, key)

        print(f"Visualizing: {key}...")

        if key == "close_friends":
            plot_network(
                q_data, key, output_dir, "Social Network: Personal Friends and Allies"
            )
            plot_bar_chart(
                q_data,
                key,
                output_dir,
                "Nomination Counts: Friends and Allies",
                "Number of Times Nominated",
            )

        elif key == "direct_ties":
            plot_network(
                q_data, key, output_dir, "Social Network: Direct Ties and Contacts"
            )
            plot_bar_chart(
                q_data,
                key,
                output_dir,
                "Reported Direct Ties and Contacts",
                "Frequency of Interaction/Contact",
            )

        elif key == "who_influences_whom":
            plot_network(
                q_data, key, output_dir, "Social Influence Network (Perceived)"
            )

        elif key == "affective_valence":
            plot_affective_heatmap(
                q_data, key, output_dir, "Affective Valence Matrix (Personal Feelings)"
            )

        elif key == "other_friendships":
            plot_network(
                q_data, key, output_dir, "Consensus View: Alliances between Others"
            )

        elif key == "perceived_centrality":
            plot_network(
                q_data, key, output_dir, "Network of Perceived Social Centrality"
            )
            plot_bar_chart(
                q_data,
                key,
                output_dir,
                "Perceived Social Centrality (Top 3 Mentions)",
                "Number of Nominations",
            )

    print(f"\nSuccess! All visualizations saved to: {output_base}")


if __name__ == "__main__":
    main()
