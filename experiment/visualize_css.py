import json
import re
import argparse
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


def load_results(filepath: str) -> dict:
    """Load the JSON results from the simulation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_json_from_text(text: str):
    """Extract JSON data from raw answer text (handles ```json ...``` blocks and bare JSON)."""
    if not isinstance(text, str):
        return text
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return text


def infer_agents_from_data(data: dict) -> list[str]:
    """Infer the agent list from any populated part of the results file."""
    agents = data.get("agents", [])
    if agents:
        return agents

    original_metadata = data.get("original_metadata", {})
    agents = original_metadata.get("agents", [])
    if agents:
        return agents

    inferred = []
    seen = set()

    def add_name(name):
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            inferred.append(name)

    objective_network = data.get("objective_network", {})
    if isinstance(objective_network, dict):
        for source, targets in objective_network.items():
            add_name(source)
            if isinstance(targets, dict):
                for target in targets.keys():
                    add_name(target)

    steps_data = data.get("steps", {})
    if isinstance(steps_data, dict):
        for step_info in steps_data.values():
            if not isinstance(step_info, dict):
                continue
            cognitive_networks = step_info.get("cognitive_networks", {})
            if isinstance(cognitive_networks, dict):
                for name, info in cognitive_networks.items():
                    add_name(name)
                    if isinstance(info, dict):
                        for value in info.values():
                            if isinstance(value, dict):
                                for nested_name in value.keys():
                                    add_name(nested_name)
                            elif isinstance(value, list):
                                for nested_name in value:
                                    add_name(nested_name)

    raw_results = data.get("raw_questionnaire_results", [])
    if isinstance(raw_results, list):
        for entry in raw_results:
            if isinstance(entry, dict):
                add_name(entry.get("character"))

    return inferred


def rebuild_networks_from_raw(raw_results: list, agents: list) -> tuple:
    """
    Rebuild cognitive_networks, consensus_network from raw_questionnaire_results
    when the pre-computed ones are empty.
    """
    cognitive_networks = defaultdict(dict)
    for entry in raw_results:
        agent = entry.get("character")
        dim = entry.get("dimension")
        val = entry.get("value")
        if val is None:
            val = _parse_json_from_text(entry.get("answer_text", ""))
        if agent and dim:
            if dim == "direct_ties" and isinstance(val, list):
                cognitive_networks[agent]["ego"] = val
                cognitive_networks[agent]["direct_ties"] = val
            elif dim == "ego_network" and isinstance(val, list):
                cognitive_networks[agent]["ego"] = val
                cognitive_networks[agent]["ego_network"] = val
            elif dim == "global_css" and isinstance(val, dict):
                cognitive_networks[agent]["global"] = val
            elif dim == "affective_valence" and isinstance(val, dict):
                cognitive_networks[agent]["valence"] = val
            elif dim == "perceived_centrality" and isinstance(val, list):
                cognitive_networks[agent]["centrality"] = val

    cognitive_networks = dict(cognitive_networks)

    # Build consensus from global views
    votes = defaultdict(lambda: defaultdict(int))
    for perceiver, data in cognitive_networks.items():
        global_view = data.get("global", {})
        if not isinstance(global_view, dict):
            continue
        for source, targets in global_view.items():
            if source not in agents or not isinstance(targets, list):
                continue
            for target in targets:
                if target in agents:
                    votes[source][target] += 1
    consensus_network = {k: dict(v) for k, v in votes.items()}

    return cognitive_networks, consensus_network


def build_objective_graph(
    objective_data: dict, agents: list[str], threshold: int = 0
) -> nx.Graph:
    """Build an undirected graph based on interactions with a minimum interaction threshold."""
    G = nx.Graph()
    G.add_nodes_from(agents)

    if not objective_data:
        return G

    for agent_a, interactions in objective_data.items():
        if agent_a not in agents:
            continue
        for agent_b, count in interactions.items():
            if agent_b in agents and count > threshold:
                G.add_edge(agent_a, agent_b, weight=count)
    return G


def build_ground_truth_graph(gt_data: dict, agents: list[str]) -> nx.Graph:
    """Build a graph from the setup's ground truth definitions."""
    G = nx.Graph()
    G.add_nodes_from(agents)

    if not gt_data:
        return G

    for source, targets in gt_data.items():
        if source not in agents:
            continue
        if isinstance(targets, list):
            for target in targets:
                if target in agents:
                    G.add_edge(source, target)
    return G


def build_consensus_graph(
    consensus_data: dict,
    agents: list[str],
    total_respondents: int,
    threshold_pct: float = 0.5,
) -> nx.DiGraph:
    """Build a directed graph based on the computed consensus votes and a percentage threshold."""
    G = nx.DiGraph()
    G.add_nodes_from(agents)

    threshold = threshold_pct * total_respondents

    for source, targets in consensus_data.items():
        for target, weight in targets.items():
            if weight >= threshold:
                G.add_edge(source, target, weight=weight)
    return G


def build_subjective_graph(
    agent_name: str, subjective_data: dict, agents: list[str]
) -> nx.DiGraph:
    """Build a directed graph for one agent's mental map."""
    G = nx.DiGraph()
    G.add_nodes_from(agents)

    direct_view = subjective_data.get("ego", subjective_data.get("direct_ties", []))
    if isinstance(direct_view, list):
        for target in direct_view:
            if target in agents and target != agent_name:
                G.add_edge(agent_name, target)

    if G.number_of_edges() == 0:
        global_view = subjective_data.get("global", {})
        if isinstance(global_view, dict):
            for perceived_source, perceived_targets in global_view.items():
                if isinstance(perceived_targets, list):
                    for target in perceived_targets:
                        if target in agents and perceived_source in agents:
                            G.add_edge(perceived_source, target)
    return G


def build_intersection_las_graph(
    cognitive_networks: dict, agents: list[str]
) -> nx.DiGraph:
    """Build the Intersection Locally Aggregated Structure (LAS).
    An edge from A to B exists only if both A and B agree it exists.
    """
    G = nx.DiGraph()
    G.add_nodes_from(agents)

    for agent_a in agents:
        for agent_b in agents:
            if agent_a == agent_b:
                continue

            # Does A perceive A -> B? (Check ego first, then global)
            a_data = cognitive_networks.get(agent_a, {})
            a_ego = a_data.get("ego", []) if isinstance(a_data.get("ego"), list) else []
            a_global_dict = a_data.get("global", {})
            a_global = (
                a_global_dict.get(agent_a, [])
                if isinstance(a_global_dict, dict)
                else []
            )
            a_sees_edge = (agent_b in a_ego) or (agent_b in a_global)

            # Does B perceive A -> B? (Check global perception of B about A)
            b_data = cognitive_networks.get(agent_b, {})
            b_global_dict = b_data.get("global", {})
            b_global = (
                b_global_dict.get(agent_a, [])
                if isinstance(b_global_dict, dict)
                else []
            )
            b_sees_edge = agent_b in b_global

            if a_sees_edge and b_sees_edge:
                G.add_edge(agent_a, agent_b)

    return G


def plot_css_grid(cognitive_networks: dict, agents: list[str], output_path: str):
    """Plot every agent's perceived 'global' network in a single grid figure."""
    n = len(agents)
    if n == 0:
        print(
            "Skipping mental-map grid: no agents could be inferred from the results file."
        )
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    fig.suptitle("Cognitive Social Structures: Individual Mental Maps", fontsize=20)

    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    mapping = {agent: str(i + 1) for i, agent in enumerate(agents)}

    # Create a unified graph to get a consistent spring layout
    G_combined = nx.Graph()
    G_combined.add_nodes_from(agents)
    for agent in agents:
        sub_data = cognitive_networks.get(agent, {})
        G_sub = build_subjective_graph(agent, sub_data, agents)
        G_combined.add_edges_from(G_sub.edges())

    G_mapped = nx.relabel_nodes(G_combined, mapping)
    pos = nx.spring_layout(G_mapped, k=0.9, seed=42)

    for i, agent in enumerate(agents):
        ax = axes[i]
        sub_data = cognitive_networks.get(agent, {})
        G = build_subjective_graph(agent, sub_data, agents)
        G_full = G.copy()
        G_full.add_nodes_from(agents)
        G_plot = nx.relabel_nodes(G_full, mapping)

        ax.set_title(f"{agent}'s Perception", fontsize=16)

        degrees = dict(G_plot.degree())
        node_sizes = [500 + 150 * degrees.get(n, 0) for n in G_plot.nodes()]

        nx.draw_networkx_nodes(
            G_plot,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color="lightgreen",
            edgecolors="gray",
        )
        nx.draw_networkx_labels(G_plot, pos, ax=ax, font_size=10, font_weight="bold")

        mutual_edges = [(u, v) for u, v in G_plot.edges() if G_plot.has_edge(v, u)]
        asym_edges = [(u, v) for u, v in G_plot.edges() if not G_plot.has_edge(v, u)]

        nx.draw_networkx_edges(
            G_plot,
            pos,
            ax=ax,
            edgelist=asym_edges,
            edge_color="gray",
            arrows=True,
            alpha=0.6,
            connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_edges(
            G_plot,
            pos,
            ax=ax,
            edgelist=mutual_edges,
            edge_color="green",
            width=2.0,
            arrows=True,
            alpha=0.9,
            connectionstyle="arc3,rad=0.1",
        )

        ax.axis("off")

    for j in range(len(agents), len(axes)):
        axes[j].axis("off")

    legend_text = "\n".join([f"{k}: {v}" for v, k in mapping.items()])
    fig.text(
        0.01,
        0.01,
        legend_text,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, facecolor="white"),
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_comparison(
    G1: nx.Graph,
    G2: nx.Graph,
    agents: list[str],
    title1: str,
    title2: str,
    main_title: str,
    pos: dict,
    output_path: str = None,
):
    """Plot two graphs side-by-side for comparison with a FIXED layout."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(main_title, fontsize=20)

    mapping = {agent: str(i + 1) for i, agent in enumerate(agents)}

    for ax, G, title, color in zip(
        axes, [G1, G2], [title1, title2], ["lightblue", "lightyellow"]
    ):
        ax.set_title(title, fontsize=16)

        G_full = G.copy()
        G_full.add_nodes_from(agents)
        G_plot = nx.relabel_nodes(G_full, mapping)

        degrees = dict(G_plot.degree())
        node_sizes = [600 + 200 * degrees.get(n, 0) for n in G_plot.nodes()]

        nx.draw_networkx_nodes(
            G_plot,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=color,
            edgecolors="gray",
        )
        nx.draw_networkx_labels(G_plot, pos, ax=ax, font_size=12, font_weight="bold")

        if isinstance(G_plot, nx.DiGraph):
            mutual_edges = [(u, v) for u, v in G_plot.edges() if G_plot.has_edge(v, u)]
            asym_edges = [
                (u, v) for u, v in G_plot.edges() if not G_plot.has_edge(v, u)
            ]

            nx.draw_networkx_edges(
                G_plot,
                pos,
                ax=ax,
                edgelist=asym_edges,
                edge_color="gray",
                arrows=True,
                alpha=0.6,
                connectionstyle="arc3,rad=0.1",
            )
            nx.draw_networkx_edges(
                G_plot,
                pos,
                ax=ax,
                edgelist=mutual_edges,
                edge_color="green",
                width=2.0,
                arrows=True,
                alpha=0.9,
                connectionstyle="arc3,rad=0.1",
            )
        else:
            weights = [
                max(1, G_plot[u][v].get("weight", 1) * 0.5) for u, v in G_plot.edges()
            ]
            nx.draw_networkx_edges(
                G_plot, pos, ax=ax, width=weights, edge_color="green", alpha=0.6
            )

        ax.axis("off")

    legend_text = "\n".join([f"{k}: {v}" for v, k in mapping.items()])
    fig.text(
        0.01,
        0.01,
        legend_text,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, facecolor="white"),
    )

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_metrics_over_time(steps: list, metrics_dict: dict, output_path: str):
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics_dict.items():
        plt.plot(steps, values, marker="o", label=metric_name)

    plt.title("Network Metrics Over Time", fontsize=16)
    plt.xlabel("Simulation Step", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cognitive Social Structures."
    )
    parser.add_argument("results_file", type=str, help="Path to the JSON results file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Consensus threshold percentage (0.0 to 1.0). Default 0.2.",
    )
    parser.add_argument(
        "--obj-min",
        type=int,
        default=0,
        help="Minimum interaction count for objective edges. Default 0.",
    )
    parser.add_argument(
        "--agent", type=str, default=None, help="Specific agent's perspective."
    )
    parser.add_argument(
        "--ref",
        type=str,
        choices=["setup", "emergent", "objective"],
        default="emergent",
        help="Reference network to show on the left side. setup=hardcoded, emergent=valence, objective=interactions. Default emergent.",
    )
    args = parser.parse_args()

    data = load_results(args.results_file)
    agents = infer_agents_from_data(data)
    if not agents:
        print("Error: no agents could be inferred from the results file.")
        return

    objective_data = data.get("objective_network", {})
    setup_gt_data = data.get(
        "ground_truth_network",
        data.get("original_metadata", {}).get("ground_truth_network", {}),
    )

    steps_data = data.get("steps", {})
    if not steps_data:
        # Fallback to old format
        steps_data = {
            "final": {
                "cognitive_networks": data.get("cognitive_networks", {}),
                "consensus_network": data.get("consensus_network", {}),
                "emergent_gt_network": data.get("emergent_gt_network", {}),
                "ground_truth_network": setup_gt_data,
                "metrics": data.get("metrics", {}),
            }
        }

    results_dir = os.path.dirname(args.results_file)
    results_name = os.path.basename(args.results_file).replace(".json", "")
    vis_dir = os.path.join(results_dir, "visualizations", results_name)
    os.makedirs(vis_dir, exist_ok=True)

    objective_G = build_objective_graph(objective_data, agents, threshold=args.obj_min)
    setup_G = build_ground_truth_graph(setup_gt_data, agents)

    metrics_over_time = defaultdict(list)
    step_labels = []

    # Sort steps numerically if possible, otherwise string sort
    def step_key_sort(k):
        try:
            return int(k)
        except ValueError:
            return 999999

    for step_key in sorted(steps_data.keys(), key=step_key_sort):
        step_info = steps_data[step_key]
        print(f"\n--- Visualizing Step: {step_key} ---")
        consensus_data = step_info.get("consensus_network", {})
        cognitive_networks = step_info.get("cognitive_networks", {})

        # Fallback for missing pre-computed networks
        if not cognitive_networks and data.get("raw_questionnaire_results"):
            print(
                f"Step {step_key}: Cognitive networks missing. Rebuilding from raw data..."
            )
            raw_results = data.get("raw_questionnaire_results", [])
            try:
                numeric_step = int(step_key)
                raw_results = [r for r in raw_results if r.get("step") == numeric_step]
            except ValueError:
                pass

            cognitive_networks, rebuilt_consensus = rebuild_networks_from_raw(
                raw_results, agents
            )
            if not consensus_data:
                consensus_data = rebuilt_consensus

        # Build graphs for this step
        consensus_G = build_consensus_graph(
            consensus_data,
            agents,
            len(cognitive_networks) or 1,
            threshold_pct=args.threshold,
        )
        intersection_las_G = build_intersection_las_graph(cognitive_networks, agents)

        emergent_gt_data = step_info.get("emergent_gt_network", {})
        emergent_G = build_ground_truth_graph(emergent_gt_data, agents)

        # Determine reference graph based on --ref flag
        if args.ref == "setup":
            ref_G = setup_G
            ref_label = "Planned Setup Network (Ground Truth)"
        elif args.ref == "objective":
            ref_G = objective_G
            ref_label = f"Actual Interactions (min {args.obj_min})"
        else:  # emergent
            if emergent_gt_data:
                ref_G = emergent_G
                ref_label = "Emergent Social Reality (Valence > 2)"
            else:
                ref_G = setup_G
                ref_label = "Planned Setup Network (Fallback)"

        # CALCULATE GLOBAL LAYOUT FOR THIS STEP
        # We use all possible edges from all sources to make the layout stable
        mapping = {agent: str(i + 1) for i, agent in enumerate(agents)}
        G_layout = nx.Graph()
        G_layout.add_nodes_from(agents)
        G_layout.add_edges_from(ref_G.edges())
        G_layout.add_edges_from(consensus_G.edges())
        G_layout.add_edges_from(intersection_las_G.edges())
        for agent_name, sub_data in cognitive_networks.items():
            G_sub = build_subjective_graph(agent_name, sub_data, agents)
            G_layout.add_edges_from(G_sub.edges())

        G_mapped = nx.relabel_nodes(G_layout, mapping)
        pos = nx.spring_layout(G_mapped, k=1.2, seed=42)  # Slightly more k for spacing

        step_suffix = f"_step{step_key}" if step_key != "final" else ""

        # 2. Plot: Reference vs Consensus
        gt_vs_con_out = os.path.join(
            vis_dir, f"ref_vs_consensus_{int(args.threshold*100)}{step_suffix}.png"
        )
        plot_comparison(
            ref_G,
            consensus_G,
            agents,
            ref_label,
            f"Consensus CSS ({int(args.threshold*100)}%)",
            f"Consensus Accuracy (Step {step_key})",
            pos,
            output_path=gt_vs_con_out,
        )

        # 4. Plot: Intersection LAS vs Reference
        las_vs_gt_out = os.path.join(
            vis_dir, f"las_intersection_vs_ref{step_suffix}.png"
        )
        plot_comparison(
            ref_G,
            intersection_las_G,
            agents,
            ref_label,
            "Intersection LAS (Mutual Agreement)",
            f"Confirmed Social Reality (Step {step_key})",
            pos,
            output_path=las_vs_gt_out,
        )

        # 5. Plot CSS Sociogram Grid
        css_grid_out = os.path.join(vis_dir, f"css_sociogram_grid{step_suffix}.png")
        plot_css_grid(cognitive_networks, agents, css_grid_out)

        # 6. Individual Perspectives vs Reference
        for agent_name, sub_data in cognitive_networks.items():
            if args.agent and agent_name != args.agent:
                continue
            subjective_G = build_subjective_graph(agent_name, sub_data, agents)
            out = os.path.join(vis_dir, f"{agent_name}_vs_ref{step_suffix}.png")
            plot_comparison(
                ref_G,
                subjective_G,
                agents,
                ref_label,
                f"{agent_name}'s Mental Map",
                f"Individual Perception: {agent_name} (Step {step_key})",
                pos,
                output_path=out,
            )

        # Collect metrics
        metrics_data = step_info.get("metrics", {})
        if metrics_data:
            step_labels.append(str(step_key))
            if "reciprocity" in metrics_data:
                metrics_over_time["Reciprocity"].append(metrics_data["reciprocity"])
            if "consensus_transitivity" in metrics_data:
                metrics_over_time["Consensus Transitivity"].append(
                    metrics_data["consensus_transitivity"]
                )
            if "structural_balance" in metrics_data:
                metrics_over_time["Structural Balance"].append(
                    metrics_data["structural_balance"]
                )
            if "town_cognitive_accuracy" in metrics_data:
                metrics_over_time["Town Cognitive Accuracy"].append(
                    metrics_data["town_cognitive_accuracy"]
                )

    if len(step_labels) > 1:
        plot_metrics_over_time(
            step_labels,
            metrics_over_time,
            os.path.join(vis_dir, "metrics_over_time.png"),
        )


if __name__ == "__main__":
    main()
