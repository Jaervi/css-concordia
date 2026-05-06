"""
SNA Metrics for Cognitive Social Structure (CSS) analysis.
Provides quantitative measures for agent social accuracy and network properties.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Any


def calculate_reciprocity(ego_networks: Dict[str, List[str]]) -> float:
    """
    Calculates the proportion of mutual ties in the ego-reported networks.
    Formula: (2 * Mutual) / (Total Reported)
    """
    total_ties = 0
    mutual_ties = 0
    seen_pairs = set()

    for u, friends in ego_networks.items():
        if not isinstance(friends, list):
            continue
        for v in friends:
            total_ties += 1
            pair = tuple(sorted((u, v)))
            if v in ego_networks and u in ego_networks[v]:
                if pair not in seen_pairs:
                    mutual_ties += 1
                    seen_pairs.add(pair)

    if total_ties == 0:
        return 0.0
    return (2 * mutual_ties) / total_ties


def calculate_accuracy(perceived: List[str], actual: List[str]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and F1-Score for an agent's perception vs. truth.
    """
    p_set = set(perceived) if isinstance(perceived, list) else set()
    a_set = set(actual) if isinstance(actual, list) else set()

    tp = len(p_set.intersection(a_set))
    fp = len(p_set - a_set)
    fn = len(a_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_transitivity(network: Dict[str, List[str]]) -> float:
    """
    Calculates the Global Clustering Coefficient (Transitivity).
    Probability that if A-B and B-C exist, A-C also exists.
    """
    triads = 0
    closed_triads = 0

    nodes = list(network.keys())
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            for k in range(len(nodes)):
                if i == j or j == k or i == k:
                    continue
                u, v, w = nodes[i], nodes[j], nodes[k]

                # Path u -> v -> w
                if v in network.get(u, []) and w in network.get(v, []):
                    triads += 1
                    if w in network.get(u, []):
                        closed_triads += 1

    if triads == 0:
        return 0.0
    return closed_triads / triads


def analyze_css_accuracy(
    cognitive_networks: Dict[str, Any], ground_truth: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Computes accuracy for every agent's mental map of the town.
    """
    results = {}
    for agent, data in cognitive_networks.items():
        ego_friends = data.get("ego", [])
        gt_friends = ground_truth.get(agent, [])
        results[agent] = calculate_accuracy(ego_friends, gt_friends)
    return results


def calculate_structural_balance(valence_data: Dict[str, Dict[str, float]]) -> float:
    """
    Heuristic for Balance Theory: 'Friend of a friend is a friend' (+ + +)
    or 'Enemy of my enemy is my friend' (- - +).
    Returns the percentage of balanced triads.
    """
    nodes = list(valence_data.keys())
    triads = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                u, v, w = nodes[i], nodes[j], nodes[k]

                # Check if we have valid dictionary data for these nodes
                if (
                    not isinstance(valence_data.get(u), dict)
                    or not isinstance(valence_data.get(v), dict)
                    or not isinstance(valence_data.get(w), dict)
                ):
                    continue

                # Get signs (positive or negative)
                s1 = np.sign(valence_data[u].get(v, 0))
                s2 = np.sign(valence_data[v].get(w, 0))
                s3 = np.sign(valence_data[u].get(w, 0))

                if s1 == 0 or s2 == 0 or s3 == 0:
                    continue

                # Balanced if product of signs is positive
                is_balanced = (s1 * s2 * s3) > 0
                triads.append(is_balanced)

    if not triads:
        return 0.0
    return sum(triads) / len(triads)


def calculate_cognitive_accuracy(
    perceived_matrix: Dict[str, Any],
    objective_matrix: Dict[str, Dict[str, int]],
    agent_names: List[str],
) -> float:
    """
    Calculates the correlation between perceived ties and objective interaction frequency.
    Handles both list-based ego networks and weighted dict-based consensus.
    """
    perceived_vec = []
    objective_vec = []

    for source in agent_names:
        data = perceived_matrix.get(source, [])

        for target in agent_names:
            if source == target:
                continue

            # Perceived: If list, 1 if in list. If dict, use the weight/count.
            val = 0
            if isinstance(data, list):
                val = 1 if target in data else 0
            elif isinstance(data, dict):
                val = data.get(target, 0)

            perceived_vec.append(val)

            # Objective: interaction count
            inter_count = objective_matrix.get(source, {}).get(target, 0)
            objective_vec.append(inter_count)

    if not perceived_vec or (sum(perceived_vec) == 0 and sum(objective_vec) == 0):
        return 1.0  # Perfect agreement on "no ties"
    if sum(perceived_vec) == 0 or sum(objective_vec) == 0:
        return 0.0  # No overlap possible

    try:
        correlation = np.corrcoef(perceived_vec, objective_vec)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    except:
        return 0.0
