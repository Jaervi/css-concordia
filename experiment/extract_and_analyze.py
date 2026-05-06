"""
Post-processing script for social simulation results.
Uses an LLM to extract structured data from natural language interviews.
Calculates SNA metrics and saves the final analyzed dataset.
"""

import argparse
import os
import json
import re
from collections import defaultdict
import numpy as np
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

from concordia.contrib.language_models.google.google_aistudio_model import (
    GoogleAIStudioLanguageModel,
)
import metrics


def setup_extraction_model():
    """Setup the Gemini model for structured extraction."""
    load_dotenv()
    API_KEY = os.environ.get("GEMINI_API_KEY")
    MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    return GoogleAIStudioLanguageModel(
        model_name=MODEL_NAME,
        api_key=API_KEY,
    )


def _extract_json_block(text: str) -> Any:
    """Extracts and parses JSON from a markdown block or bare string."""
    # Try markdown code block
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try bare JSON list/dict
    m = re.search(r"([\[{].*)", text, re.DOTALL)
    if m:
        try:
            # Basic repair: remove trailing text after the last brace/bracket
            cleaned = m.group(1).strip()
            # find last ] or }
            last_idx = max(cleaned.rfind("]"), cleaned.rfind("}"))
            if last_idx != -1:
                cleaned = cleaned[: last_idx + 1]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    return None


def _extract_direct_ties(
    text: str, agent_name: str, agent_list: List[str]
) -> List[str]:
    """Extracts a clean ordered list of direct ties from a questionnaire answer."""
    if not isinstance(text, str):
        return []

    cleaned_text = text.strip()
    if not cleaned_text or cleaned_text.lower() in {
        "none",
        "no connections",
        "no ties",
    }:
        return []

    name_lookup = {name.lower(): name for name in agent_list}
    candidate_names = [name for name in agent_list if name != agent_name]
    if not candidate_names:
        return []

    pattern = re.compile(
        r"\b("
        + "|".join(
            re.escape(name) for name in sorted(candidate_names, key=len, reverse=True)
        )
        + r")\b",
        re.IGNORECASE,
    )

    direct_ties: List[str] = []
    seen = set()
    for match in pattern.finditer(cleaned_text):
        normalized = name_lookup.get(match.group(1).lower())
        if normalized and normalized not in seen and normalized != agent_name:
            seen.add(normalized)
            direct_ties.append(normalized)

    return direct_ties


def extract_ego_network(
    model: GoogleAIStudioLanguageModel,
    agent_name: str,
    text: str,
    agent_list: List[str],
) -> List[str]:
    """Extracts a list of friends from the text."""
    prompt = (
        f"The following is an interview response from {agent_name} about their friends, allies, or regular contacts in town.\n"
        f"Agents list: {', '.join(agent_list)}\n"
        f"Interview text: {text}\n"
        f"Based on the text, extract a JSON list of {agent_name}'s direct ties. "
        f"Only include names from the provided agent list. If no friends are mentioned, return [].\n"
        f"Output ONLY the JSON list."
    )
    response = model.sample_text(prompt)
    extracted = _extract_json_block(response)
    if isinstance(extracted, list):
        return [n for n in extracted if n in agent_list]
    return []


def extract_global_css(
    model: GoogleAIStudioLanguageModel,
    agent_name: str,
    text: str,
    agent_list: List[str],
) -> Dict[str, List[str]]:
    """Extracts the perceived global social network from the text."""
    prompt = (
        f"The following is an interview response from {agent_name} about who everyone else is influenced by or spending the most time listening to.\n"
        f"Agents list: {', '.join(agent_list)}\n"
        f"Interview text: {text}\n"
        f"Based on the text, extract a JSON object where each key is a person from the list and the value is a list of the people who influence them or who they listen to.\n"
        f'Example: {{"Alice": ["Bob"], "Bob": ["Alice", "Carol"], "Carol": []}}\n'
        f"Ensure every agent in the list appears as a key. Only include people from the provided list as values.\n"
        f"Output ONLY the JSON object."
    )
    response = model.sample_text(prompt)
    extracted = _extract_json_block(response)
    if isinstance(extracted, dict):
        result = {}
        for name in agent_list:
            friends = extracted.get(name, [])
            if not isinstance(friends, list):
                friends = []
            result[name] = [n for n in friends if n in agent_list]
        return result
    return {n: [] for n in agent_list}


def extract_valence(
    model: GoogleAIStudioLanguageModel,
    agent_name: str,
    text: str,
    agent_list: List[str],
) -> Dict[str, float]:
    """Extracts affective valence scores from the text."""
    prompt = (
        f"The following is an interview response from {agent_name} describing their feelings toward other residents.\n"
        f"Agents list: {', '.join(agent_list)}\n"
        f"Interview text: {text}\n"
        f"Extract a JSON object where each key is a person (excluding {agent_name}) and the value is a score from -5 (hostile) to +5 (trusted ally).\n"
        f"Use 0 for neutral or unknown relationships.\n"
        f"Output ONLY the JSON object."
    )
    response = model.sample_text(prompt)
    extracted = _extract_json_block(response)
    if isinstance(extracted, dict):
        result = {}
        for name in agent_list:
            if name == agent_name:
                continue
            score = extracted.get(name, 0.0)
            if not isinstance(score, (int, float)):
                score = 0.0
            result[name] = float(max(-5.0, min(5.0, score)))
        return result
    return {n: 0.0 for n in agent_list if n != agent_name}


def extract_centrality(
    model: GoogleAIStudioLanguageModel,
    agent_name: str,
    text: str,
    agent_list: List[str],
) -> List[str]:
    """Extracts perceived influential people."""
    prompt = (
        f"The following is an interview response from {agent_name} about the most influential people in town.\n"
        f"Agents list: {', '.join(agent_list)}\n"
        f"Interview text: {text}\n"
        f"Extract a JSON list of the top 3 most influential people mentioned. Only use names from the agent list.\n"
        f'Example: ["Alice", "Bob", "Carol"]\n'
        f"Output ONLY the JSON list."
    )
    response = model.sample_text(prompt)
    extracted = _extract_json_block(response)
    if isinstance(extracted, list):
        return [n for n in extracted if n in agent_list][:3]
    return []


def compute_consensus_network(cognitive_networks: dict, agent_names: list[str]) -> dict:
    """Computes consensus by counting perceived ties."""
    votes = defaultdict(lambda: defaultdict(int))
    for perceiver, data in cognitive_networks.items():
        global_view = data.get("global", {})
        for source, targets in global_view.items():
            if source in agent_names:
                for target in targets:
                    if target in agent_names:
                        votes[source][target] += 1
    return {k: dict(v) for k, v in votes.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured CSS data from natural language results."
    )
    parser.add_argument("results_file", type=str, help="Path to the results.json file.")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: File {args.results_file} not found.")
        return

    with open(args.results_file, "r") as f:
        data = json.load(f)

    agent_names = data.get("agents", [])
    raw_results = data.get("raw_questionnaire_results", [])

    # Group results by step
    results_by_step = defaultdict(list)
    for entry in raw_results:
        step = entry.get("step", "final")
        results_by_step[step].append(entry)

    model = setup_extraction_model()

    print(
        f"Extracting structured data from {len(raw_results)} responses across {len(results_by_step)} steps..."
    )

    steps_data = {}

    for step, step_results in results_by_step.items():
        print(f"\n--- Processing Step: {step} ---")
        cognitive_networks = defaultdict(dict)

        for entry in step_results:
            agent = entry.get("character")
            dim = entry.get("dimension")
            text = entry.get("answer_text", "")

            if not agent or not dim or not text:
                continue

            print(f"  Processing {agent} / {dim}...")

            if dim == "direct_ties":
                val = _extract_direct_ties(text, agent, agent_names)
                if not val:
                    val = extract_ego_network(model, agent, text, agent_names)

                # Merge with existing ego ties
                existing_ego = cognitive_networks[agent].get("ego", [])
                cognitive_networks[agent]["ego"] = list(set(existing_ego) | set(val))
                cognitive_networks[agent]["direct_ties"] = val
            elif dim == "ego_network":
                val = extract_ego_network(model, agent, text, agent_names)

                # Merge with existing ego ties
                existing_ego = cognitive_networks[agent].get("ego", [])
                cognitive_networks[agent]["ego"] = list(set(existing_ego) | set(val))
                cognitive_networks[agent]["ego_network"] = val
            elif dim == "global_css":
                val = extract_global_css(model, agent, text, agent_names)
                cognitive_networks[agent]["global"] = val
            elif dim == "affective_valence":
                val = extract_valence(model, agent, text, agent_names)
                cognitive_networks[agent]["valence"] = val
            elif dim == "perceived_centrality":
                val = extract_centrality(model, agent, text, agent_names)
                cognitive_networks[agent]["centrality"] = val

        # Convert defaultdict to regular dict
        cognitive_networks = dict(cognitive_networks)

        print(f"Computing metrics for step {step}...")

        # 1. Consensus
        consensus_network = compute_consensus_network(cognitive_networks, agent_names)

        # 2. Emergent Ground Truth (Mutual Positive Valence > 2)
        emergent_gt = defaultdict(list)
        valence_data = {
            agent: info.get("valence", {}) for agent, info in cognitive_networks.items()
        }
        for a in agent_names:
            for b in agent_names:
                if a == b:
                    continue
                val_a_b = valence_data.get(a, {}).get(b, 0)
                val_b_a = valence_data.get(b, {}).get(a, 0)
                if val_a_b > 2 and val_b_a > 2:
                    emergent_gt[a].append(b)
        emergent_gt = dict(emergent_gt)

        # 3. SNA Metrics
        ego_networks = {
            agent: info.get("ego", []) for agent, info in cognitive_networks.items()
        }
        reciprocity = metrics.calculate_reciprocity(ego_networks)
        accuracy_results = metrics.analyze_css_accuracy(cognitive_networks, emergent_gt)
        consensus_transitivity = metrics.calculate_transitivity(consensus_network)
        town_cognitive_accuracy = metrics.calculate_cognitive_accuracy(
            consensus_network, data.get("objective_network", {}), agent_names
        )
        structural_balance = metrics.calculate_structural_balance(valence_data)

        steps_data[str(step)] = {
            "metrics": {
                "reciprocity": reciprocity,
                "consensus_transitivity": consensus_transitivity,
                "structural_balance": structural_balance,
                "town_cognitive_accuracy": town_cognitive_accuracy,
                "agent_accuracies": accuracy_results,
            },
            "consensus_network": consensus_network,
            "emergent_gt_network": emergent_gt,
            "cognitive_networks": cognitive_networks,
        }

    final_output = {
        "original_metadata": {
            "setup_name": data.get("setup_name"),
            "timestamp": data.get("timestamp"),
            "agents": agent_names,
        },
        "objective_network": data.get("objective_network"),
        "steps": steps_data,
        "raw_questionnaire_results": raw_results,
        "raw_data_source": args.results_file,
    }

    print(f"\nFinal output keys: {list(final_output.keys())}")

    output_path = args.results_file.replace(".json", "_analyzed.json")
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"Analysis complete! Results saved to {output_path}")

    # Final check
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"File saved successfully. Size: {size} bytes")


if __name__ == "__main__":
    main()
