"""
Streamlined simulation flow using Concordia-native components and QuestionnaireSimulation.
Saves raw natural-language interview results for separate post-processing.
"""

import argparse
import os
import datetime
import json
import re
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv

from concordia.prefabs.simulation import generic as simulation
from concordia.prefabs.simulation import questionnaire_simulation
from concordia.language_model import no_language_model
from concordia.contrib.language_models.google.google_aistudio_model import (
    GoogleAIStudioLanguageModel,
)
from concordia.typing import prefab as prefab_lib
import sentence_transformers

from setup_loader import (
    load_setup,
    build_config_from_setup,
    get_agent_names_from_setup,
    get_prefabs,
)
from questionnaires import TownBeliefsQuestionnaire


SETUPS_DIR = os.path.join(os.path.dirname(__file__), "setups")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def setup_model(disable_llm: bool = False):
    """Setup the language model and embedder."""
    load_dotenv()

    API_KEY = os.environ.get("GEMINI_API_KEY")
    MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")

    if not disable_llm and not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    if disable_llm:
        model = no_language_model.NoLanguageModel()
        embedder = lambda _: np.ones(3)
    else:
        safety_settings = (
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        )
        model = GoogleAIStudioLanguageModel(
            model_name=MODEL_NAME,
            api_key=API_KEY,
            safety_settings=safety_settings,
        )
        st_model = sentence_transformers.SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )
        embedder = lambda x: st_model.encode(x, show_progress_bar=False)

    return model, embedder


def extract_objective_network(raw_log: list, agent_names: list[str]) -> dict:
    """
    Build an objective interaction matrix based on co-occurrence in simulation events.
    Uses regex for precise name matching and captures group interactions.
    """
    interaction_matrix = defaultdict(lambda: defaultdict(int))

    # Pre-compile regex for each agent to speed up search
    agent_patterns = {
        name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        for name in agent_names
    }

    for entry in raw_log:
        text_content = str(entry)

        # Only skip very short, non-narrative system messages
        if len(text_content) < 30:
            continue

        # Find which agents are mentioned using word boundaries
        mentioned_agents = [
            name
            for name, pattern in agent_patterns.items()
            if pattern.search(text_content)
        ]

        # Capture group interactions (2 to 20 agents)
        if 2 <= len(mentioned_agents) <= 20:
            for i in range(len(mentioned_agents)):
                for j in range(i + 1, len(mentioned_agents)):
                    a, b = mentioned_agents[i], mentioned_agents[j]
                    interaction_matrix[a][b] += 1
                    interaction_matrix[b][a] += 1

    return {k: dict(v) for k, v in interaction_matrix.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Run Concordia Simulation (Natural Language Extraction)."
    )
    parser.add_argument("--setup", type=str, default="small_town_extended")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument(
        "--interview-interval",
        type=int,
        default=0,
        help="Steps between interviews (0 = only at end)",
    )
    args = parser.parse_args()

    # Load environment
    setup_path = os.path.join(SETUPS_DIR, f"{args.setup}.json")
    setup = load_setup(setup_path)
    agent_names = get_agent_names_from_setup(setup)

    print(f"Loading setup: {args.setup} with {len(agent_names)} agents.")

    model, embedder = setup_model(disable_llm=args.disable_llm)

    print("")
    main_config = build_config_from_setup(setup)

    # Premise setup - Distributed interaction model
    main_config.default_premise = (
        f"It is a busy day in Willowbrook. The residents are scattered throughout the town—"
        f"some at Rosie's Diner, others at the Hardware Store or the Library. People are meeting "
        f"in small, private groups to discuss their concerns about the new shopping center proposal. "
        f"Interactions are focused and local."
    )

    main_sim = simulation.Simulation(
        config=main_config,
        model=model,
        embedder=embedder,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_base = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{timestamp}")
    os.makedirs(checkpoint_base, exist_ok=True)

    main_log = ""
    q_log = ""
    results_list = []

    steps_completed = 0
    chunk_size = (
        args.interview_interval if args.interview_interval > 0 else args.max_steps
    )

    # Prepare Questionnaire Config
    questionnaire = TownBeliefsQuestionnaire(agent_names=agent_names)
    q_instances = []
    for agent in setup.get("agents", []):
        q_instances.append(
            prefab_lib.InstanceConfig(
                prefab=agent.get("prefab", "basic__Entity"),
                role=prefab_lib.Role.ENTITY,
                params={"name": agent["name"]},
            )
        )
    q_instances.append(
        prefab_lib.InstanceConfig(
            prefab="open_ended_interviewer__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": "TownInterviewerGM",
                "player_names": agent_names,
                "questionnaires": [questionnaire],
                "embedder": embedder,
                "sequence_of_events": [
                    "A researcher is asking residents about the community in detail."
                ],
            },
        )
    )
    q_config = prefab_lib.Config(
        default_premise="Natural language interview phase.",
        default_max_steps=100,
        prefabs=get_prefabs(),
        instances=q_instances,
    )
    from concordia.environment.engines import sequential_questionnaire

    q_engine = sequential_questionnaire.SequentialQuestionnaireEngine()

    print(f"[Phase 1 & 2] Running Simulation in chunks of {chunk_size}...")

    while steps_completed < args.max_steps:
        steps_this_round = min(chunk_size, args.max_steps - steps_completed)
        print(
            f"\n--- Running main simulation steps {steps_completed + 1} to {steps_completed + steps_this_round} ---"
        )

        main_log_chunk = main_sim.play(max_steps=steps_this_round)
        if isinstance(main_log_chunk, str):
            main_log += main_log_chunk
        else:
            main_log += str(main_log_chunk)

        steps_completed += steps_this_round

        ckpt_path = os.path.join(checkpoint_base, f"step_{steps_completed}")
        os.makedirs(ckpt_path, exist_ok=True)
        main_sim.save_checkpoint(step=steps_completed, checkpoint_path=ckpt_path)

        ckpt_file = os.path.join(ckpt_path, f"step_{steps_completed}_checkpoint.json")
        with open(ckpt_file, "r") as f:
            checkpoint_data = json.load(f)
        checkpoint_data["game_masters"] = {}

        print(f"--- Running Interviews at step {steps_completed} ---")
        q_sim = questionnaire_simulation.QuestionnaireSimulation(
            config=q_config,
            model=model,
            embedder=embedder,
            engine=q_engine,
        )
        q_sim.load_from_checkpoint(checkpoint_data)
        q_log_chunk = q_sim.play(max_steps=20)
        if isinstance(q_log_chunk, str):
            q_log += q_log_chunk
        else:
            q_log += str(q_log_chunk)

        step_results = []
        for gm in q_sim.get_game_masters():
            if gm.name == "TownInterviewerGM":
                try:
                    q_component = gm.get_component("questionnaire")
                    step_results = q_component.get_aggregated_results()
                except Exception as e:
                    print(f"Failed to extract results: {e}")
                    raw_q_log = q_sim.get_raw_log()
                    for entry in raw_q_log:
                        if hasattr(entry, "dimension") and hasattr(entry, "value"):
                            step_results.append(
                                {
                                    "character": entry.character,
                                    "dimension": entry.dimension,
                                    "value": entry.value,
                                    "answer_text": entry.answer_text,
                                }
                            )

        # Add step metadata
        for res in step_results:
            res["step"] = steps_completed
        results_list.extend(step_results)

    print("\n[Phase 3] Saving Raw Results...")

    # Compute Objective Network using full log
    raw_main_log = main_sim.get_raw_log()
    objective_network = extract_objective_network(raw_main_log, agent_names)

    final_structured_output = {
        "setup_name": args.setup,
        "timestamp": timestamp,
        "agents": agent_names,
        "objective_network": objective_network,
        "ground_truth_network": setup.get("ground_truth_network", {}),
        "raw_questionnaire_results": results_list,
    }

    # Save outputs
    output_base = os.path.join(OUTPUT_DIR, f"{args.setup}_{timestamp}")

    with open(f"{output_base}_main_log.html", "w", encoding="utf-8") as f:
        f.write(main_log)

    with open(f"{output_base}_questionnaire_log.html", "w", encoding="utf-8") as f:
        f.write(q_log)

    with open(f"{output_base}_results.json", "w", encoding="utf-8") as f:
        json.dump(final_structured_output, f, indent=4)

    print(f"\nSimulation finished. Raw results saved to {output_base}_results.json")
    print("Run the extraction script to compute metrics.")


if __name__ == "__main__":
    main()
