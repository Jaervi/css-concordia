"""
Setup loader for simulation configurations.
Loads simulation setups from JSON files.
"""

import json
import os
from typing import Any

from concordia.typing import prefab as prefab_lib
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.utils import helper_functions


def get_prefabs() -> dict:
    """Get the default prefabs dictionary."""
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }


def load_setup(setup_path: str) -> dict:
    """
    Load a simulation setup from a JSON file.

    Args:
        setup_path: Path to the JSON setup file.

    Returns:
        Dictionary containing the setup configuration.
    """
    with open(setup_path, "r", encoding="utf-8") as f:
        setup = json.load(f)
    return setup


def build_instances_from_setup(setup: dict) -> list[prefab_lib.InstanceConfig]:
    """
    Build InstanceConfig objects from a setup dictionary.
    """
    instances = []

    # Build entity instances
    for agent in setup.get("agents", []):
        instances.append(
            prefab_lib.InstanceConfig(
                prefab=agent.get("prefab", "basic__Entity"),
                role=prefab_lib.Role.ENTITY,
                params={
                    "name": agent["name"],
                    "goal": agent["goal"],
                },
            )
        )

    # Build game master instance
    gm_config = setup.get("game_master", {})
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=gm_config.get("prefab", "generic__GameMaster"),
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": gm_config.get("name", "default rules"),
                "extra_event_resolution_steps": gm_config.get(
                    "extra_event_resolution_steps", ""
                ),
            },
        )
    )

    # Build initializer instance
    init_config = setup.get("initializer", {})
    shared_memories = init_config.get("shared_memories", [])

    # NEW: Handle Agent-Specific Memories (Private Knowledge)
    agent_memories = {}
    for agent in setup.get("agents", []):
        memories = []
        if "description" in agent:
            memories.append(agent["description"])
        if "private_memories" in agent:
            memories.extend(agent["private_memories"])

        if memories:
            agent_memories[agent["name"]] = memories

    instances.append(
        prefab_lib.InstanceConfig(
            prefab=init_config.get(
                "prefab", "formative_memories_initializer__GameMaster"
            ),
            role=prefab_lib.Role.INITIALIZER,
            params={
                "name": init_config.get("name", "initial setup rules"),
                "next_game_master_name": init_config.get(
                    "next_game_master_name", "default rules"
                ),
                "shared_memories": shared_memories,
                "agent_memories": agent_memories,  # Pass private knowledge map
            },
        )
    )

    return instances


def build_config_from_setup(setup: dict) -> prefab_lib.Config:
    """
    Build a Config object from a setup dictionary.

    Args:
        setup: The setup dictionary loaded from JSON.

    Returns:
        A Config object ready for simulation.
    """
    instances = build_instances_from_setup(setup)
    prefabs = get_prefabs()

    return prefab_lib.Config(
        default_premise=setup.get("premise", ""),
        default_max_steps=setup.get("max_steps", 16),
        prefabs=prefabs,
        instances=instances,
    )


def get_agent_names_from_setup(setup: dict) -> list[str]:
    """
    Extract agent names from a setup dictionary.

    Args:
        setup: The setup dictionary loaded from JSON.

    Returns:
        List of agent names.
    """
    return [agent["name"] for agent in setup.get("agents", [])]


def list_available_setups(setups_dir: str) -> list[str]:
    """
    List all available setup files in a directory.

    Args:
        setups_dir: Path to the directory containing setup files.

    Returns:
        List of setup file names (without extension).
    """
    setups = []
    if os.path.exists(setups_dir):
        for file in os.listdir(setups_dir):
            if file.endswith(".json"):
                setups.append(file[:-5])  # Remove .json extension
    return setups
