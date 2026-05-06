"""
Natural-language questionnaires for Cognitive Social Structure (CSS) analysis.
Focuses on social influence and directed attention to break "mutual bias."
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
from concordia.contrib.data.questionnaires import base_questionnaire

Question = base_questionnaire.Question


def get_town_questions(agent_names: List[str]) -> List[Question]:
    """
    Generates questions that ask for natural language reasoning about directed social ties.
    """
    agents_str = ", ".join(agent_names)

    return [
        Question(
            statement=(
                f"Which residents from the following list do you personally consider a close friend or ally? "
                f"List: {agents_str}.\n"
                "Provide a brief list of names with a one-sentence reason for each. "
                "Focus on your most recent meaningful interactions."
            ),
            dimension="ego_network",
            preprompt=(
                "Reflect on your personal connections. "
                "My closest friends and allies are: "
            ),
            choices=[],
        ),
        Question(
            statement=(
                f"From the following list of residents, who are your friends, allies, or regular contacts? "
                f"List: {agents_str}.\n"
                "Do NOT provide any explanation or reasoning. Simply provide a comma-separated list of names. "
                "If you have no connections, answer 'None'."
            ),
            dimension="direct_ties",
            preprompt="My friends and contacts are: ",
            choices=[],
        ),
        Question(
            statement=(
                f"Who in this town is influencing whom? "
                f"List: {agents_str}.\n"
                "Briefly list the pairs of social influence you have observed. "
                "Example: [Name] is influenced by [Name]. "
                "Keep it concise; do not go person by person if you haven't observed anything."
            ),
            dimension="global_css",  # Keep as 'global_css' for extraction compatibility
            preprompt=(
                "Reflect on the social power dynamics you have observed. "
                "The primary lines of influence I see are: "
            ),
            choices=[],
        ),
        Question(
            statement=(
                f"Rate your personal feelings toward each person in the following list on a scale of -5 to +5: {agents_str}.\n"
                "Provide a list of Name: Score (e.g., Martha: +3). "
                "Briefly mention the primary reason for any score that is not neutral (0)."
            ),
            dimension="affective_valence",
            preprompt=("Reflect honestly on your feelings toward each person: "),
            choices=[],
        ),
        Question(
            statement=(
                f"List the primary friendships and alliances you have observed between OTHER residents.\n"
                f"List: {agents_str}.\n"
                "Format: [Name] and [Name] appear to be close allies. "
                "Only list relationships where you have seen actual evidence of cooperation or friendship."
            ),
            dimension="css_friendship",
            preprompt=(
                "Reflect on the friendships you have observed between other people: "
            ),
            choices=[],
        ),
        Question(
            statement=(
                f"Who are the three most socially influential or well-connected people in this community? "
                f"Choose from: {agents_str}.\n"
                "Briefly explain why each person holds social power."
            ),
            dimension="perceived_centrality",
            preprompt="Reflect on the power dynamics and social reach of each person: ",
            choices=[],
        ),
    ]


class TownBeliefsQuestionnaire(base_questionnaire.QuestionnaireBase):
    """Questionnaire to capture agent beliefs and social perceptions in natural language."""

    def __init__(self, agent_names: List[str]):
        self.agent_names = agent_names
        questions = get_town_questions(agent_names)

        super().__init__(
            name="Town_CSS_Questionnaire",
            description="A natural-language questionnaire to capture directed social influence and reasoning.",
            questionnaire_type="open-ended",
            observation_preprompt="{player_name} is carefully reviewing their memories to explain the social influence structure of Willowbrook.",
            questions=questions,
            dimensions=[
                "ego_network",
                "direct_ties",
                "global_css",
                "affective_valence",
                "css_friendship",
                "perceived_centrality",
            ],
            context="This questionnaire captures directed social ties and power dynamics.",
        )

    def process_answer(
        self, player_name: str, answer_text: str, question: Question
    ) -> Tuple[str, Any]:
        """Simply return the raw text. Extraction happens in post-processing."""
        return question.dimension, answer_text

    def aggregate_results(
        self, player_answers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregates raw answers for a single player."""
        aggregated = {}
        for q_id, question_data in player_answers.items():
            dimension = question_data["dimension"]
            value = question_data["value"]
            aggregated[dimension] = value
        return aggregated

    def get_dimension_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {"affective_valence": (-5.0, 5.0)}

    def plot_results(
        self,
        results_df: pd.DataFrame,
        label_column: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        pass
