"""Shared adapters for heavier external agent benchmarks.

WebArena and SWE-bench have different runners, but TextGrad-RL only needs a
common trajectory shape: observations, actions, rewards, failure metadata, and
the text-policy variable that should receive feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textgrad_rl.types import TextualGradient


@dataclass
class ExternalAgentStep:
    observation: str
    action: str
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalAgentEpisode:
    benchmark: str
    task_id: str
    split: str
    seed: int
    success: bool
    reward: float
    invalid_action: bool
    truncated: bool
    steps: list[ExternalAgentStep]
    target_variable: str = "general_agent_policy"
    failure_reason: str = ""


def external_episode_score(episode: ExternalAgentEpisode) -> float:
    return (
        episode.reward
        + 0.5 * (1.0 if episode.success else 0.0)
        - 0.5 * (1.0 if episode.invalid_action else 0.0)
        - 0.25 * (1.0 if episode.truncated else 0.0)
        - 0.002 * len(episode.steps)
    )


def gradients_from_external_episodes(episodes: list[ExternalAgentEpisode]) -> list[TextualGradient]:
    """Convert WebArena/SWE-bench style failures into TextGrad feedback."""

    gradients: list[TextualGradient] = []
    for target in sorted({episode.target_variable for episode in episodes}):
        group = [episode for episode in episodes if episode.target_variable == target]
        failures = [episode for episode in group if not episode.success or episode.invalid_action or episode.truncated]
        if not failures:
            continue
        repeated_actions = sum(has_repeated_actions(episode) for episode in failures)
        evidence = (
            f"{len(failures)}/{len(group)} {target} episodes failed, were invalid, or truncated; "
            f"{repeated_actions} had repeated actions."
        )
        examples = "; ".join(
            f"{episode.benchmark}:{episode.task_id}:{episode.failure_reason or 'unspecified failure'}"
            for episode in failures[:3]
        )
        gradients.append(
            TextualGradient(
                target_variable_name=target,
                failure_mode=f"{target} external benchmark failures",
                evidence_from_trajectory=f"{evidence} Examples: {examples}",
                gradient_text="Improve long-horizon tool-use planning from trajectory feedback.",
                suggested_edit=(
                    "Add a rule: keep a concise task state, verify each tool action against the latest observation, "
                    "avoid repeating failed actions, and checkpoint progress before irreversible steps."
                ),
                confidence=0.7,
                forbidden_shortcuts=["edit tests", "inspect hidden labels", "use hidden benchmark answers"],
            )
        )
    return gradients


def has_repeated_actions(episode: ExternalAgentEpisode) -> bool:
    actions = [step.action for step in episode.steps]
    return len(actions) != len(set(actions))
