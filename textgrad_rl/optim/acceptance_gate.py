"""Validation gate for accepting or rejecting prompt updates."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from textgrad_rl.agents.prompt_aware_heuristic_agent import PromptAwareHeuristicAgent
from textgrad_rl.envs.mle_repair_env import MLERepairEnv
from textgrad_rl.envs.task_specs import TaskSpec
from textgrad_rl.text_variables import summarize_text_variables
from textgrad_rl.types import Action, ExperimentMetrics, TextVariable, Trajectory
from textgrad_rl.utils.metrics import aggregate_trajectories


AgentFactory = Callable[[], Any]
EnvFactory = Callable[[TaskSpec, Path], MLERepairEnv]


class AcceptanceGate:
    """Evaluate candidate text variables on validation tasks before accepting them."""

    def __init__(
        self,
        max_steps: int = 20,
        command_timeout_sec: int = 10,
        agent_factory: AgentFactory | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.command_timeout_sec = command_timeout_sec
        self.agent_factory = agent_factory or PromptAwareHeuristicAgent
        self.env_factory = env_factory or self._default_env_factory

    def evaluate(
        self,
        text_variables: dict[str, TextVariable],
        task_specs: list[TaskSpec],
        agent_factory: Callable | None = None,
        env_factory: Callable | None = None,
    ) -> ExperimentMetrics:
        trajectories = self.evaluate_with_trajectories(text_variables, task_specs, agent_factory, env_factory)
        prompt_length = sum(len(var.value) for var in text_variables.values())
        return aggregate_trajectories(trajectories, 0, "val", prompt_length_total=prompt_length)

    def evaluate_with_trajectories(
        self,
        text_variables: dict[str, TextVariable],
        task_specs: list[TaskSpec],
        agent_factory: Callable | None = None,
        env_factory: Callable | None = None,
    ) -> list[Trajectory]:
        make_agent = agent_factory or self.agent_factory
        make_env = env_factory or self.env_factory
        trajectories: list[Trajectory] = []
        with tempfile.TemporaryDirectory(prefix="textgrad_rl_gate_") as tmp:
            base = Path(tmp)
            for task_spec in task_specs:
                agent = make_agent()
                env = make_env(task_spec, base)
                trajectories.append(_run_episode(env, agent, text_variables))
        return trajectories

    def accept_or_reject(
        self,
        old_variables: dict[str, TextVariable],
        new_variables: dict[str, TextVariable],
        val_tasks: list[TaskSpec],
        tolerance: float = 0.0,
    ) -> tuple[dict[str, TextVariable], bool, dict[str, Any]]:
        old_trajectories = self.evaluate_with_trajectories(old_variables, val_tasks)
        new_trajectories = self.evaluate_with_trajectories(new_variables, val_tasks)
        old_metrics = aggregate_trajectories(
            old_trajectories,
            0,
            "val_old",
            prompt_length_total=sum(len(v.value) for v in old_variables.values()),
        )
        new_metrics = aggregate_trajectories(
            new_trajectories,
            0,
            "val_new",
            prompt_length_total=sum(len(v.value) for v in new_variables.values()),
        )
        old_score = self._score(old_metrics)
        new_score = self._score(new_metrics)
        accepted = new_score + tolerance >= old_score
        details = {
            "old_score": old_score,
            "new_score": new_score,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "old_text_variables": summarize_text_variables(old_variables),
            "new_text_variables": summarize_text_variables(new_variables),
        }
        return (new_variables if accepted else old_variables), accepted, details

    def _score(self, metrics: ExperimentMetrics) -> float:
        return (
            10.0 * metrics.success_rate
            + metrics.average_reward
            + 1.5 * metrics.test_pass_rate
            - 2.0 * metrics.invalid_action_rate
            - 0.01 * metrics.runtime_seconds
        )

    def _default_env_factory(self, task_spec: TaskSpec, base: Path) -> MLERepairEnv:
        return MLERepairEnv(
            task_spec,
            base,
            max_steps=self.max_steps,
            command_timeout_sec=self.command_timeout_sec,
        )


def _run_episode(env: MLERepairEnv, agent: Any, text_variables: dict[str, TextVariable]) -> Trajectory:
    observation = env.reset()
    done = False
    while not done and observation.remaining_steps > 0:
        action = agent.act(observation, text_variables)
        observation, _, done, _ = env.step(action)
    if not done:
        env.step(Action(type="submit_patch", reason="automatic final submit at evaluation budget"))
    return env.get_trajectory()

