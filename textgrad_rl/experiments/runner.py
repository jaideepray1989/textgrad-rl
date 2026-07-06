"""End-to-end TextGrad-RL experiment runner."""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any

from textgrad_rl.agents.llm_actor_agent import LLMActorAgent
from textgrad_rl.agents.prompt_aware_heuristic_agent import PromptAwareHeuristicAgent
from textgrad_rl.critics.heuristic_critic import HeuristicTextualCritic
from textgrad_rl.critics.llm_critic import LLMTrajectoryCritic
from textgrad_rl.envs.mle_repair_env import MLERepairEnv
from textgrad_rl.envs.task_factory import build_task_splits
from textgrad_rl.envs.task_specs import TaskSpec
from textgrad_rl.experiments.report import generate_summary_report
from textgrad_rl.optim.acceptance_gate import AcceptanceGate
from textgrad_rl.optim.scalar_prompt_search import ScalarPromptSearch
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.text_variables import initial_text_variables, monolithic_text_variable
from textgrad_rl.types import Action, TextVariable, Trajectory
from textgrad_rl.utils.json_utils import append_jsonl, to_jsonable, write_json
from textgrad_rl.utils.logging import environment_info
from textgrad_rl.utils.metrics import aggregate_trajectories, metrics_to_row


CONSTRAINTS = [
    "Must not edit tests",
    "Must not edit hidden validation",
    "Must not change thresholds",
    "Must not hardcode outputs",
    "Must not leak labels",
]


def run_experiment(config: dict[str, Any]) -> Path:
    """Run a full experiment and return the output directory."""

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "environment_info.json", environment_info())

    train_tasks, val_tasks, test_tasks = build_task_splits(
        int(config["train_tasks"]),
        int(config["val_tasks"]),
        int(config["test_tasks"]),
        int(config["seed"]),
    )
    write_json(output_dir / "task_specs.json", [task for task in train_tasks + val_tasks + test_tasks])

    method = config["method"]
    text_variables = (
        monolithic_text_variable()
        if method == "monolithic_textgrad"
        else initial_text_variables()
    )
    write_json(output_dir / "initial_text_variables.json", text_variables)

    metrics_rows: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0

    critic = _make_critic(config)
    optimizer = TextualGradientDescent()
    scalar_search = ScalarPromptSearch()

    initial_val = _run_rollouts(val_tasks, text_variables, config, output_dir, "val", 0)
    metrics_rows.append(
        _metrics_row(
            initial_val,
            iteration=0,
            split="val",
            text_variables=text_variables,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            gradients=[],
        )
    )

    for iteration in range(1, int(config["iterations"]) + 1):
        train_trajectories = _run_rollouts(train_tasks, text_variables, config, output_dir, "train", iteration)
        gradients = []
        if method not in {"fixed_prompt", "scalar_prompt_search"}:
            for trajectory in train_trajectories:
                gradients.extend(critic.critique(trajectory, text_variables))
        write_json(output_dir / "gradients" / f"iteration_{iteration:03d}.json", gradients)

        candidate_variables = text_variables
        if method == "scalar_prompt_search":
            candidate_variables = scalar_search.propose(text_variables, int(config["seed"]), iteration)
        elif method not in {"fixed_prompt"}:
            candidate_variables = optimizer.step(text_variables, gradients, CONSTRAINTS)

        if method == "fixed_prompt":
            accepted = False
            details = {"reason": "fixed_prompt baseline keeps initial variables"}
        elif method == "no_acceptance_gate":
            text_variables = candidate_variables
            accepted = True
            details = {"reason": "no_acceptance_gate applies candidate directly"}
            accepted_count += 1
            append_jsonl(output_dir / "accepted_updates.jsonl", {"iteration": iteration, **details})
        else:
            gate = AcceptanceGate(
                max_steps=int(config["max_steps"]),
                command_timeout_sec=int(config.get("command_timeout_sec", 10)),
                agent_factory=_make_agent_factory(config),
            )
            text_variables, accepted, details = gate.accept_or_reject(
                text_variables,
                candidate_variables,
                val_tasks,
                tolerance=0.0,
            )
            if accepted:
                accepted_count += 1
                append_jsonl(output_dir / "accepted_updates.jsonl", {"iteration": iteration, **details})
            else:
                rejected_count += 1
                append_jsonl(output_dir / "rejected_updates.jsonl", {"iteration": iteration, **details})

        val_trajectories = _run_rollouts(val_tasks, text_variables, config, output_dir, "val", iteration)
        metrics_rows.append(
            _metrics_row(
                train_trajectories,
                iteration=iteration,
                split="train",
                text_variables=text_variables,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                gradients=gradients,
            )
        )
        metrics_rows.append(
            _metrics_row(
                val_trajectories,
                iteration=iteration,
                split="val",
                text_variables=text_variables,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                gradients=gradients,
            )
        )
        write_json(output_dir / "text_variables" / f"iteration_{iteration:03d}.json", text_variables)
        _write_metrics(output_dir / "metrics.csv", metrics_rows)

    test_trajectories = _run_rollouts(test_tasks, text_variables, config, output_dir, "test", int(config["iterations"]))
    metrics_rows.append(
        _metrics_row(
            test_trajectories,
            iteration=int(config["iterations"]),
            split="test",
            text_variables=text_variables,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            gradients=[],
        )
    )
    write_json(output_dir / "final_text_variables.json", text_variables)
    _write_metrics(output_dir / "metrics.csv", metrics_rows)
    generate_summary_report(output_dir)
    return output_dir


def _run_rollouts(
    task_specs: list[TaskSpec],
    text_variables: dict[str, TextVariable],
    config: dict[str, Any],
    output_dir: Path,
    split: str,
    iteration: int,
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    agent_factory = _make_agent_factory(config)
    with tempfile.TemporaryDirectory(prefix=f"textgrad_rl_{split}_") as tmp:
        base = Path(tmp)
        for task_spec in task_specs:
            agent = agent_factory()
            env = MLERepairEnv(
                task_spec,
                base,
                max_steps=int(config["max_steps"]),
                command_timeout_sec=int(config.get("command_timeout_sec", 10)),
            )
            trajectory = _run_episode(env, agent, text_variables)
            trajectories.append(trajectory)
            _save_trajectory(output_dir, split, iteration, task_spec, trajectory, env.render_trajectory())
    return trajectories


def _run_episode(env: MLERepairEnv, agent: Any, text_variables: dict[str, TextVariable]) -> Trajectory:
    observation = env.reset()
    done = False
    while not done and observation.remaining_steps > 0:
        action = agent.act(observation, text_variables)
        observation, _, done, _ = env.step(action)
    if not done:
        env.step(Action(type="submit_patch", reason="automatic final submit at rollout budget"))
    return env.get_trajectory()


def _save_trajectory(
    output_dir: Path,
    split: str,
    iteration: int,
    task_spec: TaskSpec,
    trajectory: Trajectory,
    report: str,
) -> None:
    if split == "test":
        traj_path = output_dir / "trajectories" / "test" / f"{task_spec.task_id}.json"
        report_path = output_dir / "trajectory_reports" / f"test_{task_spec.task_id}.txt"
    else:
        traj_path = (
            output_dir
            / "trajectories"
            / split
            / f"iteration_{iteration:03d}"
            / f"{task_spec.task_id}.json"
        )
        report_path = output_dir / "trajectory_reports" / f"{split}_iteration_{iteration:03d}_{task_spec.task_id}.txt"
    write_json(traj_path, trajectory)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def _make_agent_factory(config: dict[str, Any]):
    if config["agent"] == "local_llm":
        return lambda: LLMActorAgent(
            base_url=config.get("local_llm_base_url"),
            model=config.get("local_llm_model"),
        )
    return PromptAwareHeuristicAgent


def _make_critic(config: dict[str, Any]):
    if config["critic"] == "local_llm":
        return LLMTrajectoryCritic(
            base_url=config.get("local_llm_base_url"),
            model=config.get("local_llm_model"),
        )
    return HeuristicTextualCritic()


def _metrics_row(
    trajectories: list[Trajectory],
    iteration: int,
    split: str,
    text_variables: dict[str, TextVariable],
    accepted_count: int,
    rejected_count: int,
    gradients: list[Any],
) -> dict[str, Any]:
    prompt_length_by_variable = {name: len(var.value) for name, var in text_variables.items()}
    metrics = aggregate_trajectories(
        trajectories,
        iteration=iteration,
        split=split,
        accepted_updates=accepted_count,
        rejected_updates=rejected_count,
        prompt_length_total=sum(prompt_length_by_variable.values()),
    )
    extra = {
        "hidden_validation_pass_rate": metrics.success_rate,
        "forbidden_edit_count": _forbidden_edit_count(trajectories),
        "training_command_count": _command_count(trajectories, "python train.py"),
        "eval_command_count": _command_count(trajectories, "python eval.py"),
        "prompt_length_by_variable": prompt_length_by_variable,
        "number_of_gradients": len(gradients),
        "gradients_by_variable": _gradients_by_variable(gradients),
    }
    return metrics_to_row(metrics, extra)


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _forbidden_edit_count(trajectories: list[Trajectory]) -> int:
    count = 0
    for trajectory in trajectories:
        for step in trajectory.steps:
            reason = str(step.info.get("invalid_reason", "")).lower()
            if "forbidden" in reason or "test" in reason:
                count += 1
    return count


def _command_count(trajectories: list[Trajectory], command: str) -> int:
    return sum(
        1
        for trajectory in trajectories
        for step in trajectory.steps
        if (step.action.command or "") == command
    )


def _gradients_by_variable(gradients: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gradient in gradients:
        name = getattr(gradient, "target_variable_name", None)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts

