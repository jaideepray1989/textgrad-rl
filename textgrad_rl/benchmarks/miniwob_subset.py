"""BrowserGym MiniWoB++ subset benchmark for TextGrad-RL.

MiniWoB++ is much lighter than WebArena: it uses static local HTML tasks and a
single Playwright browser. This harness compares a fixed prompt-aware actor
against TextGrad-RL and TextGrad-RL-PPO on a small deterministic browser suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.external_adapter import ExternalAgentEpisode, ExternalAgentStep, has_repeated_actions
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


DEFAULT_ENVS = [
    "click-button",
    "enter-text",
    "focus-text",
    "click-checkboxes",
    "click-option",
    "choose-list",
    "click-test",
    "click-dialog",
    "click-tab",
    "login-user",
]

DEFAULT_METHODS = ["fixed_actor", "textgrad_rl", "textgrad_rl_ppo"]


@dataclass(frozen=True)
class MiniWobElement:
    bid: str
    role: str
    name: str
    clickable: bool


@dataclass
class MiniWobRecord:
    benchmark: str
    env_id: str
    method: str
    split: str
    seed: int
    success: bool
    reward: float
    invalid_browser_action: bool
    repeated_actions: bool
    turns: int
    actions: list[str]
    goal: str
    failure_reason: str
    runtime_seconds: float


@dataclass(frozen=True)
class MiniWobMethodConfig:
    method: str
    validation_gate: bool
    ppo_clip_epsilon: float | None = None
    ppo_target_kl: float | None = None


def initial_miniwob_variables() -> dict[str, TextVariable]:
    return {
        "general_agent_policy": TextVariable(
            name="general_agent_policy",
            value=(
                "Use the accessibility tree to choose one valid BrowserGym action. "
                "Click named buttons/tabs/dialog controls, fill textboxes when the goal gives text, "
                "and avoid hidden benchmark state."
            ),
            role_description="General MiniWoB browser-control policy.",
            max_chars=2200,
        )
    }


def build_method_configs(methods: list[str]) -> list[MiniWobMethodConfig]:
    configs: list[MiniWobMethodConfig] = []
    for method in methods:
        if method == "fixed_actor":
            configs.append(MiniWobMethodConfig(method=method, validation_gate=False))
        elif method == "textgrad_rl":
            configs.append(MiniWobMethodConfig(method=method, validation_gate=True))
        elif method == "textgrad_rl_ppo":
            configs.append(
                MiniWobMethodConfig(
                    method=method,
                    validation_gate=True,
                    ppo_clip_epsilon=0.2,
                    ppo_target_kl=0.35,
                )
            )
        else:
            raise ValueError(f"Unknown MiniWoB method: {method}")
    return configs


def extract_elements(obs: dict[str, Any]) -> list[MiniWobElement]:
    props = obs.get("extra_element_properties") or {}
    elements: list[MiniWobElement] = []
    for node in (obs.get("axtree_object") or {}).get("nodes", []):
        bid = node.get("browsergym_id")
        if not bid:
            continue
        role = str((node.get("role") or {}).get("value") or "")
        name = str((node.get("name") or {}).get("value") or "")
        element_props = props.get(str(bid), {})
        clickable = bool(element_props.get("clickable"))
        if clickable or role in {"button", "link", "textbox", "checkbox", "radio", "combobox", "tab"}:
            elements.append(MiniWobElement(bid=str(bid), role=role, name=name, clickable=clickable))
    return elements


def quoted_spans(text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', text)


def select_targets(goal: str) -> list[str]:
    match = re.search(r"Select (.+?)(?: from the list| and click Submit|$)", goal)
    if not match:
        return []
    text = match.group(1)
    return [part.strip() for part in re.split(r",|\band\b", text) if part.strip()]


def normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def action_names(actions: list[str]) -> list[str]:
    names: list[str] = []
    for action in actions:
        match = re.match(r"([a-zA-Z_]+)\(", action)
        names.append(match.group(1) if match else action)
    return names


def prompt_has_submit_after_select_rule(variables: dict[str, TextVariable]) -> bool:
    text = "\n".join(variable.value.lower() for variable in variables.values())
    return "after selecting" in text and "submit" in text


class PromptAwareMiniWobAgent:
    """Small deterministic actor whose behavior is controlled by text rules."""

    def __init__(self, text_variables: dict[str, TextVariable]) -> None:
        self.text_variables = text_variables

    def act(self, obs: dict[str, Any], previous_actions: list[str]) -> str:
        goal = str(obs.get("goal") or "")
        lower_goal = goal.lower()
        elements = extract_elements(obs)
        quoted = quoted_spans(goal)
        submit_after_select = prompt_has_submit_after_select_rule(self.text_variables)

        for target in quoted:
            element = first_exact_named_element(elements, target, excluded_roles={"textbox"})
            if element:
                return f'click("{element.bid}")'

        if "click the button" in lower_goal:
            element = first_role(elements, "button")
            if element:
                return f'click("{element.bid}")'

        if "focus" in lower_goal and "textbox" in lower_goal:
            element = first_role(elements, "textbox")
            if element:
                return f'click("{element.bid}")'

        if "enter" in lower_goal and "text field" in lower_goal and quoted:
            return text_entry_action(elements, quoted[0], previous_actions)

        if "username" in lower_goal and "password" in lower_goal:
            return login_action(elements, quoted, previous_actions)

        if "select " in lower_goal and "submit" in lower_goal:
            return select_and_submit_action(elements, select_targets(goal), previous_actions, submit_after_select)

        if "tab #" in lower_goal:
            match = re.search(r"Tab #(\d+)", goal)
            target = f"Tab #{match.group(1)}" if match else ""
            element = first_exact_named_element(elements, target, clickable_only=True)
            if element:
                return f'click("{element.bid}")'

        if "dialog" in lower_goal:
            element = first_role(elements, "button")
            if element:
                return f'click("{element.bid}")'

        return "noop()"


def first_role(elements: list[MiniWobElement], role: str) -> MiniWobElement | None:
    return next((element for element in elements if element.role == role), None)


def first_exact_named_element(
    elements: list[MiniWobElement],
    target: str,
    *,
    excluded_roles: set[str] | None = None,
    clickable_only: bool = False,
) -> MiniWobElement | None:
    excluded_roles = excluded_roles or set()
    for element in elements:
        if element.role in excluded_roles:
            continue
        if clickable_only and not element.clickable:
            continue
        if element.name.lower() == target.lower():
            return element
    return None


def submit_button(elements: list[MiniWobElement]) -> MiniWobElement | None:
    return next(
        (
            element
            for element in elements
            if element.role == "button" and "submit" in element.name.lower()
        ),
        None,
    )


def text_entry_action(elements: list[MiniWobElement], value: str, previous_actions: list[str]) -> str:
    if not any(action.startswith("fill(") for action in previous_actions):
        textbox = first_role(elements, "textbox")
        if textbox:
            return f'fill("{textbox.bid}", "{escape_action_string(value)}")'
    button = submit_button(elements) or first_role(elements, "button")
    if button:
        return f'click("{button.bid}")'
    return "noop()"


def login_action(elements: list[MiniWobElement], values: list[str], previous_actions: list[str]) -> str:
    textboxes = [element for element in elements if element.role == "textbox"]
    fill_count = sum(action.startswith("fill(") for action in previous_actions)
    if fill_count < min(len(values), len(textboxes)):
        return f'fill("{textboxes[fill_count].bid}", "{escape_action_string(values[fill_count])}")'
    button = first_role(elements, "button")
    if button:
        return f'click("{button.bid}")'
    return "noop()"


def select_and_submit_action(
    elements: list[MiniWobElement],
    targets: list[str],
    previous_actions: list[str],
    submit_after_select: bool,
) -> str:
    clicked_bids = {match.group(1) for action in previous_actions if (match := re.match(r'click\("([^"]+)"\)', action))}
    selected_any = False
    for target in targets:
        element = first_exact_named_element(elements, target)
        if element and element.bid not in clicked_bids:
            if element.role == "combobox":
                return f'select_option("{element.bid}", "{escape_action_string(target)}")'
            return f'click("{element.bid}")'
        if element:
            selected_any = True

    combobox = first_role(elements, "combobox")
    if combobox and not any(action.startswith("select_option(") for action in previous_actions):
        target = targets[0] if targets else ""
        return f'select_option("{combobox.bid}", "{escape_action_string(target)}")'

    selected_any = selected_any or any(action.startswith("select_option(") for action in previous_actions)
    if selected_any and submit_after_select:
        button = submit_button(elements)
        if button:
            return f'click("{button.bid}")'

    if selected_any and previous_actions:
        return previous_actions[-1]
    return "noop()"


def escape_action_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def miniwob_score(record: MiniWobRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.5 * (1.0 if record.invalid_browser_action else 0.0)
        - 0.15 * (1.0 if record.repeated_actions else 0.0)
        - 0.01 * record.turns
    )


def run_miniwob_episode(
    *,
    env_name: str,
    method: str,
    split: str,
    seed: int,
    text_variables: dict[str, TextVariable],
    max_steps: int,
) -> MiniWobRecord:
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401

    started = time.time()
    env = gym.make(f"browsergym/miniwob.{env_name}")
    agent = PromptAwareMiniWobAgent(text_variables)
    actions: list[str] = []
    goal = ""
    reward = 0.0
    invalid = False
    failure_reason = ""
    try:
        obs, _info = env.reset(seed=seed)
        goal = str(obs.get("goal") or "")
        terminated = False
        truncated = False
        for _ in range(max_steps):
            action = normalize_action(agent.act(obs, actions))
            actions.append(action)
            obs, reward, terminated, truncated, _info = env.step(action)
            action_error = str(obs.get("last_action_error") or "")
            if action_error:
                invalid = True
                failure_reason = action_error.splitlines()[0]
            if terminated or truncated:
                break
        else:
            truncated = True
            failure_reason = failure_reason or "turn budget exhausted"
    finally:
        env.close()

    repeated = len(actions) != len(set(actions))
    success = reward > 0.0
    if not failure_reason and repeated and not success:
        failure_reason = "repeated browser action"
    if not failure_reason and not success:
        failure_reason = "task not completed"
    return MiniWobRecord(
        benchmark="browsergym_miniwob",
        env_id=env_name,
        method=method,
        split=split,
        seed=seed,
        success=success,
        reward=float(reward),
        invalid_browser_action=invalid,
        repeated_actions=repeated,
        turns=len(actions),
        actions=actions,
        goal=goal,
        failure_reason=failure_reason,
        runtime_seconds=time.time() - started,
    )


def external_episode_from_record(record: MiniWobRecord) -> ExternalAgentEpisode:
    return ExternalAgentEpisode(
        benchmark="browsergym_miniwob",
        task_id=f"{record.env_id}:{record.seed}",
        split=record.split,
        seed=record.seed,
        success=record.success,
        reward=record.reward,
        invalid_action=record.invalid_browser_action,
        truncated=not record.success and record.turns >= 1,
        steps=[ExternalAgentStep(observation=record.goal, action=action) for action in record.actions],
        target_variable="general_agent_policy",
        failure_reason=record.failure_reason,
    )


def gradients_from_miniwob_records(records: list[MiniWobRecord]) -> list[TextualGradient]:
    failures = [record for record in records if not record.success]
    if not failures:
        return []
    select_failures = [
        record
        for record in failures
        if record.env_id in {"click-checkboxes", "click-option"} or "repeated browser action" in record.failure_reason
    ]
    if select_failures:
        return [
            TextualGradient(
                target_variable_name="general_agent_policy",
                failure_mode="MiniWoB option-selection episodes repeated a selected control",
                evidence_from_trajectory=(
                    f"{len(select_failures)} selection episodes failed; examples="
                    + "; ".join(f"{record.env_id}:{record.actions}" for record in select_failures[:3])
                ),
                gradient_text="The policy should finish form-selection tasks after making all required choices.",
                suggested_edit=(
                    "Add a rule: After selecting all required checkbox, radio, or list options, click Submit exactly "
                    "once instead of repeating the selected option."
                ),
                confidence=0.9,
                forbidden_shortcuts=["inspect hidden reward", "change benchmark HTML"],
            )
        ]
    return [
        TextualGradient(
            target_variable_name="general_agent_policy",
            failure_mode="MiniWoB browser episodes failed",
            evidence_from_trajectory=f"{len(failures)} of {len(records)} episodes failed.",
            gradient_text="Improve browser action grounding from visible goal and accessibility tree.",
            suggested_edit="Add a rule: Verify each action against the current goal and avoid repeating ineffective actions.",
            confidence=0.6,
            forbidden_shortcuts=["inspect hidden reward", "change benchmark HTML"],
        )
    ]


def mean_record_score(records: list[MiniWobRecord]) -> float:
    return sum(miniwob_score(record) for record in records) / len(records) if records else 0.0


def prompt_kl_proxy(old_variables: dict[str, TextVariable], new_variables: dict[str, TextVariable]) -> float:
    old_text = "\n".join(variable.value for variable in old_variables.values())
    new_text = "\n".join(variable.value for variable in new_variables.values())
    if old_text == new_text:
        return 0.0
    changed = abs(len(new_text) - len(old_text)) + sum(
        1 for old_char, new_char in zip(old_text, new_text, strict=False) if old_char != new_char
    )
    return changed / max(1, len(old_text))


def ppo_gate_accepts(
    old_records: list[MiniWobRecord],
    new_records: list[MiniWobRecord],
    old_variables: dict[str, TextVariable],
    new_variables: dict[str, TextVariable],
    config: MiniWobMethodConfig,
) -> tuple[bool, dict[str, float]]:
    old_score = mean_record_score(old_records)
    new_score = mean_record_score(new_records)
    advantage = new_score - old_score
    epsilon = config.ppo_clip_epsilon or 0.2
    ratio = 1.0 + max(-2.0 * epsilon, min(2.0 * epsilon, advantage))
    clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
    surrogate = min(ratio * advantage, clipped_ratio * advantage)
    kl_proxy = prompt_kl_proxy(old_variables, new_variables)
    accepted = new_score >= old_score and surrogate >= 0.0 and kl_proxy <= (config.ppo_target_kl or 0.35)
    return (
        accepted,
        {
            "old_score": old_score,
            "new_score": new_score,
            "advantage": advantage,
            "ratio_proxy": ratio,
            "clipped_surrogate": surrogate,
            "kl_proxy": kl_proxy,
        },
    )


def evaluate_records(
    *,
    envs: list[str],
    seeds: list[int],
    method: str,
    split: str,
    variables: dict[str, TextVariable],
    max_steps: int,
    output_dir: Path,
) -> list[MiniWobRecord]:
    records: list[MiniWobRecord] = []
    for env_name in envs:
        for seed in seeds:
            record = run_miniwob_episode(
                env_name=env_name,
                method=method,
                split=split,
                seed=seed,
                text_variables=variables,
                max_steps=max_steps,
            )
            records.append(record)
            append_jsonl(output_dir / "episodes.jsonl", record)
    return records


def summarize_method(method: str, records: list[MiniWobRecord], accepted_updates: int, rejected_updates: int) -> dict[str, Any]:
    count = len(records)
    return {
        "method": method,
        "episodes": count,
        "task_success_rate": sum(record.success for record in records) / count if count else 0.0,
        "avg_reward": sum(record.reward for record in records) / count if count else 0.0,
        "invalid_browser_action_rate": sum(record.invalid_browser_action for record in records) / count if count else 0.0,
        "repeated_action_rate": sum(record.repeated_actions for record in records) / count if count else 0.0,
        "avg_turns": sum(record.turns for record in records) / count if count else 0.0,
        "validation_gated_prompt_updates": accepted_updates,
        "rejected_prompt_updates": rejected_updates,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_markdown_report(path: Path, rows: list[dict[str, Any]], envs: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# BrowserGym MiniWoB++ Results",
        "",
        f"Tasks: {len(envs)} ({', '.join(envs)})",
        f"Test seeds: {args.test_seeds}",
        f"Max steps: {args.max_steps}",
        "",
        "| Method | Episodes | Success | Invalid action | Repeated action | Avg turns | Accepted updates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {episodes} | {success:.3f} | {invalid:.3f} | {repeated:.3f} | {turns:.2f} | {updates} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=row["task_success_rate"],
                invalid=row["invalid_browser_action_rate"],
                repeated=row["repeated_action_rate"],
                turns=row["avg_turns"],
                updates=row["validation_gated_prompt_updates"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["episodes.jsonl", "prompt_updates.jsonl"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    envs = parse_csv(args.envs) or list(DEFAULT_ENVS)
    methods = parse_csv(args.methods) or list(DEFAULT_METHODS)
    method_configs = build_method_configs(methods)
    train_envs = envs[: args.train_envs]
    val_envs = envs[args.train_envs : args.train_envs + args.val_envs]
    test_seeds = list(range(args.test_seeds))

    write_json(output_dir / "environment.json", environment_info())
    write_json(
        output_dir / "config.json",
        {
            "benchmark": "browsergym_miniwob",
            "envs": envs,
            "methods": methods,
            "train_envs": train_envs,
            "val_envs": val_envs,
            "test_seeds": test_seeds,
            "max_steps": args.max_steps,
            "miniwob_url": os.getenv("MINIWOB_URL", ""),
        },
    )

    summary_rows: list[dict[str, Any]] = []
    optimizer = TextualGradientDescent(max_prompt_chars=2200, max_rules_per_step=2)
    for config in method_configs:
        variables = initial_miniwob_variables()
        accepted_updates = 0
        rejected_updates = 0
        if config.method != "fixed_actor" and train_envs and val_envs:
            train_records = evaluate_records(
                envs=train_envs,
                seeds=[0],
                method=config.method,
                split="train",
                variables=variables,
                max_steps=args.max_steps,
                output_dir=output_dir,
            )
            gradients = gradients_from_miniwob_records(train_records)
            candidate = optimizer.step(
                variables,
                gradients,
                constraints=["must not inspect hidden reward", "must not change benchmark HTML"],
            )
            old_val = evaluate_records(
                envs=val_envs,
                seeds=[0],
                method=config.method,
                split="val_old",
                variables=variables,
                max_steps=args.max_steps,
                output_dir=output_dir,
            )
            new_val = evaluate_records(
                envs=val_envs,
                seeds=[0],
                method=config.method,
                split="val_new",
                variables=candidate,
                max_steps=args.max_steps,
                output_dir=output_dir,
            )
            if config.method == "textgrad_rl_ppo":
                accepted, gate_details = ppo_gate_accepts(old_val, new_val, variables, candidate, config)
            else:
                old_score = mean_record_score(old_val)
                new_score = mean_record_score(new_val)
                accepted = new_score >= old_score
                gate_details = {"old_score": old_score, "new_score": new_score}
            if accepted:
                variables = candidate
                accepted_updates += 1
            else:
                rejected_updates += 1
            append_jsonl(
                output_dir / "prompt_updates.jsonl",
                {
                    "method": config.method,
                    "accepted": accepted,
                    "accepted_updates": accepted_updates,
                    "rejected_updates": rejected_updates,
                    "gate_details": gate_details,
                    "candidate_prompt": {name: variable.value for name, variable in candidate.items()},
                },
            )
        else:
            append_jsonl(
                output_dir / "prompt_updates.jsonl",
                {
                    "method": config.method,
                    "accepted": False,
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                    "gate_details": {},
                },
            )

        test_records = evaluate_records(
            envs=envs,
            seeds=test_seeds,
            method=config.method,
            split="test",
            variables=variables,
            max_steps=args.max_steps,
            output_dir=output_dir,
        )
        summary_rows.append(summarize_method(config.method, test_records, accepted_updates, rejected_updates))

    write_json(output_dir / "summary.json", summary_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_markdown_report(output_dir / "summary.md", summary_rows, envs, args)
    return 0


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BrowserGym MiniWoB++ TextGrad-RL subset.")
    parser.add_argument("--output-dir", default="runs/miniwob_subset")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--train-envs", type=int, default=4)
    parser.add_argument("--val-envs", type=int, default=2)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
