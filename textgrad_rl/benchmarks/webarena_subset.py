"""Small-subset WebArena benchmark harness.

This module keeps WebArena integration honest: it selects real WebArena tasks,
checks whether the required browser/web-site infrastructure exists, and writes a
canonical experiment matrix for fixed, TextGrad-RL, and TextGrad-RL-PPO methods.
When the self-hosted WebArena stack is not available, it produces an explicit
blocked run artifact instead of fabricating task scores.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.external_adapter import (
    ExternalAgentEpisode,
    ExternalAgentStep,
    external_episode_score,
    gradients_from_external_episodes,
    has_repeated_actions,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


OFFICIAL_WEBARENA_ENV_VARS = [
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "MAP",
    "WIKIPEDIA",
    "HOMEPAGE",
]

BROWSERGYM_WEBARENA_ENV_VARS = [
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_MAP",
    "WA_WIKIPEDIA",
    "WA_HOMEPAGE",
]

DEFAULT_METHODS = ["fixed_actor", "textgrad_rl", "textgrad_rl_ppo"]
DEFAULT_REQUIRED_IMPORTS = ["playwright", "gymnasium", "beartype", "tiktoken"]


@dataclass(frozen=True)
class WebArenaTask:
    task_id: str
    sites: list[str]
    intent: str
    start_url: str
    require_login: bool
    storage_state: str
    intent_template_id: int | None
    eval_types: list[str]


@dataclass(frozen=True)
class WebArenaMethodConfig:
    method: str
    actor_policy: str
    validation_gate: bool
    ppo_clip_epsilon: float | None = None
    ppo_target_kl: float | None = None


@dataclass
class WebArenaPreflight:
    ok: bool
    backend: str
    webarena_root: str
    task_config: str
    selected_task_count: int
    missing_commands: list[str]
    missing_imports: list[str]
    missing_env_vars: list[str]
    missing_generated_configs: list[str]
    missing_auth_files: list[str]
    notes: list[str]


CommandResolver = Callable[[str], str | None]
ImportChecker = Callable[[str], bool]


class BrowserActionModel:
    """OpenAI-compatible chat client for WebArena action generation."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        timeout: int,
        max_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = os.getenv("TEXTGRAD_RL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a browser-control agent. Return exactly one WebArena action line "
                        "and no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
        return str(data["choices"][0]["message"]["content"]).strip()


def load_raw_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected WebArena raw task list at {path}, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def task_from_raw(raw: Mapping[str, Any]) -> WebArenaTask:
    task_id = raw.get("task_id")
    eval_block = raw.get("eval", {})
    eval_types = eval_block.get("eval_types", []) if isinstance(eval_block, dict) else []
    storage_state = raw.get("storage_state")
    return WebArenaTask(
        task_id=str(task_id),
        sites=[str(site) for site in raw.get("sites", [])],
        intent=str(raw.get("intent", "")),
        start_url=str(raw.get("start_url", "")),
        require_login=bool(raw.get("require_login", False)),
        storage_state=storage_state if isinstance(storage_state, str) else "",
        intent_template_id=raw.get("intent_template_id") if isinstance(raw.get("intent_template_id"), int) else None,
        eval_types=[str(item) for item in eval_types],
    )


def select_task_subset(
    raw_tasks: Iterable[Mapping[str, Any]],
    task_count: int,
    sites: set[str] | None = None,
) -> list[WebArenaTask]:
    """Select a deterministic site-balanced WebArena subset."""

    buckets: dict[str, deque[WebArenaTask]] = defaultdict(deque)
    for raw in raw_tasks:
        task = task_from_raw(raw)
        if sites and not any(site in sites for site in task.sites):
            continue
        key = "+".join(task.sites) if task.sites else "unknown"
        buckets[key].append(task)

    selected: list[WebArenaTask] = []
    keys = sorted(buckets)
    while len(selected) < task_count and any(buckets.values()):
        for key in keys:
            if not buckets[key]:
                continue
            selected.append(buckets[key].popleft())
            if len(selected) >= task_count:
                break
    return selected


def build_method_configs(methods: Iterable[str]) -> list[WebArenaMethodConfig]:
    configs: list[WebArenaMethodConfig] = []
    for method in methods:
        if method == "fixed_actor":
            configs.append(WebArenaMethodConfig(method, actor_policy="frozen_prompt", validation_gate=False))
        elif method == "textgrad_rl":
            configs.append(WebArenaMethodConfig(method, actor_policy="textgrad_policy_iteration", validation_gate=True))
        elif method == "textgrad_rl_ppo":
            configs.append(
                WebArenaMethodConfig(
                    method,
                    actor_policy="textgrad_policy_iteration_with_ppo_gate",
                    validation_gate=True,
                    ppo_clip_epsilon=0.2,
                    ppo_target_kl=0.2,
                )
            )
        else:
            raise ValueError(f"Unknown WebArena method: {method}")
    return configs


def default_import_checker(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_webarena_preflight(
    *,
    backend: str,
    webarena_root: Path,
    task_config: Path,
    selected_tasks: list[WebArenaTask],
    env: Mapping[str, str] | None = None,
    command_resolver: CommandResolver | None = None,
    import_checker: ImportChecker | None = None,
) -> WebArenaPreflight:
    env = env or os.environ
    command_resolver = command_resolver or shutil.which
    import_checker = import_checker or default_import_checker
    missing_commands: list[str] = []
    missing_imports: list[str] = []
    missing_env_vars: list[str] = []
    missing_generated_configs: list[str] = []
    missing_auth_files: list[str] = []
    notes: list[str] = []

    official_env_missing = [name for name in OFFICIAL_WEBARENA_ENV_VARS if not env.get(name)]
    browsergym_env_missing = [name for name in BROWSERGYM_WEBARENA_ENV_VARS if not env.get(name)]
    if backend == "official":
        missing_env_vars.extend(official_env_missing)
        if official_env_missing and command_resolver("docker") is None:
            missing_commands.append("docker")
            notes.append("Docker is required to self-host WebArena locally when remote site URLs are not set.")
    elif backend == "browsergym":
        missing_env_vars.extend(browsergym_env_missing)
        if browsergym_env_missing and command_resolver("docker") is None:
            missing_commands.append("docker")
            notes.append("Docker is required to self-host WebArena locally when remote BrowserGym URLs are not set.")
        missing_imports.extend(name for name in ["browsergym", "browsergym.webarena"] if not import_checker(name))
    else:
        raise ValueError(f"Unsupported WebArena backend: {backend}")

    missing_imports.extend(name for name in DEFAULT_REQUIRED_IMPORTS if not import_checker(name))
    missing_imports = sorted(set(missing_imports))

    if not webarena_root.exists():
        notes.append(f"WebArena root does not exist: {webarena_root}")
    if not task_config.exists():
        notes.append(f"Task config does not exist: {task_config}")

    for task in selected_tasks:
        generated = webarena_root / "config_files" / f"{task.task_id}.json"
        if not generated.exists():
            missing_generated_configs.append(str(generated))
        if task.require_login and task.storage_state:
            auth_file = webarena_root / task.storage_state
            if not auth_file.exists():
                missing_auth_files.append(str(auth_file))

    missing_generated_configs = sorted(set(missing_generated_configs))
    missing_auth_files = sorted(set(missing_auth_files))
    ok = not (
        missing_commands
        or missing_imports
        or missing_env_vars
        or missing_generated_configs
        or missing_auth_files
        or not webarena_root.exists()
        or not task_config.exists()
    )
    if ok:
        notes.append("Preflight passed; WebArena execution can be launched with this subset.")
    else:
        notes.append("Preflight failed; no task scores were produced.")

    return WebArenaPreflight(
        ok=ok,
        backend=backend,
        webarena_root=str(webarena_root),
        task_config=str(task_config),
        selected_task_count=len(selected_tasks),
        missing_commands=missing_commands,
        missing_imports=missing_imports,
        missing_env_vars=sorted(set(missing_env_vars)),
        missing_generated_configs=missing_generated_configs,
        missing_auth_files=missing_auth_files,
        notes=notes,
    )


def task_site_distribution(tasks: list[WebArenaTask]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts["+".join(task.sites) if task.sites else "unknown"] += 1
    return dict(sorted(counts.items()))


def task_eval_distribution(tasks: list[WebArenaTask]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        for eval_type in task.eval_types or ["unknown"]:
            counts[eval_type] += 1
    return dict(sorted(counts.items()))


def episode_to_log_record(
    *,
    method: str,
    task: WebArenaTask,
    status: str,
    episode: ExternalAgentEpisode | None = None,
) -> dict[str, Any]:
    repeated = has_repeated_actions(episode) if episode is not None else None
    return {
        "benchmark": "webarena",
        "suite": "small_subset",
        "method": method,
        "task_id": task.task_id,
        "sites": task.sites,
        "intent": task.intent,
        "status": status,
        "success": episode.success if episode is not None else None,
        "invalid_browser_action": episode.invalid_action if episode is not None else None,
        "repeated_actions": repeated,
        "turns": len(episode.steps) if episode is not None else None,
        "reward": episode.reward if episode is not None else None,
        "failure_reason": episode.failure_reason if episode is not None else "",
    }


def initial_browser_variables() -> dict[str, TextVariable]:
    return {
        "general_agent_policy": TextVariable(
            name="general_agent_policy",
            value=(
                "Solve the user request by reading the accessibility tree, choosing one valid WebArena "
                "browser action at a time, and stopping with the final answer only when the task is complete. "
                "Track progress, avoid repeating failed actions, and verify element IDs before clicking or typing."
            ),
            role_description="General WebArena browser-control policy.",
            max_chars=3000,
        )
    }


def build_browser_prompt(
    *,
    task: WebArenaTask,
    observation: str,
    text_variables: dict[str, TextVariable],
    previous_actions: list[str],
) -> str:
    variables = "\n\n".join(
        f"{variable.name} ({variable.role_description}):\n{variable.clipped_value()}"
        for variable in text_variables.values()
    )
    previous = "\n".join(previous_actions[-8:]) if previous_actions else "<none>"
    return (
        f"TEXT VARIABLES:\n{variables}\n\n"
        f"TASK:\n{task.intent}\n\n"
        "ACTION SPACE:\n"
        "- click [id]\n"
        "- type [id] [text]\n"
        "- hover [id]\n"
        "- scroll [up] or scroll [down]\n"
        "- press [key]\n"
        "- goto [url]\n"
        "- stop [answer]\n"
        "- none\n\n"
        f"PREVIOUS ACTIONS:\n{previous}\n\n"
        f"ACCESSIBILITY TREE OBSERVATION:\n{observation[-6000:]}\n\n"
        "Return exactly one action line from the action space."
    )


def normalize_official_action(raw_output: str) -> tuple[str, bool]:
    text = raw_output.strip()
    if not text:
        return "none", True
    action_line = text.splitlines()[0].strip()
    for line in text.splitlines():
        candidate = re.sub(r"^\s*(action|next action)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
        if re.match(r"^(click|type|hover|scroll|press|goto|stop|none)\b", candidate, flags=re.IGNORECASE):
            action_line = candidate
            break
    action_line = action_line.strip("` ")
    patterns = [
        r"^click\s+\[[^\]]+\]$",
        r"^type\s+\[[^\]]+\]\s+\[[\s\S]*\]$",
        r"^hover\s+\[[^\]]+\]$",
        r"^scroll\s+\[(up|down)\]$",
        r"^press\s+\[[^\]]+\]$",
        r"^goto\s+\[[^\]]+\]$",
        r"^stop\s+\[[\s\S]*\]$",
        r"^none$",
    ]
    if any(re.match(pattern, action_line, flags=re.IGNORECASE) for pattern in patterns):
        return action_line, False
    match = re.search(
        r"(click\s+\[[^\]]+\]|type\s+\[[^\]]+\]\s+\[[\s\S]*?\]|hover\s+\[[^\]]+\]|"
        r"scroll\s+\[(?:up|down)\]|press\s+\[[^\]]+\]|goto\s+\[[^\]]+\]|stop\s+\[[\s\S]*?\]|none)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(), False
    return "none", True


def import_official_webarena(webarena_root: Path) -> dict[str, Any]:
    root = str(webarena_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from browser_env import ScriptBrowserEnv, create_id_based_action, create_stop_action  # type: ignore
    from evaluation_harness import evaluator_router  # type: ignore

    return {
        "ScriptBrowserEnv": ScriptBrowserEnv,
        "create_id_based_action": create_id_based_action,
        "create_stop_action": create_stop_action,
        "evaluator_router": evaluator_router,
    }


def run_official_webarena_episode(
    *,
    task: WebArenaTask,
    split: str,
    seed: int,
    webarena_root: Path,
    model: BrowserActionModel,
    text_variables: dict[str, TextVariable],
    max_steps: int,
    render: bool = False,
) -> ExternalAgentEpisode:
    api = import_official_webarena(webarena_root)
    config_file = webarena_root / "config_files" / f"{task.task_id}.json"
    env = api["ScriptBrowserEnv"](
        headless=not render,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
    )
    steps: list[ExternalAgentStep] = []
    invalid_action = False
    truncated = False
    reward = 0.0
    failure_reason = ""
    trajectory: list[Any] = []
    previous_actions: list[str] = []
    try:
        obs, info = env.reset(options={"config_file": str(config_file)})
        trajectory.append({"observation": obs, "info": info})
        terminated = False
        for _turn in range(max_steps):
            observation_text = str(obs.get("text", ""))
            prompt = build_browser_prompt(
                task=task,
                observation=observation_text,
                text_variables=text_variables,
                previous_actions=previous_actions,
            )
            try:
                raw_output = model.complete(prompt)
            except Exception as exc:
                raw_output = ""
                action_text = "none"
                parse_invalid = True
                failure_reason = f"model_error: {exc}"
            else:
                action_text, parse_invalid = normalize_official_action(raw_output)
            invalid_action = invalid_action or parse_invalid
            previous_actions.append(action_text)
            try:
                action = api["create_id_based_action"](action_text)
                action["raw_prediction"] = raw_output
                obs, step_reward, terminated, step_truncated, info = env.step(action)
            except Exception as exc:
                invalid_action = True
                failure_reason = failure_reason or f"browser_action_error: {exc}"
                steps.append(
                    ExternalAgentStep(
                        observation=observation_text[-2000:],
                        action=action_text,
                        reward=0.0,
                        info={"raw_output": raw_output, "invalid": True, "error": str(exc)},
                    )
                )
                break
            trajectory.append(action)
            trajectory.append({"observation": obs, "info": info})
            reward = float(step_reward or 0.0)
            truncated = bool(step_truncated)
            steps.append(
                ExternalAgentStep(
                    observation=observation_text[-2000:],
                    action=action_text,
                    reward=reward,
                    info={"raw_output": raw_output, "invalid": parse_invalid},
                )
            )
            if terminated or truncated or action_text.lower().startswith("stop "):
                break
        else:
            truncated = True

        if not previous_actions or not previous_actions[-1].lower().startswith("stop "):
            trajectory.append(api["create_stop_action"](""))
        evaluator = api["evaluator_router"](str(config_file))
        reward = float(evaluator(trajectory=trajectory, config_file=str(config_file), page=env.page, client=env.get_page_client(env.page)))
    finally:
        env.close()

    repeated = len(previous_actions) != len(set(previous_actions))
    if not failure_reason and invalid_action:
        failure_reason = "invalid browser action"
    if not failure_reason and repeated:
        failure_reason = "repeated browser action"
    if not failure_reason and truncated:
        failure_reason = "turn budget exhausted"
    if not failure_reason and reward < 1.0:
        failure_reason = "task not completed"
    return ExternalAgentEpisode(
        benchmark="webarena",
        task_id=task.task_id,
        split=split,
        seed=seed,
        success=reward >= 1.0,
        reward=reward,
        invalid_action=invalid_action,
        truncated=truncated,
        steps=steps,
        target_variable="general_agent_policy",
        failure_reason=failure_reason,
    )


def mean_episode_score(episodes: list[ExternalAgentEpisode]) -> float:
    if not episodes:
        return 0.0
    return sum(external_episode_score(episode) for episode in episodes) / len(episodes)


def prompt_kl_proxy(old_variables: dict[str, TextVariable], new_variables: dict[str, TextVariable]) -> float:
    old_text = "\n".join(variable.value for variable in old_variables.values())
    new_text = "\n".join(variable.value for variable in new_variables.values())
    if old_text == new_text:
        return 0.0
    changed_chars = abs(len(new_text) - len(old_text)) + sum(
        1 for old_char, new_char in zip(old_text, new_text, strict=False) if old_char != new_char
    )
    return changed_chars / max(1, len(old_text))


def ppo_accepts_prompt_update(
    *,
    old_score: float,
    new_score: float,
    old_variables: dict[str, TextVariable],
    new_variables: dict[str, TextVariable],
    clip_epsilon: float,
    target_kl: float,
) -> tuple[bool, dict[str, float]]:
    advantage = new_score - old_score
    ratio = 1.0 + max(-clip_epsilon * 2.0, min(clip_epsilon * 2.0, advantage))
    clipped_ratio = max(1.0 - clip_epsilon, min(1.0 + clip_epsilon, ratio))
    surrogate = min(ratio * advantage, clipped_ratio * advantage)
    kl_proxy = prompt_kl_proxy(old_variables, new_variables)
    accepted = new_score >= old_score and surrogate >= 0.0 and kl_proxy <= target_kl
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


def run_episodes_for_tasks(
    *,
    tasks: list[WebArenaTask],
    split: str,
    method: str,
    webarena_root: Path,
    model: BrowserActionModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    output_dir: Path,
    seed_offset: int,
) -> list[ExternalAgentEpisode]:
    episodes: list[ExternalAgentEpisode] = []
    for index, task in enumerate(tasks):
        started = time.time()
        episode = run_official_webarena_episode(
            task=task,
            split=split,
            seed=seed_offset + index,
            webarena_root=webarena_root,
            model=model,
            text_variables=variables,
            max_steps=max_steps,
        )
        episodes.append(episode)
        record = episode_to_log_record(method=method, task=task, status="completed", episode=episode)
        record["split"] = split
        record["runtime_seconds"] = time.time() - started
        append_jsonl(output_dir / "episodes.jsonl", record)
    return episodes


def run_official_method_matrix(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    selected_tasks: list[WebArenaTask],
    method_configs: list[WebArenaMethodConfig],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    updates_path = output_dir / "prompt_updates.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    if updates_path.exists():
        updates_path.unlink()

    webarena_root = Path(args.webarena_root)
    model = BrowserActionModel(
        base_url=args.llm_base_url,
        model=args.model,
        temperature=args.temperature,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
    )
    train_count = min(args.train_tasks, len(selected_tasks))
    val_count = min(args.val_tasks, max(0, len(selected_tasks) - train_count))
    train_tasks = selected_tasks[:train_count]
    val_tasks = selected_tasks[train_count : train_count + val_count]

    summary_rows: list[dict[str, Any]] = []
    optimizer = TextualGradientDescent(max_prompt_chars=3000, max_rules_per_step=3)
    for config in method_configs:
        variables = initial_browser_variables()
        accepted_updates = 0
        rejected_updates = 0
        if config.method != "fixed_actor" and train_tasks and val_tasks:
            train_episodes = run_episodes_for_tasks(
                tasks=train_tasks,
                split="train",
                method=config.method,
                webarena_root=webarena_root,
                model=model,
                variables=variables,
                max_steps=args.max_steps,
                output_dir=output_dir,
                seed_offset=0,
            )
            gradients = gradients_from_external_episodes(train_episodes)
            candidate = optimizer.step(
                variables,
                gradients,
                constraints=["must not inspect hidden benchmark answers", "must not change WebArena task files"],
            )
            old_val = run_episodes_for_tasks(
                tasks=val_tasks,
                split="val_old",
                method=config.method,
                webarena_root=webarena_root,
                model=model,
                variables=variables,
                max_steps=args.max_steps,
                output_dir=output_dir,
                seed_offset=1000,
            )
            new_val = run_episodes_for_tasks(
                tasks=val_tasks,
                split="val_new",
                method=config.method,
                webarena_root=webarena_root,
                model=model,
                variables=candidate,
                max_steps=args.max_steps,
                output_dir=output_dir,
                seed_offset=2000,
            )
            old_score = mean_episode_score(old_val)
            new_score = mean_episode_score(new_val)
            if config.method == "textgrad_rl_ppo":
                accepted, gate_details = ppo_accepts_prompt_update(
                    old_score=old_score,
                    new_score=new_score,
                    old_variables=variables,
                    new_variables=candidate,
                    clip_epsilon=config.ppo_clip_epsilon or 0.2,
                    target_kl=config.ppo_target_kl or 0.2,
                )
            else:
                accepted = new_score >= old_score
                gate_details = {"old_score": old_score, "new_score": new_score}
            if accepted:
                variables = candidate
                accepted_updates += 1
            else:
                rejected_updates += 1
            append_jsonl(
                updates_path,
                {
                    "method": config.method,
                    "status": "completed",
                    "validation_gate": config.validation_gate,
                    "validation_gated_prompt_updates": accepted_updates,
                    "accepted_updates": accepted_updates,
                    "rejected_updates": rejected_updates,
                    "gate_details": gate_details,
                },
            )
        else:
            append_jsonl(
                updates_path,
                {
                    "method": config.method,
                    "status": "completed",
                    "validation_gate": config.validation_gate,
                    "validation_gated_prompt_updates": 0,
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                },
            )

        test_episodes = run_episodes_for_tasks(
            tasks=selected_tasks,
            split="test",
            method=config.method,
            webarena_root=webarena_root,
            model=model,
            variables=variables,
            max_steps=args.max_steps,
            output_dir=output_dir,
            seed_offset=3000,
        )
        summary_rows.append(summarize_completed_method(config.method, test_episodes, accepted_updates))

    write_completed_summary_csv(output_dir / "summary.csv", summary_rows)
    write_json(output_dir / "completed_summary.json", summary_rows)


def summarize_completed_method(
    method: str,
    episodes: list[ExternalAgentEpisode],
    accepted_updates: int,
) -> dict[str, Any]:
    n = len(episodes)
    return {
        "method": method,
        "status": "completed",
        "tasks_planned": n,
        "task_success_rate": sum(episode.success for episode in episodes) / n if n else 0.0,
        "invalid_browser_action_rate": sum(episode.invalid_action for episode in episodes) / n if n else 0.0,
        "repeated_action_rate": sum(has_repeated_actions(episode) for episode in episodes) / n if n else 0.0,
        "avg_turns": sum(len(episode.steps) for episode in episodes) / n if n else 0.0,
        "validation_gated_prompt_updates": accepted_updates,
    }


def write_completed_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "status",
                "tasks_planned",
                "task_success_rate",
                "invalid_browser_action_rate",
                "repeated_action_rate",
                "avg_turns",
                "validation_gated_prompt_updates",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_common_artifacts(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    selected_tasks: list[WebArenaTask],
    method_configs: list[WebArenaMethodConfig],
    preflight: WebArenaPreflight,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", environment_info())
    write_json(
        output_dir / "config.json",
        {
            "task_count": args.task_count,
            "backend": args.backend,
            "llm_base_url": args.llm_base_url,
            "model": args.model,
            "temperature": args.temperature,
            "llm_max_tokens": args.llm_max_tokens,
            "max_steps": args.max_steps,
            "train_tasks": args.train_tasks,
            "val_tasks": args.val_tasks,
            "methods": [config.method for config in method_configs],
        },
    )
    write_json(output_dir / "task_subset.json", selected_tasks)
    write_json(
        output_dir / "task_subset_summary.json",
        {
            "task_count": len(selected_tasks),
            "site_distribution": task_site_distribution(selected_tasks),
            "eval_distribution": task_eval_distribution(selected_tasks),
        },
    )
    write_json(output_dir / "method_configs.json", method_configs)
    write_json(output_dir / "preflight.json", preflight)


def write_blocked_artifacts(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    selected_tasks: list[WebArenaTask],
    method_configs: list[WebArenaMethodConfig],
    preflight: WebArenaPreflight,
) -> None:
    write_common_artifacts(
        output_dir,
        args=args,
        selected_tasks=selected_tasks,
        method_configs=method_configs,
        preflight=preflight,
    )

    episodes_path = output_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    for config in method_configs:
        for task in selected_tasks:
            append_jsonl(
                episodes_path,
                episode_to_log_record(method=config.method, task=task, status="not_run_preflight_failed"),
            )

    updates_path = output_dir / "prompt_updates.jsonl"
    if updates_path.exists():
        updates_path.unlink()
    for config in method_configs:
        append_jsonl(
            updates_path,
            {
                "method": config.method,
                "status": "not_run_preflight_failed",
                "validation_gate": config.validation_gate,
                "validation_gated_prompt_updates": 0,
                "accepted_updates": 0,
                "rejected_updates": 0,
                "reason": "WebArena preflight failed.",
            },
        )

    write_summary_csv(output_dir / "summary.csv", method_configs, selected_tasks, status="blocked")
    write_summary_markdown(output_dir / "summary.md", method_configs, selected_tasks, preflight)


def write_summary_csv(
    path: Path,
    method_configs: list[WebArenaMethodConfig],
    selected_tasks: list[WebArenaTask],
    *,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "status",
                "tasks_planned",
                "task_success_rate",
                "invalid_browser_action_rate",
                "repeated_action_rate",
                "avg_turns",
                "validation_gated_prompt_updates",
            ],
        )
        writer.writeheader()
        for config in method_configs:
            writer.writerow(
                {
                    "method": config.method,
                    "status": status,
                    "tasks_planned": len(selected_tasks),
                    "task_success_rate": "",
                    "invalid_browser_action_rate": "",
                    "repeated_action_rate": "",
                    "avg_turns": "",
                    "validation_gated_prompt_updates": 0,
                }
            )


def write_summary_markdown(
    path: Path,
    method_configs: list[WebArenaMethodConfig],
    selected_tasks: list[WebArenaTask],
    preflight: WebArenaPreflight,
) -> None:
    missing_sections = [
        ("Missing commands", preflight.missing_commands),
        ("Missing imports", preflight.missing_imports),
        ("Missing environment variables", preflight.missing_env_vars),
        ("Missing generated configs", preflight.missing_generated_configs[:10]),
        ("Missing auth files", preflight.missing_auth_files[:10]),
    ]
    lines = [
        "# WebArena Small-Subset Benchmark Status",
        "",
        f"Status: {'ready' if preflight.ok else 'blocked'}",
        f"Backend: {preflight.backend}",
        f"Tasks selected: {len(selected_tasks)}",
        f"Methods: {', '.join(config.method for config in method_configs)}",
        "",
        "## Task Mix",
        "",
        f"Sites: {json.dumps(task_site_distribution(selected_tasks), sort_keys=True)}",
        f"Evaluation types: {json.dumps(task_eval_distribution(selected_tasks), sort_keys=True)}",
        "",
        "## Preflight",
        "",
    ]
    for title, items in missing_sections:
        if not items:
            continue
        suffix = "" if len(items) <= 10 else f" (+{len(items) - 10} more)"
        lines.append(f"### {title}{suffix}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    if preflight.notes:
        lines.append("### Notes")
        lines.extend(f"- {note}" for note in preflight.notes)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def resolve_task_config(args: argparse.Namespace) -> Path:
    if args.task_config:
        return Path(args.task_config)
    return Path(args.webarena_root) / "config_files" / "test.raw.json"


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    return methods or list(DEFAULT_METHODS)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    webarena_root = Path(args.webarena_root)
    task_config = resolve_task_config(args)
    method_configs = build_method_configs(parse_methods(args.methods))

    selected_tasks: list[WebArenaTask] = []
    if task_config.exists():
        site_filter = {item.strip() for item in args.sites.split(",") if item.strip()} if args.sites else None
        selected_tasks = select_task_subset(load_raw_tasks(task_config), args.task_count, sites=site_filter)

    preflight = check_webarena_preflight(
        backend=args.backend,
        webarena_root=webarena_root,
        task_config=task_config,
        selected_tasks=selected_tasks,
    )
    if not preflight.ok:
        write_blocked_artifacts(
            output_dir,
            args=args,
            selected_tasks=selected_tasks,
            method_configs=method_configs,
            preflight=preflight,
        )
        return 2
    write_common_artifacts(
        output_dir,
        args=args,
        selected_tasks=selected_tasks,
        method_configs=method_configs,
        preflight=preflight,
    )
    if args.backend != "official":
        write_json(
            output_dir / "execution_ready.json",
            {
                "status": "ready_for_execution",
                "message": "BrowserGym preflight passed; use the official backend for the implemented runner.",
            },
        )
        return 0
    run_official_method_matrix(
        output_dir=output_dir,
        args=args,
        selected_tasks=selected_tasks,
        method_configs=method_configs,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or preflight a small WebArena subset for TextGrad-RL.")
    parser.add_argument("--webarena-root", default=os.getenv("WEBARENA_ROOT", "external/webarena"))
    parser.add_argument("--task-config", default="")
    parser.add_argument("--output-dir", default="runs/webarena_small_subset")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--sites", default="", help="Optional comma-separated WebArena site filter.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--backend", choices=["official", "browsergym"], default="official")
    parser.add_argument("--llm-base-url", default=os.getenv("TEXTGRAD_RL_LLM_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("TEXTGRAD_RL_LLM_MODEL", "gpt-oss:20b"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--train-tasks", type=int, default=8)
    parser.add_argument("--val-tasks", type=int, default=4)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
