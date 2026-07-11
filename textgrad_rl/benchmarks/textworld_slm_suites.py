"""Real local-SLM TextWorldExpress and TextWorld benchmark suites.

This module drives local text-game environments with an OpenAI-compatible
chat model. It compares a frozen prompt-only SLM policy against one
TextGrad-RL train/validation-gated textual policy update.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_slm_compare import OpenAICompatibleChatModel
from textgrad_rl.benchmarks.textworld_24_suite import (
    FAMILY_CATEGORIES,
    RULE_TEXTS as TEXTWORLD_RULE_TEXTS,
    TextWorldSpec,
    default_specs,
    ensure_games,
)
from textgrad_rl.benchmarks.textworld_express_suite import GAME_CATEGORIES, RULE_BY_GAME, TWX_GAMES
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


METHODS = ["no_textgrad", "textgrad_rl"]


class ModelCallTimeout(RuntimeError):
    """Raised when a local model call exceeds the per-action wall-clock budget."""


@contextlib.contextmanager
def model_call_timer(seconds: int):
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise ModelCallTimeout(f"model call exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@dataclass
class SLMTextGameRecord:
    benchmark: str
    problem_id: str
    family: str
    category: str
    method: str
    split: str
    seed: int
    model: str
    reward: float
    success: bool
    invalid_action: bool
    parse_failed: bool
    repeated_actions: bool
    turns: int
    final_score: float
    max_score: float
    actions: list[str]
    raw_outputs: list[str]
    failure_reason: str
    runtime_seconds: float


class IndexedActionSLMAgent:
    """Frozen SLM actor that must select one visible action by index."""

    def __init__(
        self,
        *,
        model: OpenAICompatibleChatModel,
        variables: dict[str, TextVariable],
        benchmark: str,
        problem_id: str,
        model_call_timeout: int = 0,
    ) -> None:
        self.model = model
        self.variables = variables
        self.benchmark = benchmark
        self.problem_id = problem_id
        self.model_call_timeout = model_call_timeout
        self.previous_actions: list[str] = []

    def act(self, observation: str, valid_actions: list[str]) -> tuple[str | None, str]:
        prompt = self._prompt(observation, valid_actions)
        try:
            if self.model_call_timeout > 0:
                raw = complete_with_hard_timeout(self.model, prompt, self.model_call_timeout)
            else:
                with model_call_timer(self.model_call_timeout):
                    raw = self.model.complete(prompt)
        except ModelCallTimeout as exc:
            return None, f"REQUEST_TIMEOUT: {exc}"
        except Exception as exc:
            return None, f"REQUEST_FAILED: {type(exc).__name__}: {exc}"
        action = parse_indexed_action(raw, valid_actions)
        if action is not None:
            self.previous_actions.append(action)
        return action, raw

    def _prompt(self, observation: str, valid_actions: list[str]) -> str:
        variables = "\n\n".join(
            f"{variable.name} ({variable.role_description}):\n{variable.clipped_value()}"
            for variable in self.variables.values()
        )
        actions = "\n".join(f"[{index}] {action}" for index, action in enumerate(valid_actions, start=1))
        previous = ", ".join(self.previous_actions[-8:]) if self.previous_actions else "<none>"
        return (
            f"TEXT VARIABLES:\n{variables}\n\n"
            f"BENCHMARK: {self.benchmark}\n"
            f"PROBLEM: {self.problem_id}\n"
            f"PREVIOUS ACTIONS: {previous}\n\n"
            f"OBSERVATION:\n{observation}\n\n"
            f"VALID ACTIONS:\n{actions}\n\n"
            "Choose exactly one valid action. Return only the bracketed number, for example [1]."
        )


def complete_with_hard_timeout(model: OpenAICompatibleChatModel, prompt: str, seconds: int) -> str:
    """Call an OpenAI-compatible endpoint with an OS-enforced wall-clock cap."""

    payload = {
        "model": model.model,
        "messages": [
            {
                "role": "system",
                "content": "You choose actions in text games. Return exactly one bracketed number like [1]. Do not explain.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "stream": False,
    }
    command = [
        "curl",
        "-sS",
        "--max-time",
        str(seconds),
        "-X",
        "POST",
        f"{model.base_url}/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {model.api_key}",
        "--data-binary",
        "@-",
    ]
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=seconds + 5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelCallTimeout(f"curl subprocess exceeded {seconds + 5}s") from exc
    if result.returncode != 0:
        if result.returncode == 28:
            raise ModelCallTimeout(f"curl exceeded {seconds}s")
        raise RuntimeError(result.stderr.strip() or f"curl failed with exit code {result.returncode}")
    try:
        data = json.loads(result.stdout)
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError(f"could not parse chat response: {result.stdout[:500]}") from exc


def initial_variables() -> dict[str, TextVariable]:
    return {
        "text_game_slm_policy": TextVariable(
            name="text_game_slm_policy",
            value=(
                "Use only one of the numbered valid actions. Track the objective, inventory, room, and last actions. "
                "Prefer progress toward the visible goal, avoid repeating no-progress commands, inspect/read useful "
                "documents when available, and use look/inventory only when needed."
            ),
            role_description="General local text-game policy for a frozen SLM.",
            max_chars=3200,
        )
    }


def parse_indexed_action(raw: str, valid_actions: list[str]) -> str | None:
    text = raw.strip()
    for pattern in [r"\[(\d+)\]", r"^\s*(\d+)\s*$", r"action\s*(?:#|number)?\s*:?\s*(\d+)"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            index = int(match.group(1))
            if 1 <= index <= len(valid_actions):
                return valid_actions[index - 1]
    cleaned = clean_output_action(text)
    by_norm = {normalize_action(action): action for action in valid_actions}
    if normalize_action(cleaned) in by_norm:
        return by_norm[normalize_action(cleaned)]
    quoted = re.findall(r"[\"'`](.+?)[\"'`]", text)
    for item in quoted:
        normalized = normalize_action(item)
        if normalized in by_norm:
            return by_norm[normalized]
    lowered = normalize_action(text)
    matches = [action for action in valid_actions if normalize_action(action) in lowered]
    if matches:
        return max(matches, key=len)
    return None


def clean_output_action(text: str) -> str:
    text = text.splitlines()[0] if text.splitlines() else text
    text = re.sub(r"^[-*\s]*(?:action|command|answer)\s*:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" []\"'`.")


def normalize_action(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\[(]\s*\d+\s*[\])]\s*", "", text)
    text = re.sub(r"\b(the|a|an|some)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sorted_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def textgame_score(record: SLMTextGameRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.4 * (1.0 if record.invalid_action else 0.0)
        - 0.2 * (1.0 if record.parse_failed else 0.0)
        - 0.1 * (1.0 if record.repeated_actions else 0.0)
        - 0.004 * record.turns
    )


def textworld_express_observation(observation: str, info: dict[str, Any]) -> str:
    pieces = [str(observation)]
    for key in ["taskDesc", "inventory", "score"]:
        value = info.get(key)
        if value:
            pieces.append(f"{key}: {value}")
    return "\n".join(pieces)


def run_textworld_express_episode(
    *,
    game: str,
    method: str,
    split: str,
    seed: int,
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    model_call_timeout: int,
) -> SLMTextGameRecord:
    try:
        from textworld_express import TextWorldExpressEnv
    except ImportError as exc:
        raise SystemExit("TextWorldExpress is not installed. Run: python -m pip install textworld-express") from exc

    started = time.perf_counter()
    env = TextWorldExpressEnv(envStepLimit=max_steps)
    agent = IndexedActionSLMAgent(
        model=model,
        variables=variables,
        benchmark="textworld_express",
        problem_id=game,
        model_call_timeout=model_call_timeout,
    )
    actions: list[str] = []
    raw_outputs: list[str] = []
    success = False
    invalid = False
    parse_failed = False
    reward_total = 0.0
    final_score = 0.0
    failure = ""
    turns = 0
    try:
        observation, info = env.reset(seed=seed, gameFold=split, gameName=game, gameParams="", generateGoldPath=False)
        for _ in range(max_steps):
            valid_actions = sorted_unique(list(info.get("validActions") or []))
            action, raw = agent.act(textworld_express_observation(str(observation), info), valid_actions)
            raw_outputs.append(raw)
            if action is None:
                invalid = True
                parse_failed = True
                failure = f"could not parse model action: {raw[:160]}"
                break
            if action not in set(valid_actions):
                invalid = True
                failure = f"invalid action: {action}"
                break
            observation, reward, done, info = env.step(action)
            actions.append(action)
            reward_total += float(reward)
            final_score = float(info.get("score", final_score))
            success = bool(info.get("tasksuccess")) or final_score >= 1.0
            turns += 1
            if done:
                if not success and bool(info.get("taskfailure")):
                    failure = "task failure"
                break
        if not success and not failure:
            failure = "step budget exhausted" if turns >= max_steps else "score below success"
    except Exception as exc:
        invalid = True
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    return SLMTextGameRecord(
        benchmark="textworld_express",
        problem_id=game,
        family=game,
        category=GAME_CATEGORIES.get(game, "other"),
        method=method,
        split=split,
        seed=seed,
        model=model.model,
        reward=final_score if final_score else reward_total,
        success=success,
        invalid_action=invalid,
        parse_failed=parse_failed,
        repeated_actions=len(actions) != len(set(actions)),
        turns=turns,
        final_score=final_score,
        max_score=1.0,
        actions=actions,
        raw_outputs=raw_outputs,
        failure_reason=failure,
        runtime_seconds=time.perf_counter() - started,
    )


def compact_text(text: str, limit: int) -> str:
    """Trim banner-heavy TextWorld text while preserving semantic lines."""

    lines: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped:
            continue
        alnum = sum(char.isalnum() for char in stripped)
        if len(stripped) > 20 and alnum / max(len(stripped), 1) < 0.35:
            continue
        lines.append(stripped)
    return " ".join(lines)[:limit]


def textworld_observation(state: Any) -> str:
    return "\n".join(
        [
            f"Objective: {compact_text(str(state.objective or ''), 900)}",
            f"Description: {compact_text(str(state.description or ''), 900)}",
            f"Inventory: {compact_text(str(state.inventory or ''), 300)}",
            f"Last feedback: {compact_text(str(state.feedback or ''), 500)}",
            f"Score: {state.score or 0}/{state.max_score or 1}",
        ]
    )


def run_textworld_episode(
    *,
    spec: TextWorldSpec,
    game_path: Path,
    method: str,
    split: str,
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    model_call_timeout: int,
) -> SLMTextGameRecord:
    try:
        import textworld
    except ImportError as exc:
        raise SystemExit("TextWorld is not installed. Run: python -m pip install textworld==1.7.0") from exc

    infos = textworld.EnvInfos(
        admissible_commands=True,
        objective=True,
        inventory=True,
        description=True,
        score=True,
        max_score=True,
        won=True,
        lost=True,
        intermediate_reward=True,
        moves=True,
    )
    started = time.perf_counter()
    env = textworld.start(str(game_path), infos)
    agent = IndexedActionSLMAgent(
        model=model,
        variables=variables,
        benchmark="textworld_24",
        problem_id=spec.spec_id,
        model_call_timeout=model_call_timeout,
    )
    actions: list[str] = []
    raw_outputs: list[str] = []
    invalid = False
    parse_failed = False
    failure = ""
    final_score = 0.0
    max_score = 1.0
    success = False
    turns = 0
    try:
        state = env.reset()
        max_score = float(state.max_score or 1.0)
        for _ in range(max_steps):
            admissible = sorted_unique(list(state.admissible_commands or []))
            action, raw = agent.act(textworld_observation(state), admissible)
            raw_outputs.append(raw)
            if action is None:
                invalid = True
                parse_failed = True
                failure = f"could not parse model action: {raw[:160]}"
                break
            if action not in set(admissible):
                invalid = True
                failure = f"invalid action: {action}"
                break
            state, _score, done = env.step(action)
            actions.append(action)
            final_score = float(state.score or 0.0)
            success = bool(state.won) or final_score >= max_score
            turns += 1
            if done:
                if not success and bool(state.lost):
                    failure = "lost"
                break
        if not success and not failure:
            failure = "step budget exhausted" if turns >= max_steps else "score below success"
    except Exception as exc:
        invalid = True
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    return SLMTextGameRecord(
        benchmark="textworld_24",
        problem_id=spec.spec_id,
        family=spec.family,
        category=spec.category,
        method=method,
        split=split,
        seed=spec.seed,
        model=model.model,
        reward=final_score / max(max_score, 1.0),
        success=success,
        invalid_action=invalid,
        parse_failed=parse_failed,
        repeated_actions=len(actions) != len(set(actions)),
        turns=turns,
        final_score=final_score,
        max_score=max_score,
        actions=actions,
        raw_outputs=raw_outputs,
        failure_reason=failure,
        runtime_seconds=time.perf_counter() - started,
    )


def append_records(path: Path, records: list[SLMTextGameRecord]) -> None:
    if path.exists():
        path.unlink()
    for record in records:
        append_jsonl(path, record)


def run_twx_records(
    *,
    games: list[str],
    method: str,
    split: str,
    seeds: list[int],
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    model_call_timeout: int,
    output_jsonl: Path,
) -> list[SLMTextGameRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[SLMTextGameRecord] = []
    for game in games:
        for seed in seeds:
            record = run_textworld_express_episode(
                game=game,
                method=method,
                split=split,
                seed=seed,
                model=model,
                variables=variables,
                max_steps=max_steps,
                model_call_timeout=model_call_timeout,
            )
            records.append(record)
            append_jsonl(output_jsonl, record)
            print(
                f"{method}/{split}/{game}: reward={record.reward:.3f} success={int(record.success)} "
                f"invalid={int(record.invalid_action)} turns={record.turns}",
                flush=True,
            )
    return records


def run_tw_records(
    *,
    specs: list[TextWorldSpec],
    game_paths: dict[str, Path],
    method: str,
    split: str,
    model: OpenAICompatibleChatModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    model_call_timeout: int,
    output_jsonl: Path,
) -> list[SLMTextGameRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[SLMTextGameRecord] = []
    for spec in specs:
        record = run_textworld_episode(
            spec=spec,
            game_path=game_paths[spec.spec_id],
            method=method,
            split=split,
            model=model,
            variables=variables,
            max_steps=max_steps,
            model_call_timeout=model_call_timeout,
        )
        records.append(record)
        append_jsonl(output_jsonl, record)
        print(
            f"{method}/{split}/{spec.spec_id}: reward={record.reward:.3f} success={int(record.success)} "
            f"invalid={int(record.invalid_action)} turns={record.turns}",
            flush=True,
        )
    return records


def twx_gradients(records: list[SLMTextGameRecord]) -> list[TextualGradient]:
    gradients: list[TextualGradient] = []
    for game in TWX_GAMES:
        group = [record for record in records if record.problem_id == game]
        if not group:
            continue
        success = mean([float(record.success) for record in group])
        if success >= 1.0 and not any(record.invalid_action or record.repeated_actions for record in group):
            continue
        gradients.append(
            TextualGradient(
                target_variable_name="text_game_slm_policy",
                failure_mode=f"textworld_express:{game}",
                evidence_from_trajectory=trajectory_evidence(group),
                gradient_text=f"Add a TextWorldExpress strategy for {game}.",
                suggested_edit=f"Add a rule: {RULE_BY_GAME[game]}",
                confidence=0.82,
                forbidden_shortcuts=["gold path", "hidden state", "hardcoded seed answer"],
            )
        )
    return gradients


def tw_gradients(records: list[SLMTextGameRecord]) -> list[TextualGradient]:
    needed: set[str] = set()
    for record in records:
        if record.success and not record.invalid_action and not record.repeated_actions:
            continue
        if record.family in {"tw-simple", "tw-coin_collector", "tw-treasure_hunter"}:
            needed.add("objective_sequence")
            needed.add("graph_exploration")
        if record.family == "tw-cooking":
            needed.add("cooking_recipe")
            needed.add("graph_exploration")
    return [
        TextualGradient(
            target_variable_name="text_game_slm_policy",
            failure_mode=f"textworld_24:{rule_id}",
            evidence_from_trajectory=trajectory_evidence([record for record in records if not record.success][:4]),
            gradient_text=f"Add a reusable TextWorld strategy for {rule_id}.",
            suggested_edit=f"Add a rule: {TEXTWORLD_RULE_TEXTS[rule_id]}",
            confidence=0.82,
            forbidden_shortcuts=["oracle walkthrough", "policy_commands", "hidden state"],
        )
        for rule_id in sorted(needed)
    ]


def trajectory_evidence(records: list[SLMTextGameRecord]) -> str:
    examples = []
    for record in records[:4]:
        tail = " -> ".join(record.actions[-5:]) if record.actions else "<none>"
        examples.append(
            f"{record.problem_id}/seed={record.seed}: success={record.success}, invalid={record.invalid_action}, "
            f"repeated={record.repeated_actions}, turns={record.turns}, reason={record.failure_reason}, actions={tail}"
        )
    return "; ".join(examples)


def train_and_gate(
    *,
    benchmark: str,
    train_runner: Any,
    val_runner: Any,
    gradient_fn: Any,
    base_variables: dict[str, TextVariable],
    output_dir: Path,
    min_mean_delta: float,
    max_rules_per_step: int,
) -> tuple[dict[str, TextVariable], dict[str, Any]]:
    train = train_runner(base_variables, "train_base")
    gradients = gradient_fn(train)
    candidate = TextualGradientDescent(max_prompt_chars=3200, max_rules_per_step=max_rules_per_step).step(
        base_variables,
        gradients,
        constraints=["must choose only visible valid actions", "must not use hidden state", "must not hardcode seeds"],
    )
    val_old = val_runner(base_variables, "val_old")
    val_new = val_runner(candidate, "val_candidate")
    old_score = mean([textgame_score(record) for record in val_old])
    new_score = mean([textgame_score(record) for record in val_new])
    old_invalid = mean([float(record.invalid_action) for record in val_old])
    new_invalid = mean([float(record.invalid_action) for record in val_new])
    old_parse = mean([float(record.parse_failed) for record in val_old])
    new_parse = mean([float(record.parse_failed) for record in val_new])
    accepted = (
        bool(gradients)
        and new_score >= old_score + min_mean_delta
        and new_invalid <= old_invalid
        and new_parse <= old_parse
    )
    gate = {
        "benchmark": benchmark,
        "method": "textgrad_rl",
        "accepted": accepted,
        "old_val_score": old_score,
        "new_val_score": new_score,
        "old_invalid_rate": old_invalid,
        "new_invalid_rate": new_invalid,
        "old_parse_fail_rate": old_parse,
        "new_parse_fail_rate": new_parse,
        "gradient_count": len(gradients),
    }
    write_json(output_dir / "textgrad_rl" / "gradients.json", gradients)
    write_json(output_dir / "textgrad_rl" / "gate_decision.json", gate)
    final_variables = candidate if accepted else base_variables
    write_json(output_dir / "textgrad_rl" / "text_variables.json", final_variables)
    return final_variables, gate


def summarize_records(records: list[SLMTextGameRecord]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "average_reward": mean([record.reward for record in records]),
        "average_score": mean([textgame_score(record) for record in records]),
        "success_rate": mean([float(record.success) for record in records]),
        "invalid_action_rate": mean([float(record.invalid_action) for record in records]),
        "parse_fail_rate": mean([float(record.parse_failed) for record in records]),
        "repeated_action_rate": mean([float(record.repeated_actions) for record in records]),
        "average_turns": mean([record.turns for record in records]),
        "runtime_seconds": sum(record.runtime_seconds for record in records),
    }


def overall_rows(records: list[SLMTextGameRecord], gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        group = [record for record in records if record.method == method and record.split == "test"]
        row = {
            "method": method,
            "problems": len({record.problem_id for record in group}),
            **summarize_records(group),
            "accepted_updates": 1 if gates.get(method, {}).get("accepted") else 0,
            "gradient_count": int(gates.get(method, {}).get("gradient_count", 0)),
        }
        rows.append(row)
    return rows


def slice_rows(records: list[SLMTextGameRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test = [record for record in records if record.split == "test"]
    for method in METHODS:
        method_records = [record for record in test if record.method == method]
        for key in ["problem_id", "family", "category"]:
            for value in sorted({getattr(record, key) for record in method_records}):
                group = [record for record in method_records if getattr(record, key) == value]
                rows.append(
                    {
                        "method": method,
                        "slice": key,
                        "value": value,
                        "problems": len({record.problem_id for record in group}),
                        **summarize_records(group),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary(path: Path, title: str, summary_rows: list[dict[str, Any]], groups: list[dict[str, Any]], gates: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        "| method | problems | episodes | reward | score | success | invalid | parse_fail | repeated | turns | updates | gradients |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {problems} | {episodes} | {average_reward:.3f} | {average_score:.3f} | "
            "{success_rate:.3f} | {invalid_action_rate:.3f} | {parse_fail_rate:.3f} | "
            "{repeated_action_rate:.3f} | {average_turns:.2f} | {accepted_updates} | {gradient_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per Problem",
            "",
            "| method | problem | reward | score | success | invalid | parse_fail | repeated | turns |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [row for row in groups if row["slice"] == "problem_id"]:
        lines.append(
            "| {method} | {value} | {average_reward:.3f} | {average_score:.3f} | {success_rate:.3f} | "
            "{invalid_action_rate:.3f} | {parse_fail_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} |".format(**row)
        )
    lines.extend(["", "## Gate Decisions", ""])
    for method, gate in gates.items():
        lines.append(f"- `{method}`: `{gate}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_textworld_express_suite(args: argparse.Namespace, model: OpenAICompatibleChatModel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    games = list(TWX_GAMES)
    train_seeds = [args.seed + index for index in range(args.train_seeds)]
    val_seeds = [args.seed + 10_000 + index for index in range(args.val_seeds)]
    test_seeds = [args.seed + 20_000 + index for index in range(args.test_seeds)]
    write_json(output_dir / "config.json", {
        "benchmark": "textworld_express_slm",
        "games": games,
        "model": model.model,
        "temperature": model.temperature,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "max_steps": args.max_steps,
    })
    write_json(output_dir / "environment_info.json", environment_info())
    all_records: list[SLMTextGameRecord] = []
    gates = {"no_textgrad": {"accepted": False, "gradient_count": 0}}
    base = initial_variables()
    write_json(output_dir / "no_textgrad" / "text_variables.json", base)
    no_textgrad = run_twx_records(
        games=games,
        method="no_textgrad",
        split="test",
        seeds=test_seeds,
        model=model,
        variables=base,
        max_steps=args.max_steps,
        model_call_timeout=args.model_call_timeout,
        output_jsonl=output_dir / "no_textgrad" / "test.jsonl",
    )
    all_records.extend(no_textgrad)

    textgrad_base = initial_variables()

    def train_runner(variables: dict[str, TextVariable], label: str) -> list[SLMTextGameRecord]:
        return run_twx_records(
            games=games,
            method="textgrad_rl",
            split="train",
            seeds=train_seeds,
            model=model,
            variables=variables,
            max_steps=args.max_steps,
            model_call_timeout=args.model_call_timeout,
            output_jsonl=output_dir / "textgrad_rl" / f"{label}.jsonl",
        )

    def val_runner(variables: dict[str, TextVariable], label: str) -> list[SLMTextGameRecord]:
        return run_twx_records(
            games=games,
            method="textgrad_rl",
            split="validation",
            seeds=val_seeds,
            model=model,
            variables=variables,
            max_steps=args.max_steps,
            model_call_timeout=args.model_call_timeout,
            output_jsonl=output_dir / "textgrad_rl" / f"{label}.jsonl",
        )

    final_variables, gate = train_and_gate(
        benchmark="textworld_express",
        train_runner=train_runner,
        val_runner=val_runner,
        gradient_fn=twx_gradients,
        base_variables=textgrad_base,
        output_dir=output_dir,
        min_mean_delta=args.min_mean_delta,
        max_rules_per_step=args.max_rules_per_step,
    )
    gates["textgrad_rl"] = gate
    textgrad_test = run_twx_records(
        games=games,
        method="textgrad_rl",
        split="test",
        seeds=test_seeds,
        model=model,
        variables=final_variables,
        max_steps=args.max_steps,
        model_call_timeout=args.model_call_timeout,
        output_jsonl=output_dir / "textgrad_rl" / "test.jsonl",
    )
    all_records.extend(textgrad_test)
    summary = overall_rows(all_records, gates)
    groups = slice_rows(all_records)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "slice_summary.json", groups)
    write_json(output_dir / "gate_decisions.json", gates)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "slice_summary.csv", groups)
    write_summary(output_dir / "summary.md", "TextWorldExpress qwen2.5:7b SLM Suite", summary, groups, gates)
    return output_dir


def training_specs(specs: list[TextWorldSpec], offset: int, per_family: int) -> list[TextWorldSpec]:
    selected: list[TextWorldSpec] = []
    for family in sorted({spec.family for spec in specs}):
        group = [spec for spec in specs if spec.family == family]
        selected.extend(group[offset : offset + per_family])
    return selected


def run_textworld_24_suite(args: argparse.Namespace, model: OpenAICompatibleChatModel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = default_specs(args.textworld_seed)
    game_paths = ensure_games(specs, output_dir / "games", force=args.force_regenerate)
    train_specs = training_specs(specs, 0, args.textworld_train_per_family)
    val_specs = training_specs(specs, args.textworld_train_per_family, args.textworld_val_per_family)
    write_json(output_dir / "config.json", {
        "benchmark": "textworld_24_slm",
        "specs": specs,
        "model": model.model,
        "temperature": model.temperature,
        "train_specs": [spec.spec_id for spec in train_specs],
        "val_specs": [spec.spec_id for spec in val_specs],
        "test_specs": [spec.spec_id for spec in specs],
        "max_steps": args.max_steps,
    })
    write_json(output_dir / "environment_info.json", environment_info())
    all_records: list[SLMTextGameRecord] = []
    gates = {"no_textgrad": {"accepted": False, "gradient_count": 0}}
    base = initial_variables()
    write_json(output_dir / "no_textgrad" / "text_variables.json", base)
    no_textgrad = run_tw_records(
        specs=specs,
        game_paths=game_paths,
        method="no_textgrad",
        split="test",
        model=model,
        variables=base,
        max_steps=args.max_steps,
        model_call_timeout=args.model_call_timeout,
        output_jsonl=output_dir / "no_textgrad" / "test.jsonl",
    )
    all_records.extend(no_textgrad)

    textgrad_base = initial_variables()

    def train_runner(variables: dict[str, TextVariable], label: str) -> list[SLMTextGameRecord]:
        return run_tw_records(
            specs=train_specs,
            game_paths=game_paths,
            method="textgrad_rl",
            split="train",
            model=model,
            variables=variables,
            max_steps=args.max_steps,
            model_call_timeout=args.model_call_timeout,
            output_jsonl=output_dir / "textgrad_rl" / f"{label}.jsonl",
        )

    def val_runner(variables: dict[str, TextVariable], label: str) -> list[SLMTextGameRecord]:
        return run_tw_records(
            specs=val_specs,
            game_paths=game_paths,
            method="textgrad_rl",
            split="validation",
            model=model,
            variables=variables,
            max_steps=args.max_steps,
            model_call_timeout=args.model_call_timeout,
            output_jsonl=output_dir / "textgrad_rl" / f"{label}.jsonl",
        )

    final_variables, gate = train_and_gate(
        benchmark="textworld_24",
        train_runner=train_runner,
        val_runner=val_runner,
        gradient_fn=tw_gradients,
        base_variables=textgrad_base,
        output_dir=output_dir,
        min_mean_delta=args.min_mean_delta,
        max_rules_per_step=args.max_rules_per_step,
    )
    gates["textgrad_rl"] = gate
    textgrad_test = run_tw_records(
        specs=specs,
        game_paths=game_paths,
        method="textgrad_rl",
        split="test",
        model=model,
        variables=final_variables,
        max_steps=args.max_steps,
        model_call_timeout=args.model_call_timeout,
        output_jsonl=output_dir / "textgrad_rl" / "test.jsonl",
    )
    all_records.extend(textgrad_test)
    summary = overall_rows(all_records, gates)
    groups = slice_rows(all_records)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "slice_summary.json", groups)
    write_json(output_dir / "gate_decisions.json", gates)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "slice_summary.csv", groups)
    write_summary(output_dir / "summary.md", "TextWorld 24 qwen2.5:7b SLM Suite", summary, groups, gates)
    return output_dir


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qwen-style SLM TextWorld suites.")
    parser.add_argument("--suites", default="textworld_express,textworld_24")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--model-call-timeout", type=int, default=45)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--train-seeds", type=int, default=1)
    parser.add_argument("--val-seeds", type=int, default=1)
    parser.add_argument("--test-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--textworld-seed", type=int, default=62001)
    parser.add_argument("--textworld-train-per-family", type=int, default=1)
    parser.add_argument("--textworld-val-per-family", type=int, default=1)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--max-rules-per-step", type=int, default=8)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--output-dir", default="runs/qwen25_7b_textworld_slm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = OpenAICompatibleChatModel(
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    selected = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
    if "textworld_express" in selected:
        run_textworld_express_suite(args, model, output_dir / "textworld_express")
    if "textworld_24" in selected:
        run_textworld_24_suite(args, model, output_dir / "textworld_24")
    lines = ["# qwen2.5:7b TextWorld SLM Suites", ""]
    if "textworld_express" in selected:
        lines.append("- TextWorldExpress: `textworld_express/summary.md`")
    if "textworld_24" in selected:
        lines.append("- TextWorld 24: `textworld_24/summary.md`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TextWorld SLM artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
