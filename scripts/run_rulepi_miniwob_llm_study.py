#!/usr/bin/env python3
"""Run the preregistered RulePI MiniWoB study with a direct LLM actor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable

from textgrad_rl.benchmarks.miniwob_subset import (
    MiniWobActorConfig,
    MiniWobRecord,
    category_for_env,
    initial_miniwob_variables,
    miniwob_score,
    run_miniwob_episode,
)
from textgrad_rl.types import TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json


CALIBRATION_ENVS = [
    "click-button-sequence",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-checkboxes-transfer",
    "click-menu",
    "click-menu-2",
    "click-scroll-list",
    "click-tab-2",
    "click-widget",
    "copy-paste-2",
    "email-inbox-delete",
    "email-inbox-reply",
    "enter-date",
    "enter-time",
    "form-sequence",
    "form-sequence-2",
    "form-sequence-3",
    "login-user-popup",
    "navigate-tree",
    "use-autocomplete",
]

CALIBRATION_SEEDS = [11_000, 11_001, 11_002]
TRAIN_SEEDS = [21_000, 21_001]
VALIDATION_SEEDS = [31_000, 31_001]
TEST_SEEDS = list(range(41_000, 41_010))
RECORD_FIELDS = {field.name for field in fields(MiniWobRecord)}
WRITE_LOCK = threading.Lock()
DIAGNOSTIC_LOCK = threading.Lock()


def prompt_hash(variables: dict[str, TextVariable]) -> str:
    text = "\n".join(f"{name}:{variable.value}" for name, variable in sorted(variables.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def variables_with_rule(rule: str) -> dict[str, TextVariable]:
    base = initial_miniwob_variables()["general_agent_policy"]
    return {
        base.name: TextVariable(
            name=base.name,
            value=f"{base.value}\n\nAdditional reusable rule:\n{rule.strip()}",
            role_description=base.role_description,
            max_chars=base.max_chars,
            version=1,
        )
    }


def actor_config(args: argparse.Namespace, *, seed_offset: int = 0) -> MiniWobActorConfig:
    return MiniWobActorConfig(
        actor="llm",
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        generation_seed_offset=seed_offset,
    )


def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["env_id"],
        int(row["seed"]),
        row["method"],
        row["split"],
        int(row.get("attempt", 1)),
        row.get("prompt_hash", ""),
    )


def load_records(path: Path) -> dict[tuple[Any, ...], MiniWobRecord]:
    loaded: dict[tuple[Any, ...], MiniWobRecord] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        loaded[record_key(row)] = MiniWobRecord(**{name: row.get(name) for name in RECORD_FIELDS})
    return loaded


def save_record(path: Path, record: MiniWobRecord, *, variables: dict[str, TextVariable], attempt: int) -> None:
    row = {**asdict(record), "prompt_hash": prompt_hash(variables), "attempt": attempt}
    with WRITE_LOCK:
        append_jsonl(path, row)


def run_episode_job(job: tuple[Any, ...]) -> MiniWobRecord:
    env_id, method, split, seed, variables, max_steps, config = job
    return run_miniwob_episode(
        env_name=env_id,
        method=method,
        split=split,
        seed=seed,
        text_variables=variables,
        max_steps=max_steps,
        actor_config=config,
    )


def evaluate_matrix(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    envs: list[str],
    seeds: list[int],
    method: str,
    split: str,
    variables: dict[str, TextVariable],
    attempt: int = 1,
    seed_offset: int = 0,
) -> list[MiniWobRecord]:
    records_path = output_dir / "episodes.jsonl"
    existing = load_records(records_path)
    p_hash = prompt_hash(variables)
    config = actor_config(args, seed_offset=seed_offset)
    results: dict[tuple[str, int], MiniWobRecord] = {}
    pending: list[tuple[str, int]] = []
    for env_id in envs:
        for seed in seeds:
            key = (env_id, seed, method, split, attempt, p_hash)
            if key in existing:
                results[(env_id, seed)] = existing[key]
            else:
                pending.append((env_id, seed))

    if pending:
        jobs = {
            spec: (spec[0], method, split, spec[1], variables, args.max_steps, config)
            for spec in pending
        }
        if args.parallelism == 1:
            completed = [(spec, run_episode_job(job)) for spec, job in jobs.items()]
        else:
            completed = []
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.parallelism, mp_context=context) as executor:
                futures = {executor.submit(run_episode_job, job): spec for spec, job in jobs.items()}
                for future in as_completed(futures):
                    completed.append((futures[future], future.result()))
        for index, (spec, record) in enumerate(completed, start=1):
            env_id, seed = spec
            save_record(records_path, record, variables=variables, attempt=attempt)
            results[(env_id, seed)] = record
            print(
                f"[{split} {index:03d}/{len(pending):03d}] {method} {env_id} seed={seed} "
                f"success={int(record.success)} turns={record.turns} invalid={int(record.invalid_browser_action)}",
                flush=True,
            )
    return [results[(env_id, seed)] for env_id in envs for seed in seeds]


def modal_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("MODAL_PROXY_TOKEN_ID")
    secret = os.getenv("MODAL_PROXY_TOKEN_SECRET")
    if key and secret:
        headers.update({"Modal-Key": key, "Modal-Secret": secret})
    return headers


def chat_complete(
    args: argparse.Namespace,
    *,
    system: str,
    user: str,
    temperature: float,
    seed: int,
    max_tokens: int = 1200,
) -> str:
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=modal_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
    return str(data["choices"][0]["message"].get("content") or "").strip()


def first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON value in model response")


def normalize_rule(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("rule") or value.get("text") or value.get("policy_update")
    if not isinstance(value, str):
        raise ValueError("Rule candidate is not text")
    rule = " ".join(value.split()).strip()
    if rule.lower().startswith("add a rule:"):
        rule = rule.split(":", 1)[1].strip()
    if not rule.endswith("."):
        rule += "."
    if not 30 <= len(rule) <= 600:
        raise ValueError(f"Rule has invalid length: {len(rule)}")
    lowered = rule.lower()
    forbidden = ["reward", "seed", "hidden", "benchmark html", "task id"]
    if any(token in lowered for token in forbidden):
        raise ValueError(f"Rule contains forbidden shortcut language: {rule}")
    return rule


def failure_evidence(records: list[MiniWobRecord], limit: int = 16) -> str:
    failures = [record for record in records if not record.success]
    lines = []
    for record in failures[:limit]:
        raw_tail = (record.raw_outputs or [])[-2:]
        lines.append(
            f"task={record.env_id}; goal={record.goal}; actions={record.actions}; "
            f"failure={record.failure_reason}; raw_tail={raw_tail}"
        )
    return "\n".join(lines)


def generate_candidates(args: argparse.Namespace, output_dir: Path, train: list[MiniWobRecord]) -> list[str]:
    path = output_dir / "candidate_generation.json"
    if path.exists():
        return [str(item) for item in json.loads(path.read_text(encoding="utf-8"))["candidates"]]
    evidence = failure_evidence(train)
    if not evidence:
        raise RuntimeError("Training produced no failures, so no diagnostic update can be generated")
    system = (
        "You improve a browser-control policy from failed trajectories. Propose reusable procedural rules, not "
        "answers for individual tasks. Use only the visible goal, accessibility elements, and action history."
    )
    user = (
        f"FAILED TRAINING TRAJECTORIES:\n{evidence}\n\n"
        f"Return exactly one JSON object with key \"candidates\" containing exactly {args.candidate_count} "
        "distinct rule strings. Each rule must be one concrete sentence that changes action selection. "
        "Do not mention task IDs, seeds, hidden state, benchmark internals, or rewards."
    )
    raw = chat_complete(
        args,
        system=system,
        user=user,
        temperature=args.candidate_temperature,
        seed=args.candidate_seed,
    )
    parsed = first_json_value(raw)
    values = parsed.get("candidates") if isinstance(parsed, dict) else parsed
    if not isinstance(values, list) or len(values) != args.candidate_count:
        raise ValueError(f"Expected exactly {args.candidate_count} candidate rules")
    candidates = [normalize_rule(value) for value in values]
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate generator returned duplicate rules")
    write_json(path, {"raw_output": raw, "candidates": candidates, "evidence": evidence})
    return candidates


def success_rate(records: Iterable[MiniWobRecord]) -> float:
    items = list(records)
    return sum(record.success for record in items) / len(items) if items else 0.0


def invalid_rate(records: Iterable[MiniWobRecord]) -> float:
    items = list(records)
    return sum(record.invalid_browser_action for record in items) / len(items) if items else 0.0


def select_calibration_tasks(records: list[MiniWobRecord], count: int) -> list[str]:
    stats = []
    for env_id in CALIBRATION_ENVS:
        group = [record for record in records if record.env_id == env_id]
        rate = success_rate(group)
        stats.append(
            {
                "env_id": env_id,
                "category": category_for_env(env_id),
                "success": rate,
                "invalid": invalid_rate(group),
                "avg_turns": statistics.mean(record.turns for record in group),
            }
        )
    viable = sorted(
        [row for row in stats if 0.0 < row["success"] < 1.0],
        key=lambda row: (abs(row["success"] - 0.5), row["invalid"], row["env_id"]),
    )
    non_ceiling = sorted(
        [row for row in stats if row["success"] < 1.0],
        key=lambda row: (abs(row["success"] - 0.5), row["invalid"], row["env_id"]),
    )
    ordered = viable + [row for row in non_ceiling if row not in viable] + [row for row in stats if row["success"] == 1.0]
    selected: list[str] = []
    seen_categories: set[str] = set()
    for row in ordered:
        if row["category"] not in seen_categories:
            selected.append(row["env_id"])
            seen_categories.add(row["category"])
        if len(selected) == count:
            break
    for row in ordered:
        if row["env_id"] not in selected:
            selected.append(row["env_id"])
        if len(selected) == count:
            break
    return selected


def run_smoke(args: argparse.Namespace, output_dir: Path) -> None:
    records = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=["click-button", "click-checkboxes-large"],
        seeds=[9_001],
        method="fixed_prompt",
        split="smoke",
        variables=initial_miniwob_variables(),
    )
    model_errors = [record for record in records if record.failure_reason.startswith("model_error:")]
    non_noop = sum(action != "noop()" for record in records for action in record.actions)
    write_json(
        output_dir / "smoke_summary.json",
        {"records": records, "model_errors": len(model_errors), "non_noop_actions": non_noop},
    )
    if model_errors or non_noop == 0:
        raise RuntimeError("Modal smoke test failed; refusing to launch calibration")


def run_calibration(args: argparse.Namespace, output_dir: Path) -> list[str]:
    manifest_path = output_dir / "frozen_manifest.json"
    if manifest_path.exists():
        return list(json.loads(manifest_path.read_text(encoding="utf-8"))["tasks"])
    records = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=CALIBRATION_ENVS,
        seeds=CALIBRATION_SEEDS,
        method="fixed_prompt",
        split="calibration",
        variables=initial_miniwob_variables(),
    )
    selected = select_calibration_tasks(records, args.task_count)
    manifest = {
        "tasks": selected,
        "categories": {env_id: category_for_env(env_id) for env_id in selected},
        "calibration_seeds": CALIBRATION_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "selection_rule": "closest to 50% calibration success, category coverage first, then non-ceiling tasks",
        "created_at_unix": time.time(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(manifest_path, manifest)
    print(f"Frozen tasks: {', '.join(selected)}", flush=True)
    return selected


def select_candidate(
    args: argparse,
    output_dir: Path,
    tasks: list[str],
    candidates: list[str],
) -> tuple[dict[str, TextVariable], dict[str, Any]]:
    base_variables = initial_miniwob_variables()
    base_val = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=tasks,
        seeds=VALIDATION_SEEDS,
        method="validation_base",
        split="validation",
        variables=base_variables,
    )
    rows = []
    policies = []
    for index, candidate in enumerate(candidates):
        variables = variables_with_rule(candidate)
        policies.append(variables)
        records = evaluate_matrix(
            args=args,
            output_dir=output_dir,
            envs=tasks,
            seeds=VALIDATION_SEEDS,
            method=f"validation_candidate_{index:02d}",
            split="validation",
            variables=variables,
        )
        rows.append(
            {
                "candidate_index": index,
                "rule": candidate,
                "successes": sum(record.success for record in records),
                "success_rate": success_rate(records),
                "invalids": sum(record.invalid_browser_action for record in records),
                "mean_score": statistics.mean(miniwob_score(record) for record in records),
            }
        )
    base_successes = sum(record.success for record in base_val)
    base_invalids = sum(record.invalid_browser_action for record in base_val)
    selected = max(rows, key=lambda row: (row["successes"], row["mean_score"], -row["invalids"], -row["candidate_index"]))
    accepted = selected["successes"] >= base_successes + 1 and selected["invalids"] <= base_invalids
    gate = {
        "accepted": accepted,
        "base": {
            "successes": base_successes,
            "success_rate": success_rate(base_val),
            "invalids": base_invalids,
            "mean_score": statistics.mean(miniwob_score(record) for record in base_val),
        },
        "candidates": rows,
        "selected_candidate_index": selected["candidate_index"],
        "selected_rule": selected["rule"],
        "criterion": "at least one additional validation success and no additional invalid actions",
    }
    write_json(output_dir / "gate_decision.json", gate)
    return (policies[int(selected["candidate_index"])] if accepted else base_variables), gate


def diagnostic_for_retry(args: argparse.Namespace, output_dir: Path, record: MiniWobRecord) -> str:
    diagnostics_path = output_dir / "retry_diagnostics.json"
    key = f"{record.env_id}:{record.seed}"
    with DIAGNOSTIC_LOCK:
        saved = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
        if key in saved:
            return str(saved[key]["rule"])
    system = (
        "Diagnose one failed browser-control attempt. Return exactly one reusable procedural rule sentence for "
        "the next attempt. Use only the visible goal, accessibility elements implied by the actions, and history."
    )
    user = (
        f"Goal: {record.goal}\nActions: {record.actions}\nFailure: {record.failure_reason}\n"
        f"Model outputs: {(record.raw_outputs or [])[-3:]}\nReturn one rule sentence and no prose."
    )
    raw = chat_complete(
        args,
        system=system,
        user=user,
        temperature=args.candidate_temperature,
        seed=args.candidate_seed + record.seed,
        max_tokens=300,
    )
    rule = normalize_rule(raw)
    with DIAGNOSTIC_LOCK:
        saved = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
        saved[key] = {"rule": rule, "raw_output": raw}
        write_json(diagnostics_path, saved)
    return rule


def run_retries(
    args: argparse,
    output_dir: Path,
    fixed: list[MiniWobRecord],
) -> dict[tuple[str, int], MiniWobRecord]:
    failures = [record for record in fixed if not record.success]
    records_path = output_dir / "episodes.jsonl"
    existing = load_records(records_path)
    results: dict[tuple[str, int], MiniWobRecord] = {}

    def run_one(record: MiniWobRecord) -> tuple[MiniWobRecord, dict[str, TextVariable]]:
        rule = diagnostic_for_retry(args, output_dir, record)
        variables = variables_with_rule(rule)
        key = (
            record.env_id,
            record.seed,
            "retry_with_diagnostics",
            "test",
            2,
            prompt_hash(variables),
        )
        if key in existing:
            return existing[key], variables
        retry = run_miniwob_episode(
            env_name=record.env_id,
            method="retry_with_diagnostics",
            split="test",
            seed=record.seed,
            text_variables=variables,
            max_steps=args.max_steps,
            actor_config=actor_config(args, seed_offset=args.retry_seed_offset),
        )
        return retry, variables

    if failures:
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {executor.submit(run_one, record): record for record in failures}
            for index, future in enumerate(as_completed(futures), start=1):
                source = futures[future]
                retry, variables = future.result()
                key = (
                    retry.env_id,
                    retry.seed,
                    retry.method,
                    retry.split,
                    2,
                    prompt_hash(variables),
                )
                if key not in existing:
                    save_record(records_path, retry, variables=variables, attempt=2)
                results[(retry.env_id, retry.seed)] = retry
                print(
                    f"[retry {index:03d}/{len(failures):03d}] {source.env_id} seed={source.seed} "
                    f"success={int(retry.success)} turns={retry.turns}",
                    flush=True,
                )
    return results


def retry_outcome(record: MiniWobRecord, retry: MiniWobRecord | None) -> dict[str, Any]:
    return {
        "success": record.success or bool(retry and retry.success),
        "invalid": record.invalid_browser_action or bool(retry and retry.invalid_browser_action),
        "turns": record.turns + (retry.turns if retry else 0),
        "attempts": 1 + int(retry is not None),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(index, len(ordered) - 1))]


def hierarchical_delta_ci(
    left: dict[tuple[str, int], float],
    right: dict[tuple[str, int], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    keys = sorted(set(left) & set(right))
    tasks = sorted({env_id for env_id, _ in keys})
    seeds_by_task = {env_id: sorted(seed_id for task, seed_id in keys if task == env_id) for env_id in tasks}
    observed = statistics.mean(right[key] - left[key] for key in keys)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        differences = []
        for task in rng.choices(tasks, k=len(tasks)):
            task_seeds = seeds_by_task[task]
            for sampled_seed in rng.choices(task_seeds, k=len(task_seeds)):
                key = (task, sampled_seed)
                differences.append(right[key] - left[key])
        boot.append(statistics.mean(differences))
    return observed, percentile(boot, 0.025), percentile(boot, 0.975)


def summarize_test(
    args: argparse,
    output_dir: Path,
    tasks: list[str],
    fixed: list[MiniWobRecord],
    rulepi: list[MiniWobRecord],
    retries: dict[tuple[str, int], MiniWobRecord],
    gate: dict[str, Any],
) -> None:
    fixed_map = {(record.env_id, record.seed): record for record in fixed}
    rulepi_map = {(record.env_id, record.seed): record for record in rulepi}
    retry_map = {
        key: retry_outcome(record, retries.get(key))
        for key, record in fixed_map.items()
    }
    metrics = {
        "fixed_prompt": {
            "success": {key: float(record.success) for key, record in fixed_map.items()},
            "invalid": {key: float(record.invalid_browser_action) for key, record in fixed_map.items()},
            "turns": {key: float(record.turns) for key, record in fixed_map.items()},
        },
        "retry_with_diagnostics": {
            "success": {key: float(value["success"]) for key, value in retry_map.items()},
            "invalid": {key: float(value["invalid"]) for key, value in retry_map.items()},
            "turns": {key: float(value["turns"]) for key, value in retry_map.items()},
        },
        "rulepi": {
            "success": {key: float(record.success) for key, record in rulepi_map.items()},
            "invalid": {key: float(record.invalid_browser_action) for key, record in rulepi_map.items()},
            "turns": {key: float(record.turns) for key, record in rulepi_map.items()},
        },
    }
    rows = []
    for method, values in metrics.items():
        rows.append(
            {
                "method": method,
                "test_instances": len(values["success"]),
                "success_rate": statistics.mean(values["success"].values()),
                "invalid_rate": statistics.mean(values["invalid"].values()),
                "avg_actions_per_instance": statistics.mean(values["turns"].values()),
                "avg_attempts": (
                    statistics.mean(value["attempts"] for value in retry_map.values())
                    if method == "retry_with_diagnostics"
                    else 1.0
                ),
            }
        )
    comparisons = {}
    for left, right in [
        ("fixed_prompt", "rulepi"),
        ("fixed_prompt", "retry_with_diagnostics"),
        ("retry_with_diagnostics", "rulepi"),
    ]:
        comparisons[f"{right} - {left}"] = {
            metric: hierarchical_delta_ci(
                metrics[left][metric],
                metrics[right][metric],
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed,
            )
            for metric in ["success", "invalid", "turns"]
        }
    write_json(output_dir / "summary.json", {"methods": rows, "comparisons": comparisons, "gate": gate})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# RulePI GPT-OSS-120B MiniWoB++ Study",
        "",
        f"Frozen tasks: {len(tasks)}. Held-out test instances: {len(tasks) * len(TEST_SEEDS)}.",
        f"Actor: `{args.model}` at temperature `{args.temperature}`; horizon: `{args.max_steps}` actions.",
        f"Validation update accepted: `{gate['accepted']}`.",
        "",
        "| Method | Test instances | Success | Invalid | Avg actions | Avg attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['test_instances']} | {100 * row['success_rate']:.1f}% | "
            f"{100 * row['invalid_rate']:.1f}% | {row['avg_actions_per_instance']:.2f} | {row['avg_attempts']:.2f} |"
        )
    lines += ["", "## Paired Hierarchical Bootstrap", ""]
    for name, values in comparisons.items():
        success = values["success"]
        invalid = values["invalid"]
        turns = values["turns"]
        lines.append(
            f"- **{name}:** success {100 * success[0]:+.1f} pp "
            f"[{100 * success[1]:+.1f}, {100 * success[2]:+.1f}]; "
            f"invalid {100 * invalid[0]:+.1f} pp [{100 * invalid[1]:+.1f}, {100 * invalid[2]:+.1f}]; "
            f"actions {turns[0]:+.2f} [{turns[1]:+.2f}, {turns[2]:+.2f}]."
        )
    lines += ["", "## Accepted Rule", "", gate["selected_rule"] if gate["accepted"] else "No update accepted."]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_main(args: argparse.Namespace, output_dir: Path, tasks: list[str]) -> None:
    base_variables = initial_miniwob_variables()
    train = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=tasks,
        seeds=TRAIN_SEEDS,
        method="training_base",
        split="train",
        variables=base_variables,
    )
    candidates = generate_candidates(args, output_dir, train)
    rulepi_variables, gate = select_candidate(args, output_dir, tasks, candidates)
    fixed = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=tasks,
        seeds=TEST_SEEDS,
        method="fixed_prompt",
        split="test",
        variables=base_variables,
    )
    rulepi = evaluate_matrix(
        args=args,
        output_dir=output_dir,
        envs=tasks,
        seeds=TEST_SEEDS,
        method="rulepi",
        split="test",
        variables=rulepi_variables,
    )
    retries = run_retries(args, output_dir, fixed)
    summarize_test(args, output_dir, tasks, fixed, rulepi, retries, gate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/rulepi_miniwob_gptoss120b_t07")
    parser.add_argument("--phase", choices=["smoke", "calibrate", "main", "all"], default="all")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--candidate-temperature", type=float, default=0.2)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--candidate-seed", type=int, default=73_001)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--retry-seed-offset", type=int, default=100_000)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=88_001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preregistered_config.json", vars(args))
    if args.phase in {"smoke", "all"}:
        run_smoke(args, output_dir)
    tasks: list[str] = []
    if args.phase in {"calibrate", "all"}:
        tasks = run_calibration(args, output_dir)
    if args.phase in {"main", "all"}:
        if not tasks:
            manifest_path = output_dir / "frozen_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError("Run calibration before the main phase")
            tasks = list(json.loads(manifest_path.read_text(encoding="utf-8"))["tasks"])
        run_main(args, output_dir, tasks)


if __name__ == "__main__":
    main()
