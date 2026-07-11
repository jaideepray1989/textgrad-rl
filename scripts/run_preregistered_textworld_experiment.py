"""Run the preregistered automatic TextGrad-RL TextWorld experiment.

The runner separates train, validation, and test procedural game seeds. It
creates candidates from train trajectories only, evaluates candidates on
validation only, then freezes fixed, ungated, and gated policies for test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textgrad_rl.benchmarks.textworld_24_suite import TextWorldSpec, default_specs, ensure_games
from textgrad_rl.benchmarks.textworld_slm_suites import (
    initial_variables,
    run_textworld_episode,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


FAMILY_ORDER = ("tw-simple", "tw-coin_collector", "tw-treasure_hunter", "tw-cooking")
METHODS = ("fixed_prompt", "ungated_textgrad", "validation_gated_textgrad")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260709
FORBIDDEN_CANDIDATE_TERMS = (
    "seed",
    "hidden state",
    "oracle",
    "walkthrough",
    "policy_commands",
    "benchmark id",
    "game file",
    "simple_",
    "coin_level",
    "treasure_level",
    "cooking_recipe",
)


class CandidateGenerationError(ValueError):
    """Raised when raw generator output cannot become a valid candidate pool."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class TextWorldChatModel:
    """Minimal TextWorld-specific OpenAI-compatible actor client."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, watchdog_seconds: int) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.watchdog_seconds = watchdog_seconds

    def complete(self, prompt: str) -> str:
        return complete_chat(
            base_url=self.base_url,
            model=self.model,
            system=(
                "You choose one action in a TextWorld game. Return exactly one bracketed number selecting a visible "
                "admissible action, such as [1]. Do not explain."
            ),
            user=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            watchdog_seconds=self.watchdog_seconds,
        )


def evaluate_episode_worker(
    spec: TextWorldSpec,
    game_path: Path,
    method: str,
    split: str,
    actor_config: tuple[str, str, float, int, int],
    variables: dict[str, TextVariable],
    max_steps: int,
) -> tuple[TextWorldSpec, Any]:
    """Run one TextWorld episode in an isolated process.

    TextWorld's Inform7 parser is not thread-safe, while independent processes
    can safely share the local model server.
    """

    actor = TextWorldChatModel(*actor_config)
    record = run_textworld_episode(
        spec=spec,
        game_path=game_path,
        method=method,
        split=split,
        model=actor,
        variables=variables,
        max_steps=max_steps,
        model_call_timeout=0,
    )
    return spec, record


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def record_score(record: dict[str, Any]) -> float:
    return (
        float(record["reward"])
        + 0.5 * float(bool(record["success"]))
        - 0.4 * float(bool(record["invalid_action"]))
        - 0.2 * float(bool(record["parse_failed"]))
        - 0.1 * float(bool(record["repeated_actions"]))
        - 0.004 * float(record["turns"])
    )


def select_specs(specs: list[TextWorldSpec], *, offset: int, count: int) -> list[TextWorldSpec]:
    selected: list[TextWorldSpec] = []
    for family in FAMILY_ORDER:
        family_specs = [spec for spec in specs if spec.family == family]
        if len(family_specs) != 6:
            raise ValueError(f"Expected six specs for {family}, found {len(family_specs)}")
        selected.extend(family_specs[(offset + index) % len(family_specs)] for index in range(count))
    return selected


def run_records(
    *,
    specs: list[TextWorldSpec],
    game_paths: dict[str, Path],
    method: str,
    split: str,
    actor: TextWorldChatModel,
    variables: dict[str, TextVariable],
    max_steps: int,
    actor_parallelism: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    completed = {(row["problem_id"], int(row["seed"])): row for row in read_jsonl(output_path)}
    pending = [spec for spec in specs if (spec.spec_id, spec.seed) not in completed]

    actor_config = (
        actor.base_url,
        actor.model,
        actor.temperature,
        actor.max_tokens,
        actor.watchdog_seconds,
    )

    if actor_parallelism == 1:
        results = (
            evaluate_episode_worker(
                spec,
                game_paths[spec.spec_id],
                method,
                split,
                actor_config,
                variables,
                max_steps,
            )
            for spec in pending
        )
        for spec, record in results:
            append_jsonl(output_path, record)
            completed[(spec.spec_id, spec.seed)] = {}
            print(
                f"{method}/{split}/{spec.spec_id}: success={int(record.success)} "
                f"reward={record.reward:.3f} turns={record.turns}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=actor_parallelism) as executor:
            futures = [
                executor.submit(
                    evaluate_episode_worker,
                    spec,
                    game_paths[spec.spec_id],
                    method,
                    split,
                    actor_config,
                    variables,
                    max_steps,
                )
                for spec in pending
            ]
            for future in as_completed(futures):
                spec, record = future.result()
                append_jsonl(output_path, record)
                completed[(spec.spec_id, spec.seed)] = {}
                print(
                    f"{method}/{split}/{spec.spec_id}: success={int(record.success)} "
                    f"reward={record.reward:.3f} turns={record.turns}",
                    flush=True,
                )
    records = read_jsonl(output_path)
    expected = {(spec.spec_id, spec.seed) for spec in specs}
    found = {(row["problem_id"], int(row["seed"])) for row in records}
    if found != expected or len(records) != len(found):
        raise RuntimeError(f"Incomplete record set in {output_path}")
    return records


def trajectory_evidence(records: list[dict[str, Any]], limit: int = 6) -> str:
    failures = [row for row in records if not row["success"] or row["invalid_action"]]
    selected = failures[:limit] or records[:limit]
    summaries = []
    for row in selected:
        tail = " -> ".join(row["actions"][-6:]) if row["actions"] else "<none>"
        summaries.append(
            f"{row['family']}/{row['problem_id']}: success={row['success']}; reward={row['reward']:.2f}; "
            f"turns={row['turns']}; invalid={row['invalid_action']}; repeated={row['repeated_actions']}; "
            f"failure={row['failure_reason']}; tail={tail}"
        )
    return "\n".join(summaries)


def complete_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    watchdog_seconds: int,
    json_object: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=watchdog_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
    return str(data["choices"][0]["message"]["content"]).strip()


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
    raise ValueError("Candidate generator did not return JSON")


def normalize_candidate(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("policy_update") or value.get("rule") or value.get("text")
    if not isinstance(value, str):
        raise ValueError("Candidate must be a JSON string or object with policy_update")
    candidate = re.sub(r"\s+", " ", value).strip()
    if candidate.lower().startswith("add a rule:"):
        candidate = candidate.split(":", 1)[1].strip()
    if not candidate.endswith("."):
        candidate += "."
    if not 40 <= len(candidate) <= 700:
        raise ValueError(f"Candidate has invalid length: {len(candidate)}")
    lowered = candidate.lower()
    blocked = [term for term in FORBIDDEN_CANDIDATE_TERMS if term in lowered]
    if blocked:
        raise ValueError(f"Candidate contains forbidden terms: {blocked}")
    return candidate


def generate_candidates(
    *,
    args: argparse.Namespace,
    evidence: str,
) -> tuple[str, list[str]]:
    system = (
        "You are a policy-improvement critic for a text-game agent. Propose general, reusable policy rules only. "
        "Never use hidden state, random seeds, game files, oracle walkthroughs, task IDs, or object-specific commands."
    )
    user = (
        "The frozen actor selects an index from visible admissible TextWorld commands. Here are training traces:\n\n"
        f"{evidence}\n\n"
        f"Return exactly one JSON object with key 'candidates' containing exactly {args.candidate_count} distinct strings. "
        "Each string is one concise, general policy update that can help with ordered objectives, exploration, or recipes. "
        "Do not include markdown, rationale, task names, seeds, hidden-state references, or example commands."
    )
    raw = complete_chat(
        base_url=args.generator_base_url,
        model=args.generator_model,
        system=system,
        user=user,
        temperature=args.generator_temperature,
        max_tokens=args.generator_max_tokens,
        watchdog_seconds=args.watchdog_seconds,
        json_object=True,
    )
    try:
        parsed = first_json_value(raw)
        candidates = parsed.get("candidates") if isinstance(parsed, dict) else parsed
        if not isinstance(candidates, list) or len(candidates) != args.candidate_count:
            raise ValueError(f"Expected exactly {args.candidate_count} candidates")
        normalized = [normalize_candidate(item) for item in candidates]
        if len({item.lower() for item in normalized}) != len(normalized):
            raise ValueError("Candidate generator returned duplicate candidates")
        return raw, normalized
    except ValueError as exc:
        raise CandidateGenerationError(str(exc), raw) from exc


def candidate_variables(
    *,
    base: dict[str, TextVariable],
    candidate: str,
    outer_index: int,
    candidate_index: int,
    evidence: str,
) -> dict[str, TextVariable]:
    gradient = TextualGradient(
        target_variable_name="text_game_slm_policy",
        failure_mode=f"outer_{outer_index:02d}_candidate_{candidate_index:02d}",
        evidence_from_trajectory=evidence,
        gradient_text="Convert the observed failure pattern into a general reusable policy rule.",
        suggested_edit=candidate,
        confidence=0.8,
        forbidden_shortcuts=["hidden state", "seeds", "oracle walkthroughs", "task identifiers"],
    )
    updated = TextualGradientDescent(max_prompt_chars=3200, max_rules_per_step=1).step(
        base,
        [gradient],
        constraints=[
            "must not use hidden state",
            "must not hardcode seeds",
            "must not use oracle walkthroughs",
            "must not use task identifiers",
        ],
    )
    if updated["text_game_slm_policy"].value == base["text_game_slm_policy"].value:
        raise ValueError("Candidate was rejected by the textual optimizer")
    return updated


def validation_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "score": mean([record_score(record) for record in records]),
        "success": mean([float(bool(record["success"])) for record in records]),
        "invalid": mean([float(bool(record["invalid_action"])) for record in records]),
        "parse_failed": mean([float(bool(record["parse_failed"])) for record in records]),
        "turns": mean([float(record["turns"]) for record in records]),
    }


def outer_seeds(seed: int, outer_index: int) -> dict[str, int]:
    base = seed + outer_index * 100_000
    return {"train": base + 10_000, "validation": base + 40_000, "test": base + 70_000}


def run_outer(args: argparse.Namespace, outer_index: int, actor: TextWorldChatModel) -> dict[str, Any]:
    outer_dir = Path(args.output_dir) / f"outer_{outer_index:02d}"
    outer_dir.mkdir(parents=True, exist_ok=True)
    seeds = outer_seeds(args.seed, outer_index)
    all_train_specs = default_specs(seeds["train"])
    all_val_specs = default_specs(seeds["validation"])
    all_test_specs = default_specs(seeds["test"])
    train_specs = select_specs(all_train_specs, offset=(2 * outer_index) % 6, count=args.train_per_family)
    val_specs = select_specs(all_val_specs, offset=(2 * outer_index + 2) % 6, count=args.val_per_family)
    train_paths = ensure_games(train_specs, outer_dir / "games" / "train")
    val_paths = ensure_games(val_specs, outer_dir / "games" / "validation")
    test_paths = ensure_games(all_test_specs, outer_dir / "games" / "test")
    split_manifest = {
        "outer_index": outer_index,
        "seeds": seeds,
        "train_specs": [spec.spec_id for spec in train_specs],
        "validation_specs": [spec.spec_id for spec in val_specs],
        "test_specs": [spec.spec_id for spec in all_test_specs],
    }
    write_json(outer_dir / "split_manifest.json", split_manifest)

    base = initial_variables()
    write_json(outer_dir / "fixed_prompt" / "text_variables.json", base)
    train_records = run_records(
        specs=train_specs,
        game_paths=train_paths,
        method="train_base",
        split="train",
        actor=actor,
        variables=base,
        max_steps=args.max_steps,
        actor_parallelism=args.actor_parallelism,
        output_path=outer_dir / "train" / "fixed_prompt.jsonl",
    )
    evidence = trajectory_evidence(train_records)
    candidates_path = outer_dir / "candidates.json"
    if candidates_path.exists():
        saved = read_json(candidates_path)
        raw_candidate_output = str(saved["raw_output"])
        candidates = [str(item) for item in saved["candidates"]]
    else:
        try:
            raw_candidate_output, candidates = generate_candidates(args=args, evidence=evidence)
        except CandidateGenerationError as exc:
            write_json(
                outer_dir / "candidate_generation_failure.json",
                {"error": str(exc), "raw_output": exc.raw_output, "train_evidence": evidence},
            )
            raise
        write_json(
            candidates_path,
            {
                "raw_output": raw_candidate_output,
                "candidates": candidates,
                "train_evidence": evidence,
                "generator_model": args.generator_model,
                "generator_temperature": args.generator_temperature,
            },
        )
    candidate_policies = [
        candidate_variables(
            base=base,
            candidate=candidate,
            outer_index=outer_index,
            candidate_index=index,
            evidence=evidence,
        )
        for index, candidate in enumerate(candidates)
    ]
    for index, variables in enumerate(candidate_policies):
        write_json(outer_dir / "candidates" / f"candidate_{index:02d}_variables.json", variables)

    base_val = run_records(
        specs=val_specs,
        game_paths=val_paths,
        method="fixed_prompt",
        split="validation",
        actor=actor,
        variables=base,
        max_steps=args.max_steps,
        actor_parallelism=args.actor_parallelism,
        output_path=outer_dir / "validation" / "fixed_prompt.jsonl",
    )
    base_metrics = validation_metrics(base_val)
    candidate_metrics: list[dict[str, Any]] = []
    for index, variables in enumerate(candidate_policies):
        records = run_records(
            specs=val_specs,
            game_paths=val_paths,
            method=f"candidate_{index:02d}",
            split="validation",
            actor=actor,
            variables=variables,
            max_steps=args.max_steps,
            actor_parallelism=args.actor_parallelism,
            output_path=outer_dir / "validation" / "candidates" / f"candidate_{index:02d}.jsonl",
        )
        candidate_metrics.append({"candidate_index": index, **validation_metrics(records)})
    selected = max(candidate_metrics, key=lambda row: (float(row["score"]), -int(row["candidate_index"])))
    accepted = (
        selected["score"] >= base_metrics["score"] + args.min_validation_score_delta
        and selected["invalid"] <= base_metrics["invalid"]
        and selected["parse_failed"] <= base_metrics["parse_failed"]
    )
    gate = {
        "accepted": accepted,
        "base_validation": base_metrics,
        "candidates": candidate_metrics,
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_candidate": candidates[int(selected["candidate_index"])],
        "min_validation_score_delta": args.min_validation_score_delta,
    }
    write_json(outer_dir / "gate_decision.json", gate)
    gated = candidate_policies[int(selected["candidate_index"])] if accepted else base
    write_json(outer_dir / "validation_gated_textgrad" / "text_variables.json", gated)
    write_json(outer_dir / "ungated_textgrad" / "text_variables.json", candidate_policies[0])

    test_policies = {
        "fixed_prompt": base,
        "ungated_textgrad": candidate_policies[0],
        "validation_gated_textgrad": gated,
    }
    test_records: dict[str, list[dict[str, Any]]] = {}
    for method, variables in test_policies.items():
        test_records[method] = run_records(
            specs=all_test_specs,
            game_paths=test_paths,
            method=method,
            split="test",
            actor=actor,
            variables=variables,
            max_steps=args.max_steps,
            actor_parallelism=args.actor_parallelism,
            output_path=outer_dir / "test" / f"{method}.jsonl",
        )
    outer_summary = {
        "outer_index": outer_index,
        "gate": gate,
        "test": {method: validation_metrics(records) for method, records in test_records.items()},
    }
    write_json(outer_dir / "outer_summary.json", outer_summary)
    return outer_summary


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(fraction * len(ordered)))))
    return ordered[index]


def bootstrap(
    by_outer: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]],
    statistic: Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float | None],
) -> tuple[float, float, float]:
    observed_pairs = [pair for outer in sorted(by_outer) for pair in by_outer[outer]]
    observed = statistic(observed_pairs)
    if observed is None:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(BOOTSTRAP_SEED + len(observed_pairs))
    outer_ids = sorted(by_outer)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for _ in outer_ids:
            outer = outer_ids[rng.randrange(len(outer_ids))]
            pairs = by_outer[outer]
            sample.extend(pairs[rng.randrange(len(pairs))] for _ in pairs)
        estimate = statistic(sample)
        if estimate is not None:
            estimates.append(estimate)
    return observed, percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_test_records(
    run_dir: Path,
    left_method: str,
    right_method: str,
) -> dict[int, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for outer_dir in sorted(run_dir.glob("outer_*")):
        outer_index = int(outer_dir.name.rsplit("_", 1)[1])
        base = {row["problem_id"]: row for row in read_jsonl(outer_dir / "test" / f"{left_method}.jsonl")}
        right = {row["problem_id"]: row for row in read_jsonl(outer_dir / "test" / f"{right_method}.jsonl")}
        if base.keys() != right.keys() or len(base) != 24:
            raise RuntimeError(f"Invalid held-out test pairing in {outer_dir}")
        grouped[outer_index] = [(base[task], right[task]) for task in sorted(base)]
    return grouped


def rate(which: int) -> Callable[[list[tuple[dict[str, Any], dict[str, Any]]]], float]:
    return lambda pairs: mean([float(bool(pair[which]["success"])) for pair in pairs])


def success_delta(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    return mean([float(bool(new["success"])) - float(bool(old["success"])) for old, new in pairs])


def common_step_delta(pairs: list[tuple[dict[str, Any], dict[str, Any]]], median: bool) -> float | None:
    values = [
        float(old["turns"]) - float(new["turns"])
        for old, new in pairs
        if old["success"] and new["success"]
    ]
    if not values:
        return None
    return float(statistics.median(values) if median else mean(values))


def ci_text(value: tuple[float, float, float], *, percent: bool = False, pp: bool = False) -> str:
    observed, low, high = value
    if pp:
        return f"{100 * observed:+.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]"
    if percent:
        return f"{100 * observed:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"
    return f"{observed:+.2f} [{low:+.2f}, {high:+.2f}]"


def write_final_report(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    run_dir = Path(args.output_dir)
    comparisons = {
        "validation_gated_textgrad - fixed_prompt": paired_test_records(
            run_dir, "fixed_prompt", "validation_gated_textgrad"
        ),
        "ungated_textgrad - fixed_prompt": paired_test_records(
            run_dir, "fixed_prompt", "ungated_textgrad"
        ),
        "validation_gated_textgrad - ungated_textgrad": paired_test_records(
            run_dir, "ungated_textgrad", "validation_gated_textgrad"
        ),
    }
    rows: list[dict[str, Any]] = []
    for comparison, grouped in comparisons.items():
        pairs = [pair for outer in grouped for pair in grouped[outer]]
        rows.append(
            {
                "comparison": comparison,
                "outer_repetitions": len(grouped),
                "paired_test_episodes": len(pairs),
                "fixed_success": bootstrap(grouped, rate(0)),
                "method_success": bootstrap(grouped, rate(1)),
                "success_delta": bootstrap(grouped, success_delta),
                "common_successes": sum(old["success"] and new["success"] for old, new in pairs),
                "mean_step_decrease": bootstrap(grouped, lambda sample: common_step_delta(sample, False)),
                "median_step_decrease": bootstrap(grouped, lambda sample: common_step_delta(sample, True)),
            }
        )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "comparison": row["comparison"],
                "outer_repetitions": row["outer_repetitions"],
                "paired_test_episodes": row["paired_test_episodes"],
                "fixed_success": row["fixed_success"][0],
                "fixed_success_ci_low": row["fixed_success"][1],
                "fixed_success_ci_high": row["fixed_success"][2],
                "method_success": row["method_success"][0],
                "method_success_ci_low": row["method_success"][1],
                "method_success_ci_high": row["method_success"][2],
                "success_delta": row["success_delta"][0],
                "success_delta_ci_low": row["success_delta"][1],
                "success_delta_ci_high": row["success_delta"][2],
                "common_successes": row["common_successes"],
                "mean_step_decrease": row["mean_step_decrease"][0],
                "mean_step_decrease_ci_low": row["mean_step_decrease"][1],
                "mean_step_decrease_ci_high": row["mean_step_decrease"][2],
                "median_step_decrease": row["median_step_decrease"][0],
                "median_step_decrease_ci_low": row["median_step_decrease"][1],
                "median_step_decrease_ci_high": row["median_step_decrease"][2],
            }
        )
    output_csv = run_dir / "paired_bootstrap_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    accepted = sum(bool(summary["gate"]["accepted"]) for summary in summaries)
    lines = [
        "# Preregistered TextWorld TextGrad-RL Results",
        "",
        f"Actor: `{args.actor_model}` at temperature `{args.actor_temperature}`. Candidate generator: `{args.generator_model}` at temperature `{args.generator_temperature}`.",
        f"Outer repetitions: `{args.outer_repetitions}`. Each repetition uses disjoint procedural train, validation, and full 24-game test seeds. Test horizon: `{args.max_steps}` actions.",
        "",
        "| Comparison | Fixed success | Method success | Paired success delta | Common successes | Mean step decrease | Median step decrease |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {ci_text(row['fixed_success'], percent=True)} | "
            f"{ci_text(row['method_success'], percent=True)} | {ci_text(row['success_delta'], pp=True)} | "
            f"{row['common_successes']} | {ci_text(row['mean_step_decrease'])} | "
            f"{ci_text(row['median_step_decrease'])} |"
        )
    lines.extend(
        [
            "",
            f"Validation gate accepted `{accepted}/{len(summaries)}` outer-repetition updates.",
            "A positive step decrease means the compared method used fewer actions on paired episodes both methods solved.",
            "",
            "## Preregistered Decision",
            "",
            "The planned positive result requires the lower confidence bound for `validation_gated_textgrad - fixed_prompt` success to exceed zero and a non-negative point estimate for `validation_gated_textgrad - ungated_textgrad`.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "outer_summaries.json", summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/preregistered_textworld_qwen14b_t0_3outer")
    parser.add_argument("--actor-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--actor-model", default="qwen2.5:14b")
    parser.add_argument("--actor-temperature", type=float, default=0.0)
    parser.add_argument("--generator-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--generator-model", default="gpt-oss:20b")
    parser.add_argument("--generator-temperature", type=float, default=0.7)
    parser.add_argument("--generator-max-tokens", type=int, default=1200)
    parser.add_argument("--outer-repetitions", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-per-family", type=int, default=2)
    parser.add_argument("--val-per-family", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--actor-parallelism", type=int, default=1)
    parser.add_argument("--watchdog-seconds", type=int, default=300)
    parser.add_argument("--min-validation-score-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=93001)
    args = parser.parse_args()
    if args.outer_repetitions < 1 or args.candidate_count < 2 or args.actor_parallelism < 1:
        raise SystemExit("outer-repetitions, candidate-count, and actor-parallelism must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "actor_model": args.actor_model,
            "actor_temperature": args.actor_temperature,
            "generator_model": args.generator_model,
            "generator_temperature": args.generator_temperature,
            "outer_repetitions": args.outer_repetitions,
            "candidate_count": args.candidate_count,
            "train_per_family": args.train_per_family,
            "val_per_family": args.val_per_family,
            "max_steps": args.max_steps,
            "actor_parallelism": args.actor_parallelism,
            "watchdog_seconds": args.watchdog_seconds,
            "seed": args.seed,
            "preregistration": "PREREGISTERED_TEXTWORLD_EXPERIMENT.md",
        },
    )
    write_json(output_dir / "environment_info.json", environment_info())
    actor = TextWorldChatModel(
        args.actor_base_url,
        args.actor_model,
        args.actor_temperature,
        16,
        args.watchdog_seconds,
    )
    summaries = [run_outer(args, outer_index, actor) for outer_index in range(args.outer_repetitions)]
    write_final_report(args, summaries)
    print(f"Preregistered experiment artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
