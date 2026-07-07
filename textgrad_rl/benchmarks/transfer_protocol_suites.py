"""Local transfer-protocol probes for external TextGrad-RL benchmark claims.

These probes are not replacements for official WebArena/WorkArena, tau-bench,
or SWE-bench Docker/service runs. They are small, reproducible transfer
protocols that use the shared external trajectory adapter and make the source
to target transfer assumptions explicit:

* MiniWoB-style browser failures -> WebArena/WorkArena-style browser tasks.
* tau-bench retail tool policy -> airline/banking tool policy.
* SWE-bench dev repairs -> SWE-bench Lite-style held-out repairs.

The official backends are intentionally reported separately in
``official_backend_status.json`` so a local protocol table cannot be confused
with a heavyweight benchmark execution.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.external_adapter import (
    ExternalAgentEpisode,
    ExternalAgentStep,
    external_episode_score,
)
from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


@dataclass(frozen=True)
class TransferRule:
    rule_id: str
    text: str
    target_variable: str


@dataclass(frozen=True)
class TransferTask:
    suite: str
    task_id: str
    source: str
    target: str
    split: str
    category: str
    required_rules: tuple[str, ...]
    safety_rules: tuple[str, ...] = ()
    base_turns: int = 3


@dataclass
class TransferRecord:
    suite: str
    method: str
    task_id: str
    source: str
    target: str
    split: str
    category: str
    success: bool
    reward: float
    invalid_action: bool
    repeated_actions: bool
    turns: int
    matched_rules: list[str]
    missing_rules: list[str]
    failure_reason: str


TRANSFER_RULES: dict[str, TransferRule] = {
    "fill_before_submit": TransferRule(
        "fill_before_submit",
        "Fill all required fields before submitting or saving a browser form.",
        "browser_policy",
    ),
    "select_then_submit": TransferRule(
        "select_then_submit",
        "After selecting required checkbox, radio, filter, or list options, click the final Submit, Save, Apply, or Checkout control exactly once.",
        "browser_policy",
    ),
    "avoid_repeated_browser_action": TransferRule(
        "avoid_repeated_browser_action",
        "Avoid repeating the same browser action after it fails or stops changing the page; choose the next required control instead.",
        "browser_policy",
    ),
    "search_before_detail": TransferRule(
        "search_before_detail",
        "Use search or filtering controls to locate the target item before opening or editing its detail page.",
        "browser_policy",
    ),
    "confirm_site_state": TransferRule(
        "confirm_site_state",
        "Confirm that multi-page web navigation has landed on the intended site state before taking the final irreversible action.",
        "browser_policy",
    ),
    "verify_identity": TransferRule(
        "verify_identity",
        "Verify the customer, account, order, or reservation identity before any tool mutation.",
        "tool_policy",
    ),
    "lookup_before_mutation": TransferRule(
        "lookup_before_mutation",
        "Look up the current database record and relevant constraints before calling update, cancel, refund, or transfer tools.",
        "tool_policy",
    ),
    "check_policy": TransferRule(
        "check_policy",
        "Check domain policy and eligibility before irreversible customer-service actions.",
        "tool_policy",
    ),
    "confirm_final_state": TransferRule(
        "confirm_final_state",
        "After tool updates, confirm the final state against the user's requested goal before responding.",
        "tool_policy",
    ),
    "explain_domain_fee": TransferRule(
        "explain_domain_fee",
        "Explain domain-specific fees, fare differences, or account charges before finalizing customer-facing changes.",
        "tool_policy",
    ),
    "reproduce_failure": TransferRule(
        "reproduce_failure",
        "Reproduce or inspect the failing test, traceback, or issue condition before editing code.",
        "coding_policy",
    ),
    "minimal_patch": TransferRule(
        "minimal_patch",
        "Make the smallest root-cause code patch rather than broad rewrites.",
        "coding_policy",
    ),
    "targeted_tests": TransferRule(
        "targeted_tests",
        "Run targeted regression tests for the changed behavior after patching.",
        "coding_policy",
    ),
    "preserve_api": TransferRule(
        "preserve_api",
        "Preserve public APIs and backward compatibility unless the issue explicitly asks for a breaking change.",
        "coding_policy",
    ),
    "trace_cross_file_flow": TransferRule(
        "trace_cross_file_flow",
        "For multi-file bugs, trace the data flow across callers and callees before patching.",
        "coding_policy",
    ),
    "dependency_edge_case": TransferRule(
        "dependency_edge_case",
        "Check dependency-version, async, or concurrency edge cases when the issue mentions environment-specific failures.",
        "coding_policy",
    ),
}


BROWSER_TASKS = [
    TransferTask("browser_transfer", "miniwob_train_select_submit", "MiniWoB", "MiniWoB", "train", "selection", ("select_then_submit",), ("avoid_repeated_browser_action",), 3),
    TransferTask("browser_transfer", "miniwob_train_form_submit", "MiniWoB", "MiniWoB", "train", "forms", ("fill_before_submit",), (), 3),
    TransferTask("browser_transfer", "miniwob_train_search_result", "MiniWoB", "MiniWoB", "train", "search", ("search_before_detail",), (), 4),
    TransferTask("browser_transfer", "miniwob_train_repeat_recovery", "MiniWoB", "MiniWoB", "train", "recovery", ("avoid_repeated_browser_action",), ("avoid_repeated_browser_action",), 4),
    TransferTask("browser_transfer", "webarena_shop_filter_checkout", "MiniWoB", "WebArena", "test", "shopping", ("search_before_detail", "select_then_submit"), ("avoid_repeated_browser_action",), 7),
    TransferTask("browser_transfer", "webarena_reddit_update_profile", "MiniWoB", "WebArena", "test", "forum", ("fill_before_submit", "avoid_repeated_browser_action"), ("avoid_repeated_browser_action",), 5),
    TransferTask("browser_transfer", "webarena_gitlab_issue_label", "MiniWoB", "WebArena", "test", "collaboration", ("search_before_detail", "select_then_submit"), ("avoid_repeated_browser_action",), 7),
    TransferTask("browser_transfer", "webarena_cms_publish_edit", "MiniWoB", "WebArena", "test", "cms", ("fill_before_submit", "select_then_submit"), ("avoid_repeated_browser_action",), 6),
    TransferTask("browser_transfer", "workarena_incident_reassign", "MiniWoB", "WorkArena", "test", "service", ("search_before_detail", "select_then_submit", "confirm_site_state"), ("avoid_repeated_browser_action",), 8),
    TransferTask("browser_transfer", "workarena_knowledge_article", "MiniWoB", "WorkArena", "test", "knowledge", ("fill_before_submit", "select_then_submit"), ("avoid_repeated_browser_action",), 6),
]


TAU_TASKS = [
    TransferTask("tau_transfer", "retail_train_cancel_order", "tau-retail", "tau-retail", "train", "retail", ("lookup_before_mutation", "check_policy"), ("verify_identity", "check_policy"), 4),
    TransferTask("tau_transfer", "retail_train_refund_item", "tau-retail", "tau-retail", "train", "retail", ("verify_identity", "lookup_before_mutation", "check_policy"), ("verify_identity", "check_policy"), 5),
    TransferTask("tau_transfer", "retail_train_update_address", "tau-retail", "tau-retail", "train", "retail", ("verify_identity", "lookup_before_mutation", "confirm_final_state"), ("verify_identity",), 5),
    TransferTask("tau_transfer", "airline_change_flight", "tau-retail", "tau-airline", "test", "airline", ("verify_identity", "lookup_before_mutation", "check_policy", "confirm_final_state", "explain_domain_fee"), ("verify_identity", "check_policy"), 7),
    TransferTask("tau_transfer", "airline_cancel_refund", "tau-retail", "tau-airline", "test", "airline", ("verify_identity", "lookup_before_mutation", "check_policy"), ("verify_identity", "check_policy"), 6),
    TransferTask("tau_transfer", "airline_add_bag", "tau-retail", "tau-airline", "test", "airline", ("lookup_before_mutation", "confirm_final_state"), (), 5),
    TransferTask("tau_transfer", "banking_wire_transfer", "tau-retail", "tau-banking", "test", "banking", ("verify_identity", "lookup_before_mutation", "check_policy", "confirm_final_state"), ("verify_identity", "check_policy"), 7),
    TransferTask("tau_transfer", "banking_card_dispute", "tau-retail", "tau-banking", "test", "banking", ("verify_identity", "lookup_before_mutation", "check_policy"), ("verify_identity", "check_policy"), 6),
    TransferTask("tau_transfer", "banking_update_contact", "tau-retail", "tau-banking", "test", "banking", ("verify_identity", "lookup_before_mutation", "confirm_final_state"), ("verify_identity",), 5),
]


SWE_TASKS = [
    TransferTask("swe_transfer", "dev_traceback_type_error", "SWE-bench-dev", "SWE-bench-dev", "train", "dev", ("reproduce_failure", "minimal_patch", "targeted_tests"), (), 5),
    TransferTask("swe_transfer", "dev_api_regression", "SWE-bench-dev", "SWE-bench-dev", "train", "dev", ("minimal_patch", "preserve_api", "targeted_tests"), ("preserve_api",), 5),
    TransferTask("swe_transfer", "dev_cross_file_bug", "SWE-bench-dev", "SWE-bench-dev", "train", "dev", ("reproduce_failure", "trace_cross_file_flow", "minimal_patch"), (), 6),
    TransferTask("swe_transfer", "lite_datetime_parser", "SWE-bench-dev", "SWE-bench-Lite", "test", "lite", ("reproduce_failure", "minimal_patch", "targeted_tests"), (), 5),
    TransferTask("swe_transfer", "lite_public_api_alias", "SWE-bench-dev", "SWE-bench-Lite", "test", "lite", ("minimal_patch", "preserve_api", "targeted_tests"), ("preserve_api",), 5),
    TransferTask("swe_transfer", "lite_multi_file_cache", "SWE-bench-dev", "SWE-bench-Lite", "test", "lite", ("reproduce_failure", "trace_cross_file_flow", "minimal_patch", "targeted_tests", "dependency_edge_case"), (), 7),
    TransferTask("swe_transfer", "lite_test_selection", "SWE-bench-dev", "SWE-bench-Lite", "test", "lite", ("reproduce_failure", "targeted_tests"), (), 4),
    TransferTask("swe_transfer", "lite_backward_compat", "SWE-bench-dev", "SWE-bench-Lite", "test", "lite", ("minimal_patch", "preserve_api"), ("preserve_api",), 4),
]


TASKS_BY_SUITE = {
    "browser_transfer": BROWSER_TASKS,
    "tau_transfer": TAU_TASKS,
    "swe_transfer": SWE_TASKS,
}


INITIAL_RULES = {
    "browser_transfer": ("fill_before_submit",),
    "tau_transfer": ("lookup_before_mutation",),
    "swe_transfer": ("minimal_patch",),
}


def initial_variables(suite: str) -> dict[str, TextVariable]:
    target = target_variable_for_suite(suite)
    lines = [
        "Follow visible instructions, use only allowed actions, and avoid hidden benchmark answers.",
        "",
        "Seed rules:",
    ]
    for rule_id in INITIAL_RULES[suite]:
        lines.append(f"- {TRANSFER_RULES[rule_id].text}")
    return {
        target: TextVariable(
            name=target,
            value="\n".join(lines),
            role_description=f"{suite} transfer policy.",
            max_chars=2400,
        )
    }


def target_variable_for_suite(suite: str) -> str:
    if suite == "browser_transfer":
        return "browser_policy"
    if suite == "tau_transfer":
        return "tool_policy"
    if suite == "swe_transfer":
        return "coding_policy"
    raise ValueError(f"Unknown transfer suite: {suite}")


def policy_rule_ids(variables: dict[str, TextVariable]) -> set[str]:
    text = "\n".join(variable.value.lower() for variable in variables.values())
    return {rule_id for rule_id, rule in TRANSFER_RULES.items() if rule.text.lower() in text}


def evaluate_task(task: TransferTask, method: str, variables: dict[str, TextVariable], seed: int) -> tuple[TransferRecord, ExternalAgentEpisode]:
    known_rules = policy_rule_ids(variables)
    matched = [rule_id for rule_id in task.required_rules if rule_id in known_rules]
    missing = [rule_id for rule_id in task.required_rules if rule_id not in known_rules]
    reward = len(matched) / len(task.required_rules) if task.required_rules else 1.0
    success = not missing
    invalid = any(rule_id not in known_rules for rule_id in task.safety_rules)
    repeated = "avoid_repeated_browser_action" in missing
    turns = task.base_turns + 2 * len(missing) + (2 if repeated else 0)
    failure_reason = "" if success and not invalid else "missing transfer rules: " + ", ".join(missing)
    record = TransferRecord(
        suite=task.suite,
        method=method,
        task_id=task.task_id,
        source=task.source,
        target=task.target,
        split=task.split,
        category=task.category,
        success=success,
        reward=reward,
        invalid_action=invalid,
        repeated_actions=repeated,
        turns=turns,
        matched_rules=matched,
        missing_rules=missing,
        failure_reason=failure_reason,
    )
    steps = [
        ExternalAgentStep(
            observation=f"{task.target}:{task.category}: requires {', '.join(task.required_rules)}",
            action=f"apply_policy_rules:{','.join(sorted(known_rules)) or 'none'}",
            reward=reward,
            info={"matched_rules": matched, "missing_rules": missing},
        )
    ]
    episode = ExternalAgentEpisode(
        benchmark=task.suite,
        task_id=task.task_id,
        split=task.split,
        seed=seed,
        success=success,
        reward=reward,
        invalid_action=invalid,
        truncated=False,
        steps=steps,
        target_variable=target_variable_for_suite(task.suite),
        failure_reason=failure_reason,
    )
    return record, episode


def gradients_from_transfer_records(records: list[TransferRecord]) -> list[TextualGradient]:
    gradients: list[TextualGradient] = []
    for record in records:
        if record.success and not record.invalid_action:
            continue
        for rule_id in record.missing_rules:
            rule = TRANSFER_RULES[rule_id]
            gradients.append(
                TextualGradient(
                    target_variable_name=rule.target_variable,
                    failure_mode=f"{record.suite}:{record.category}:{rule_id}",
                    evidence_from_trajectory=(
                        f"{record.task_id} failed on {record.target}; missing {rule_id}; "
                        f"matched={record.matched_rules}"
                    ),
                    gradient_text=f"Transfer source-domain failure into a reusable rule: {rule.text}",
                    suggested_edit=f"Add a rule: {rule.text}",
                    confidence=0.82,
                    forbidden_shortcuts=["inspect hidden labels", "mutate benchmark data", "edit tests to pass"],
                )
            )
    return gradients


def evaluate_records(
    tasks: list[TransferTask],
    method: str,
    variables: dict[str, TextVariable],
    seed: int,
    output_jsonl: Path,
) -> tuple[list[TransferRecord], list[ExternalAgentEpisode]]:
    records: list[TransferRecord] = []
    episodes: list[ExternalAgentEpisode] = []
    if output_jsonl.exists():
        output_jsonl.unlink()
    for index, task in enumerate(tasks):
        record, episode = evaluate_task(task, method, variables, seed + index)
        records.append(record)
        episodes.append(episode)
        append_jsonl(output_jsonl, record)
    return records, episodes


def run_transfer_suite(suite: str, output_dir: Path, seed: int) -> dict[str, Any]:
    suite_dir = output_dir / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    tasks = TASKS_BY_SUITE[suite]
    train_tasks = [task for task in tasks if task.split == "train"]
    test_tasks = [task for task in tasks if task.split == "test"]
    fixed_variables = initial_variables(suite)
    fixed_train, _ = evaluate_records(train_tasks, "fixed_policy", fixed_variables, seed, suite_dir / "fixed_train.jsonl")
    fixed_test, fixed_episodes = evaluate_records(test_tasks, "fixed_policy", fixed_variables, seed + 1000, suite_dir / "fixed_test.jsonl")

    optimizer = TextualGradientDescent(max_prompt_chars=2400, max_rules_per_step=12)
    gradients = gradients_from_transfer_records(fixed_train)
    candidate_variables = optimizer.step(
        fixed_variables,
        gradients,
        constraints=["must not inspect hidden labels", "must not mutate benchmark data", "must not edit tests to pass"],
    )
    old_score = mean([record_score(record) for record in fixed_train])
    candidate_train, _ = evaluate_records(
        train_tasks,
        "textgrad_rl",
        candidate_variables,
        seed + 2000,
        suite_dir / "textgrad_train.jsonl",
    )
    new_score = mean([record_score(record) for record in candidate_train])
    accepted = new_score >= old_score
    textgrad_variables = candidate_variables if accepted else fixed_variables
    textgrad_test, textgrad_episodes = evaluate_records(
        test_tasks,
        "textgrad_rl",
        textgrad_variables,
        seed + 3000,
        suite_dir / "textgrad_test.jsonl",
    )

    write_json(suite_dir / "tasks.json", tasks)
    write_json(suite_dir / "fixed_policy_variables.json", fixed_variables)
    write_json(suite_dir / "textgrad_policy_variables.json", textgrad_variables)
    write_json(suite_dir / "gradients.json", gradients)
    write_json(
        suite_dir / "update_decision.json",
        {
            "accepted": accepted,
            "old_train_score": old_score,
            "new_train_score": new_score,
            "gradient_count": len(gradients),
            "learned_rule_ids": sorted(policy_rule_ids(textgrad_variables) - policy_rule_ids(fixed_variables)),
        },
    )
    summary_rows = [
        summarize_records(suite, "fixed_policy", fixed_test, fixed_episodes, accepted_updates=0),
        summarize_records(suite, "textgrad_rl", textgrad_test, textgrad_episodes, accepted_updates=1 if accepted else 0),
    ]
    write_csv(suite_dir / "summary.csv", summary_rows)
    write_json(suite_dir / "summary.json", summary_rows)
    write_suite_summary(suite_dir / "summary.md", suite, summary_rows, accepted, gradients, textgrad_variables)
    return {"suite": suite, "summary": summary_rows, "accepted": accepted}


def record_score(record: TransferRecord) -> float:
    return record.reward + (0.5 if record.success else 0.0) - (0.4 if record.invalid_action else 0.0) - 0.01 * record.turns


def summarize_records(
    suite: str,
    method: str,
    records: list[TransferRecord],
    episodes: list[ExternalAgentEpisode],
    accepted_updates: int,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "method": method,
        "source": records[0].source if records else "",
        "target": "+".join(sorted({record.target for record in records})),
        "episodes": len(records),
        "success_rate": mean([float(record.success) for record in records]),
        "avg_reward": mean([record.reward for record in records]),
        "invalid_action_rate": mean([float(record.invalid_action) for record in records]),
        "repeated_action_rate": mean([float(record.repeated_actions) for record in records]),
        "avg_turns": mean([record.turns for record in records]),
        "external_score": mean([external_episode_score(episode) for episode in episodes]),
        "accepted_updates": accepted_updates,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_suite_summary(
    path: Path,
    suite: str,
    rows: list[dict[str, Any]],
    accepted: bool,
    gradients: list[TextualGradient],
    variables: dict[str, TextVariable],
) -> None:
    learned = sorted(policy_rule_ids(variables) - set(INITIAL_RULES[suite]))
    lines = [
        f"# {suite} Transfer Probe",
        "",
        f"Accepted TextGrad update: {str(accepted).lower()}",
        f"Gradient count: {len(gradients)}",
        f"Learned rule ids: {', '.join(learned) if learned else 'none'}",
        "",
        "| Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {source} | {target} | {episodes} | {success_rate:.3f} | {avg_reward:.3f} | "
            "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {avg_turns:.2f} | {accepted_updates} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def official_backend_status() -> dict[str, Any]:
    webarena_urls = ["SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE"]
    workarena_vars = ["SNOW_INSTANCE_URL", "SERVICENOW_INSTANCE_URL", "WORKARENA_INSTANCE_URL"]
    llm_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY"]
    return {
        "webarena": {
            "can_run_official_backend": all(os.getenv(name) for name in webarena_urls),
            "missing_url_vars": [name for name in webarena_urls if not os.getenv(name)],
        },
        "workarena": {
            "browsergym_workarena_installed": module_available("browsergym.workarena"),
            "has_servicenow_instance": any(os.getenv(name) for name in workarena_vars),
            "missing_instance_vars": [name for name in workarena_vars if not os.getenv(name)],
        },
        "tau_bench": {
            "has_llm_api_key": any(os.getenv(name) for name in llm_keys),
            "note": "Official tau2/tau3 evaluation usually requires an LLM user simulator or provider key.",
        },
        "swe_bench": {
            "docker_available": shutil.which("docker") is not None,
            "swebench_package_available": module_available("swebench"),
            "note": "Official SWE-bench scoring applies generated patches in Docker and runs repository tests.",
        },
    }


def module_available(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def write_overall_summary(path: Path, suite_results: list[dict[str, Any]], backend_status: dict[str, Any]) -> None:
    lines = [
        "# External Transfer Protocol Results",
        "",
        "These are local, reproducible transfer probes for the requested source->target benchmark protocols. "
        "They are not official WebArena/WorkArena, tau-bench, or SWE-bench leaderboard runs.",
        "",
        "## Transfer Results",
        "",
        "| Transfer | Method | Source | Target | Episodes | Success | Reward | Invalid | Repeated | Turns | Updates |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite_result in suite_results:
        for row in suite_result["summary"]:
            lines.append(
                "| {suite} | {method} | {source} | {target} | {episodes} | {success_rate:.3f} | {avg_reward:.3f} | "
                "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {avg_turns:.2f} | {accepted_updates} |".format(
                    **row
                )
            )
    lines.extend(["", "## Official Backend Status", ""])
    for name, status in backend_status.items():
        lines.append(f"- `{name}`: `{status}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_suites(value: str) -> list[str]:
    suites = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(suites) - set(TASKS_BY_SUITE))
    if unknown:
        raise ValueError(f"Unknown transfer suites: {', '.join(unknown)}")
    return suites or list(TASKS_BY_SUITE)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = parse_suites(args.suites)
    write_json(output_dir / "environment_info.json", environment_info())
    write_json(output_dir / "config.json", {"suites": suites, "seed": args.seed})
    suite_results = [run_transfer_suite(suite, output_dir, args.seed + index * 10_000) for index, suite in enumerate(suites)]
    backend_status = official_backend_status()
    rows = [row for suite_result in suite_results for row in suite_result["summary"]]
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    write_json(output_dir / "official_backend_status.json", backend_status)
    write_overall_summary(output_dir / "summary.md", suite_results, backend_status)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local external-transfer protocol probes.")
    parser.add_argument("--suites", default=",".join(TASKS_BY_SUITE))
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument("--output-dir", default="runs/external_transfer_protocols")
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    print(f"External transfer protocol artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
