"""Offline heuristic trajectory critic."""

from __future__ import annotations

from textgrad_rl.types import TextualGradient, TextVariable, Trajectory


class HeuristicTextualCritic:
    """Detect common ML-engineering trajectory failures and emit textual gradients."""

    def critique(
        self,
        trajectory: Trajectory,
        text_variables: dict[str, TextVariable],
    ) -> list[TextualGradient]:
        gradients: list[TextualGradient] = []
        text = _trajectory_text(trajectory).lower()
        actions = [step.action.type for step in trajectory.steps]
        commands = [step.action.command or "" for step in trajectory.steps]

        def add(
            target: str,
            failure_mode: str,
            evidence: str,
            gradient_text: str,
            suggested_edit: str,
            confidence: float = 0.85,
            forbidden: list[str] | None = None,
        ) -> None:
            gradients.append(
                TextualGradient(
                    target_variable_name=_target(target, text_variables),
                    failure_mode=failure_mode,
                    evidence_from_trajectory=evidence,
                    gradient_text=gradient_text,
                    suggested_edit=suggested_edit,
                    confidence=confidence,
                    forbidden_shortcuts=forbidden or ["edit tests", "change thresholds", "hardcode outputs"],
                )
            )

        if trajectory.invalid_action_count:
            add(
                "patch_planning_prompt",
                "Invalid or forbidden edit attempted",
                f"{trajectory.invalid_action_count} invalid action(s) were recorded.",
                "The patch planner should explicitly avoid forbidden shortcuts and repair source code instead.",
                "Add a rule: avoid forbidden shortcuts such as editing tests, metadata, hidden validation, thresholds, data, or expected outputs.",
                0.95,
            )

        if "matmul" in text or "shape mismatch" in text or "dimension" in text:
            add(
                "log_interpretation_prompt",
                "Shape mismatch not converted into targeted inspection",
                "Trajectory output contained shape/dimension evidence.",
                "Shape errors usually identify the file where feature dimensionality is assumed. Inspect that code before another full run.",
                "Add a rule: for shape mismatch errors, inspect training/preprocessing feature dimensions before editing or rerunning full training.",
                0.9,
            )

        if "python train.py" in commands and commands.count("python train.py") >= 2 and "inspect_logs" in actions:
            add(
                "experiment_planning_prompt",
                "Repeated full training after deterministic crash",
                "The agent reran python train.py after a deterministic failure.",
                "The agent should use cheaper reproducers and inspect logs before repeating full training.",
                "Add a rule: after a deterministic training crash, inspect stack traces and run the cheapest reproducer before full training.",
                0.88,
            )

        if "age_years" in text or "keyerror" in text or "schema" in text:
            add(
                "triage_prompt",
                "Schema mismatch not triaged through preprocessing",
                "The failure mentioned a missing/stale column name.",
                "Schema drift should cause the agent to inspect data columns and preprocessing aliases before model code.",
                "Add a rule: for schema mismatch errors, inspect data columns and preprocessing assumptions before editing model code.",
                0.9,
            )

        if "reproduc" in text or "random_state" in text or "deterministic" in text:
            add(
                "validation_prompt",
                "Reproducibility failure needs explicit validation",
                "The trajectory contained nondeterminism or reproducibility evidence.",
                "Validation should include repeated eval/tests when the task is about determinism.",
                "Add a rule: for reproducibility tasks, set random_state consistently and rerun eval or tests before submit.",
                0.9,
            )

        metric = trajectory.metric_results.get("accuracy")
        if (metric is not None and metric < 0.75) or "metric" in text or "threshold" in text:
            add(
                "patch_planning_prompt",
                "Metric regression requires source-level root-cause repair",
                "Eval metric was below threshold or metric tests failed.",
                "Metric failures should lead to checking label mapping, feature construction, and split logic without leaking validation labels.",
                "Add a rule: for metric regressions, inspect feature construction and label mapping, and never leak validation/test labels into training.",
                0.86,
            )

        if "latency" in text or "runtime" in text or "too slow" in text:
            add(
                "patch_planning_prompt",
                "Latency regression from repeated inference work",
                "The trajectory mentioned runtime or latency failures.",
                "Latency fixes should cache invariant loading and vectorize batch prediction while preserving correctness.",
                "Add a rule: for latency regressions, inspect inference hot paths for repeated loading and vectorize batch prediction.",
                0.9,
            )

        if "submit_patch" in actions:
            before_submit = trajectory.steps[: actions.index("submit_patch")]
            if not any(step.action.type == "run_eval" for step in before_submit):
                add(
                    "validation_prompt",
                    "Submitted without eval validation",
                    "submit_patch occurred before an explicit run_eval action.",
                    "The agent should validate the visible eval path before final submission.",
                    "Add a rule: before submit, run visible tests, training, and eval unless the task has no eval path.",
                    0.82,
                )

        read_before_edit: set[str] = set()
        for step in trajectory.steps:
            if step.action.type == "read_file" and step.action.path:
                read_before_edit.add(step.action.path)
            if step.action.type == "edit_file" and step.action.path and step.action.path not in read_before_edit:
                add(
                    "patch_planning_prompt",
                    "Patched before inspecting target file",
                    f"Edited {step.action.path} before reading it.",
                    "Patches should be grounded in the current file content and failure evidence.",
                    "Add a rule: read a source file before editing it, except for generated scratch artifacts.",
                    0.8,
                )
                break

        if not trajectory.success and not gradients:
            add(
                "triage_prompt",
                "Unresolved repair trajectory",
                trajectory.failure_summary or "Episode failed without a specific classified failure.",
                "The triage prompt should prioritize concrete failing evidence and next diagnostic actions.",
                "Add a rule: when a repair fails, classify the first concrete error and inspect the smallest relevant source file next.",
                0.7,
            )

        return _dedupe_gradients(gradients)


def _target(name: str, text_variables: dict[str, TextVariable]) -> str:
    if name in text_variables:
        return name
    if "monolithic_prompt" in text_variables:
        return "monolithic_prompt"
    return next(iter(text_variables))


def _trajectory_text(trajectory: Trajectory) -> str:
    chunks = [trajectory.failure_summary, trajectory.final_status]
    for step in trajectory.steps:
        chunks.extend(
            [
                step.action.type,
                step.action.path or "",
                step.action.command or "",
                step.action.reason or "",
                step.stdout,
                step.stderr,
            ]
        )
    chunks.extend(trajectory.file_diffs.values())
    return "\n".join(chunks)


def _dedupe_gradients(gradients: list[TextualGradient]) -> list[TextualGradient]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TextualGradient] = []
    for gradient in gradients:
        key = (gradient.target_variable_name, gradient.failure_mode)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gradient)
    return deduped

