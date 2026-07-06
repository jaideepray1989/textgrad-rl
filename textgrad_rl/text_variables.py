"""Initial modular text variables for TextGrad-RL experiments."""

from __future__ import annotations

from textgrad_rl.types import TextVariable


ROLE_DESCRIPTIONS: dict[str, str] = {
    "triage_prompt": (
        "Diagnose ML-engineering failures from task descriptions, logs, stack traces, "
        "test output, metrics, and file contents."
    ),
    "log_interpretation_prompt": (
        "Extract useful signal from stack traces, pytest output, schema errors, shape "
        "mismatches, metric regressions, profiler output, and runtime logs."
    ),
    "patch_planning_prompt": (
        "Choose minimal code/config edits that fix the root cause without hiding "
        "failures, changing tests, hardcoding outputs, or leaking labels."
    ),
    "experiment_planning_prompt": (
        "Choose when to run unit tests, smoke tests, training, evaluation, or artifact "
        "inspection under a limited step/runtime budget."
    ),
    "validation_prompt": (
        "Decide whether the patch is real, reproducible, non-leaky, and ready to submit."
    ),
}


INITIAL_VALUES: dict[str, str] = {
    "triage_prompt": (
        "Start from the observable failure and identify the likely subsystem. Prefer "
        "reading code near the reported traceback before broad edits. Keep notes about "
        "which files explain the failure."
    ),
    "log_interpretation_prompt": (
        "Summarize the first concrete error in pytest, training, or eval output. Use "
        "tracebacks to connect a failure to a source file and line. Treat repeated "
        "failures as evidence that the next action should gather more information."
    ),
    "patch_planning_prompt": (
        "Make the smallest source-code change that addresses the observed root cause. "
        "Prefer fixes in preprocessing, training, feature, or inference code over broad "
        "rewrites. Re-run checks after editing."
    ),
    "experiment_planning_prompt": (
        "Use the limited step budget to alternate between tests, code inspection, and "
        "targeted edits. Avoid repeating the same failing command many times in a row."
    ),
    "validation_prompt": (
        "Before submitting, make sure the visible tests and the main training/eval path "
        "have completed successfully. Do not submit when the latest output is an error."
    ),
}


def initial_text_variables(max_chars: int = 2500) -> dict[str, TextVariable]:
    """Return fresh modular text variables."""

    return {
        name: TextVariable(
            name=name,
            value=value,
            role_description=ROLE_DESCRIPTIONS[name],
            max_chars=max_chars,
        )
        for name, value in INITIAL_VALUES.items()
    }


def monolithic_text_variable(max_chars: int = 8000) -> dict[str, TextVariable]:
    """Return one concatenated variable for the monolithic ablation."""

    joined = "\n\n".join(f"[{name}]\n{value}" for name, value in INITIAL_VALUES.items())
    return {
        "monolithic_prompt": TextVariable(
            name="monolithic_prompt",
            value=joined,
            role_description="All agent instructions concatenated into a single text variable.",
            max_chars=max_chars,
        )
    }


def summarize_text_variables(text_variables: dict[str, TextVariable]) -> dict[str, str]:
    """Compact summaries for observations and artifacts."""

    return {
        name: f"v{var.version}, {len(var.value)} chars, {len(var.gradient_history)} gradients"
        for name, var in text_variables.items()
    }

