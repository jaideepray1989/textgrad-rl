"""Offline prompt-aware heuristic actor.

The agent is intentionally frozen. Its policy code never changes, but it checks the
current text variables for learned rules and chooses different actions when those
rules are present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from textgrad_rl.types import Action, Observation, TextVariable


@dataclass
class _TaskState:
    commands: list[str] = field(default_factory=list)
    attempted_test_edit: bool = False
    inspected_after_failure: bool = False
    reran_after_deterministic_failure: bool = False
    edited_files: set[str] = field(default_factory=set)
    reran_eval_for_validation: bool = False


class PromptAwareHeuristicAgent:
    """A deterministic local actor whose behavior depends on text variables."""

    def __init__(self) -> None:
        self._states: dict[str, _TaskState] = {}

    def act(self, observation: Observation, text_variables: dict[str, TextVariable]) -> Action:
        state = self._states.setdefault(observation.task_id, _TaskState())
        prompt_text = _prompt_text(text_variables)
        output = observation.recent_output.lower()

        repair = self._repair_action(observation, state)
        if repair is not None:
            return repair

        if observation.remaining_steps <= 1:
            return Action(type="submit_patch", reason="step budget nearly exhausted")

        if not state.commands and not observation.recent_output:
            if "smoke" in prompt_text and "smoke_test.py" in observation.file_tree:
                state.commands.append("python smoke_test.py")
                return Action(
                    type="run_training",
                    command="python smoke_test.py",
                    reason="learned rule says to run the cheapest smoke test first",
                )
            state.commands.append("pytest -q")
            return Action(type="run_tests", reason="start with visible tests")

        if _latest_output_is_failure(observation):
            if self._should_attempt_bad_test_edit(prompt_text, observation, state):
                state.attempted_test_edit = True
                test_path = next((p for p in observation.file_tree if p.startswith("tests/")), "tests/test_placeholder.py")
                return Action(
                    type="edit_file",
                    path=test_path,
                    content="def test_placeholder():\n    assert True\n",
                    reason="naive shortcut attempt before learned forbidden-shortcut rule",
                )
            if not state.inspected_after_failure:
                state.inspected_after_failure = True
                return Action(type="inspect_logs", reason="inspect the latest failing output")
            read_action = self._choose_file_to_read(observation, prompt_text, state)
            if read_action is not None:
                return read_action

        if observation.test_status == "passed" and observation.training_status not in {"passed"}:
            state.commands.append("python train.py")
            return Action(type="run_training", reason="visible tests pass; run training")

        if observation.training_status == "passed" and observation.eval_status != "passed":
            state.commands.append("python eval.py")
            return Action(type="run_eval", reason="training completed; run eval")

        metric = observation.metric_summary.get("accuracy")
        if observation.eval_status == "passed" and metric is not None and metric < 0.75:
            read_action = self._choose_file_to_read(observation, prompt_text, state)
            if read_action is not None:
                return read_action

        if self._ready_to_submit(observation):
            if (
                "reproducibility" in prompt_text
                and "reproducibility" in observation.task_description.lower()
                and not state.reran_eval_for_validation
            ):
                state.reran_eval_for_validation = True
                state.commands.append("python eval.py")
                return Action(type="run_eval", reason="learned validation rule asks for reproducibility recheck")
            return Action(type="submit_patch", reason="tests, training, and eval are green")

        if observation.test_status != "passed":
            state.commands.append("pytest -q")
            return Action(type="run_tests", reason="run visible tests after changes")

        state.commands.append("python train.py")
        return Action(type="run_training", reason="continue validating the repair")

    def _should_attempt_bad_test_edit(
        self,
        prompt_text: str,
        observation: Observation,
        state: _TaskState,
    ) -> bool:
        return (
            observation.test_status == "failed"
            and not state.attempted_test_edit
            and "forbidden shortcut" not in prompt_text
            and any(path.startswith("tests/") for path in observation.file_tree)
        )

    def _choose_file_to_read(
        self,
        observation: Observation,
        prompt_text: str,
        state: _TaskState,
    ) -> Action | None:
        output = observation.recent_output.lower()
        visible = observation.visible_files

        if ("matmul" in output or "shape" in output or "dimension" in output) and "train.py" not in visible:
            if "shape mismatch" not in prompt_text and not state.reran_after_deterministic_failure:
                state.reran_after_deterministic_failure = True
                state.commands.append("python train.py")
                return Action(
                    type="run_training",
                    reason="without a shape-mismatch rule, retry full training once",
                )
            return Action(type="read_file", path="train.py", reason="inspect shape-related training code")

        if ("age_years" in output or "keyerror" in output or "schema" in output) and "preprocess.py" not in visible:
            if "schema mismatch" in prompt_text or "data columns" in prompt_text:
                return Action(type="read_file", path="preprocess.py", reason="inspect preprocessing schema handling")
            if "train.py" not in visible:
                return Action(type="read_file", path="train.py", reason="inspect caller before schema rule is learned")
            return Action(type="read_file", path="preprocess.py", reason="inspect preprocessing after caller")

        if ("reproduc" in output or "random_state" in output or "deterministic" in output) and "train.py" not in visible:
            return Action(type="read_file", path="train.py", reason="inspect random state usage")

        if ("accuracy" in output or "threshold" in output or "metric" in output) and "train.py" not in visible:
            return Action(type="read_file", path="train.py", reason="inspect training target and metric path")

        if ("latency" in output or "runtime" in output or "slow" in output) and "inference.py" not in visible:
            return Action(type="read_file", path="inference.py", reason="inspect inference hot path")

        traceback_path = _path_from_traceback(observation.recent_output, observation.file_tree)
        if traceback_path and traceback_path not in visible:
            return Action(type="read_file", path=traceback_path, reason="inspect traceback source file")

        for preferred in ["preprocess.py", "features.py", "inference.py", "train.py", "eval.py"]:
            if preferred in observation.file_tree and preferred not in visible:
                return Action(type="read_file", path=preferred, reason="inspect likely repair file")
        return None

    def _repair_action(self, observation: Observation, state: _TaskState) -> Action | None:
        for path, content in observation.visible_files.items():
            if path in state.edited_files:
                continue
            repaired = _repair_content(path, content)
            if repaired is not None and repaired != content:
                state.edited_files.add(path)
                state.inspected_after_failure = False
                return Action(
                    type="edit_file",
                    path=path,
                    content=repaired,
                    reason=f"apply pattern-based source repair for {path}",
                )
        return None

    def _ready_to_submit(self, observation: Observation) -> bool:
        return (
            observation.test_status == "passed"
            and observation.training_status == "passed"
            and observation.eval_status == "passed"
        )


def _prompt_text(text_variables: dict[str, TextVariable]) -> str:
    return "\n".join(var.value for var in text_variables.values()).lower()


def _latest_output_is_failure(observation: Observation) -> bool:
    if observation.test_status == "failed" or observation.training_status == "failed" or observation.eval_status == "failed":
        return True
    output = observation.recent_output.lower()
    return any(token in output for token in ["failed", "error", "traceback", "assertionerror", "keyerror"])


def _path_from_traceback(output: str, file_tree: list[str]) -> str | None:
    for match in re.findall(r'File "([^"]+\.py)"', output):
        cleaned = match.replace("\\", "/").split("/")[-1]
        for path in file_tree:
            if path.endswith(cleaned):
                return path
    return None


def _repair_content(path: str, content: str) -> str | None:
    if path == "train.py" and "weights = np.ones(3)" in content:
        return content.replace("weights = np.ones(3)", "weights = np.ones(X.shape[1])")

    if path == "preprocess.py" and 'FEATURE_COLUMNS = ["age_years", "income"]' in content:
        return (
            'FEATURE_COLUMNS = ["age_years", "income"]\n\n\n'
            "def make_features(df):\n"
            "    working = df.copy()\n"
            '    if "age_years" not in working.columns and "age" in working.columns:\n'
            '        working = working.rename(columns={"age": "age_years"})\n'
            "    missing = [column for column in FEATURE_COLUMNS if column not in working.columns]\n"
            "    if missing:\n"
            '        raise ValueError(f"Missing required columns: {missing}")\n'
            "    features = working[FEATURE_COLUMNS].copy()\n"
            '    features["age_years"] = features["age_years"] / 100.0\n'
            '    features["income"] = features["income"] / 100000.0\n'
            "    return features.to_numpy(dtype=float)\n"
        )

    if path == "train.py" and "train_test_split(X, y, test_size=0.3)" in content:
        repaired = content.replace(
            "train_test_split(X, y, test_size=0.3)",
            "train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)",
        )
        repaired = repaired.replace(
            "RandomForestClassifier(n_estimators=15)",
            "RandomForestClassifier(n_estimators=15, random_state=42)",
        )
        return repaired

    if path == "train.py" and 'target = 1 - df["label"].to_numpy()' in content:
        return content.replace('target = 1 - df["label"].to_numpy()', 'target = df["label"].to_numpy()')

    if path == "inference.py" and "for row in rows:" in content and "_load_model()" in content:
        return (
            "import json\n"
            "import time\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n\n\n"
            "_MODEL = None\n\n\n"
            "def _load_model():\n"
            "    time.sleep(0.03)\n"
            '    return json.loads(Path("artifacts/model.json").read_text())\n\n\n'
            "def _get_model():\n"
            "    global _MODEL\n"
            "    if _MODEL is None:\n"
            "        _MODEL = _load_model()\n"
            "    return _MODEL\n\n\n"
            "def predict_batch(rows):\n"
            "    model = _get_model()\n"
            "    X = np.asarray(rows, dtype=float)\n"
            '    scores = X @ np.asarray(model["coef"], dtype=float) + float(model["intercept"])\n'
            "    return (scores >= 0.0).astype(int).tolist()\n"
        )

    return None
