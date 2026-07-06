"""Secure CPU-only ML-engineering repair environment."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from textgrad_rl.envs.task_specs import TaskSpec
from textgrad_rl.types import Action, Observation, StepRecord, Trajectory
from textgrad_rl.utils.diffs import unified_diff
from textgrad_rl.utils.files import list_files, read_text_if_exists, write_files
from textgrad_rl.utils.metrics import extract_json_metrics
from textgrad_rl.utils.security import looks_like_forbidden_content, normalize_relative_path
from textgrad_rl.utils.subprocesses import CommandResult, run_command


ALLOWED_COMMANDS = {
    "pytest -q",
    "python train.py",
    "python eval.py",
    "python smoke_test.py",
    "python -m pytest -q",
}


class MLERepairEnv:
    """A small local environment that resembles an ML engineer's repair loop."""

    def __init__(
        self,
        task_spec: TaskSpec,
        work_dir: Path,
        max_steps: int = 20,
        command_timeout_sec: int = 10,
        allow_test_edits: bool = False,
        allow_network: bool = False,
    ):
        self.task_spec = task_spec
        self.work_dir = Path(work_dir)
        self.max_steps = max_steps
        self.command_timeout_sec = command_timeout_sec
        self.allow_test_edits = allow_test_edits
        self.allow_network = allow_network
        self.task_dir = self.work_dir / task_spec.task_id
        self._start_time = 0.0
        self._initial_files: dict[str, str] = {}
        self._visible_cache: dict[str, str] = {}
        self._steps: list[StepRecord] = []
        self._recent_output = ""
        self._test_status = "not_run"
        self._training_status = "not_run"
        self._eval_status = "not_run"
        self._metric_summary: dict[str, float] = {}
        self._done = False
        self._success = False
        self._final_status = "not_started"
        self._failure_summary = ""
        self._command_count = 0
        self._invalid_action_count = 0
        self._last_failed_command: str | None = None
        self._failed_command_repeats = 0
        self._best_metric = float("-inf")
        self._subprocess_seconds = 0.0

    def reset(self) -> Observation:
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        write_files(self.task_dir, self.task_spec.files)
        write_files(self.task_dir, self.task_spec.hidden_files)
        self._initial_files = {
            **self.task_spec.files,
            **self.task_spec.hidden_files,
        }
        self._visible_cache = {}
        self._steps = []
        self._recent_output = ""
        self._test_status = "not_run"
        self._training_status = "not_run"
        self._eval_status = "not_run"
        self._metric_summary = {}
        self._done = False
        self._success = False
        self._final_status = "running"
        self._failure_summary = ""
        self._command_count = 0
        self._invalid_action_count = 0
        self._last_failed_command = None
        self._failed_command_repeats = 0
        self._best_metric = float("-inf")
        self._subprocess_seconds = 0.0
        self._start_time = time.perf_counter()
        return self._observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._done:
            return self._observation(), 0.0, True, {"message": "episode already done"}

        step_index = len(self._steps)
        before = self._snapshot_visible_files()
        stdout = ""
        stderr = ""
        info: dict[str, Any] = {}
        reward = -0.05
        invalid_reason: str | None = None

        try:
            if action.type == "read_file":
                reward += self._read_file(action)
                stdout = self._recent_output
            elif action.type == "edit_file":
                edit_reward, invalid_reason = self._edit_file(action)
                reward += edit_reward
                stderr = invalid_reason or ""
            elif action.type == "run_tests":
                command = action.command or self.task_spec.visible_test_command
                result, invalid_reason = self._run_allowed_command(command, status_kind="tests")
                reward += self._reward_for_command(result, command) if result else -2.0
                stdout = result.stdout if result else ""
                stderr = result.stderr if result else invalid_reason or ""
            elif action.type == "run_training":
                command = action.command or self.task_spec.train_command
                result, invalid_reason = self._run_allowed_command(command, status_kind="training")
                reward += self._reward_for_command(result, command) if result else -2.0
                stdout = result.stdout if result else ""
                stderr = result.stderr if result else invalid_reason or ""
            elif action.type == "run_eval":
                command = action.command or self.task_spec.eval_command
                result, invalid_reason = self._run_allowed_command(command, status_kind="eval")
                reward += self._reward_for_command(result, command) if result else -2.0
                stdout = result.stdout if result else ""
                stderr = result.stderr if result else invalid_reason or ""
            elif action.type == "inspect_logs":
                reward += 0.1
                self._failed_command_repeats = 0
                stdout = self._inspect_logs()
                self._recent_output = stdout
            elif action.type == "submit_patch":
                submit_reward, stdout, stderr, info = self._submit_patch()
                reward += submit_reward
            elif action.type == "noop":
                reward -= 0.1
                self._recent_output = action.reason or "noop"
                stdout = self._recent_output
            else:
                invalid_reason = f"unsupported action type {action.type}"
        except Exception as exc:
            invalid_reason = f"environment error: {exc}"

        if invalid_reason:
            self._invalid_action_count += 1
            reward -= 2.0
            info["invalid"] = True
            info["invalid_reason"] = invalid_reason
            self._recent_output = invalid_reason
            stderr = stderr or invalid_reason
            if "forbidden" in invalid_reason or "test" in invalid_reason:
                reward -= 6.0

        if self._failed_command_repeats >= 2 and action.type in {"run_tests", "run_training", "run_eval"}:
            reward -= 1.0

        reward -= 0.02 * self._subprocess_seconds

        if len(self._steps) + 1 >= self.max_steps and not self._done:
            self._done = True
            self._final_status = "max_steps_exceeded"
            self._failure_summary = "Episode ended before submit_patch succeeded."

        after = self._snapshot_visible_files()
        files_changed = sorted(path for path in after if before.get(path) != after.get(path))
        record = StepRecord(
            step_index=step_index,
            observation_summary=self._summarize_observation(),
            action=action,
            reward=round(float(reward), 4),
            done=self._done,
            info=info,
            stdout=stdout,
            stderr=stderr,
            files_changed=files_changed,
        )
        self._steps.append(record)
        return self._observation(), record.reward, self._done, info

    def render_trajectory(self) -> str:
        trajectory = self.get_trajectory()
        lines = [
            f"Task: {trajectory.task_id}",
            f"Family: {trajectory.task_family}",
            f"Description: {self.task_spec.description}",
            f"Success: {trajectory.success}",
            f"Final status: {trajectory.final_status}",
            f"Total reward: {trajectory.total_reward:.3f}",
            f"Failure summary: {trajectory.failure_summary}",
            "",
            "Steps:",
        ]
        for step in trajectory.steps:
            lines.append(
                f"- Step {step.step_index}: {step.action.type} "
                f"path={step.action.path!r} command={step.action.command!r} "
                f"reward={step.reward} done={step.done}"
            )
            if step.action.reason:
                lines.append(f"  reason: {step.action.reason}")
            if step.stdout:
                lines.append("  stdout:")
                lines.append(_indent(_clip(step.stdout, 1600), "    "))
            if step.stderr:
                lines.append("  stderr:")
                lines.append(_indent(_clip(step.stderr, 1600), "    "))
            if step.files_changed:
                lines.append(f"  files_changed: {', '.join(step.files_changed)}")
        lines.extend(["", "File diffs:"])
        for path, diff in trajectory.file_diffs.items():
            lines.append(f"--- {path}")
            lines.append(_indent(_clip(diff, 2400), "  "))
        lines.extend(
            [
                "",
                f"Test results: {json.dumps(trajectory.test_results, sort_keys=True)}",
                f"Metric results: {json.dumps(trajectory.metric_results, sort_keys=True)}",
            ]
        )
        return "\n".join(lines) + "\n"

    def get_trajectory(self) -> Trajectory:
        diffs = self._file_diffs()
        test_results = {
            "status": self._test_status,
            "passed": self._test_status == "passed",
        }
        return Trajectory(
            task_id=self.task_spec.task_id,
            task_family=self.task_spec.family,
            steps=list(self._steps),
            total_reward=sum(step.reward for step in self._steps),
            success=self._success,
            final_status=self._final_status,
            failure_summary=self._failure_summary or self._infer_failure_summary(),
            file_diffs=diffs,
            test_results=test_results,
            metric_results=dict(self._metric_summary),
            command_count=self._command_count,
            invalid_action_count=self._invalid_action_count,
            runtime_seconds=time.perf_counter() - self._start_time if self._start_time else 0.0,
        )

    def _read_file(self, action: Action) -> float:
        path, error = normalize_relative_path(self.task_dir, action.path)
        if error:
            raise ValueError(error)
        relative = _relative_to_root(self.task_dir, path)
        if relative in self.task_spec.hidden_files:
            raise ValueError("reading hidden files before submit is forbidden")
        if not path.exists() or not path.is_file():
            raise ValueError(f"file does not exist: {relative}")
        content = path.read_text(encoding="utf-8", errors="replace")
        self._visible_cache[relative] = content
        self._recent_output = f"READ {relative}\n{content}"
        return 0.05

    def _edit_file(self, action: Action) -> tuple[float, str | None]:
        path, error = normalize_relative_path(self.task_dir, action.path)
        if error:
            return -2.0, error
        if action.content is None:
            return -2.0, "edit_file requires full replacement content"
        relative = _relative_to_root(self.task_dir, path)
        forbidden = self._forbidden_edit_reason(relative, action.content)
        if forbidden:
            return -8.0, forbidden
        if not path.exists() or not path.is_file():
            return -2.0, f"file does not exist: {relative}"
        path.write_text(action.content, encoding="utf-8")
        self._visible_cache[relative] = action.content
        self._recent_output = f"EDITED {relative}"
        return 0.2, None

    def _forbidden_edit_reason(self, relative: str, content: str) -> str | None:
        if relative in self.task_spec.hidden_files:
            return "editing hidden validation files is forbidden"
        if relative in {"task_metadata.json", "reward.py"}:
            return "editing task metadata or reward code is forbidden"
        if relative in self.task_spec.forbidden_paths:
            if relative.startswith("tests/") and self.allow_test_edits:
                return None
            return f"editing forbidden path is not allowed: {relative}"
        if relative.startswith("tests/") and not self.allow_test_edits:
            return "editing tests is forbidden"
        if "threshold" in relative.lower() or "threshold" in content.lower() and relative == "eval.py":
            return "changing eval thresholds is forbidden"
        return looks_like_forbidden_content(content)

    def _run_allowed_command(
        self,
        command: str,
        status_kind: str,
    ) -> tuple[CommandResult | None, str | None]:
        if command not in ALLOWED_COMMANDS:
            return None, f"shell command is not allow-listed: {command}"
        result = run_command(command, self.task_dir, self.command_timeout_sec, self.allow_network)
        self._command_count += 1
        self._subprocess_seconds += result.runtime_seconds
        output = (result.stdout + "\n" + result.stderr).strip()
        self._recent_output = output
        self._update_status_from_command(status_kind, result, command)
        if result.returncode != 0:
            if self._last_failed_command == command:
                self._failed_command_repeats += 1
            else:
                self._failed_command_repeats = 1
            self._last_failed_command = command
        else:
            self._failed_command_repeats = 0
            self._last_failed_command = None
        return result, None

    def _update_status_from_command(self, status_kind: str, result: CommandResult, command: str) -> None:
        status = "passed" if result.returncode == 0 else "failed"
        if command == "python smoke_test.py":
            self._training_status = "smoke_passed" if result.returncode == 0 else "smoke_failed"
        elif status_kind == "tests":
            self._test_status = status
        elif status_kind == "training":
            self._training_status = status
        elif status_kind == "eval":
            self._eval_status = status
            metrics = extract_json_metrics(result.stdout)
            if metrics:
                self._metric_summary.update(metrics)
                metric = metrics.get(self.task_spec.metric_name)
                if metric is not None:
                    self._best_metric = max(self._best_metric, metric)

    def _reward_for_command(self, result: CommandResult, command: str) -> float:
        if result.returncode != 0:
            return -0.4
        reward = 0.2
        if command in {"pytest -q", "python -m pytest -q"}:
            reward += 2.0
        elif command == "python train.py":
            reward += 1.5
        elif command == "python eval.py":
            metric = extract_json_metrics(result.stdout).get(self.task_spec.metric_name, 0.0)
            normalized_gain = max(0.0, metric / max(self.task_spec.metric_threshold, 1e-9))
            reward += min(2.0, 2.0 * normalized_gain)
        elif command == "python smoke_test.py":
            reward += 0.5
        return reward

    def _inspect_logs(self) -> str:
        chunks = ["Recent output:", self._recent_output or "<none>"]
        for path in sorted(self.task_dir.glob("*.log")):
            chunks.append(f"\nLog file {path.name}:")
            chunks.append(read_text_if_exists(path))
        return "\n".join(chunks)

    def _submit_patch(self) -> tuple[float, str, str, dict[str, Any]]:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        info: dict[str, Any] = {}
        visible, error = self._run_internal_validation(self.task_spec.visible_test_command)
        stdout_parts.append(visible.stdout)
        stderr_parts.append(visible.stderr)
        self._test_status = "passed" if visible.returncode == 0 else "failed"
        if error:
            self._done = True
            self._success = False
            self._final_status = "visible_tests_failed_on_submit"
            self._failure_summary = error
            return -1.0, "\n".join(stdout_parts), "\n".join(stderr_parts), info

        hidden, error = self._run_internal_validation(self.task_spec.hidden_validation_command)
        stdout_parts.append(hidden.stdout)
        stderr_parts.append(hidden.stderr)
        metrics = extract_json_metrics(hidden.stdout)
        if metrics:
            self._metric_summary.update(metrics)
        self._done = True
        self._success = error is None
        self._final_status = "success" if self._success else "hidden_validation_failed"
        self._failure_summary = "" if self._success else (error or hidden.stderr or hidden.stdout)
        info["hidden_validation_passed"] = self._success
        reward = 5.0 if self._success else -2.0
        return reward, "\n".join(stdout_parts), "\n".join(stderr_parts), info

    def _run_internal_validation(self, command: str) -> tuple[CommandResult, str | None]:
        result = run_command(command, self.task_dir, self.command_timeout_sec, self.allow_network)
        self._command_count += 1
        self._subprocess_seconds += result.runtime_seconds
        self._recent_output = (result.stdout + "\n" + result.stderr).strip()
        error = None if result.returncode == 0 else f"validation command failed: {command}"
        return result, error

    def _observation(self) -> Observation:
        return Observation(
            task_id=self.task_spec.task_id,
            task_description=self.task_spec.description,
            visible_files=dict(self._visible_cache),
            file_tree=[
                path
                for path in list_files(self.task_dir)
                if path not in self.task_spec.hidden_files and "__pycache__" not in path
            ],
            recent_output=_clip(self._recent_output, 5000),
            test_status=self._test_status,
            training_status=self._training_status,
            eval_status=self._eval_status,
            metric_summary=dict(self._metric_summary),
            remaining_steps=max(0, self.max_steps - len(self._steps)),
            forbidden_paths=list(self.task_spec.forbidden_paths),
            text_variable_summary={},
        )

    def _snapshot_visible_files(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        if not self.task_dir.exists():
            return snapshot
        hidden = set(self.task_spec.hidden_files)
        for relative in list_files(self.task_dir):
            if relative in hidden or "__pycache__" in relative:
                continue
            snapshot[relative] = (self.task_dir / relative).read_text(encoding="utf-8", errors="replace")
        return snapshot

    def _file_diffs(self) -> dict[str, str]:
        diffs: dict[str, str] = {}
        current = self._snapshot_visible_files()
        for path, content in current.items():
            original = self.task_spec.files.get(path, "")
            diff = unified_diff(original, content, path)
            if diff:
                diffs[path] = diff
        return diffs

    def _summarize_observation(self) -> str:
        return (
            f"tests={self._test_status}, training={self._training_status}, "
            f"eval={self._eval_status}, metrics={self._metric_summary}, "
            f"remaining={max(0, self.max_steps - len(self._steps))}"
        )

    def _infer_failure_summary(self) -> str:
        if self._success:
            return ""
        if self._recent_output:
            return _clip(self._recent_output, 800)
        return "No successful submit_patch action was recorded."


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 40] + "\n... [clipped] ...\n" + text[-20:]


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _relative_to_root(root: Path, path: Path) -> str:
    return Path(os.path.relpath(os.path.realpath(path), os.path.realpath(root))).as_posix()
