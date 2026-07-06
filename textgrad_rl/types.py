"""Core dataclasses shared by agents, environments, critics, and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal[
    "read_file",
    "edit_file",
    "run_tests",
    "run_training",
    "run_eval",
    "inspect_logs",
    "submit_patch",
    "noop",
]

VALID_ACTION_TYPES: set[str] = {
    "read_file",
    "edit_file",
    "run_tests",
    "run_training",
    "run_eval",
    "inspect_logs",
    "submit_patch",
    "noop",
}


@dataclass
class TextVariable:
    """A mutable text module optimized by trajectory-level textual gradients."""

    name: str
    value: str
    role_description: str
    requires_grad: bool = True
    gradient_history: list[str] = field(default_factory=list)
    version: int = 0
    max_chars: int = 2500

    def clipped_value(self) -> str:
        """Return a max-length-safe prompt value."""

        return self.value[-self.max_chars :] if len(self.value) > self.max_chars else self.value


@dataclass
class Action:
    """Structured environment action emitted by an agent."""

    type: ActionType
    path: str | None = None
    content: str | None = None
    command: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action type: {self.type!r}")


@dataclass
class Observation:
    """Visible state returned to an agent after each environment step."""

    task_id: str
    task_description: str
    visible_files: dict[str, str]
    file_tree: list[str]
    recent_output: str
    test_status: str
    training_status: str
    eval_status: str
    metric_summary: dict[str, float]
    remaining_steps: int
    forbidden_paths: list[str]
    text_variable_summary: dict[str, str]


@dataclass
class StepRecord:
    """One transition in a repair trajectory."""

    step_index: int
    observation_summary: str
    action: Action
    reward: float
    done: bool
    info: dict[str, Any]
    stdout: str
    stderr: str
    files_changed: list[str]


@dataclass
class Trajectory:
    """Complete episode trace used by the textual critic."""

    task_id: str
    task_family: str
    steps: list[StepRecord]
    total_reward: float
    success: bool
    final_status: str
    failure_summary: str
    file_diffs: dict[str, str]
    test_results: dict[str, Any]
    metric_results: dict[str, float]
    command_count: int
    invalid_action_count: int
    runtime_seconds: float


@dataclass
class TextualGradient:
    """A natural-language gradient targeted at a specific text variable."""

    target_variable_name: str
    failure_mode: str
    evidence_from_trajectory: str
    gradient_text: str
    suggested_edit: str
    confidence: float
    forbidden_shortcuts: list[str] = field(default_factory=list)


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for a split and iteration."""

    iteration: int
    split: str
    success_rate: float
    average_reward: float
    test_pass_rate: float
    invalid_action_rate: float
    average_steps: float
    command_count: float
    accepted_updates: int
    rejected_updates: int
    runtime_seconds: float
    prompt_length_total: int

