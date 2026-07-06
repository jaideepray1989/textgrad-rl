"""Environment protocol definitions."""

from __future__ import annotations

from typing import Any, Protocol

from textgrad_rl.types import Action, Observation, Trajectory


class RepairEnv(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]: ...

    def render_trajectory(self) -> str: ...

    def get_trajectory(self) -> Trajectory: ...

