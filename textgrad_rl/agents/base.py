"""Agent protocol."""

from __future__ import annotations

from typing import Protocol

from textgrad_rl.types import Action, Observation, TextVariable


class Agent(Protocol):
    def act(self, observation: Observation, text_variables: dict[str, TextVariable]) -> Action: ...

