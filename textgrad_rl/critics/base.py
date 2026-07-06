"""Textual critic protocol."""

from __future__ import annotations

from typing import Protocol

from textgrad_rl.types import TextualGradient, TextVariable, Trajectory


class TextualCritic(Protocol):
    def critique(
        self,
        trajectory: Trajectory,
        text_variables: dict[str, TextVariable],
    ) -> list[TextualGradient]: ...

