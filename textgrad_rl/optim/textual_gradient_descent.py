"""Textual gradient descent over modular prompts."""

from __future__ import annotations

import copy
import re
from typing import Literal

from textgrad_rl.types import TextualGradient, TextVariable


class TextualGradientDescent:
    """Apply trajectory-level textual gradients by editing text variables only."""

    def __init__(
        self,
        max_prompt_chars: int = 2500,
        max_rules_per_step: int = 3,
        mode: Literal["heuristic", "llm"] = "heuristic",
    ) -> None:
        self.max_prompt_chars = max_prompt_chars
        self.max_rules_per_step = max_rules_per_step
        self.mode = mode

    def step(
        self,
        text_variables: dict[str, TextVariable],
        gradients: list[TextualGradient],
        constraints: list[str],
    ) -> dict[str, TextVariable]:
        updated = copy.deepcopy(text_variables)
        grouped: dict[str, list[TextualGradient]] = {}
        for gradient in gradients:
            grouped.setdefault(gradient.target_variable_name, []).append(gradient)

        for name, target_gradients in grouped.items():
            if name not in updated or not updated[name].requires_grad:
                continue
            variable = updated[name]
            existing_rules = {_normalize_rule(rule) for rule in _extract_rules(variable.value)}
            additions: list[str] = []
            for gradient in target_gradients[: self.max_rules_per_step]:
                rule = _clean_suggested_edit(gradient.suggested_edit or gradient.gradient_text)
                if not rule or _violates_constraints(rule, constraints):
                    continue
                normalized = _normalize_rule(rule)
                if normalized in existing_rules:
                    continue
                existing_rules.add(normalized)
                additions.append(rule)
                variable.gradient_history.append(f"{gradient.failure_mode}: {gradient.gradient_text}")
            if additions:
                if "Learned rules:" not in variable.value:
                    variable.value = variable.value.rstrip() + "\n\nLearned rules:"
                for rule in additions:
                    variable.value += f"\n- {rule}"
                variable.version += 1
                variable.max_chars = min(variable.max_chars, self.max_prompt_chars)
                variable.value = _trim_prompt(variable.value, variable.max_chars)
        return updated


def _extract_rules(text: str) -> list[str]:
    return [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _clean_suggested_edit(text: str) -> str:
    text = re.sub(r"^\s*add a rule:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".") + "."


def _normalize_rule(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _violates_constraints(rule: str, constraints: list[str]) -> bool:
    lowered = rule.lower()
    forbidden_tokens = ["hidden_validation.py", "hardcode the answer", "edit tests to pass"]
    if any(token in lowered for token in forbidden_tokens):
        return True
    return any("must not" in constraint.lower() and constraint.lower().replace("must not", "").strip() in lowered for constraint in constraints)


def _trim_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    header, _, learned = text.partition("Learned rules:")
    learned = learned[-max(0, max_chars - len(header) - 20) :]
    return (header.rstrip() + "\n\nLearned rules:" + learned)[-max_chars:]

