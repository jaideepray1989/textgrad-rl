"""Parse JSON-only model outputs into validated environment actions."""

from __future__ import annotations

import json
import re
from typing import Any

from textgrad_rl.types import Action, VALID_ACTION_TYPES


def parse_action(model_output: str) -> Action:
    """Parse a model response as a structured action.

    The parser first tries strict JSON. If that fails, it extracts the first fenced
    JSON block or object-looking span. Invalid schemas become a clear noop action.
    """

    try:
        payload = json.loads(model_output)
    except json.JSONDecodeError:
        extracted = _extract_json(model_output)
        if extracted is None:
            return Action(type="noop", reason="invalid model output: no JSON action found")
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return Action(type="noop", reason=f"invalid model output: {exc}")
    try:
        return _action_from_payload(payload)
    except (TypeError, ValueError) as exc:
        return Action(type="noop", reason=f"invalid action schema: {exc}")


def _extract_json(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _action_from_payload(payload: Any) -> Action:
    if not isinstance(payload, dict):
        raise TypeError("action payload must be an object")
    action_type = payload.get("type")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unknown action type {action_type!r}")
    return Action(
        type=action_type,
        path=_optional_str(payload.get("path")),
        content=_optional_str(payload.get("content")),
        command=_optional_str(payload.get("command")),
        reason=_optional_str(payload.get("reason")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)

