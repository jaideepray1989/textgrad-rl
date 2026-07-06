"""Unified-diff helpers."""

from __future__ import annotations

import difflib


def unified_diff(original: str, current: str, path: str) -> str:
    if original == current:
        return ""
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )

