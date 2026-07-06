"""Security checks for local repair task sandboxes."""

from __future__ import annotations

import os
from pathlib import Path


SUSPICIOUS_EDIT_SNIPPETS = (
    "pytest.skip",
    "@pytest.mark.skip",
    "unittest.skip",
    "hidden_validation.py",
    "subprocess.run(['true']",
    "subprocess.run([\"true\"]",
    "socket.",
    "urllib.request",
    "requests.",
    "curl ",
    "wget ",
    "nvidia-smi",
    "cuda",
)


def normalize_relative_path(root: Path, relative: str | None) -> tuple[Path | None, str | None]:
    """Resolve a relative path and reject absolute/path-traversal writes."""

    if not relative:
        return None, "missing path"
    candidate = Path(relative)
    if candidate.is_absolute():
        return None, "absolute paths are not allowed"
    root_resolved = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(root / candidate))
    if os.path.commonpath([str(root_resolved), str(resolved)]) != str(root_resolved):
        return None, "path traversal outside task directory is not allowed"
    return resolved, None


def looks_like_forbidden_content(content: str) -> str | None:
    lowered = content.lower()
    for snippet in SUSPICIOUS_EDIT_SNIPPETS:
        if snippet.lower() in lowered:
            return f"forbidden edit content: {snippet}"
    return None
