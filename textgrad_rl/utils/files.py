"""Filesystem helpers used by generated task repositories."""

from __future__ import annotations

from pathlib import Path


def write_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def list_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def read_text_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

