"""Controlled subprocess execution for task environments."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    runtime_seconds: float
    timed_out: bool = False


def run_command(command: str, cwd: Path, timeout_sec: int, allow_network: bool = False) -> CommandResult:
    """Run an exact allow-listed command without invoking a shell."""

    argv = shlex.split(command)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    elif argv and argv[0] == "pytest":
        argv = [sys.executable, "-m", "pytest", *argv[1:]]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    if not allow_network:
        env.setdefault("TEXTGRAD_RL_OFFLINE", "1")
    start = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        runtime = time.time() - start
        return CommandResult(command, proc.returncode, proc.stdout, proc.stderr, runtime, False)
    except subprocess.TimeoutExpired as exc:
        runtime = time.time() - start
        return CommandResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout_sec}s.",
            runtime_seconds=runtime,
            timed_out=True,
        )
