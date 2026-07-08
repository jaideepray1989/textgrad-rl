"""Preflight checks for official external benchmark executions.

The local transfer probes in this repository are lightweight research
protocols. Official leaderboard runs need the benchmark owners' harnesses,
credentials, service URLs, and sometimes Docker. This module makes those
requirements explicit before a launch script starts a costly run.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from textgrad_rl.utils.json_utils import write_json


Env = Mapping[str, str]
ExecutableFinder = Callable[[str], str | None]
ModuleChecker = Callable[[str], bool]
PathChecker = Callable[[str | Path], bool]
CommandChecker = Callable[[Sequence[str]], bool]


WEB_ARENA_URL_VARS = (
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "MAP",
    "WIKIPEDIA",
    "HOMEPAGE",
)
WORKARENA_ENV_VARS = ("SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD")
LLM_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "TEXTGRAD_RL_LLM_API_KEY",
)
LOCAL_LLM_ENDPOINT_VARS = ("TEXTGRAD_RL_LLM_BASE_URL", "OPENAI_BASE_URL", "OLLAMA_HOST")
DEFAULT_SUITES = ("browser", "tau2", "swebench")


@dataclass(frozen=True)
class Requirement:
    name: str
    ok: bool
    detail: str
    fix: str
    required: bool = True


@dataclass(frozen=True)
class SuitePreflight:
    suite: str
    display_name: str
    can_run: bool
    requirements: list[Requirement]
    launch_script: str
    launch_command: list[str]
    official_reference: str
    notes: list[str]


def module_available(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def executable_path(name: str) -> str | None:
    return shutil.which(name)


def path_exists(path: str | Path) -> bool:
    return Path(path).expanduser().exists()


def command_succeeds(command: Sequence[str]) -> bool:
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=8)
    except Exception:
        return False
    return True


def has_any_env(env: Env, names: Sequence[str]) -> bool:
    return any(bool(env.get(name)) for name in names)


def missing_env(env: Env, names: Sequence[str]) -> list[str]:
    return [name for name in names if not env.get(name)]


def requirement(name: str, ok: bool, detail: str, fix: str, required: bool = True) -> Requirement:
    return Requirement(name=name, ok=ok, detail=detail, fix=fix, required=required)


def suite_can_run(requirements: Sequence[Requirement]) -> bool:
    return all(item.ok for item in requirements if item.required)


def build_browser_preflight(
    env: Env,
    which: ExecutableFinder = executable_path,
    module_check: ModuleChecker = module_available,
    exists: PathChecker = path_exists,
    targets: Sequence[str] = ("webarena", "workarena"),
) -> SuitePreflight:
    normalized_targets = tuple(target.strip().lower() for target in targets if target.strip())
    requirements: list[Requirement] = []
    notes: list[str] = []

    if "webarena" in normalized_targets:
        webarena_repo = Path(env.get("WEBARENA_REPO", "external/webarena")).expanduser()
        missing_urls = missing_env(env, WEB_ARENA_URL_VARS)
        requirements.extend(
            [
                requirement(
                    "WebArena repository",
                    exists(webarena_repo / "run.py") and exists(webarena_repo / "scripts" / "generate_test_data.py"),
                    f"WEBARENA_REPO={webarena_repo}",
                    "Clone https://github.com/web-arena-x/webarena and set WEBARENA_REPO to that checkout.",
                ),
                requirement(
                    "WebArena site URLs",
                    not missing_urls,
                    "missing=" + ",".join(missing_urls) if missing_urls else "all WebArena URL env vars set",
                    "Set SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, and HOMEPAGE.",
                ),
                requirement(
                    "WebArena OpenAI key",
                    bool(env.get("OPENAI_API_KEY")),
                    "OPENAI_API_KEY is set" if env.get("OPENAI_API_KEY") else "OPENAI_API_KEY is missing",
                    "Set OPENAI_API_KEY or adapt the official WebArena runner to the target provider.",
                ),
            ]
        )
    else:
        notes.append("WebArena checks skipped because it is not in BROWSER_OFFICIAL_TARGETS.")

    if "workarena" in normalized_targets:
        missing_workarena_env = missing_env(env, WORKARENA_ENV_VARS)
        requirements.extend(
            [
                requirement(
                    "WorkArena Python package",
                    module_check("browsergym.workarena"),
                    "browsergym.workarena import check",
                    "Install with `pip install browsergym-workarena` in the environment used by the runner.",
                ),
                requirement(
                    "WorkArena ServiceNow credentials",
                    not missing_workarena_env,
                    "missing=" + ",".join(missing_workarena_env)
                    if missing_workarena_env
                    else "ServiceNow URL, username, and password env vars set",
                    "Set SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME, and SNOW_INSTANCE_PWD.",
                ),
                requirement(
                    "workarena-install command",
                    which("workarena-install") is not None,
                    "workarena-install is on PATH" if which("workarena-install") else "workarena-install not found",
                    "Run the BrowserGym WorkArena install in this Python environment.",
                    required=False,
                ),
            ]
        )
    else:
        notes.append("WorkArena checks skipped because it is not in BROWSER_OFFICIAL_TARGETS.")

    has_playwright = which("playwright") is not None or module_check("playwright")
    requirements.append(
        requirement(
            "Playwright browser runtime",
            has_playwright,
            "playwright executable/module check",
            "Install Playwright and run `playwright install chromium`.",
        )
    )

    command = [
        "bash",
        "scripts/run_official_browser_benchmarks.sh",
        "--launch",
    ]
    return SuitePreflight(
        suite="browser",
        display_name="WebArena/WorkArena official browser benchmarks",
        can_run=suite_can_run(requirements),
        requirements=requirements,
        launch_script="scripts/run_official_browser_benchmarks.sh",
        launch_command=command,
        official_reference="https://github.com/web-arena-x/webarena and https://github.com/ServiceNow/BrowserGym",
        notes=notes
        + [
            "Set BROWSER_OFFICIAL_TARGETS=webarena,workarena to require both targets, or a single target for a partial official run.",
            "The script defaults to preflight mode; pass --launch after requirements are satisfied.",
        ],
    )


def build_tau2_preflight(
    env: Env,
    which: ExecutableFinder = executable_path,
    module_check: ModuleChecker = module_available,
) -> SuitePreflight:
    has_tau2 = which("tau2") is not None or module_check("tau2")
    has_llm = has_any_env(env, LLM_KEY_VARS) or has_any_env(env, LOCAL_LLM_ENDPOINT_VARS)
    domains = env.get("TAU2_DOMAINS", "retail,airline,telecom,banking_knowledge")
    requirements = [
        requirement(
            "tau2 CLI or package",
            has_tau2,
            "tau2 executable/package check",
            "Install tau2-bench from https://github.com/sierra-research/tau2-bench.",
        ),
        requirement(
            "agent/user simulator LLM access",
            has_llm,
            "found an LLM API key or local endpoint" if has_llm else "no provider key or local endpoint env var found",
            "Set an LLM key such as OPENAI_API_KEY, or configure a compatible local endpoint.",
        ),
        requirement(
            "official domains selected",
            bool(domains.strip()),
            f"TAU2_DOMAINS={domains}",
            "Use retail, airline, telecom, and banking_knowledge for a complete current text submission.",
        ),
    ]
    return SuitePreflight(
        suite="tau2",
        display_name="tau2-bench official text leaderboard",
        can_run=suite_can_run(requirements),
        requirements=requirements,
        launch_script="scripts/run_official_taubench.sh",
        launch_command=["bash", "scripts/run_official_taubench.sh", "--launch"],
        official_reference="https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md",
        notes=[
            "Official submissions should use consistent agent/user simulator settings across domains.",
            "The leaderboard guide strongly prefers 4+ trials per domain.",
        ],
    )


def build_swebench_preflight(
    env: Env,
    which: ExecutableFinder = executable_path,
    module_check: ModuleChecker = module_available,
    command_ok: CommandChecker = command_succeeds,
) -> SuitePreflight:
    docker_cli = which("docker") is not None
    docker_ready = docker_cli and command_ok(("docker", "info"))
    predictions_path = env.get("SWE_BENCH_PREDICTIONS_PATH", "")
    predictions_ok = bool(predictions_path) and (
        predictions_path == "gold" or Path(predictions_path).expanduser().exists()
    )
    try:
        free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    except Exception:
        free_gb = 0.0
    requirements = [
        requirement(
            "Docker CLI",
            docker_cli,
            "docker executable check",
            "Install Docker Desktop or Docker Engine.",
        ),
        requirement(
            "Docker daemon",
            docker_ready,
            "docker info check passed" if docker_ready else "docker info failed or Docker is not running",
            "Start Docker before launching the SWE-bench harness.",
        ),
        requirement(
            "SWE-bench package",
            module_check("swebench"),
            "swebench import check",
            "Install SWE-bench from https://github.com/SWE-bench/SWE-bench.",
        ),
        requirement(
            "predictions JSONL",
            predictions_ok,
            f"SWE_BENCH_PREDICTIONS_PATH={predictions_path or '<missing>'}",
            "Set SWE_BENCH_PREDICTIONS_PATH to a model predictions JSONL file, or `gold` for harness validation only.",
        ),
        requirement(
            "recommended free disk",
            free_gb >= 100.0,
            f"{free_gb:.1f} GB free in current filesystem",
            "Official SWE-bench docs recommend roughly 120 GB free for Docker evaluation.",
            required=False,
        ),
    ]
    notes = [
        "The runner targets princeton-nlp/SWE-bench_Lite by default.",
        "On ARM machines, the shell runner automatically passes an empty namespace unless SWE_BENCH_NAMESPACE is set.",
    ]
    if platform.machine().lower() in {"arm64", "aarch64"}:
        notes.append("ARM64 detected; official SWE-bench marks ARM support as experimental.")
    return SuitePreflight(
        suite="swebench",
        display_name="SWE-bench Lite official harness",
        can_run=suite_can_run(requirements),
        requirements=requirements,
        launch_script="scripts/run_official_swebench_lite.sh",
        launch_command=["bash", "scripts/run_official_swebench_lite.sh", "--launch"],
        official_reference="https://github.com/SWE-bench/SWE-bench",
        notes=notes,
    )


def parse_suites(value: str) -> list[str]:
    aliases = {
        "all": list(DEFAULT_SUITES),
        "browser": ["browser"],
        "webarena": ["browser"],
        "workarena": ["browser"],
        "tau": ["tau2"],
        "tau2": ["tau2"],
        "taubench": ["tau2"],
        "tau-bench": ["tau2"],
        "swe": ["swebench"],
        "swebench": ["swebench"],
        "swe-bench": ["swebench"],
    }
    suites: list[str] = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown official suite: {part}")
        suites.extend(aliases[key])
    deduped = []
    for suite in suites or list(DEFAULT_SUITES):
        if suite not in deduped:
            deduped.append(suite)
    return deduped


def parse_targets(value: str) -> list[str]:
    valid = {"webarena", "workarena"}
    targets = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(targets) - valid)
    if unknown:
        raise ValueError(f"Unknown browser target: {', '.join(unknown)}")
    return targets or ["webarena", "workarena"]


def collect_preflights(
    suites: Sequence[str],
    env: Env | None = None,
    which: ExecutableFinder = executable_path,
    module_check: ModuleChecker = module_available,
    exists: PathChecker = path_exists,
    command_ok: CommandChecker = command_succeeds,
    browser_targets: Sequence[str] = ("webarena", "workarena"),
) -> list[SuitePreflight]:
    selected_env = env if env is not None else os.environ
    statuses: list[SuitePreflight] = []
    for suite in suites:
        if suite == "browser":
            statuses.append(build_browser_preflight(selected_env, which, module_check, exists, browser_targets))
        elif suite == "tau2":
            statuses.append(build_tau2_preflight(selected_env, which, module_check))
        elif suite == "swebench":
            statuses.append(build_swebench_preflight(selected_env, which, module_check, command_ok))
        else:
            raise ValueError(f"Unknown official suite: {suite}")
    return statuses


def blocking_requirements(status: SuitePreflight) -> list[Requirement]:
    return [item for item in status.requirements if item.required and not item.ok]


def render_markdown(statuses: Sequence[SuitePreflight]) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Official Leaderboard Preflight",
        "",
        f"Generated: {generated}",
        "",
        "| Suite | Can Launch Now | Blocking Requirements | Launch Script |",
        "|---|---:|---|---|",
    ]
    for status in statuses:
        blockers = blocking_requirements(status)
        blocker_text = ", ".join(item.name for item in blockers) if blockers else "none"
        lines.append(f"| {status.display_name} | {'yes' if status.can_run else 'no'} | {blocker_text} | `{status.launch_script}` |")

    for status in statuses:
        lines.extend(["", f"## {status.display_name}", "", f"Official reference: {status.official_reference}", ""])
        lines.append("Requirements:")
        for item in status.requirements:
            mark = "x" if item.ok else " "
            required = "required" if item.required else "recommended"
            lines.append(f"- [{mark}] {item.name} ({required}): {item.detail}")
            if not item.ok:
                lines.append(f"  Fix: {item.fix}")
        lines.extend(["", "Launch command:", "", "```bash", " ".join(status.launch_command), "```"])
        if status.notes:
            lines.extend(["", "Notes:"])
            for note in status.notes:
                lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> tuple[Path, list[SuitePreflight]]:
    suites = parse_suites(args.suite)
    browser_targets = parse_targets(args.browser_targets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = collect_preflights(suites, browser_targets=browser_targets)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.executable,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "suites": statuses,
    }
    write_json(output_dir / "official_leaderboard_preflight.json", payload)
    (output_dir / "official_leaderboard_preflight.md").write_text(render_markdown(statuses), encoding="utf-8")
    return output_dir, statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether official external leaderboard harnesses can run.")
    parser.add_argument("--suite", default="all", help="Comma-separated suite list: browser,tau2,swebench,all.")
    parser.add_argument(
        "--browser-targets",
        default=os.environ.get("BROWSER_OFFICIAL_TARGETS", "webarena,workarena"),
        help="Comma-separated browser targets: webarena,workarena.",
    )
    parser.add_argument("--output-dir", default="runs/official_leaderboard_preflight")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any selected suite cannot launch.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir, statuses = run(args)
    for status in statuses:
        blockers = blocking_requirements(status)
        if blockers:
            names = ", ".join(item.name for item in blockers)
            print(f"{status.suite}: cannot launch ({names})")
        else:
            print(f"{status.suite}: ready to launch")
    print(f"Preflight artifacts saved to {output_dir}")
    if args.strict and any(not status.can_run for status in statuses):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
