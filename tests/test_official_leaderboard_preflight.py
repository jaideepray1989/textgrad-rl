from pathlib import Path

from textgrad_rl.benchmarks.official_leaderboard_preflight import (
    collect_preflights,
    parse_suites,
    parse_targets,
)


def test_parse_suites_accepts_aliases() -> None:
    assert parse_suites("taubench,swe-bench,webarena") == ["tau2", "swebench", "browser"]


def test_parse_targets_rejects_unknown_browser_target() -> None:
    try:
        parse_targets("webarena,nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("Expected unknown browser target error")


def test_tau2_preflight_blocks_without_cli_or_llm() -> None:
    status = collect_preflights(
        ["tau2"],
        env={},
        which=lambda _name: None,
        module_check=lambda _name: False,
    )[0]

    assert not status.can_run
    blockers = {item.name for item in status.requirements if item.required and not item.ok}
    assert "tau2 CLI or package" in blockers
    assert "agent/user simulator LLM access" in blockers


def test_tau2_preflight_passes_with_cli_and_key() -> None:
    status = collect_preflights(
        ["tau2"],
        env={"OPENAI_API_KEY": "sk-test"},
        which=lambda name: f"/bin/{name}" if name == "tau2" else None,
        module_check=lambda _name: False,
    )[0]

    assert status.can_run


def test_swebench_preflight_allows_gold_harness_validation() -> None:
    status = collect_preflights(
        ["swebench"],
        env={"SWE_BENCH_PREDICTIONS_PATH": "gold"},
        which=lambda name: f"/bin/{name}" if name == "docker" else None,
        module_check=lambda name: name == "swebench",
        command_ok=lambda command: list(command) == ["docker", "info"],
    )[0]

    assert status.can_run


def test_browser_preflight_can_require_only_webarena(tmp_path: Path) -> None:
    repo = tmp_path / "webarena"
    (repo / "scripts").mkdir(parents=True)
    (repo / "run.py").write_text("", encoding="utf-8")
    (repo / "scripts" / "generate_test_data.py").write_text("", encoding="utf-8")
    env = {name: "http://localhost" for name in ["SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE"]}
    env["OPENAI_API_KEY"] = "sk-test"
    env["WEBARENA_REPO"] = str(repo)

    status = collect_preflights(
        ["browser"],
        env=env,
        which=lambda name: f"/bin/{name}" if name == "playwright" else None,
        module_check=lambda _name: False,
        browser_targets=["webarena"],
    )[0]

    assert status.can_run


def test_browser_preflight_reports_workarena_credentials() -> None:
    status = collect_preflights(
        ["browser"],
        env={},
        which=lambda _name: None,
        module_check=lambda name: name == "browsergym.workarena",
        browser_targets=["workarena"],
    )[0]

    assert not status.can_run
    blockers = {item.name for item in status.requirements if item.required and not item.ok}
    assert "WorkArena ServiceNow credentials" in blockers
    assert "Playwright browser runtime" in blockers
