from textgrad_rl.benchmarks.textarena_multienv_compare import EpisodeRecord
from textgrad_rl.benchmarks.textarena_textgrad_plus import (
    bootstrap_delta_ci,
    causal_assignments_from_records,
    record_score,
)


def _record(env_id: str, reward: float, success: bool, invalid: bool = False, turns: int = 5) -> EpisodeRecord:
    return EpisodeRecord(
        env_id=env_id,
        variant="unit",
        split="unit",
        seed=1,
        target_side=None,
        reward=reward,
        success=success,
        done=True,
        turns=turns,
        invalid_move=invalid,
        reason="",
        actions=[],
        runtime_seconds=0.0,
    )


def test_causal_assignment_targets_failed_environment_variable() -> None:
    assignments = causal_assignments_from_records(
        [
            _record("Nim-v0", 0.0, False),
            _record("GuessTheNumber-v0", 1.0, True),
        ]
    )

    assert len(assignments) == 1
    assert assignments[0].env_id == "Nim-v0"
    assert assignments[0].target_variable_name == "nim_strategy_prompt"


def test_bootstrap_delta_ci_tracks_candidate_improvement() -> None:
    old = [_record("Nim-v0", 0.0, False, turns=8) for _ in range(4)]
    new = [_record("Nim-v0", 1.0, True, turns=6) for _ in range(4)]
    mean, lo, hi = bootstrap_delta_ci(old, new, seed=7, samples=100)

    assert mean > 1.0
    assert lo > 1.0
    assert hi >= lo


def test_record_score_penalizes_invalid_moves() -> None:
    valid = _record("Mastermind-v0", 0.5, False, invalid=False)
    invalid = _record("Mastermind-v0", 0.5, False, invalid=True)

    assert record_score(valid) > record_score(invalid)
