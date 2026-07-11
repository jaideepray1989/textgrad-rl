"""One-game TextWorld probe for structured controller rules with qwen2.5:7b.

The goal is to test whether TextGrad-RL should optimize structured controller
rules instead of only a flat prompt for a frozen SLM actor.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textgrad_rl.benchmarks.textarena_slm_compare import OpenAICompatibleChatModel
from textgrad_rl.benchmarks.textworld_24_suite import (
    TextWorldAgentState,
    cooking_action,
    default_specs,
    ensure_games,
    graph_exploration_action,
    match_wanted_action,
    parse_objective_actions,
    parse_objective_piece,
    sorted_unique,
    update_room_state,
)
from textgrad_rl.benchmarks.textworld_slm_suites import (
    complete_with_hard_timeout,
    initial_variables as initial_slm_variables,
    parse_indexed_action,
    run_textworld_episode,
    textworld_observation,
)
from textgrad_rl.utils.json_utils import write_json


@dataclass
class ProbeRecord:
    variant: str
    success: bool
    reward: float
    final_score: float
    max_score: float
    turns: int
    invalid: bool
    repeated: bool
    timeout_or_parse_failures: int
    qwen_calls: int
    fallback_calls: int
    runtime_seconds: float
    actions: list[str]
    raw_outputs: list[str]
    failure_reason: str


def patched_objective_actions(objective: str, *, textgrad_rule: bool) -> list[str]:
    if not textgrad_rule:
        return list(parse_objective_actions(objective))
    return ordered_textgrad_objective_actions(objective)


def ordered_textgrad_objective_actions(objective: str) -> list[str]:
    """Parse objective clauses into executable subgoals while preserving order."""

    text = re.sub(r"\s+", " ", objective)
    clauses = re.split(
        r"(?<=[.!?])\s+| then,? | and then,? | after that,? | following that,? | next,? | "
        r"with that (?:done|accomplished|over with),? | that done,? | once you (?:do|finish) that,? | "
        r"if you can [^,]+,\s*",
        text,
        flags=re.IGNORECASE,
    )
    actions: list[str] = []
    for clause in clauses:
        for action in ordered_clause_actions(clause):
            if action and action != "put it in places" and not same_as_previous(actions, action):
                actions.append(action)
    return actions


def ordered_clause_actions(clause: str) -> list[str]:
    lower = clause.lower().strip(" .!?:;")
    if not lower:
        return []
    events: list[tuple[int, str]] = []
    for pattern in [
        r"\b(?:put|place|rest|leave) (?:the |a |an )?(.+?) (on|in) (?:the |a |an )?(.+?)(?: within| inside|[.,;]|$)",
    ]:
        for match in re.finditer(pattern, lower):
            events.append(
                (
                    match.start(),
                    f"put {clean_object(match.group(1))} {match.group(2)} {clean_object(match.group(3))}",
                )
            )
    for match in re.finditer(
        r"\b(?:go(?: to)?|travel|head|move|venture|take a trip)(?: to)? (north|south|east|west)\b",
        lower,
    ):
        events.append((match.start(), f"go {match.group(1)}"))
    for match in re.finditer(
        r"\b(?:try|attempt|make an effort|make an attempt)[^.!?;]{0,80}?\b(north|south|east|west)\b",
        lower,
    ):
        events.append((match.start(), f"go {match.group(1)}"))
    for match in re.finditer(
        r"\b(?:pick up|get your hands on|recover|retrieve|lift|take(?! a trip)|pick-up) "
        r"(?:the |a |an )?(.+?)(?: from| on| in|[.,;]|$)",
        lower,
    ):
        events.append((match.start(), f"take {clean_object(match.group(1))}"))
    for match in re.finditer(
        r"\b(?:unlock|unlocked) (?:the |a |an )?(.+?) with (?:the |a |an )?(.+?)(?:[.,;]|$)",
        lower,
    ):
        events.append((match.start(), f"unlock {clean_object(match.group(1))} with {clean_object(match.group(2))}"))
    for match in re.finditer(r"\bunlock (?:the |a |an )?([a-z][a-z0-9 -]+?)(?:[.,;]|$)", lower):
        if " with " not in match.group(0):
            events.append((match.start(), f"unlock {clean_object(match.group(1))}"))
    for match in re.finditer(
        r"\b(?:assure|ensure|check|doublecheck|look and see|make it so) that (?:the |a |an )?(.+?)"
        r"(?: (?:inside|within|in) .+?)? is (?:wide open|open|opened|ajar)\b",
        lower,
    ):
        events.append((match.start(), f"open {clean_object(match.group(1))}"))

    parsed = parse_objective_piece(lower)
    if parsed and not events:
        events.append((0, parsed))
    ordered: list[str] = []
    seen_at_position: set[tuple[int, str]] = set()
    for start, action in sorted(events, key=lambda item: (item[0], action_priority(item[1]))):
        key = (start, action)
        if key in seen_at_position or not action:
            continue
        seen_at_position.add(key)
        if not same_as_previous(ordered, action):
            ordered.append(action)
    return ordered


def action_priority(action: str) -> int:
    if action.startswith("unlock "):
        return 0
    if action.startswith("open "):
        return 1
    if action.startswith("take "):
        return 2
    if action.startswith("go "):
        return 3
    if action.startswith("put "):
        return 4
    return 5


def same_as_previous(actions: list[str], action: str) -> bool:
    return bool(actions and actions[-1] == action)


def clean_object(text: str) -> str:
    text = re.sub(r"\b(?:inside|within|from|floor|room|of)\b.*$", "", text)
    text = re.sub(r"\b(the|a|an|some)\b", " ", text)
    text = re.sub(r"[^a-z0-9 -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class StructuredController:
    def __init__(self, objective: str, *, textgrad_rule: bool) -> None:
        self.objective = objective
        self.state = TextWorldAgentState(objective_actions=patched_objective_actions(objective, textgrad_rule=textgrad_rule))
        self.textgrad_rule = textgrad_rule
        objective_lower = objective.lower()
        self.use_cooking_rule = textgrad_rule and (
            "cookbook" in objective_lower or "recipe" in objective_lower or "delicious meal" in objective_lower
        )
        self.last_target = ""

    def candidates(self, state: Any) -> list[str]:
        admissible = sorted_unique(str(command) for command in (state.admissible_commands or []))
        if not admissible:
            return ["look"]
        update_room_state(self.state, str(state.description or state.feedback or ""))
        candidates: list[str] = []
        objective = self.objective_candidate(admissible)
        if objective:
            candidates.append(objective)
        if self.use_cooking_rule:
            cooking = cooking_action(self.state, state, admissible)
            if cooking:
                candidates.append(cooking)
        candidates.extend(priority_candidates(admissible, self.state.previous_actions))
        explore = graph_exploration_action(self.state, admissible)
        if explore:
            candidates.append(explore)
        candidates.extend(action for action in admissible if action not in self.state.previous_actions)
        candidates.extend(admissible)
        return dedupe([action for action in candidates if action in set(admissible)])[:8]

    def objective_candidate(self, admissible: list[str]) -> str:
        for index in range(self.state.objective_index, len(self.state.objective_actions)):
            wanted = self.state.objective_actions[index]
            match = match_wanted_action(wanted, admissible)
            if match:
                self.state.objective_index = index + 1
                self.last_target = wanted
                return match
        self.last_target = ""
        return ""

    def record(self, action: str) -> None:
        if action.startswith("go "):
            self.state.last_move_direction = action.split(maxsplit=1)[1]
        elif action.startswith("open "):
            self.state.opened_actions.add(action)
            self.state.last_move_direction = None
        else:
            self.state.last_move_direction = None
        self.state.previous_actions.append(action)


def priority_candidates(admissible: list[str], previous: list[str]) -> list[str]:
    previous_set = set(previous)
    patterns = [
        r"^unlock .+ with .+",
        r"^open .+",
        r"^take .*(key|coin|milk|cookbook)",
        r"^put .+ (?:on|in) .+",
        r"^go .+",
        r"^read .+",
    ]
    out: list[str] = []
    for pattern in patterns:
        out.extend(action for action in admissible if re.match(pattern, action) and action not in previous_set)
    return out


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def qwen_choose(
    *,
    model: OpenAICompatibleChatModel,
    problem_id: str,
    objective: str,
    observation: str,
    controller: StructuredController,
    candidates: list[str],
    timeout: int,
) -> tuple[str, str, bool]:
    candidate_text = "\n".join(
        f"[{index}] {action}{'  # controller recommendation' if index == 1 else ''}"
        for index, action in enumerate(candidates, start=1)
    )
    previous = ", ".join(controller.state.previous_actions[-8:]) if controller.state.previous_actions else "<none>"
    prompt = (
        f"PROBLEM: {problem_id}\n"
        f"OBJECTIVE: {objective}\n"
        f"PARSED SUBGOALS: {controller.state.objective_actions}\n"
        f"CURRENT TARGET: {controller.last_target or '<controller is exploring>'}\n"
        f"PREVIOUS ACTIONS: {previous}\n\n"
        f"OBSERVATION:\n{observation}\n\n"
        f"CANDIDATE ACTIONS:\n{candidate_text}\n\n"
        "Choose the action that advances the next unmet objective clause. "
        "Avoid repeated no-progress examine/inventory actions. Return only one bracketed number."
    )
    try:
        raw = model.complete(prompt) if timeout <= 0 else complete_with_hard_timeout(model, prompt, timeout)
    except Exception as exc:
        return candidates[0], f"FALLBACK: {type(exc).__name__}: {exc}", True
    action = parse_indexed_action(raw, candidates)
    if action is None:
        return candidates[0], f"FALLBACK_PARSE: {raw}", True
    return action, raw, False


def run_structured_variant(
    *,
    variant: str,
    game_path: Path,
    model: OpenAICompatibleChatModel | None,
    use_qwen_ranker: bool,
    textgrad_rule: bool,
    timeout: int,
    max_steps: int,
) -> ProbeRecord:
    import textworld

    infos = textworld.EnvInfos(
        admissible_commands=True,
        objective=True,
        inventory=True,
        description=True,
        score=True,
        max_score=True,
        won=True,
        lost=True,
        intermediate_reward=True,
        moves=True,
    )
    started = time.perf_counter()
    env = textworld.start(str(game_path), infos)
    actions: list[str] = []
    raw_outputs: list[str] = []
    invalid = False
    failure = ""
    final_score = 0.0
    max_score = 1.0
    success = False
    fallback_calls = 0
    qwen_calls = 0
    turns = 0
    try:
        state = env.reset()
        max_score = float(state.max_score or 1.0)
        controller = StructuredController(str(state.objective or ""), textgrad_rule=textgrad_rule)
        for _ in range(max_steps):
            admissible = set(str(command) for command in (state.admissible_commands or []))
            candidates = controller.candidates(state)
            if use_qwen_ranker:
                assert model is not None
                qwen_calls += 1
                action, raw, fallback = qwen_choose(
                    model=model,
                    problem_id=game_path.stem,
                    objective=str(state.objective or ""),
                    observation=textworld_observation(state),
                    controller=controller,
                    candidates=candidates,
                    timeout=timeout,
                )
                fallback_calls += int(fallback)
            else:
                action, raw = candidates[0], f"controller:{candidates[0]}"
            raw_outputs.append(raw)
            if action not in admissible:
                invalid = True
                failure = f"invalid action: {action}"
                break
            state, _score, done = env.step(action)
            controller.record(action)
            actions.append(action)
            final_score = float(state.score or 0.0)
            success = bool(state.won) or final_score >= max_score
            turns += 1
            if done:
                if not success and bool(state.lost):
                    failure = "lost"
                break
        if not success and not failure:
            failure = "step budget exhausted" if turns >= max_steps else "score below success"
    except Exception as exc:
        invalid = True
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    return ProbeRecord(
        variant=variant,
        success=success,
        reward=final_score / max(max_score, 1.0),
        final_score=final_score,
        max_score=max_score,
        turns=turns,
        invalid=invalid,
        repeated=len(actions) != len(set(actions)),
        timeout_or_parse_failures=fallback_calls,
        qwen_calls=qwen_calls,
        fallback_calls=fallback_calls,
        runtime_seconds=time.perf_counter() - started,
        actions=actions,
        raw_outputs=raw_outputs,
        failure_reason=failure,
    )


def summarize(record: ProbeRecord) -> dict[str, Any]:
    return {
        "variant": record.variant,
        "success": record.success,
        "reward": record.reward,
        "final_score": record.final_score,
        "max_score": record.max_score,
        "turns": record.turns,
        "invalid": record.invalid,
        "repeated": record.repeated,
        "qwen_calls": record.qwen_calls,
        "fallback_calls": record.fallback_calls,
        "runtime_seconds": record.runtime_seconds,
        "failure_reason": record.failure_reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, records: list[ProbeRecord]) -> None:
    lines = [
        "# TextWorld simple_1 Structured Controller Probe",
        "",
        "Problem: `simple_1_dense_detailed` from the TextWorld 24 suite.",
        "",
        "| variant | success | reward | score | turns | invalid | repeated | qwen calls | fallbacks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record.variant} | {int(record.success)} | {record.reward:.3f} | "
            f"{record.final_score:.1f}/{record.max_score:.1f} | {record.turns} | {int(record.invalid)} | "
            f"{int(record.repeated)} | {record.qwen_calls} | {record.fallback_calls} |"
        )
    lines.extend(["", "## Action Traces", ""])
    for record in records:
        lines.append(f"### {record.variant}")
        lines.append("")
        lines.append(f"- Failure reason: `{record.failure_reason}`")
        lines.append(f"- Actions: `{' -> '.join(record.actions)}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/textworld24_simple1_structured_probe")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = [item for item in default_specs(62001) if item.spec_id == "simple_1_dense_detailed"][0]
    game_paths = ensure_games([spec], output_dir / "games")
    model = OpenAICompatibleChatModel(
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=16,
        timeout=120,
    )

    records: list[ProbeRecord] = []
    raw = run_textworld_episode(
        spec=spec,
        game_path=game_paths[spec.spec_id],
        method="raw_qwen",
        split="probe",
        model=model,
        variables=initial_slm_variables(),
        max_steps=args.max_steps,
        model_call_timeout=args.timeout,
    )
    records.append(
        ProbeRecord(
            variant="raw_qwen",
            success=raw.success,
            reward=raw.reward,
            final_score=raw.final_score,
            max_score=raw.max_score,
            turns=raw.turns,
            invalid=raw.invalid_action,
            repeated=raw.repeated_actions,
            timeout_or_parse_failures=int(raw.parse_failed),
            qwen_calls=len(raw.raw_outputs),
            fallback_calls=sum(output.startswith("REQUEST_") for output in raw.raw_outputs),
            runtime_seconds=raw.runtime_seconds,
            actions=raw.actions,
            raw_outputs=raw.raw_outputs,
            failure_reason=raw.failure_reason,
        )
    )
    records.append(
        run_structured_variant(
            variant="controller_direct_base",
            game_path=game_paths[spec.spec_id],
            model=None,
            use_qwen_ranker=False,
            textgrad_rule=False,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
    )
    records.append(
        run_structured_variant(
            variant="controller_direct_textgrad_rule",
            game_path=game_paths[spec.spec_id],
            model=None,
            use_qwen_ranker=False,
            textgrad_rule=True,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
    )
    records.append(
        run_structured_variant(
            variant="qwen_ranker_base",
            game_path=game_paths[spec.spec_id],
            model=model,
            use_qwen_ranker=True,
            textgrad_rule=False,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
    )
    records.append(
        run_structured_variant(
            variant="qwen_ranker_textgrad_rule",
            game_path=game_paths[spec.spec_id],
            model=model,
            use_qwen_ranker=True,
            textgrad_rule=True,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
    )

    write_json(output_dir / "records.json", records)
    write_csv(output_dir / "summary.csv", [summarize(record) for record in records])
    write_markdown(output_dir / "summary.md", records)
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
