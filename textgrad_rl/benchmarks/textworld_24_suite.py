"""Local Microsoft TextWorld 24-game benchmark.

This suite generates local `.z8` games from TextWorld's built-in challenge
generators and evaluates fixed text prompts against TextGrad-RL policy updates.
It does not need API keys, hosted services, or benchmark credentials.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


METHODS = [
    "fixed_prompt",
    "retry_with_diagnostics",
    "ungated_persistent_rules",
    "textgrad_policy_iteration",
]
FAMILY_CATEGORIES = {
    "tw-simple": "objective_sequence",
    "tw-coin_collector": "navigation",
    "tw-treasure_hunter": "navigation_objective",
    "tw-cooking": "recipe_planning",
}


@dataclass(frozen=True)
class TextWorldSpec:
    spec_id: str
    family: str
    category: str
    seed: int
    args: tuple[str, ...]


@dataclass
class TextWorldRecord:
    benchmark: str
    spec_id: str
    family: str
    category: str
    method: str
    split: str
    seed: int
    success: bool
    reward: float
    invalid_action: bool
    repeated_actions: bool
    turns: int
    final_score: float
    max_score: float
    actions: list[str]
    failure_reason: str
    runtime_seconds: float
    attempts: int = 1
    total_turns: int = 0


@dataclass
class TextWorldAgentState:
    objective_actions: list[str] = field(default_factory=list)
    objective_index: int = 0
    previous_actions: list[str] = field(default_factory=list)
    visited_rooms: set[str] = field(default_factory=set)
    graph: dict[str, dict[str, str]] = field(default_factory=dict)
    last_room: str | None = None
    last_move_direction: str | None = None
    opened_actions: set[str] = field(default_factory=set)
    recipe_lines: list[str] = field(default_factory=list)
    recipe_index: int = 0
    found_cookbook: bool = False


def default_specs(seed: int = 62001) -> list[TextWorldSpec]:
    specs: list[TextWorldSpec] = []
    simple_settings = [
        ("dense", "detailed"),
        ("balanced", "detailed"),
        ("sparse", "detailed"),
        ("dense", "brief"),
        ("balanced", "brief"),
        ("sparse", "brief"),
    ]
    for index, (rewards, goal) in enumerate(simple_settings):
        specs.append(
            TextWorldSpec(
                spec_id=f"simple_{index + 1}_{rewards}_{goal}",
                family="tw-simple",
                category=FAMILY_CATEGORIES["tw-simple"],
                seed=seed + index,
                args=("tw-simple", "--rewards", rewards, "--goal", goal),
            )
        )
    for index, level in enumerate([1, 3, 5, 10, 20, 30]):
        specs.append(
            TextWorldSpec(
                spec_id=f"coin_level_{level}",
                family="tw-coin_collector",
                category=FAMILY_CATEGORIES["tw-coin_collector"],
                seed=seed + 100 + index,
                args=("tw-coin_collector", "--level", str(level)),
            )
        )
    for index, level in enumerate([1, 3, 5, 10, 15, 20]):
        specs.append(
            TextWorldSpec(
                spec_id=f"treasure_level_{level}",
                family="tw-treasure_hunter",
                category=FAMILY_CATEGORIES["tw-treasure_hunter"],
                seed=seed + 200 + index,
                args=("tw-treasure_hunter", "--level", str(level)),
            )
        )
    cooking_settings = [
        ("recipe1_take1_go1", ("--recipe", "1", "--take", "1", "--go", "1")),
        ("recipe1_take1_go6_open", ("--recipe", "1", "--take", "1", "--go", "6", "--open")),
        ("recipe2_take2_go6_open", ("--recipe", "2", "--take", "2", "--go", "6", "--open")),
        ("recipe2_take2_go6_cook", ("--recipe", "2", "--take", "2", "--go", "6", "--open", "--cook")),
        ("recipe3_take3_go9_cut", ("--recipe", "3", "--take", "3", "--go", "9", "--open", "--cook", "--cut")),
        ("recipe4_take4_go12_cut", ("--recipe", "4", "--take", "4", "--go", "12", "--open", "--cook", "--cut")),
    ]
    for index, (name, extra) in enumerate(cooking_settings):
        specs.append(
            TextWorldSpec(
                spec_id=f"cooking_{name}",
                family="tw-cooking",
                category=FAMILY_CATEGORIES["tw-cooking"],
                seed=seed + 300 + index,
                args=("tw-cooking", *extra, "--recipe-seed", str(seed + 900 + index), "--split", "train"),
            )
        )
    return specs


def initial_textworld_variables() -> dict[str, TextVariable]:
    return {
        "textworld_policy": TextVariable(
            name="textworld_policy",
            value=(
                "Use only admissible TextWorld commands. Prefer useful visible actions, take obvious target objects, "
                "open containers or doors when needed, and avoid repeating commands that did not change progress."
            ),
            role_description="Local TextWorld command policy.",
            max_chars=2600,
        )
    }


def policy_text(variables: dict[str, TextVariable]) -> str:
    return "\n".join(variable.value.lower() for variable in variables.values())


def learned_rule_ids(variables: dict[str, TextVariable]) -> set[str]:
    text = policy_text(variables)
    rules: set[str] = set()
    if "ordered subgoals" in text and "visible objective" in text:
        rules.add("objective_sequence")
    if "room graph" in text and "unvisited exits" in text:
        rules.add("graph_exploration")
    if "cookbook" in text and "recipe directions" in text:
        rules.add("cooking_recipe")
    return rules


RULE_TEXTS = {
    "objective_sequence": (
        "For TextWorld objective games, convert the visible objective into ordered subgoals: movement directions, "
        "open/unlock commands, take target objects, and put/place target objects on or in the requested destination."
    ),
    "graph_exploration": (
        "For TextWorld exploration, maintain a room graph, prefer unopened containers and unvisited exits, "
        "and avoid immediate backtracking unless no other admissible progress action exists."
    ),
    "cooking_recipe": (
        "For TextWorld cooking, search for the kitchen/cookbook, take and read the cookbook, collect listed ingredients, "
        "follow recipe directions for cutting/cooking, then prepare and eat the meal."
    ),
}


def tw_make_executable() -> str:
    local = Path(sys.executable).with_name("tw-make")
    if local.exists():
        return str(local)
    found = shutil.which("tw-make")
    if found:
        return found
    raise SystemExit("TextWorld CLI not found. Install with: python -m pip install textworld==1.7.0")


def ensure_games(specs: list[TextWorldSpec], games_dir: Path, force: bool = False) -> dict[str, Path]:
    games_dir.mkdir(parents=True, exist_ok=True)
    executable = tw_make_executable()
    paths: dict[str, Path] = {}
    for spec in specs:
        path = games_dir / f"{spec.spec_id}.z8"
        paths[spec.spec_id] = path
        if path.exists() and not force:
            continue
        command = [
            executable,
            *spec.args,
            "--seed",
            str(spec.seed),
            "--output",
            str(path),
            "--force",
            "--silent",
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return paths


class TextWorldPromptAgent:
    def __init__(self, variables: dict[str, TextVariable], objective: str) -> None:
        self.variables = variables
        self.state = TextWorldAgentState(objective_actions=parse_objective_actions(objective))

    def act(self, state: Any) -> str:
        admissible = sorted_unique(str(command) for command in (state.admissible_commands or []))
        if not admissible:
            return "look"
        update_room_state(self.state, str(state.description or state.feedback or ""))
        rules = learned_rule_ids(self.variables)
        if "objective_sequence" in rules:
            action = self.objective_action(admissible)
            if action:
                self.record(action)
                return action
        if "cooking_recipe" in rules:
            action = cooking_action(self.state, state, admissible)
            if action:
                self.record(action)
                return action
        if "graph_exploration" in rules:
            action = graph_exploration_action(self.state, admissible)
            self.record(action)
            return action
        action = base_action(self.state, state, admissible)
        self.record(action)
        return action

    def objective_action(self, admissible: list[str]) -> str:
        for index in range(self.state.objective_index, len(self.state.objective_actions)):
            wanted = self.state.objective_actions[index]
            match = match_wanted_action(wanted, admissible)
            if match:
                self.state.objective_index = index + 1
                return match
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


def base_action(agent_state: TextWorldAgentState, state: Any, admissible: list[str]) -> str:
    del state
    for pattern in [r"^take .*(coin|key|cookbook)", r"^read cookbook$", r"^open .+", r"^unlock .+ with .+"]:
        action = first_matching(admissible, pattern, agent_state.previous_actions)
        if action:
            return action
    take = first_matching(admissible, r"^take .+", agent_state.previous_actions)
    if take:
        return take
    return graph_exploration_action(agent_state, admissible, prefer_unseen=False)


def graph_exploration_action(
    state: TextWorldAgentState,
    admissible: list[str],
    *,
    prefer_unseen: bool = True,
) -> str:
    for action in admissible:
        if action.startswith("open ") and action not in state.opened_actions:
            return action
    moves = [action for action in admissible if action.startswith("go ")]
    if moves:
        current = state.last_room
        if prefer_unseen and current:
            for action in moves:
                direction = action.split(maxsplit=1)[1]
                destination = state.graph.get(current, {}).get(direction)
                if not destination or destination not in state.visited_rooms:
                    return action
        if state.last_move_direction:
            opposite = opposite_direction(state.last_move_direction)
            for action in moves:
                if action != f"go {opposite}":
                    return action
        return first_non_repeated(moves, state.previous_actions[-6:]) or moves[0]
    return first_non_repeated(admissible, state.previous_actions, avoid={"inventory", "look"}) or admissible[0]


def cooking_action(agent_state: TextWorldAgentState, state: Any, admissible: list[str]) -> str:
    feedback = str(state.feedback or "")
    update_recipe(agent_state, feedback)
    if "eat meal" in admissible:
        return "eat meal"
    if "prepare meal" in admissible:
        return "prepare meal"
    for action in ["take cookbook", "read cookbook"]:
        if action in admissible and (action != "read cookbook" or not agent_state.recipe_lines):
            return action
    if not agent_state.recipe_lines:
        for action in admissible:
            if action.startswith("take ") and "cookbook" in action:
                return action
        return graph_exploration_action(agent_state, admissible)
    recipe_action = next_recipe_action(agent_state, admissible)
    if recipe_action:
        return recipe_action
    return graph_exploration_action(agent_state, admissible)


def update_recipe(state: TextWorldAgentState, feedback: str) -> None:
    if "Ingredients:" not in feedback and "Directions:" not in feedback:
        return
    lines = [clean_text(line) for line in feedback.splitlines()]
    recipe: list[str] = []
    capture = False
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("ingredients"):
            capture = True
            continue
        if lowered.startswith("directions"):
            capture = True
            continue
        if capture and line and not line.startswith(">"):
            if any(token in lowered for token in ["ingredient", "direction"]):
                continue
            recipe.append(line)
    if recipe:
        state.recipe_lines = recipe


def next_recipe_action(state: TextWorldAgentState, admissible: list[str]) -> str:
    for line in state.recipe_lines:
        for action in admissible:
            if recipe_line_matches_action(line, action):
                return action
    for action in admissible:
        if action.startswith("take ") and not any(action == previous for previous in state.previous_actions):
            return action
    return ""


def recipe_line_matches_action(line: str, action: str) -> bool:
    line_norm = normalize(line)
    action_norm = normalize(action)
    if line_norm == action_norm:
        return True
    if line_norm.startswith("slice ") or line_norm.startswith("dice ") or line_norm.startswith("chop "):
        verb, item = line_norm.split(" ", maxsplit=1)
        return action_norm.startswith(verb + " ") and item in action_norm
    if any(line_norm.startswith(prefix + " ") for prefix in ["fry", "roast", "grill", "bake", "cook"]):
        _, item = line_norm.split(" ", maxsplit=1)
        return action_norm.startswith("cook ") and item in action_norm
    if line_norm.startswith("take "):
        return action_norm.startswith(line_norm)
    return False


def parse_objective_actions(objective: str) -> list[str]:
    text = re.sub(r"\s+", " ", objective)
    pieces = re.split(r"(?<=[.!?])\s+| then,? | and then,? | after that,? | following that,?", text, flags=re.IGNORECASE)
    actions: list[str] = []
    for piece in pieces:
        piece = piece.strip(" .!?:;")
        if not piece:
            continue
        action = parse_objective_piece(piece)
        if action:
            actions.append(action)
    return actions


def parse_objective_piece(piece: str) -> str:
    lower = piece.lower().replace("pick-up", "pick up")
    direction = re.search(r"\b(north|south|east|west)\b", lower)
    if direction and any(token in lower for token in ["go", "travel", "head", "trip", "make an effort", "attempt"]):
        return f"go {direction.group(1)}"
    put = re.search(r"(?:put|place) (?:the |a |an )?(.+?) (?:on|in) (?:the |a |an )?([a-z][a-z0-9 -]+)", lower)
    if put:
        preposition = "on" if " on " in put.group(0) else "in"
        return f"put {clean_object(put.group(1))} {preposition} {clean_object(put.group(2))}"
    open_state = re.search(
        r"(?:make it so that|ensure that|see that|look and see that) (?:the |a |an )?(.+?) (?:is |inside .+ is )?(?:wide open|open|ajar)",
        lower,
    )
    if open_state:
        return f"open {clean_object(open_state.group(1))}"
    passive_unlock = re.search(
        r"(?:check that )?(?:the |a |an )?(.+?) is unlocked with (?:the |a |an )?([a-z][a-z0-9 -]+)",
        lower,
    )
    if passive_unlock:
        return f"unlock {clean_object(passive_unlock.group(1))} with {clean_object(passive_unlock.group(2))}"
    bare_unlock = re.search(r"unlock (?:the |a |an )?([a-z][a-z0-9 -]+)", lower)
    if bare_unlock:
        return f"unlock {clean_object(bare_unlock.group(1))}"
    unlock = re.search(r"(?:unlock|unlocked) (?:the |a |an )?(.+?) with (?:the |a |an )?([a-z][a-z0-9 -]+)", lower)
    if unlock:
        return f"unlock {clean_object(unlock.group(1))} with {clean_object(unlock.group(2))}"
    open_match = re.search(r"open (?:the |a |an )?([a-z][a-z0-9 -]+)", lower)
    if open_match:
        return f"open {clean_object(open_match.group(1))}"
    take = re.search(
        r"(?:pick up|retrieve|get your hands on|recover|lift|take|pick-up) (?:the |a |an )?(.+?)(?: from| on| in|$)",
        lower,
    )
    if take:
        return f"take {clean_object(take.group(1))}"
    return ""


def match_wanted_action(wanted: str, admissible: list[str]) -> str:
    wanted_norm = normalize(wanted)
    for action in admissible:
        if normalize(action) == wanted_norm:
            return action
    if wanted.startswith("take "):
        item = normalize(wanted[5:])
        for action in admissible:
            action_norm = normalize(action)
            if action_norm.startswith("take ") and item in action_norm:
                return action
    if wanted.startswith("open "):
        item = normalize(wanted[5:])
        for action in admissible:
            if normalize(action).startswith("open ") and item in normalize(action):
                return action
    if wanted.startswith("unlock "):
        wanted_parts = normalize(wanted).split(" with ")
        for action in admissible:
            action_norm = normalize(action)
            if action_norm.startswith("unlock ") and all(part in action_norm for part in wanted_parts):
                return action
            if len(wanted_parts) == 1 and action_norm.startswith("unlock ") and wanted_parts[0] in action_norm:
                return action
    if wanted.startswith("put "):
        wanted_norm = normalize(wanted)
        for action in admissible:
            action_norm = normalize(action)
            if action_norm.startswith("put ") and all(token in action_norm for token in wanted_norm.split() if token not in {"put", "on", "in"}):
                return action
    return ""


def update_room_state(state: TextWorldAgentState, description: str) -> None:
    room = parse_room(description)
    if not room:
        return
    if state.last_room and state.last_move_direction and state.last_room != room:
        state.graph.setdefault(state.last_room, {})[state.last_move_direction] = room
        state.graph.setdefault(room, {})[opposite_direction(state.last_move_direction)] = state.last_room
    state.visited_rooms.add(room)
    state.last_room = room


def parse_room(description: str) -> str | None:
    match = re.search(r"-=\s*([^=]+?)\s*=-", description)
    return normalize(match.group(1)) if match else None


def clean_object(text: str) -> str:
    text = re.sub(r"\b(?:inside|within|from|floor|room|of)\b.*$", "", text)
    return clean_text(text)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" -*.\t")).strip()


def normalize(text: str) -> str:
    text = text.lower().replace("pick up", "take").replace("pick-up", "take")
    text = re.sub(r"\b(the|a|an|some)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def opposite_direction(direction: str) -> str:
    return {"north": "south", "south": "north", "east": "west", "west": "east"}.get(direction, direction)


def first_matching(admissible: list[str], pattern: str, previous: list[str]) -> str:
    for action in admissible:
        if re.match(pattern, action) and action not in previous:
            return action
    return ""


def first_non_repeated(admissible: list[str], previous: list[str], avoid: set[str] | None = None) -> str:
    avoid = avoid or set()
    filtered = [action for action in admissible if action not in avoid]
    for action in filtered:
        if action not in previous:
            return action
    return filtered[0] if filtered else ""


def sorted_unique(commands: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for command in commands:
        command = " ".join(command.split())
        if command and command not in seen:
            seen.add(command)
            result.append(command)
    return result


def textworld_score(record: TextWorldRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.4 * (1.0 if record.invalid_action else 0.0)
        - 0.1 * (1.0 if record.repeated_actions else 0.0)
        - 0.004 * record.turns
    )


def run_episode(
    *,
    spec: TextWorldSpec,
    game_path: Path,
    method: str,
    split: str,
    variables: dict[str, TextVariable],
    max_steps: int,
) -> TextWorldRecord:
    try:
        import textworld
    except ImportError as exc:
        raise SystemExit("TextWorld is not installed. Run: python -m pip install textworld==1.7.0") from exc

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
    invalid = False
    failure = ""
    final_score = 0.0
    max_score = 1.0
    success = False
    turns = 0
    try:
        state = env.reset()
        max_score = float(state.max_score or 1.0)
        agent = TextWorldPromptAgent(variables, str(state.objective or ""))
        for _ in range(max_steps):
            admissible = set(str(command) for command in (state.admissible_commands or []))
            action = agent.act(state)
            if action not in admissible:
                invalid = True
                failure = f"invalid action: {action}"
                break
            state, _score, done = env.step(action)
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

    return TextWorldRecord(
        benchmark="textworld_24",
        spec_id=spec.spec_id,
        family=spec.family,
        category=spec.category,
        method=method,
        split=split,
        seed=spec.seed,
        success=success,
        reward=final_score / max(max_score, 1.0),
        invalid_action=invalid,
        repeated_actions=len(actions) != len(set(actions)),
        turns=turns,
        final_score=final_score,
        max_score=max_score,
        actions=actions,
        failure_reason=failure,
        runtime_seconds=time.perf_counter() - started,
        total_turns=turns,
    )


def run_records(
    *,
    specs: list[TextWorldSpec],
    game_paths: dict[str, Path],
    method: str,
    split: str,
    variables: dict[str, TextVariable],
    max_steps: int,
    output_jsonl: Path,
) -> list[TextWorldRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[TextWorldRecord] = []
    for spec in specs:
        record = run_episode(
            spec=spec,
            game_path=game_paths[spec.spec_id],
            method=method,
            split=split,
            variables=variables,
            max_steps=max_steps,
        )
        records.append(record)
        append_jsonl(output_jsonl, record)
    return records


def training_specs(specs: list[TextWorldSpec], offset: int) -> list[TextWorldSpec]:
    selected: list[TextWorldSpec] = []
    for family in sorted({spec.family for spec in specs}):
        group = [spec for spec in specs if spec.family == family]
        selected.extend(group[offset : offset + 2])
    return selected


def auxiliary_specs(seed: int, split: str) -> list[TextWorldSpec]:
    """Create two separately seeded optimization games per family."""
    return [
        replace(spec, spec_id=f"{split}_{spec.spec_id}")
        for spec in training_specs(default_specs(seed), 0)
    ]


def gradients_from_records(records: list[TextWorldRecord]) -> list[TextualGradient]:
    rules: set[str] = set()
    for record in records:
        if record.success and not record.repeated_actions:
            continue
        if record.family in {"tw-simple", "tw-coin_collector", "tw-treasure_hunter"}:
            rules.add("objective_sequence")
        if record.family == "tw-cooking":
            rules.add("cooking_recipe")
        if record.repeated_actions or record.turns > 20:
            rules.add("graph_exploration")
    gradients = []
    for rule_id in sorted(rules):
        gradients.append(
            TextualGradient(
                target_variable_name="textworld_policy",
                failure_mode=f"textworld_24:{rule_id}",
                evidence_from_trajectory=f"Training failures indicate missing rule {rule_id}.",
                gradient_text=f"Add a reusable TextWorld rule for {rule_id}.",
                suggested_edit=f"Add a rule: {RULE_TEXTS[rule_id]}",
                confidence=0.82,
                forbidden_shortcuts=["use oracle walkthrough", "use policy_commands", "inspect hidden labels"],
            )
        )
    return gradients


def variables_from_diagnostics(
    records: list[TextWorldRecord],
) -> tuple[dict[str, TextVariable], list[TextualGradient]]:
    """Build a temporary policy edit from trajectory diagnostics."""
    variables = initial_textworld_variables()
    gradients = gradients_from_records(records)
    updated = TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=4).step(
        variables,
        gradients,
        constraints=["must not use oracle walkthrough", "must not use policy_commands", "must not inspect hidden labels"],
    )
    return updated, gradients


def run_retry_with_diagnostics(
    *,
    specs: list[TextWorldSpec],
    game_paths: dict[str, Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[TextWorldRecord], dict[str, Any]]:
    """Retry failed test tasks once with a task-local diagnostic policy edit."""
    method = "retry_with_diagnostics"
    method_dir = output_dir / method
    first_attempts = run_records(
        specs=specs,
        game_paths=game_paths,
        method=method,
        split="test_attempt_1",
        variables=initial_textworld_variables(),
        max_steps=args.max_steps,
        output_jsonl=method_dir / "test_attempt_1.jsonl",
    )
    retry_path = method_dir / "test_attempt_2.jsonl"
    final_path = method_dir / "test.jsonl"
    for path in (retry_path, final_path):
        if path.exists():
            path.unlink()

    final_records: list[TextWorldRecord] = []
    diagnostics: list[dict[str, Any]] = []
    total_gradients = 0
    retries = 0
    for spec, first in zip(specs, first_attempts):
        first = replace(first, total_turns=first.turns)
        if first.success:
            final = replace(first, split="test_all_24")
        else:
            temporary_variables, gradients = variables_from_diagnostics([first])
            total_gradients += len(gradients)
            retries += 1
            diagnostics.append(
                {
                    "spec_id": spec.spec_id,
                    "first_attempt": first,
                    "gradients": gradients,
                    "temporary_rule_ids": sorted(learned_rule_ids(temporary_variables)),
                }
            )
            second = run_episode(
                spec=spec,
                game_path=game_paths[spec.spec_id],
                method=method,
                split="test_attempt_2",
                variables=temporary_variables,
                max_steps=args.max_steps,
            )
            append_jsonl(retry_path, second)
            final = replace(
                second,
                split="test_all_24",
                attempts=2,
                total_turns=first.turns + second.turns,
            )
        final_records.append(final)
        append_jsonl(final_path, final)

    gate: dict[str, Any] = {
        "method": method,
        "accepted": False,
        "gradient_count": total_gradients,
        "diagnosed_tasks": retries,
        "retry_count": retries,
        "maximum_attempts_per_task": 2,
        "persistent_policy_update": False,
        "validation_gate": False,
        "optimization_episodes": 0,
        "optimization_environment_turns": 0,
    }
    write_json(method_dir / "diagnostics.json", diagnostics)
    write_json(method_dir / "gate_decision.json", gate)
    write_json(method_dir / "text_variables.json", initial_textworld_variables())
    return final_records, gate


def train_method(
    *,
    train_specs: list[TextWorldSpec],
    val_specs: list[TextWorldSpec],
    game_paths: dict[str, Path],
    method: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, TextVariable], dict[str, Any]]:
    base = initial_textworld_variables()
    train_records = run_records(
        specs=train_specs,
        game_paths=game_paths,
        method=method,
        split="train",
        variables=base,
        max_steps=args.max_steps,
        output_jsonl=output_dir / method / "train_base.jsonl",
    )
    gradients = gradients_from_records(train_records)
    candidate = TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=4).step(
        base,
        gradients,
        constraints=["must not use oracle walkthrough", "must not use policy_commands", "must not inspect hidden labels"],
    )
    old_val: list[TextWorldRecord] = []
    new_val: list[TextWorldRecord] = []
    if method == "ungated_persistent_rules":
        old_score = None
        new_score = None
        accepted = bool(learned_rule_ids(candidate) - learned_rule_ids(base))
    else:
        old_val = run_records(
            specs=val_specs,
            game_paths=game_paths,
            method=method,
            split="validation",
            variables=base,
            max_steps=args.max_steps,
            output_jsonl=output_dir / method / "val_base.jsonl",
        )
        new_val = run_records(
            specs=val_specs,
            game_paths=game_paths,
            method=method,
            split="validation",
            variables=candidate,
            max_steps=args.max_steps,
            output_jsonl=output_dir / method / "val_candidate.jsonl",
        )
        old_score = mean([textworld_score(record) for record in old_val])
        new_score = mean([textworld_score(record) for record in new_val])
        accepted = new_score >= old_score + args.min_mean_delta
    gate: dict[str, Any] = {
        "method": method,
        "accepted": accepted,
        "old_val_score": old_score,
        "new_val_score": new_score,
        "gradient_count": len(gradients),
        "learned_rule_ids": sorted(learned_rule_ids(candidate) - learned_rule_ids(base)),
        "validation_gate": method != "ungated_persistent_rules",
        "persistent_policy_update": True,
        "optimization_episodes": len(train_records) + len(old_val) + len(new_val),
        "optimization_environment_turns": sum(
            record.turns for record in [*train_records, *old_val, *new_val]
        ),
    }
    variables = candidate if accepted else base
    write_json(output_dir / method / "gradients.json", gradients)
    write_json(output_dir / method / "gate_decision.json", gate)
    write_json(output_dir / method / "text_variables.json", variables)
    return variables, gate


def summarize_records(method: str, records: list[TextWorldRecord], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "games": len({record.spec_id for record in records}),
        "episodes": len(records),
        "average_reward": mean([record.reward for record in records]),
        "success_rate": mean([float(record.success) for record in records]),
        "invalid_action_rate": mean([float(record.invalid_action) for record in records]),
        "repeated_action_rate": mean([float(record.repeated_actions) for record in records]),
        "average_turns": mean([record.turns for record in records]),
        "average_attempts": mean([record.attempts for record in records]),
        "average_environment_turns": mean([record.total_turns for record in records]),
        "accepted_updates": 1 if gate.get("accepted") else 0,
        "gradient_count": int(gate.get("gradient_count", 0)),
    }


def summarize_groups(records: list[TextWorldRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({record.method for record in records}):
        method_records = [record for record in records if record.method == method]
        for key in ["family", "category"]:
            for value in sorted({getattr(record, key) for record in method_records}):
                group = [record for record in method_records if getattr(record, key) == value]
                rows.append(
                    {
                        "method": method,
                        "slice": key,
                        "value": value,
                        "games": len({record.spec_id for record in group}),
                        "episodes": len(group),
                        "average_reward": mean([record.reward for record in group]),
                        "success_rate": mean([float(record.success) for record in group]),
                        "invalid_action_rate": mean([float(record.invalid_action) for record in group]),
                        "repeated_action_rate": mean([float(record.repeated_actions) for record in group]),
                        "average_turns": mean([record.turns for record in group]),
                        "average_attempts": mean([record.attempts for record in group]),
                        "average_environment_turns": mean([record.total_turns for record in group]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary_markdown(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# TextWorld 24-Game Local Suite",
        "",
        "This suite generates 24 local Microsoft TextWorld `.z8` games from built-in challenge generators.",
        "It uses admissible commands and visible observations/objectives, without oracle walkthrough commands.",
        "RulePI optimization uses separately seeded training and validation games that are disjoint from these 24 tests.",
        "",
        f"Games: 24",
        f"Step budget: {args.max_steps}",
        "",
        "## Overall Results",
        "",
        "| Method | Games | Reward | Success | Invalid | Repeated | Final Turns | Attempts | Test Turns | Updates | Gradients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {games} | {average_reward:.3f} | {success_rate:.3f} | {invalid_action_rate:.3f} | "
            "{repeated_action_rate:.3f} | {average_turns:.2f} | {average_attempts:.2f} | "
            "{average_environment_turns:.2f} | {accepted_updates} | {gradient_count} |".format(
                **row
            )
        )
    lines.extend(["", "## Family Results", "", "| Method | Family | Games | Reward | Success | Invalid | Repeated | Final Turns | Attempts | Test Turns |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in [row for row in group_rows if row["slice"] == "family"]:
        lines.append(
            "| {method} | {value} | {games} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} | "
            "{average_attempts:.2f} | {average_environment_turns:.2f} |".format(**row)
        )
    lines.extend(["", "## Gate Decisions", ""])
    for method, gate in gates.items():
        lines.append(f"- `{method}`: `{gate}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods or list(METHODS)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = default_specs(args.seed)
    train_specs = auxiliary_specs(args.seed + 10_000, "train")
    val_specs = auxiliary_specs(args.seed + 20_000, "validation")
    methods = parse_methods(args.methods)
    game_paths = ensure_games(
        [*specs, *train_specs, *val_specs],
        output_dir / "games",
        force=args.force_regenerate,
    )
    write_json(
        output_dir / "config.json",
        {
            "benchmark": "textworld_24",
            "methods": methods,
            "specs": specs,
            "test_specs": specs,
            "train_specs": train_specs,
            "validation_specs": val_specs,
        },
    )
    write_json(output_dir / "environment_info.json", environment_info())
    all_records: list[TextWorldRecord] = []
    summary_rows: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method == "retry_with_diagnostics":
            records, gate = run_retry_with_diagnostics(
                specs=specs,
                game_paths=game_paths,
                output_dir=output_dir,
                args=args,
            )
        elif method == "fixed_prompt":
            variables = initial_textworld_variables()
            gate = {
                "method": method,
                "accepted": False,
                "gradient_count": 0,
                "optimization_episodes": 0,
                "optimization_environment_turns": 0,
            }
            write_json(output_dir / method / "text_variables.json", variables)
        else:
            variables, gate = train_method(
                train_specs=train_specs,
                val_specs=val_specs,
                game_paths=game_paths,
                method=method,
                output_dir=output_dir,
                args=args,
            )
        gates[method] = gate
        if method != "retry_with_diagnostics":
            records = run_records(
                specs=specs,
                game_paths=game_paths,
                method=method,
                split="test_all_24",
                variables=variables,
                max_steps=args.max_steps,
                output_jsonl=output_dir / method / "test.jsonl",
            )
        all_records.extend(records)
        summary_rows.append(summarize_records(method, records, gate))
    group_rows = summarize_groups(all_records)
    write_json(output_dir / "summary.json", summary_rows)
    write_json(output_dir / "slice_summary.json", group_rows)
    write_json(output_dir / "gate_decisions.json", gates)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "slice_summary.csv", group_rows)
    write_summary_markdown(output_dir / "summary.md", summary_rows=summary_rows, group_rows=group_rows, gates=gates, args=args)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Microsoft TextWorld 24-game benchmark.")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seed", type=int, default=62001)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--output-dir", default="runs/textworld_24_suite")
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    print(f"TextWorld 24-game artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
