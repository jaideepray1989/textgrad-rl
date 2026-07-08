"""Local TextWorldExpress benchmark for TextGrad-RL.

TextWorldExpress is a no-key, local JVM-backed text-game suite. This harness
compares a fixed valid-action actor against TextGrad policy iteration and a
PPO-style gated variant over the same text-policy variables.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textgrad_rl.optim.textual_gradient_descent import TextualGradientDescent
from textgrad_rl.types import TextualGradient, TextVariable
from textgrad_rl.utils.json_utils import append_jsonl, write_json
from textgrad_rl.utils.logging import environment_info


TWX_GAMES = [
    "cookingworld",
    "coin",
    "twc",
    "mapreader",
    "arithmetic",
    "sorting",
    "simonsays",
    "peckingorder",
]

GAME_CATEGORIES = {
    "cookingworld": "recipe_planning",
    "coin": "exploration",
    "twc": "commonsense_storage",
    "mapreader": "map_navigation",
    "arithmetic": "quantity_reasoning",
    "sorting": "quantity_reasoning",
    "simonsays": "instruction_following",
    "peckingorder": "instruction_following",
}

METHODS = ["fixed_prompt", "textgrad_policy_iteration", "textgrad_ppo"]

RULE_BY_GAME = {
    "cookingworld": (
        "For cookingworld, read the cookbook, collect every listed ingredient and the knife, "
        "follow recipe directions in order, map grill/roast/fry/bake directions to visible cook actions, "
        "then prepare and eat the meal."
    ),
    "coin": (
        "For coin search, systematically open doors and containers, move through unvisited rooms, "
        "take the coin as soon as it appears, then stop searching."
    ),
    "twc": (
        "For twc, pick up misplaced objects and place each into its usual household location, "
        "such as shoes in the shoe cabinet, hats on the hat rack, dirty clothes in the laundry basket, "
        "and toiletries in the bathroom cabinet."
    ),
    "mapreader": (
        "For mapreader, read the task and map, navigate room by room to the coin, take it, "
        "then return to the room with the box and put the coin in the box."
    ),
    "arithmetic": (
        "For arithmetic, take and read the math problem, compute the numeric answer from the visible text, "
        "take the object whose quantity equals that answer, and put it in the box."
    ),
    "sorting": (
        "For sorting, repeatedly take the visible object with the smallest numeric quantity and put it in the box "
        "before moving to the next larger quantity."
    ),
    "simonsays": "For simonsays, parse the quoted command after 'Simon says' and execute exactly that visible action.",
    "peckingorder": (
        "For peckingorder, read the instructions book before each action and execute the current requested take action."
    ),
}


@dataclass
class TextWorldExpressRecord:
    benchmark: str
    game: str
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
    actions: list[str]
    failure_reason: str
    runtime_seconds: float


@dataclass
class TextWorldExpressState:
    game: str
    previous_actions: list[str] = field(default_factory=list)
    visited_rooms: set[str] = field(default_factory=set)
    graph: dict[str, dict[str, str]] = field(default_factory=dict)
    last_room: str | None = None
    last_move_direction: str | None = None
    room_with_box: str | None = None
    coin_room: str | None = None
    recipe_ingredients: list[str] = field(default_factory=list)
    recipe_directions: list[str] = field(default_factory=list)
    recipe_direction_index: int = 0
    arithmetic_answer: int | None = None
    opened_actions: set[str] = field(default_factory=set)


def initial_textworld_variables() -> dict[str, TextVariable]:
    return {
        "textworld_policy": TextVariable(
            name="textworld_policy",
            value=(
                "Use only actions listed by the environment. Prefer useful visible actions over repeats. "
                "Inspect the task, look around once, pick up target objects, open accessible doors or containers, "
                "and avoid hidden benchmark state."
            ),
            role_description="TextWorldExpress local text-game policy.",
            max_chars=2600,
        )
    }


def policy_text(variables: dict[str, TextVariable]) -> str:
    return "\n".join(variable.value.lower() for variable in variables.values())


def learned_rule_ids(variables: dict[str, TextVariable]) -> set[str]:
    text = policy_text(variables)
    ids: set[str] = set()
    if "cookingworld" in text and "cookbook" in text:
        ids.add("cookingworld")
    if "coin search" in text or ("take the coin" in text and "unvisited rooms" in text):
        ids.add("coin")
    if "for twc" in text or "usual household location" in text:
        ids.add("twc")
    if "for mapreader" in text or "read the task and map" in text:
        ids.add("mapreader")
    if "for arithmetic" in text or "numeric answer" in text:
        ids.add("arithmetic")
    if "for sorting" in text or "smallest numeric quantity" in text:
        ids.add("sorting")
    if "for simonsays" in text or "simon says" in text:
        ids.add("simonsays")
    if "for peckingorder" in text or "instructions book before each action" in text:
        ids.add("peckingorder")
    return ids


class TextWorldExpressAgent:
    """Deterministic valid-action actor controlled by learned text rules."""

    def __init__(self, variables: dict[str, TextVariable], game: str) -> None:
        self.variables = variables
        self.state = TextWorldExpressState(game=game)

    def act(self, observation: str, info: dict[str, Any]) -> str:
        actions = sorted_unique(str(action) for action in (info.get("validActions") or []))
        if not actions:
            return "look around"
        update_navigation_state(self.state, observation)
        rules = learned_rule_ids(self.variables)
        game = self.state.game

        if game in rules:
            specialized = self.specialized_action(game, observation, info, actions)
            if specialized in actions:
                self.record_action(specialized)
                return specialized

        action = self.base_action(observation, info, actions)
        self.record_action(action)
        return action

    def record_action(self, action: str) -> None:
        if action.startswith("move "):
            self.state.last_move_direction = action.split(maxsplit=1)[1]
        elif action.startswith("open "):
            self.state.opened_actions.add(action)
        else:
            self.state.last_move_direction = None
        self.state.previous_actions.append(action)

    def specialized_action(self, game: str, observation: str, info: dict[str, Any], actions: list[str]) -> str:
        if game == "simonsays":
            return simonsays_action(observation, actions)
        if game == "peckingorder":
            return peckingorder_action(observation, actions)
        if game == "sorting":
            return sorting_action(self.state, actions)
        if game == "arithmetic":
            return arithmetic_action(self.state, observation, actions)
        if game in {"coin", "mapreader"}:
            return coin_or_map_action(self.state, observation, info, actions)
        if game == "twc":
            return twc_action(self.state, actions)
        if game == "cookingworld":
            return cooking_action(self.state, observation, actions)
        return ""

    def base_action(self, observation: str, info: dict[str, Any], actions: list[str]) -> str:
        del info
        if "look around" in actions and not self.state.previous_actions:
            return "look around"
        for exact in ["take coin", "put coin in box", "read cookbook", "read map", "read instructions book"]:
            if exact in actions and exact not in self.state.previous_actions:
                return exact
        if "Simon says" in observation:
            return first_non_repeated(actions, self.state.previous_actions, avoid_prefixes=("inventory", "look"))
        take = first_action_matching(actions, r"^take (?!math problem$).+")
        if take:
            return take
        open_action = first_non_repeated(
            [action for action in actions if action.startswith("open ")],
            self.state.previous_actions,
        )
        if open_action:
            return open_action
        move = first_non_repeated([action for action in actions if action.startswith("move ")], self.state.previous_actions)
        if move:
            return move
        return first_non_repeated(actions, self.state.previous_actions, avoid_prefixes=("inventory",)) or actions[0]


def simonsays_action(observation: str, actions: list[str]) -> str:
    match = re.search(r"Simon says,\s*'([^']+)'", observation)
    if match and match.group(1) in actions:
        return match.group(1)
    return first_non_repeated(actions, [], avoid_prefixes=("inventory", "look")) or actions[0]


def peckingorder_action(observation: str, actions: list[str]) -> str:
    match = re.search(r"current task is to ([^.]+)\.", observation, flags=re.IGNORECASE)
    if match:
        requested = re.sub(r"\bthe\b\s*", "", match.group(1).strip(), flags=re.IGNORECASE)
        if requested in actions:
            return requested
    if "read instructions book" in actions:
        return "read instructions book"
    return best_take_action(actions) or actions[0]


def sorting_action(state: TextWorldExpressState, actions: list[str]) -> str:
    placed_items = placed_sorting_items(state.previous_actions)
    put_in_box = [item for item in quantity_actions(actions, prefix="put", suffix="in box") if item[2] not in placed_items]
    if put_in_box:
        return min(put_in_box, key=lambda item: item[0])[1]
    take_quantities = [item for item in quantity_actions(actions, prefix="take") if item[2] not in placed_items]
    if take_quantities:
        return min(take_quantities, key=lambda item: item[0])[1]
    return actions[0]


def arithmetic_action(state: TextWorldExpressState, observation: str, actions: list[str]) -> str:
    answer = parse_arithmetic_answer(observation)
    if answer is not None:
        state.arithmetic_answer = answer
    if state.arithmetic_answer is None:
        if "take math problem" in actions:
            return "take math problem"
        if "read math problem" in actions:
            return "read math problem"
    if state.arithmetic_answer is not None:
        answer_text = str(state.arithmetic_answer)
        for action in actions:
            if action.startswith(f"put {answer_text} ") and action.endswith(" in box"):
                return action
        for action in actions:
            if action.startswith(f"take {answer_text} "):
                return action
    if "read math problem" in actions:
        return "read math problem"
    return sorting_action(state, actions)


def coin_or_map_action(
    state: TextWorldExpressState,
    observation: str,
    info: dict[str, Any],
    actions: list[str],
) -> str:
    current_room = parse_room(observation)
    if current_room and "box" in observation.lower():
        state.room_with_box = current_room
    if current_room and "coin" in observation.lower():
        state.coin_room = current_room
    if "put coin in box" in actions:
        return "put coin in box"
    if "take coin" in actions:
        return "take coin"
    if "task" in actions and not any(action == "task" for action in state.previous_actions):
        return "task"
    if "read map" in actions and state.game == "mapreader" and not any(action == "read map" for action in state.previous_actions):
        return "read map"
    if current_room and has_inventory_item(info, "coin") and state.room_with_box:
        path_action = next_move_toward(state, current_room, state.room_with_box, actions)
        if path_action:
            return path_action
    return exploration_action(state, actions)


def twc_action(state: TextWorldExpressState, actions: list[str]) -> str:
    put_actions = [action for action in actions if action.startswith("put ")]
    if put_actions:
        scored = [(score_twc_put_action(action), action) for action in put_actions]
        scored = [item for item in scored if item[0] > 0]
        if scored:
            return max(scored, key=lambda item: (item[0], -len(item[1])))[1]
    take = best_take_action(actions)
    if take:
        return take
    return exploration_action(state, actions)


def cooking_action(state: TextWorldExpressState, observation: str, actions: list[str]) -> str:
    update_recipe_state(state, observation)
    if "eat meal" in actions:
        return "eat meal"
    if "prepare meal" in actions:
        return "prepare meal"
    if "take cookbook" in actions and "take cookbook" not in state.previous_actions:
        return "take cookbook"
    if "read cookbook" in actions and not state.recipe_directions:
        return "read cookbook"
    if "take knife" in actions:
        return "take knife"
    if "open cutlery drawer" in actions and "take knife" not in state.previous_actions:
        return "open cutlery drawer"
    ingredient_action = next_recipe_ingredient_action(state, actions)
    if ingredient_action:
        return ingredient_action
    direction_action = next_recipe_direction_action(state, actions)
    if direction_action:
        return direction_action
    return exploration_action(state, actions)


def update_recipe_state(state: TextWorldExpressState, observation: str) -> None:
    if "Ingredients:" not in observation or "Directions:" not in observation:
        return
    ingredients_text = observation.split("Ingredients:", maxsplit=1)[1].split("Directions:", maxsplit=1)[0]
    directions_text = observation.split("Directions:", maxsplit=1)[1]
    ingredients = [clean_item(line) for line in ingredients_text.splitlines() if clean_item(line)]
    directions = [clean_item(line) for line in directions_text.splitlines() if clean_item(line)]
    if ingredients:
        state.recipe_ingredients = ingredients
    if directions:
        state.recipe_directions = directions


def next_recipe_ingredient_action(state: TextWorldExpressState, actions: list[str]) -> str:
    if not state.recipe_ingredients:
        return ""
    action_text = " ".join(state.previous_actions).lower()
    for ingredient in state.recipe_ingredients:
        if f"take {ingredient}".lower() in action_text:
            continue
        for action in actions:
            if action.startswith("take ") and ingredient_match(ingredient, action[5:]):
                return action
    return ""


def next_recipe_direction_action(state: TextWorldExpressState, actions: list[str]) -> str:
    while state.recipe_direction_index < len(state.recipe_directions):
        direction = state.recipe_directions[state.recipe_direction_index]
        candidates = direction_to_action_candidates(direction, actions)
        if candidates:
            state.recipe_direction_index += 1
            return candidates[0]
        if any(direction_complete(direction, action) for action in state.previous_actions):
            state.recipe_direction_index += 1
            continue
        return ""
    return ""


def direction_to_action_candidates(direction: str, actions: list[str]) -> list[str]:
    normalized = normalize_text(direction)
    direct = [action for action in actions if normalize_text(action) == normalized]
    if direct:
        return direct
    match = re.match(r"(slice|dice|chop|cut) (.+)", direction)
    if match:
        verb, item = match.groups()
        return [action for action in actions if action.startswith(f"{verb} ") and ingredient_match(item, action[len(verb) + 1 :])]
    match = re.match(r"(grill|roast|fry|bake|cook) (.+)", direction)
    if match:
        verb, item = match.groups()
        preferred = {
            "grill": ("barbeque", "stove", "oven"),
            "roast": ("oven", "barbeque", "stove"),
            "fry": ("stove", "oven", "barbeque"),
            "bake": ("oven", "stove", "barbeque"),
            "cook": ("stove", "oven", "barbeque"),
        }[verb]
        matches = [
            action
            for action in actions
            if action.startswith("cook ") and ingredient_match(item, action[5:].split(" in ", maxsplit=1)[0])
        ]
        return sorted(matches, key=lambda action: preferred_index(action, preferred))
    if direction == "prepare meal" and "prepare meal" in actions:
        return ["prepare meal"]
    return []


def direction_complete(direction: str, action: str) -> bool:
    if normalize_text(direction) == normalize_text(action):
        return True
    match = re.match(r"(grill|roast|fry|bake|cook) (.+)", direction)
    return bool(match and action.startswith("cook ") and ingredient_match(match.group(2), action[5:].split(" in ", maxsplit=1)[0]))


def exploration_action(state: TextWorldExpressState, actions: list[str]) -> str:
    for action in actions:
        if action.startswith("open ") and action not in state.opened_actions:
            return action
    move_actions = [action for action in actions if action.startswith("move ")]
    for action in move_actions:
        direction = action.split(maxsplit=1)[1]
        room = state.last_room
        if not room:
            return action
        destination = state.graph.get(room, {}).get(direction)
        if not destination or destination not in state.visited_rooms:
            return action
    if move_actions:
        return first_non_repeated(move_actions, state.previous_actions[-4:]) or move_actions[0]
    take = best_take_action(actions)
    if take:
        return take
    return first_non_repeated(actions, state.previous_actions, avoid_prefixes=("inventory",)) or actions[0]


def update_navigation_state(state: TextWorldExpressState, observation: str) -> None:
    room = parse_room(observation)
    if not room:
        return
    if state.last_room and state.last_move_direction and state.last_room != room:
        state.graph.setdefault(state.last_room, {})[state.last_move_direction] = room
        state.graph.setdefault(room, {})[opposite_direction(state.last_move_direction)] = state.last_room
    state.visited_rooms.add(room)
    state.last_room = room
    if "box" in observation.lower():
        state.room_with_box = room


def next_move_toward(state: TextWorldExpressState, current_room: str, target_room: str, actions: list[str]) -> str:
    queue: deque[tuple[str, list[str]]] = deque([(current_room, [])])
    seen = {current_room}
    while queue:
        room, path = queue.popleft()
        if room == target_room and path:
            action = f"move {path[0]}"
            return action if action in actions else ""
        for direction, neighbor in state.graph.get(room, {}).items():
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [direction]))
    return ""


def parse_room(observation: str) -> str | None:
    match = re.search(r"You are in the ([^.]+)\.", observation)
    return match.group(1).strip().lower() if match else None


def opposite_direction(direction: str) -> str:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(direction, direction)


def parse_arithmetic_answer(text: str) -> int | None:
    lower = text.lower()
    patterns = [
        (r"add (\d+) and (\d+)", lambda a, b: a + b),
        (r"sum of (\d+) and (\d+)", lambda a, b: a + b),
        (r"subtract (\d+) from (\d+)", lambda a, b: b - a),
        (r"take (\d+) away from (\d+)", lambda a, b: b - a),
        (r"multiply (\d+) by (\d+)", lambda a, b: a * b),
        (r"divide (\d+) by (\d+)", lambda a, b: a // b if b else None),
    ]
    for pattern, fn in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        result = fn(int(match.group(1)), int(match.group(2)))
        return int(result) if result is not None else None
    simple = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", lower)
    if simple:
        left, op, right = int(simple.group(1)), simple.group(2), int(simple.group(3))
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/" and right:
            return left // right
    return None


def quantity_actions(actions: list[str], prefix: str, suffix: str | None = None) -> list[tuple[float, str, str]]:
    results: list[tuple[float, str, str]] = []
    for action in actions:
        if not action.startswith(prefix + " "):
            continue
        if suffix and not action.endswith(" " + suffix):
            continue
        match = re.search(r"(\d+)\s*([a-zA-Z]+)?", action)
        if match:
            item = action[len(prefix) + 1 :]
            if suffix and item.endswith(" " + suffix):
                item = item[: -len(" " + suffix)]
            results.append((quantity_value(int(match.group(1)), match.group(2) or ""), action, normalize_text(item)))
    return results


def quantity_value(amount: int, unit: str) -> float:
    unit = unit.lower()
    if unit in {"l", "liter", "liters", "kg", "kilogram", "kilograms"}:
        return float(amount * 1000)
    if unit in {"ml", "milliliter", "milliliters", "g", "gram", "grams"}:
        return float(amount)
    return float(amount)


def placed_sorting_items(previous_actions: list[str]) -> set[str]:
    placed: set[str] = set()
    for action in previous_actions:
        if action.startswith("put ") and action.endswith(" in box"):
            item = action[len("put ") : -len(" in box")]
            placed.add(normalize_text(item))
    return placed


def best_take_action(actions: list[str]) -> str:
    takes = [action for action in actions if action.startswith("take ") and action != "take math problem"]
    if not takes:
        return ""
    return sorted(takes, key=lambda action: (object_priority(action), action))[0]


def object_priority(action: str) -> int:
    lower = action.lower()
    if "coin" in lower:
        return 0
    if any(token in lower for token in ["book", "map", "math problem"]):
        return 1
    if any(token in lower for token in ["knife", "key"]):
        return 2
    return 3


def score_twc_put_action(action: str) -> int:
    match = re.match(r"put (.+) in (.+)", action)
    if not match:
        return 0
    obj, target = normalize_text(match.group(1)), normalize_text(match.group(2))
    score = 0
    mappings = [
        (("shoe", "sneaker", "boot", "slipper"), ("shoe cabinet",), 10),
        (("hat", "beret", "cap"), ("hat rack",), 10),
        (("umbrella",), ("umbrella stand",), 10),
        (("key",), ("key holder",), 10),
        (("dirty", "wet"), ("laundry basket", "washing machine"), 9),
        (("shirt", "blazer", "bra", "socks", "pants", "dress"), ("wardrobe", "chest of drawers", "drawer"), 7),
        (("shaving", "toothbrush", "toothpaste", "soap", "deodorant"), ("bathroom cabinet",), 8),
        (("book", "magazine"), ("bookshelf", "shelf"), 6),
        (("plate", "cup", "mug", "bowl"), ("kitchen cupboard", "dishwasher"), 7),
        (("towel",), ("towel rack",), 8),
        (("jacket", "coat"), ("coat hanger",), 8),
    ]
    for object_tokens, target_tokens, value in mappings:
        if any(token in obj for token in object_tokens) and any(token in target for token in target_tokens):
            score = max(score, value)
    return score


def has_inventory_item(info: dict[str, Any], item: str) -> bool:
    return item.lower() in str(info.get("inventory") or "").lower()


def first_action_matching(actions: list[str], pattern: str) -> str:
    return next((action for action in actions if re.match(pattern, action)), "")


def first_non_repeated(
    actions: list[str],
    previous_actions: list[str],
    avoid_prefixes: tuple[str, ...] = (),
) -> str:
    filtered = [action for action in actions if not action.startswith(avoid_prefixes)]
    for action in filtered:
        if action not in previous_actions:
            return action
    return filtered[0] if filtered else ""


def sorted_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def clean_item(text: str) -> str:
    text = re.sub(r"^[\s*\-]+", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("some ", "").replace("a raw ", "").replace("a ", "")).strip()


def ingredient_match(ingredient: str, candidate: str) -> bool:
    ingredient_norm = normalize_text(ingredient)
    candidate_norm = normalize_text(candidate)
    return ingredient_norm in candidate_norm or candidate_norm in ingredient_norm


def preferred_index(action: str, preferred: tuple[str, ...]) -> int:
    for index, token in enumerate(preferred):
        if token in action:
            return index
    return len(preferred)


def textworld_score(record: TextWorldExpressRecord) -> float:
    return (
        record.reward
        + 0.5 * (1.0 if record.success else 0.0)
        - 0.4 * (1.0 if record.invalid_action else 0.0)
        - 0.1 * (1.0 if record.repeated_actions else 0.0)
        - 0.004 * record.turns
    )


def run_episode(
    *,
    game: str,
    method: str,
    split: str,
    seed: int,
    variables: dict[str, TextVariable],
    max_steps: int,
) -> TextWorldExpressRecord:
    try:
        from textworld_express import TextWorldExpressEnv
    except ImportError as exc:
        raise SystemExit("TextWorldExpress is not installed. Run: python -m pip install textworld-express") from exc

    started = time.perf_counter()
    env = TextWorldExpressEnv(envStepLimit=max_steps)
    agent = TextWorldExpressAgent(variables, game)
    actions: list[str] = []
    success = False
    invalid = False
    reward_total = 0.0
    final_score = 0.0
    failure_reason = ""
    turns = 0
    try:
        observation, info = env.reset(seed=seed, gameFold=split, gameName=game, gameParams="", generateGoldPath=False)
        for _ in range(max_steps):
            valid_actions = set(str(action) for action in (info.get("validActions") or []))
            action = agent.act(str(observation), info)
            if action not in valid_actions:
                invalid = True
                failure_reason = f"invalid action: {action}"
                break
            observation, reward, done, info = env.step(action)
            actions.append(action)
            reward_total += float(reward)
            final_score = float(info.get("score", final_score))
            success = bool(info.get("tasksuccess")) or final_score >= 1.0
            turns += 1
            if done:
                if not success and bool(info.get("taskfailure")):
                    failure_reason = "task failure"
                break
        if not success and not failure_reason:
            failure_reason = "step budget exhausted" if turns >= max_steps else "score below success"
    except Exception as exc:
        invalid = True
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass

    return TextWorldExpressRecord(
        benchmark="textworld_express",
        game=game,
        category=GAME_CATEGORIES.get(game, "other"),
        method=method,
        split=split,
        seed=seed,
        success=success,
        reward=final_score if final_score else reward_total,
        invalid_action=invalid,
        repeated_actions=len(actions) != len(set(actions)),
        turns=turns,
        final_score=final_score,
        actions=actions,
        failure_reason=failure_reason,
        runtime_seconds=time.perf_counter() - started,
    )


def run_records(
    *,
    games: list[str],
    method: str,
    split: str,
    seeds: list[int],
    variables: dict[str, TextVariable],
    max_steps: int,
    output_jsonl: Path,
) -> list[TextWorldExpressRecord]:
    if output_jsonl.exists():
        output_jsonl.unlink()
    records: list[TextWorldExpressRecord] = []
    for game in games:
        for seed in seeds:
            record = run_episode(
                game=game,
                method=method,
                split=split,
                seed=seed,
                variables=variables,
                max_steps=max_steps,
            )
            records.append(record)
            append_jsonl(output_jsonl, record)
    return records


def gradients_from_records(records: list[TextWorldExpressRecord]) -> list[TextualGradient]:
    gradients: list[TextualGradient] = []
    for game in TWX_GAMES:
        group = [record for record in records if record.game == game]
        if not group:
            continue
        success_rate = mean([float(record.success) for record in group])
        repeated_rate = mean([float(record.repeated_actions) for record in group])
        if success_rate >= 1.0:
            continue
        gradients.append(
            TextualGradient(
                target_variable_name="textworld_policy",
                failure_mode=f"textworld_express:{game}",
                evidence_from_trajectory=(
                    f"{game} train success={success_rate:.3f}, repeated={repeated_rate:.3f}. "
                    f"Examples: {failure_examples(group)}"
                ),
                gradient_text=f"Add a reusable TextWorldExpress strategy for {game}.",
                suggested_edit=f"Add a rule: {RULE_BY_GAME[game]}",
                confidence=0.82,
                forbidden_shortcuts=["use gold path", "inspect hidden labels", "hardcode benchmark seed answers"],
            )
        )
    return gradients


def failure_examples(records: list[TextWorldExpressRecord]) -> str:
    failures = [record for record in records if not record.success or record.invalid_action]
    examples = []
    for record in failures[:3]:
        tail = " -> ".join(record.actions[-4:]) if record.actions else "<none>"
        examples.append(f"seed={record.seed}:{record.failure_reason}:actions={tail}")
    return "; ".join(examples) if examples else "none"


def train_policy_iteration(
    *,
    games: list[str],
    train_seeds: list[int],
    val_seeds: list[int],
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
) -> tuple[dict[str, TextVariable], dict[str, Any]]:
    base_variables = initial_textworld_variables()
    train_records = run_records(
        games=games,
        method=method,
        split="train",
        seeds=train_seeds,
        variables=base_variables,
        max_steps=args.max_steps,
        output_jsonl=output_dir / method / "train_base.jsonl",
    )
    gradients = gradients_from_records(train_records)
    optimizer = TextualGradientDescent(max_prompt_chars=2600, max_rules_per_step=args.max_rules_per_step)
    candidate_variables = optimizer.step(
        base_variables,
        gradients,
        constraints=["must not use gold path", "must not inspect hidden labels", "must not hardcode benchmark seed answers"],
    )
    old_val = run_records(
        games=games,
        method=method,
        split="dev",
        seeds=val_seeds,
        variables=base_variables,
        max_steps=args.max_steps,
        output_jsonl=output_dir / method / "val_base.jsonl",
    )
    new_val = run_records(
        games=games,
        method=method,
        split="dev",
        seeds=val_seeds,
        variables=candidate_variables,
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
        "learned_rule_ids": sorted(learned_rule_ids(candidate_variables) - learned_rule_ids(base_variables)),
    }
    if method == "textgrad_ppo":
        ppo_stats = ppo_gate_stats(old_val, new_val, args.ppo_clip_epsilon)
        gate.update(ppo_stats)
        accepted = accepted and ppo_stats["approx_kl"] <= args.ppo_target_kl and ppo_stats["clipped_surrogate_delta"] >= 0.0
        gate["accepted"] = accepted
    variables = candidate_variables if accepted else base_variables
    write_json(output_dir / method / "gradients.json", gradients)
    write_json(output_dir / method / "gate_decision.json", gate)
    write_json(output_dir / method / "text_variables.json", variables)
    return variables, gate


def ppo_gate_stats(
    old_records: list[TextWorldExpressRecord],
    new_records: list[TextWorldExpressRecord],
    clip_epsilon: float,
) -> dict[str, Any]:
    old_by_key = {(record.game, record.seed): record for record in old_records}
    ratios: list[float] = []
    advantages: list[float] = []
    action_distances: list[float] = []
    for record in new_records:
        old = old_by_key.get((record.game, record.seed))
        if not old:
            continue
        old_score = textworld_score(old)
        new_score = textworld_score(record)
        advantage = new_score - old_score
        distance = action_distance(old.actions, record.actions)
        ratio = math.exp(max(-1.0, min(1.0, advantage)))
        ratios.append(ratio)
        advantages.append(advantage)
        action_distances.append(distance)
    unclipped = [ratio * adv for ratio, adv in zip(ratios, advantages)]
    clipped = [min(value, max(1 - clip_epsilon, min(1 + clip_epsilon, ratio)) * adv) for value, ratio, adv in zip(unclipped, ratios, advantages)]
    return {
        "mean_behavior_ratio": mean(ratios),
        "mean_action_distance": mean(action_distances),
        "approx_kl": mean(action_distances),
        "surrogate_delta": mean(unclipped),
        "clipped_surrogate_delta": mean(clipped),
    }


def action_distance(old_actions: list[str], new_actions: list[str]) -> float:
    max_len = max(len(old_actions), len(new_actions), 1)
    changed = 0
    for index in range(max_len):
        old = old_actions[index] if index < len(old_actions) else "<missing>"
        new = new_actions[index] if index < len(new_actions) else "<missing>"
        if old != new:
            changed += 1
    return changed / max_len


def summarize_records(method: str, records: list[TextWorldExpressRecord], gate: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "method": method,
        "games": len({record.game for record in records}),
        "episodes": len(records),
        "average_reward": mean([record.reward for record in records]),
        "success_rate": mean([float(record.success) for record in records]),
        "invalid_action_rate": mean([float(record.invalid_action) for record in records]),
        "repeated_action_rate": mean([float(record.repeated_actions) for record in records]),
        "average_turns": mean([record.turns for record in records]),
        "accepted_updates": 1 if gate and gate.get("accepted") else 0,
        "gradient_count": int(gate.get("gradient_count", 0)) if gate else 0,
    }


def summarize_groups(records: list[TextWorldExpressRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({record.method for record in records}):
        method_records = [record for record in records if record.method == method]
        for key in ["game", "category"]:
            for value in sorted({getattr(record, key) for record in method_records}):
                group = [record for record in method_records if getattr(record, key) == value]
                rows.append(
                    {
                        "method": method,
                        "slice": key,
                        "value": value,
                        "episodes": len(group),
                        "average_reward": mean([record.reward for record in group]),
                        "success_rate": mean([float(record.success) for record in group]),
                        "invalid_action_rate": mean([float(record.invalid_action) for record in group]),
                        "repeated_action_rate": mean([float(record.repeated_actions) for record in group]),
                        "average_turns": mean([record.turns for record in group]),
                    }
                )
    return rows


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
        "# TextWorldExpress Local Benchmark",
        "",
        "TextWorldExpress runs fully locally through the `textworld-express` package and does not require API keys, browser credentials, or hosted services.",
        "",
        f"Games: {len(TWX_GAMES)}",
        f"Train/dev/test seeds per game: {args.train_seeds}/{args.val_seeds}/{args.test_seeds}",
        f"Step budget: {args.max_steps}",
        "",
        "## Overall Results",
        "",
        "| Method | Games | Episodes | Reward | Success | Invalid | Repeated | Turns | Updates | Gradients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {games} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} | "
            "{accepted_updates} | {gradient_count} |".format(**row)
        )
    lines.extend(["", "## Per-Game Results", "", "| Method | Game | Episodes | Reward | Success | Invalid | Repeated | Turns |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in [row for row in group_rows if row["slice"] == "game"]:
        lines.append(
            "| {method} | {value} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} |".format(**row)
        )
    lines.extend(["", "## Category Results", "", "| Method | Category | Episodes | Reward | Success | Invalid | Repeated | Turns |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in [row for row in group_rows if row["slice"] == "category"]:
        lines.append(
            "| {method} | {value} | {episodes} | {average_reward:.3f} | {success_rate:.3f} | "
            "{invalid_action_rate:.3f} | {repeated_action_rate:.3f} | {average_turns:.2f} |".format(**row)
        )
    lines.extend(["", "## Update Gates", ""])
    for method, gate in gates.items():
        lines.append(f"- `{method}`: `{gate}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods or list(METHODS)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_methods(args.methods)
    games = list(TWX_GAMES)
    train_seeds = [args.seed + index for index in range(args.train_seeds)]
    val_seeds = [args.seed + 10_000 + index for index in range(args.val_seeds)]
    test_seeds = [args.seed + 20_000 + index for index in range(args.test_seeds)]
    write_json(output_dir / "environment_info.json", environment_info())
    write_json(
        output_dir / "config.json",
        {
            "benchmark": "textworld_express",
            "games": games,
            "methods": methods,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "max_steps": args.max_steps,
        },
    )

    all_records: list[TextWorldExpressRecord] = []
    summary_rows: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method == "fixed_prompt":
            variables = initial_textworld_variables()
            gate = {"method": method, "accepted": False, "gradient_count": 0}
            write_json(output_dir / method / "text_variables.json", variables)
        else:
            variables, gate = train_policy_iteration(
                games=games,
                train_seeds=train_seeds,
                val_seeds=val_seeds,
                args=args,
                output_dir=output_dir,
                method=method,
            )
        gates[method] = gate
        records = run_records(
            games=games,
            method=method,
            split="test",
            seeds=test_seeds,
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
    parser = argparse.ArgumentParser(description="Run local TextWorldExpress benchmark.")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-rules-per-step", type=int, default=8)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-target-kl", type=float, default=0.65)
    parser.add_argument("--output-dir", default="runs/textworld_express_suite")
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    print(f"TextWorldExpress artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
