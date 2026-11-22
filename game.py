"""
Level setup, apples, birds, scoring, timer and finish rules (handbook ch. 12).

The original 2D rules are preserved: 4 apples to unlock the exit, birds follow
corridor neighbours, score persists across levels, R restarts score and level.
Positions are world coordinates; birds hover at BIRD_HEIGHT, apples bob around
APPLE_HEIGHT.
"""

import math
import random

import numpy as np

from maze import (
    generate_maze,
    get_available_neighbors,
    cell_center_3d,
    START_CELL,
    FINISH_CELL,
    COLS,
    ROWS,
    BIRD_SPEED,
    WATER_HOLE_RADIUS,
)
from geometry import maze_wall_boxes
from physics import GROUND_Y

APPLE_COUNT = 4
WATER_HOLE_COUNT = 9
APPLE_VALUE_SILVER = 50
APPLE_VALUE_GOLD = 100

BIRD_HEIGHT = 0.55
APPLE_HEIGHT = 0.28
APPLE_BOB_AMP = 0.05
APPLE_BOB_FREQ = 1.5
# How many recently visited cells a bird remembers; it prefers unvisited
# corridor neighbours so it patrols a wider area instead of oscillating
# between the same two or three cells.
BIRD_MEMORY = 6

REWARD_VALUES = [APPLE_VALUE_SILVER, APPLE_VALUE_SILVER,
                 APPLE_VALUE_SILVER, APPLE_VALUE_GOLD]


def _candidate_cells(maze, cols, rows, blocked):
    return [(c, r) for r in range(rows) for c in range(cols)
            if (c, r) not in blocked]


def build_level(level, seed=None, cols=COLS, rows=ROWS,
                start=START_CELL, finish=FINISH_CELL):
    """Create a fresh maze with water holes, apples and birds."""
    rng = random.Random(seed)
    maze = generate_maze(cols=cols, rows=rows,
                         start_cell=start, finish_cell=finish)
    boxes = maze_wall_boxes(maze, cols, rows)

    blocked = {start, finish}

    # Water holes
    water_cells = rng.sample(_candidate_cells(maze, cols, rows, blocked),
                             min(WATER_HOLE_COUNT, cols * rows - 2))
    blocked.update(water_cells)
    water_holes = [cell_center_3d(c, r, 0.0)[[0, 2]] for c, r in water_cells]

    # Apples: three silver (50) and one gold (100)
    apple_cells = rng.sample(_candidate_cells(maze, cols, rows, blocked),
                             min(APPLE_COUNT, cols * rows - len(blocked)))
    blocked.update(apple_cells)
    apples = []
    for cell, value in zip(apple_cells, REWARD_VALUES):
        pos = cell_center_3d(cell[0], cell[1], APPLE_HEIGHT)
        apples.append({
            "cell": cell,
            "pos": pos,
            "value": value,
            "collected": False,
            "phase": rng.uniform(0, 2 * math.pi),
        })

    # Birds follow corridor neighbours; count grows with level, capped at 8.
    bird_count = min(2 + level, 8)
    bird_cells = rng.sample(_candidate_cells(maze, cols, rows, blocked),
                            min(bird_count, cols * rows - len(blocked)))
    birds = []
    for cell in bird_cells:
        neighbours = get_available_neighbors(maze, cell, cols, rows)
        target = rng.choice(neighbours) if neighbours else cell
        birds.append({
            "cell": cell,
            "target": target,
            "previous": cell,
            "recent": [cell],
            "yaw": 0.0,
            "pos": cell_center_3d(cell[0], cell[1], BIRD_HEIGHT),
            "speed": BIRD_SPEED + rng.uniform(-0.2, 0.4),
        })

    return {
        "level": level,
        "maze": maze,
        "wall_boxes": boxes,
        "water_holes": water_holes,
        "apples": apples,
        "birds": birds,
        "position": cell_center_3d(START_CELL[0], START_CELL[1], GROUND_Y),
        "score": 0,
        "collected": 0,
        "finished": False,
    }


def apple_world_pos(apple, t):
    """Bobbed apple position: small time-dependent vertical offset (12.2)."""
    pos = apple["pos"].copy()
    pos[1] = APPLE_HEIGHT + APPLE_BOB_AMP * math.sin(
        2 * math.pi * APPLE_BOB_FREQ * t + apple["phase"])
    return pos


def update_birds(state, dt):
    """Move birds toward their target cell; pick a new corridor neighbour."""
    cols, rows = COLS, ROWS
    for bird in state["birds"]:
        target_pos = cell_center_3d(bird["target"][0], bird["target"][1],
                                     BIRD_HEIGHT)
        direction = target_pos - bird["pos"]
        distance = np.linalg.norm(direction)

        if distance <= 0.05:
            bird["pos"] = target_pos.copy()
            bird["previous"] = bird["cell"]
            bird["cell"] = bird["target"]
            recent = bird.setdefault("recent", [bird["cell"]])
            recent.append(bird["cell"])
            del recent[:-BIRD_MEMORY]
            neighbours = get_available_neighbors(
                state["maze"], bird["cell"], cols, rows)
            # Prefer a neighbour not visited recently, then one that is not
            # the cell just left; fall back to any neighbour in a dead end.
            fresh = [n for n in neighbours if n not in recent]
            alternatives = [n for n in neighbours if n != bird["previous"]]
            choices = fresh or alternatives or neighbours
            if choices:
                bird["target"] = random.choice(choices)
        else:
            bird["yaw"] = math.atan2(direction[0], direction[2])
            bird["pos"] += direction / distance * bird["speed"] * dt


def collect_apple(state, apple):
    apple["collected"] = True
    state["score"] += apple["value"]
    state["collected"] += 1


def at_finish(state, radius=0.25):
    finish = cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], GROUND_Y)
    return np.linalg.norm(state["position"] - finish) < radius


def check_finish(state):
    """Advance to the next level only after all apples are collected."""
    if at_finish(state) and state["collected"] == APPLE_COUNT:
        state["finished"] = True
        return True
    return False


def reset_player(state):
    state["position"] = cell_center_3d(START_CELL[0], START_CELL[1], GROUND_Y)


def complete_level(state):
    """Start a new level; score persists unless the run is restarted."""
    next_level = state["level"] + 1
    score = state["score"]
    new_state = build_level(next_level)
    new_state["score"] = score
    return new_state


def restart_run(state):
    """R key: reset score and level to 1 (original 2D behaviour)."""
    return build_level(1)
