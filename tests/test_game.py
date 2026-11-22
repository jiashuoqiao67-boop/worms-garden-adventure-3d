r"""
M10 self-check: level setup, apples, birds, scoring and finish rules.

Run from the project root:
    .venv/Scripts/python.exe tests/test_game.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import (
    build_level,
    apple_world_pos,
    collect_apple,
    check_finish,
    reset_player,
    complete_level,
    update_birds,
    APPLE_COUNT,
    WATER_HOLE_COUNT,
    REWARD_VALUES,
)
from maze import START_CELL, FINISH_CELL, cell_center_3d
from physics import GROUND_Y


def test_level_spawn_counts():
    state = build_level(1, seed=1)
    assert len(state["apples"]) == APPLE_COUNT
    assert len(state["water_holes"]) == WATER_HOLE_COUNT
    assert len(state["birds"]) == min(2 + 1, 8)
    assert sorted(a["value"] for a in state["apples"]) == sorted(REWARD_VALUES)


def test_start_finish_not_blocked():
    state = build_level(1, seed=2)
    apple_cells = {a["cell"] for a in state["apples"]}
    assert START_CELL not in apple_cells
    assert FINISH_CELL not in apple_cells


def test_apple_bob_moves_with_time():
    state = build_level(1, seed=1)
    apple = state["apples"][0]
    p0 = apple_world_pos(apple, 0.0)
    p1 = apple_world_pos(apple, 1.0)
    assert not np.allclose(p0, p1)


def test_collect_apple_scores():
    state = build_level(1, seed=1)
    apple = state["apples"][0]
    collect_apple(state, apple)
    assert apple["collected"]
    assert state["score"] == apple["value"]
    assert state["collected"] == 1


def test_finish_requires_all_apples():
    state = build_level(1, seed=1)
    finish = cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], GROUND_Y)
    state["position"] = finish.copy()

    # Not enough apples -> finish does not advance.
    assert not check_finish(state)

    state["collected"] = APPLE_COUNT
    assert check_finish(state)


def test_complete_level_keeps_score():
    state = build_level(1, seed=1)
    state["score"] = 175
    nxt = complete_level(state)
    assert nxt["level"] == 2
    assert nxt["score"] == 175
    assert nxt["collected"] == 0


def test_reset_player_returns_to_start():
    state = build_level(1, seed=1)
    state["position"] = np.array([9.0, GROUND_Y, 9.0])
    reset_player(state)
    start = cell_center_3d(START_CELL[0], START_CELL[1], GROUND_Y)
    assert np.allclose(state["position"], start)


def test_birds_update_stay_in_bounds():
    state = build_level(1, seed=1)
    update_birds(state, 0.1)
    for bird in state["birds"]:
        assert 0 <= bird["pos"][0] <= 23
        assert 0 <= bird["pos"][2] <= 15


if __name__ == "__main__":
    test_level_spawn_counts()
    test_start_finish_not_blocked()
    test_apple_bob_moves_with_time()
    test_collect_apple_scores()
    test_finish_requires_all_apples()
    test_complete_level_keeps_score()
    test_reset_player_returns_to_start()
    test_birds_update_stay_in_bounds()
    print("M10 game rules tests OK")
