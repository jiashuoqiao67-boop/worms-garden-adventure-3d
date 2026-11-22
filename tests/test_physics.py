r"""
M8/M9 self-check: wall collision, sliding, jump gravity and height-aware hits.

Run from the project root:
    .venv/Scripts/python.exe tests/test_physics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics import (
    circle_hits_box_xz,
    hits_any_wall,
    move_xz,
    JumpState,
    request_jump,
    update_jump,
    hits_water,
    hits_bird,
    collects_apple,
    GROUND_Y,
    JUMP_SPEED,
    GRAVITY,
)

# A north/south wall running along X: centre (2.5, ., 1), size (1, h, 0.16)
WALL = (np.array([2.5, 0.375, 1.0]), np.array([1.0, 0.75, 0.16]))


def test_wall_hit_inside():
    assert circle_hits_box_xz(2.5, 1.0, 0.22, WALL[0], WALL[1])


def test_wall_clear_outside():
    assert not circle_hits_box_xz(2.5, 0.4, 0.22, WALL[0], WALL[1])


def test_slide_along_wall_x():
    # Moving along X parallel to the wall must not be blocked.
    pos = np.array([2.0, GROUND_Y, 0.5])
    move_xz(pos, [1, 0, 0], 4.6, 0.1, [WALL], 0.22)
    assert pos[0] > 2.0
    assert np.isclose(pos[2], 0.5)


def test_blocked_into_wall_z():
    # Moving into the wall along Z must be stopped at the footprint.
    pos = np.array([2.5, GROUND_Y, 0.5])
    move_xz(pos, [0, 0, 1], 4.6, 0.1, [WALL], 0.22)
    # Did not cross into the wall footprint (z around 1.0 +/- half thickness).
    assert pos[2] < 1.0 - 0.08 + 0.22


def test_jump_rises_and_lands():
    state = JumpState()
    request_jump(state)
    assert np.isclose(state.vy, JUMP_SPEED)
    assert not state.grounded

    # Integrate until it returns to the ground.
    dt = 0.01
    for _ in range(200):
        update_jump(state, dt)
        if state.grounded:
            break
    assert state.grounded
    assert np.isclose(state.y, GROUND_Y, atol=1e-6)


def test_jump_cooldown():
    state = JumpState()
    request_jump(state)
    vy_before = state.vy
    request_jump(state)  # airborne, must not trigger a second jump
    assert np.isclose(state.vy, vy_before)
    assert not state.grounded


def test_water_only_when_grounded():
    holes = [np.array([1.0, 0.0, 1.0])]
    pos_ground = np.array([1.0, GROUND_Y, 1.0])
    pos_air = np.array([1.0, 0.8, 1.0])
    assert hits_water(pos_ground, holes, 0.22, 0.26)
    assert not hits_water(pos_air, holes, 0.22, 0.26)


def test_bird_and_apple_distance():
    worm = np.array([0.0, GROUND_Y, 0.0])
    assert hits_bird(worm, np.array([0.1, GROUND_Y, 0.1]))
    assert not hits_bird(worm, np.array([2.0, GROUND_Y, 2.0]))
    assert collects_apple(worm, np.array([0.15, GROUND_Y, 0.0]))
    assert not collects_apple(worm, np.array([1.5, GROUND_Y, 0.0]))


if __name__ == "__main__":
    test_wall_hit_inside()
    test_wall_clear_outside()
    test_slide_along_wall_x()
    test_blocked_into_wall_z()
    test_jump_rises_and_lands()
    test_jump_cooldown()
    test_water_only_when_grounded()
    test_bird_and_apple_distance()
    print("M8/M9 physics tests OK")
