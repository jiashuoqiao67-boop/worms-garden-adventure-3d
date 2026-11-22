r"""
M1 self-check: world coordinate conversion and maze topology.

Run from the project root:
    .venv/Scripts/python.exe tests/test_maze.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maze
from maze import (
    cell_center_3d,
    old_screen_to_world,
    generate_maze,
    get_available_neighbors,
)


def test_cell_center():
    # handbook 4.1 worked example: cell (2, 1) -> [2.5, 0, 1.5]
    assert np.allclose(cell_center_3d(2, 1), [2.5, 0.0, 1.5])
    # height becomes world Y, not Z
    assert np.allclose(cell_center_3d(0, 0, height=0.22), [0.5, 0.22, 0.5])


def test_screen_to_world():
    # handbook 4.4: start pixel (65, 470) at height 0.22 -> [0.5, 0.22, 7.5]
    assert np.allclose(old_screen_to_world(65, 470, 0.22),
                       [0.5, 0.22, 7.5])


def test_conversion_consistency():
    # Cell (2, 1) old pixel centre: x = 40 + 2*50 + 25 = 165,
    # y = 95 + 1*50 + 25 = 170. Both conversions must agree.
    a = cell_center_3d(2, 1, 0.0)
    b = old_screen_to_world(165, 170, 0.0)
    assert np.allclose(a, b)


def test_world_constants():
    assert np.isclose(maze.WALL_THICKNESS_WORLD, 8 / 50)   # 0.16
    assert np.isclose(maze.WORM_RADIUS, 11 / 50)          # 0.22
    assert np.isclose(maze.WORM_SPEED, 230 / 50)         # 4.6
    assert np.isclose(maze.BIRD_RADIUS, 10 / 50)          # 0.20
    assert np.isclose(maze.BIRD_SPEED, 75 / 50)           # 1.5
    assert np.isclose(maze.WATER_HOLE_RADIUS, 13 / 50)    # 0.26


def test_maze_openings():
    small = generate_maze(cols=3, rows=3, start_cell=(0, 1),
                          finish_cell=(2, 1), seed=1)
    assert small[1][0]["walls"]["W"] is False   # entrance open
    assert small[1][2]["walls"]["E"] is False   # exit open


def test_maze_connectivity():
    # A generated maze must connect every cell (depth-first backtracking).
    cols, rows = 3, 3
    start = (0, 1)
    m = generate_maze(cols=cols, rows=rows, start_cell=start,
                       finish_cell=(2, 1), seed=1)

    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nxt in get_available_neighbors(m, cell, cols, rows):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    assert len(seen) == cols * rows, f"unreachable cells: {seen}"


if __name__ == "__main__":
    test_cell_center()
    test_screen_to_world()
    test_conversion_consistency()
    test_world_constants()
    test_maze_openings()
    test_maze_connectivity()
    print("M1 maze / coordinate tests OK")
