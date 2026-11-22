r"""
M6 self-check: maze wall flags map to 3D wall boxes without duplicates.

Run from the project root:
    .venv/Scripts/python.exe tests/test_maze_world.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze import generate_maze
from geometry import maze_wall_boxes, WALL_HEIGHT, WALL_THICKNESS


def _expected_box_count(maze, cols, rows):
    count = 0
    for row in range(rows):
        for col in range(cols):
            walls = maze[row][col]["walls"]
            if walls["N"]:
                count += 1
            if walls["W"]:
                count += 1
            if col == cols - 1 and walls["E"]:
                count += 1
            if row == rows - 1 and walls["S"]:
                count += 1
    return count


def test_wall_boxes_match_flags():
    cols, rows = 3, 3
    maze = generate_maze(cols=cols, rows=rows, start_cell=(0, 1),
                         finish_cell=(2, 1), seed=1)
    boxes = maze_wall_boxes(maze, cols, rows)
    assert len(boxes) == _expected_box_count(maze, cols, rows)


def test_box_dimensions_are_valid():
    maze = generate_maze(cols=3, rows=3, start_cell=(0, 1),
                          finish_cell=(2, 1), seed=1)
    for center, size in maze_wall_boxes(maze, 3, 3):
        span = 1.0 + WALL_THICKNESS
        assert size[1] == WALL_HEIGHT
        assert size[0] in (span, WALL_THICKNESS)
        assert size[2] in (span, WALL_THICKNESS)
        assert np.isclose(center[1], WALL_HEIGHT / 2.0)


def test_entrance_exit_have_no_wall():
    cols, rows = 3, 3
    start, finish = (0, 1), (2, 1)
    maze = generate_maze(cols=cols, rows=rows, start_cell=start,
                         finish_cell=finish, seed=1)
    boxes = maze_wall_boxes(maze, cols, rows)

    # Start W opening: no west wall box on the start cell (centre x = 0).
    assert not any(np.isclose(c[0], 0.0) and np.isclose(c[2], start[1] + 0.5)
                   for c, _ in boxes)
    # Finish E opening: no east wall box on the finish cell (centre x = cols).
    assert not any(np.isclose(c[0], float(cols)) and np.isclose(c[2], finish[1] + 0.5)
                   for c, _ in boxes)


if __name__ == "__main__":
    test_wall_boxes_match_flags()
    test_box_dimensions_are_valid()
    test_entrance_exit_have_no_wall()
    print("M6 maze-to-world wall tests OK")
