r"""
M3 self-check: cube / ground / wall-box geometry.

Run from the project root:
    .venv/Scripts/python.exe tests/test_geometry.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import (
    unit_cube,
    cube_mesh,
    ground_mesh,
    wall_box,
    WALL_HEIGHT,
    WALL_THICKNESS,
)


def test_cube_topology():
    cube = unit_cube()
    assert cube.vertices.shape == (8, 3)
    assert cube.triangles.shape == (12, 3)
    assert np.allclose(cube.vertices.min(axis=0), [-0.5, -0.5, -0.5])
    assert np.allclose(cube.vertices.max(axis=0), [0.5, 0.5, 0.5])


def test_cube_placement():
    cube = cube_mesh(size=(2, 1, 1), center=(1, 0, 0))
    assert np.allclose(cube.vertices.min(axis=0), [0.0, -0.5, -0.5])
    assert np.allclose(cube.vertices.max(axis=0), [2.0, 0.5, 0.5])


def test_ground_grid():
    g = ground_mesh(cols=3, rows=3)
    assert g.vertices.shape == ((3 + 1) * (3 + 1), 3)
    assert g.triangles.shape == (3 * 3 * 2, 3)


def test_ground_normal_points_up():
    g = ground_mesh(cols=2, rows=2)
    a, b, c = g.vertices[g.triangles[0]]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    assert normal[1] > 0.99   # +Y


def test_wall_box_dimensions():
    n_center, n_size = wall_box(2, 1, "N")
    assert np.allclose(n_center, [2.5, WALL_HEIGHT / 2, 1.0])
    assert np.allclose(n_size, [1.0, WALL_HEIGHT, WALL_THICKNESS])

    s_center, s_size = wall_box(2, 1, "S")
    assert np.allclose(s_center, [2.5, WALL_HEIGHT / 2, 2.0])

    e_center, e_size = wall_box(2, 1, "E")
    assert np.allclose(e_center, [3.0, WALL_HEIGHT / 2, 1.5])
    assert np.allclose(e_size, [WALL_THICKNESS, WALL_HEIGHT, 1.0])

    w_center, w_size = wall_box(2, 1, "W")
    assert np.allclose(w_center, [2.0, WALL_HEIGHT / 2, 1.5])


if __name__ == "__main__":
    test_cube_topology()
    test_cube_placement()
    test_ground_grid()
    test_ground_normal_points_up()
    test_wall_box_dimensions()
    print("M3 geometry tests OK")
