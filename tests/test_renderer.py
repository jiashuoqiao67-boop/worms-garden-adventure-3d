r"""
M5 self-check: rasterization, barycentric weights and Z-buffer occlusion.

Run from the project root:
    .venv/Scripts/python.exe tests/test_renderer.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer import Renderer, barycentric


def test_clear():
    r = Renderer(80, 80)
    assert np.all(r.depthbuf == np.inf)
    assert np.all(r.colorbuf == 0)


def test_barycentric_inside_outside():
    a = np.array([10.0, 10.0])
    b = np.array([50.0, 10.0])
    c = np.array([10.0, 50.0])
    w = barycentric(np.array([20.0, 20.0]), a, b, c)
    assert w is not None
    assert np.isclose(sum(w), 1.0)
    assert barycentric(np.array([70.0, 70.0]), a, b, c) is None


def test_rasterize_fills_inside():
    r = Renderer(80, 80)
    screen = [np.array([10.0, 10.0]), np.array([50.0, 10.0]),
              np.array([10.0, 50.0])]
    r.rasterize_triangle(screen, [2.0, 2.0, 2.0], (255, 0, 0))
    assert np.allclose(r.colorbuf[20, 20], [255, 0, 0])
    assert r.depthbuf[20, 20] < np.inf


def test_occlusion_near_overwrites_far():
    r = Renderer(80, 80)
    tri = [np.array([10.0, 10.0]), np.array([50.0, 10.0]),
           np.array([10.0, 50.0])]
    r.rasterize_triangle(tri, [3.0, 3.0, 3.0], (0, 0, 255))   # far blue
    r.rasterize_triangle(tri, [1.0, 1.0, 1.0], (255, 0, 0))   # near red
    assert np.allclose(r.colorbuf[20, 20], [255, 0, 0])
    assert np.isclose(r.depthbuf[20, 20], 1.0)


def test_far_does_not_overwrite_near():
    r = Renderer(80, 80)
    tri = [np.array([10.0, 10.0]), np.array([50.0, 10.0]),
           np.array([10.0, 50.0])]
    r.rasterize_triangle(tri, [1.0, 1.0, 1.0], (255, 0, 0))   # near red first
    r.rasterize_triangle(tri, [3.0, 3.0, 3.0], (0, 0, 255))   # far blue later
    assert np.allclose(r.colorbuf[20, 20], [255, 0, 0])


if __name__ == "__main__":
    test_clear()
    test_barycentric_inside_outside()
    test_rasterize_fills_inside()
    test_occlusion_near_overwrites_far()
    test_far_does_not_overwrite_near()
    print("M5 renderer tests OK")
