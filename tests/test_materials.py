r"""
M7 self-check: lighting model, texture sampling and material rendering.

Run from the project root:
    .venv/Scripts/python.exe tests/test_materials.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials import (
    make_grass_texture,
    make_stone_texture,
    sample_nearest,
    shade,
    face_normal,
    AMBIENT,
    DIFFUSE,
)
from geometry import ground_mesh
from camera import Camera
from renderer import Renderer
from transforms import translation, scale, model_matrix
from materials import Material


def test_texture_shapes():
    grass = make_grass_texture(64)
    stone = make_stone_texture(64)
    assert grass.shape == (64, 64, 3)
    assert stone.shape == (64, 64, 3)
    assert grass.dtype == np.uint8


def test_nearest_sample_wrap_and_bounds():
    tex = np.zeros((8, 8, 3), dtype=np.uint8)
    tex[0, 0] = [255, 0, 0]
    assert np.allclose(sample_nearest(tex, 0.0, 0.0), [255, 0, 0])
    # u = 1.0 wraps back to column 0
    assert np.allclose(sample_nearest(tex, 1.0, 0.0), [255, 0, 0])


def test_lighting_front_vs_back():
    light = (0.0, 1.0, 0.0)
    front = shade([100, 100, 100], [0, 1, 0], light)
    back = shade([100, 100, 100], [0, -1, 0], light)
    # Front face fully lit; back face receives ambient only.
    assert np.allclose(front, [100, 100, 100])
    assert np.allclose(back, np.full(3, 100 * AMBIENT), atol=1.0)


def test_face_normal():
    a = np.array([0.0, 0, 0])
    b = np.array([1.0, 0, 0])
    c = np.array([0.0, 0, 1.0])
    n = face_normal(a, b, c)
    assert np.allclose(np.abs(n), [0, 1, 0], atol=1e-9)


def test_render_textured_ground():
    renderer = Renderer(160, 160)
    cam = Camera(eye=(1.0, 2.5, 3.5), target=(1.0, 0.0, 1.0),
                 width=160, height=160)
    ground = ground_mesh(2, 2)
    material = Material(base_color=(35, 105, 48),
                        texture=make_grass_texture(64))
    model = model_matrix((1, 1, 1), 0.0, (0.0, 0.0, 0.0))
    renderer.draw_mesh(cam, ground, model, material)
    # Some pixel must be painted (not the cleared background).
    assert renderer.colorbuf.sum() > 0


if __name__ == "__main__":
    test_texture_shapes()
    test_nearest_sample_wrap_and_bounds()
    test_lighting_front_vs_back()
    test_face_normal()
    test_render_textured_ground()
    print("M7 material / lighting / texture tests OK")
