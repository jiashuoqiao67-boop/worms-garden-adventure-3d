r"""
M2 self-check: vectors, homogeneous 4x4 transforms, point vs direction,
and normal transform.

Run from the project root:
    .venv/Scripts/python.exe tests/test_transforms.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms import (
    normalized,
    vector_length,
    dot,
    cross,
    translation,
    rotation_y,
    scale,
    transform_point,
    transform_direction,
    transform_normal,
)


def test_rotation_y_then_translate():
    # handbook 5.2 worked example: point (1,0,0), rotate 90 deg about Y,
    # then translate by (2,0,3) -> (2, 0, 2).
    M = translation(2, 0, 3) @ rotation_y(np.pi / 2)
    assert np.allclose(transform_point(M, [1, 0, 0]), [2.0, 0.0, 2.0])


def test_direction_is_not_translated():
    # A direction (w = 0) must be unaffected by translation.
    M = translation(5, 5, 5)
    assert np.allclose(transform_direction(M, [1, 2, 3]), [1, 2, 3])


def test_scale_translate_order_matters():
    # Translate-then-scale differs from scale-then-translate (handbook 5.3).
    v = [1.0, 0.0, 0.0]
    scale_first = translation(1, 0, 0) @ scale(2, 2, 2)   # S then T
    translate_first = scale(2, 2, 2) @ translation(1, 0, 0)  # T then S
    assert np.allclose(transform_point(scale_first, v), [3.0, 0.0, 0.0])
    assert np.allclose(transform_point(translate_first, v), [4.0, 0.0, 0.0])


def test_uniform_scale_keeps_normal_direction():
    M = scale(2, 2, 2)
    n = normalized([1, 1, 0])
    assert np.allclose(transform_normal(M, n), n)


def test_nonuniform_normal_uses_inverse_transpose():
    # S(2,1,1): local normal (1,1,0) -> (0.5,1,0), normalized.
    M = scale(2, 1, 1)
    expected = normalized([0.5, 1.0, 0.0])
    assert np.allclose(transform_normal(M, [1, 1, 0]), expected)


def test_normalized_and_length():
    assert np.isclose(vector_length(normalized([3, 4, 0])), 1.0)
    try:
        normalized([0, 0, 0])
        assert False, "zero vector should raise"
    except ValueError:
        pass


def test_dot_cross_properties():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.isclose(dot(a, b), 0.0)
    c = cross(a, b)
    assert np.allclose(c, [0.0, 0.0, 1.0])
    # cross product is perpendicular to both inputs
    assert np.isclose(dot(c, a), 0.0)
    assert np.isclose(dot(c, b), 0.0)


if __name__ == "__main__":
    test_rotation_y_then_translate()
    test_direction_is_not_translated()
    test_scale_translate_order_matters()
    test_uniform_scale_keeps_normal_direction()
    test_nonuniform_normal_uses_inverse_transpose()
    test_normalized_and_length()
    test_dot_cross_properties()
    print("M2 transform tests OK")
