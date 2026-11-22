r"""
M4 self-check: camera basis, perspective/orthographic projection, clipping.

Run from the project root:
    .venv/Scripts/python.exe tests/test_camera.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import (
    Camera,
    perspective_project,
    orthographic_project,
    clip_triangle_near,
)


def test_basis_sanity():
    cam = Camera(eye=(0, 0, 0), target=(0, 0, 1))
    forward, right, up = cam.basis()
    assert np.allclose(right, [1, 0, 0])
    assert np.allclose(up, [0, 1, 0])
    assert np.allclose(forward, [0, 0, 1])


def test_camera_space():
    cam = Camera(eye=(0, 0, 0), target=(0, 0, 1))
    assert np.allclose(cam.to_camera_space([1, 2, 3]), [1, 2, 3])


def test_perspective_shrinks_with_depth():
    # Same x, farther depth -> smaller screen offset in perspective.
    near = perspective_project([1.0, 0.0, 2.0], fov_y=70.0)
    far = perspective_project([1.0, 0.0, 5.0], fov_y=70.0)
    assert abs(near[0] - 320) > abs(far[0] - 320)


def test_orthographic_size_independent_of_depth():
    a = orthographic_project([1.0, 0.0, 2.0])
    b = orthographic_project([1.0, 0.0, 5.0])
    assert np.allclose(a, b)


def test_near_plane_rejects_behind():
    assert perspective_project([1.0, 0.0, 0.05]) is None
    assert orthographic_project([1.0, 0.0, 0.05]) is None


def test_clip_one_vertex_behind():
    # Handbook 7.6: one vertex behind the near plane -> 4-vertex polygon
    # (fan-triangulated into two triangles later).
    tri = [[-0.4, 0.0, 0.05], [0.0, 0.0, 0.5], [0.6, 0.0, 0.5]]
    poly = clip_triangle_near(tri, near=0.1)
    assert poly is not None
    assert len(poly) == 4


def test_clip_triangle_entirely_behind():
    tri = [[-0.4, 0.0, 0.02], [0.0, 0.0, 0.05], [0.6, 0.0, 0.08]]
    assert clip_triangle_near(tri, near=0.1) is None


def test_toggle_projection():
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0))
    assert cam.projection == "perspective"
    cam.toggle_projection()
    assert cam.projection == "orthographic"
    cam.toggle_projection()
    assert cam.projection == "perspective"


if __name__ == "__main__":
    test_basis_sanity()
    test_camera_space()
    test_perspective_shrinks_with_depth()
    test_orthographic_size_independent_of_depth()
    test_near_plane_rejects_behind()
    test_clip_one_vertex_behind()
    test_clip_triangle_entirely_behind()
    test_toggle_projection()
    print("M4 camera / projection tests OK")
