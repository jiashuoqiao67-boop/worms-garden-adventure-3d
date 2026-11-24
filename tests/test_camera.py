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


def test_screen_ray_hits_the_ground_plane():
    """A click straight at the centre must land where the camera looks."""
    cam = Camera(eye=(4.0, 4.0, 4.0), target=(2.0, 0.0, 2.0),
                 width=320, height=240)
    hit = cam.ground_hit(160, 120, ground_y=0.0)
    assert hit is not None
    assert abs(hit[1]) < 1e-9
    # The pixel ray through the centre is the eye->target direction, whose
    # intersection with y = 0 is exactly the target.
    assert np.allclose(hit[[0, 2]], [2.0, 2.0], atol=1e-6)


def test_screen_ray_corners_differ():
    cam = Camera(eye=(4.0, 4.0, 4.0), target=(2.0, 0.0, 2.0),
                 width=320, height=240)
    left = cam.ground_hit(20, 120)
    right = cam.ground_hit(300, 120)
    assert left is not None and right is not None
    assert float(np.linalg.norm(left - right)) > 0.5


def test_orthographic_ray_is_parallel():
    cam = Camera(eye=(0.0, 5.0, 5.0), target=(0.0, 0.0, 0.0),
                 projection="orthographic", width=320, height=240,
                 pixels_per_unit=30.0)
    _, d1 = cam.screen_ray(20, 100)
    _, d2 = cam.screen_ray(300, 200)
    assert np.allclose(d1, d2)


if __name__ == "__main__":
    test_basis_sanity()
    test_camera_space()
    test_perspective_shrinks_with_depth()
    test_orthographic_size_independent_of_depth()
    test_near_plane_rejects_behind()
    test_clip_one_vertex_behind()
    test_clip_triangle_entirely_behind()
    test_toggle_projection()
    test_screen_ray_hits_the_ground_plane()
    test_screen_ray_corners_differ()
    test_orthographic_ray_is_parallel()
    print("M4 camera / projection tests OK")
