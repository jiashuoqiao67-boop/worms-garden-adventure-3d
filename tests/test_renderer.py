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


def test_draw_mesh_textured_batch():
    """The batched mesh path must shade and texture a real mesh."""
    from camera import Camera
    from geometry import ground_mesh
    from materials import Material
    from transforms import model_matrix

    cam = Camera(eye=(1.0, 2.0, 3.0), target=(1.0, 0.0, 1.0),
                 width=120, height=120)
    r = Renderer(120, 120)
    tex = np.zeros((8, 8, 3), dtype=np.uint8)
    tex[:, :] = (200, 120, 60)
    mat = Material(base_color=(255, 255, 255), texture=tex)
    ground = ground_mesh(2, 2)
    r.clear()
    r.draw_mesh(cam, ground, model_matrix((1, 1, 1), 0.0, (0, 0, 0)), mat)
    assert r.stats["submitted"] == 8
    painted = int((r.colorbuf.sum(axis=2) > 0).sum())
    assert painted > 200
    # Untextured material on the same mesh uses the flat colour instead.
    r2 = Renderer(120, 120)
    r2.clear()
    r2.draw_mesh(cam, ground, model_matrix((1, 1, 1), 0.0, (0, 0, 0)),
                 Material(base_color=(10, 200, 10)))
    assert r2.colorbuf[:, :, 1].max() >= 100
    assert r2.colorbuf[:, :, 0].max() <= 20


def test_depth_image_separates_near_and_far():
    from camera import Camera
    from geometry import cube_mesh
    from materials import Material
    from transforms import model_matrix

    cam = Camera(eye=(0.0, 0.0, 6.0), target=(0.0, 0.0, 0.0),
                 width=160, height=160)
    r = Renderer(160, 160)
    r.clear()
    near = cube_mesh((1.0, 1.0, 1.0), (0.0, 0.0, 2.0))
    far = cube_mesh((1.0, 1.0, 1.0), (0.0, 0.0, -2.0))
    r.draw_mesh(cam, near, model_matrix((1, 1, 1), 0.0, (0, 0, 0)),
                Material(base_color=(200, 200, 200)))
    r.draw_mesh(cam, far, model_matrix((1, 1, 1), 0.0, (0, 0, 0)),
                Material(base_color=(200, 200, 200)))
    depth = r.depth_image()
    assert depth.shape == (160, 160, 3)
    # The near cube's pixels must differ from the far cube's pixels.
    assert int(depth[:, :, 2].max()) > int(depth[:, :, 2].min())


def test_wireframe_mode_draws_edges():
    r = Renderer(80, 80)
    r.wireframe = True
    tri = [np.array([10.0, 10.0]), np.array([60.0, 12.0]),
           np.array([12.0, 60.0])]
    r.rasterize_triangle(tri, [2.0, 2.0, 2.0], (30, 30, 30))
    edge = np.array([140, 255, 180], dtype=np.uint8)
    assert (r.colorbuf == edge).all(axis=2).any()


if __name__ == "__main__":
    test_clear()
    test_barycentric_inside_outside()
    test_rasterize_fills_inside()
    test_occlusion_near_overwrites_far()
    test_far_does_not_overwrite_near()
    test_draw_mesh_textured_batch()
    test_depth_image_separates_near_and_far()
    test_wireframe_mode_draws_edges()
    print("M5 renderer tests OK")
