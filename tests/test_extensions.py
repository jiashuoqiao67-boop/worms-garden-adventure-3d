r"""
Part B extension self-checks: atmospheric fog and bilinear filtering.

Run from the project root:
    .venv/Scripts/python.exe tests/test_extensions.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials import sample_bilinear, Material
from renderer import Renderer


def _four_tex():
    # 2x2 texture: R, G / B, yellow
    tex = np.zeros((2, 2, 3), dtype=np.uint8)
    tex[0, 0] = (255, 0, 0)
    tex[0, 1] = (0, 255, 0)
    tex[1, 0] = (0, 0, 255)
    tex[1, 1] = (255, 255, 0)
    return tex


def test_bilinear_integer_texel_matches_nearest():
    tex = _four_tex()
    # u=0.25 -> x=0, v=0.25 -> y=0 => red texel
    out = sample_bilinear(tex, 0.25, 0.25)
    assert np.allclose(out, [255, 0, 0], atol=1.0)


def test_bilinear_center_is_average_of_four():
    tex = _four_tex()
    # u=v=0.5 -> halfway between all four texels
    out = sample_bilinear(tex, 0.5, 0.5)
    expected = np.array([127.5, 127.5, 63.75])
    assert np.allclose(out, expected, atol=1.0)


def test_bilinear_interpolates_two_texels_horizontally():
    tex = _four_tex()
    # v=0.25 -> y=0 (top row); u=0.5 -> halfway R and G
    out = sample_bilinear(tex, 0.5, 0.25)
    assert np.allclose(out, [127.5, 127.5, 0.0], atol=1.0)


def test_fog_blends_distant_fragment_toward_fog_color():
    fog = {"color": np.array([128, 128, 128]), "near": 1.0, "far": 3.0}
    r = Renderer(40, 40, fog=fog)
    tri = [np.array([5.0, 5.0]), np.array([35.0, 5.0]), np.array([5.0, 35.0])]
    # Far triangle: fog factor = 1, fragment should be almost pure fog colour.
    r.rasterize_triangle(tri, [3.0, 3.0, 3.0], (255, 0, 0))
    assert np.allclose(r.colorbuf[15, 15], [128, 128, 128], atol=3.0)


def test_fog_leaves_near_fragment_untouched():
    fog = {"color": np.array([128, 128, 128]), "near": 1.0, "far": 3.0}
    r = Renderer(40, 40, fog=fog)
    tri = [np.array([5.0, 5.0]), np.array([35.0, 5.0]), np.array([5.0, 35.0])]
    # Near triangle: fog factor = 0, fragment keeps its own colour.
    r.rasterize_triangle(tri, [1.0, 1.0, 1.0], (255, 0, 0))
    assert np.allclose(r.colorbuf[15, 15], [255, 0, 0], atol=3.0)


def test_default_material_filter_is_nearest():
    m = Material(base_color=(1, 2, 3))
    assert m.filter_mode == "nearest"


if __name__ == "__main__":
    test_bilinear_integer_texel_matches_nearest()
    test_bilinear_center_is_average_of_four()
    test_bilinear_interpolates_two_texels_horizontally()
    test_fog_blends_distant_fragment_toward_fog_color()
    test_fog_leaves_near_fragment_untouched()
    test_default_material_filter_is_nearest()
    print("Part B extension tests OK")
