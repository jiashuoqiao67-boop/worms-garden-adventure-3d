r"""
M11 resize / aspect-ratio check (handbook 14.8, 16.2).

The internal framebuffer must keep the window aspect ratio after a resize, so
the maze is neither squashed nor stretched.

Run from the project root:
    .venv/Scripts/python.exe tests/test_resize.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import fit_buffer


def test_buffer_matches_window_aspect_ratio():
    # Wide window, balanced, narrow portrait window.
    for w, h in [(1280, 800), (1000, 1000), (800, 1200)]:
        bw, bh = fit_buffer(w, h, 700)
        assert abs(bw / bh - w / h) < 0.02, (w, h, bw, bh)


def test_native_preset_keeps_window_size():
    bw, bh = fit_buffer(1280, 800, None)
    assert (bw, bh) == (1280, 800)


if __name__ == "__main__":
    test_buffer_matches_window_aspect_ratio()
    test_native_preset_keeps_window_size()
    print("resize / aspect-ratio tests OK")
