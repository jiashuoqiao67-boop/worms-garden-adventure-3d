"""
Offline performance benchmark (handbook 14.7).

Renders the same level and camera offscreen at every internal resolution
preset, measures the render time after a warm-up frame (so Numba compilation
is not counted), and writes a table to ``results/benchmark.txt`` plus a
machine-readable ``results/benchmark.json`` used by the technical report.

The scene, the camera, the maze seed and the window size are fixed, so the
only variable between rows is the internal framebuffer resolution.

Run from the project root:
    .venv/Scripts/python.exe benchmark.py
"""

import json
import os
import statistics
import time

import pygame

from maze import COLS, ROWS, cell_center_3d
from game import build_level
from renderer import Renderer
from main import (
    QUALITY_PRESETS, fit_buffer, make_camera, build_materials,
    build_static_scene, draw_static, draw_apples, draw_birds, draw_worm,
    reset_trail, WINDOW_W, WINDOW_H,
)

FRAMES_PER_PRESET = 12


def render_frame(renderer, cam, state, scene, mats, t, worm_yaw=0.0):
    renderer.clear()
    draw_static(renderer, cam, scene, mats)
    draw_apples(renderer, cam, state, mats, t)
    draw_birds(renderer, cam, state, mats, t)
    draw_worm(renderer, cam, state, mats, t, worm_yaw)
    return renderer.stats["submitted"]


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  # offscreen; no real window needed

    mats = build_materials(bilinear=False)   # measure the shipped default
    state = build_level(1, seed=12)
    state["position"] = cell_center_3d(11, 7, state["position"][1])
    reset_trail(state)
    scene = build_static_scene(state)

    results = []
    for preset_id in sorted(QUALITY_PRESETS):
        label, target_h = QUALITY_PRESETS[preset_id]
        buf_w, buf_h = fit_buffer(WINDOW_W, WINDOW_H, target_h)
        cam = make_camera(0, state["position"], "perspective", buf_w, buf_h)
        renderer = Renderer(buf_w, buf_h)

        # One warm-up frame: the first call compiles the Numba kernels.
        triangles = render_frame(renderer, cam, state, scene, mats, 1.0)

        times = []
        for _ in range(FRAMES_PER_PRESET):
            t0 = time.perf_counter()
            render_frame(renderer, cam, state, scene, mats, 1.0)
            times.append((time.perf_counter() - t0) * 1000.0)

        results.append({
            "preset": label,
            "resolution": f"{buf_w}x{buf_h}",
            "pixels": buf_w * buf_h,
            "triangles": triangles,
            "render_ms_mean": round(statistics.mean(times), 2),
            "render_ms_median": round(statistics.median(times), 2),
            "render_ms_min": round(min(times), 2),
            "render_ms_max": round(max(times), 2),
            "frames_per_s": round(1000.0 / statistics.median(times), 1),
        })

    pygame.quit()

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = (f"{'PRESET':<11}{'RESOLUTION':<13}{'PIXELS':>10}{'TRIS':>7}"
              f"{'MEAN ms':>10}{'MEDIAN ms':>11}{'FPS':>8}")
    lines = ["Performance benchmark (offscreen, perspective, overview camera)",
             "same maze seed, camera and scene for every row",
             f"frames/preset = {FRAMES_PER_PRESET} (after 1 warm-up frame)",
             "", header]
    for r in results:
        lines.append(
            f"{r['preset']:<11}{r['resolution']:<13}{r['pixels']:>10}"
            f"{r['triangles']:>7}{r['render_ms_mean']:>10}"
            f"{r['render_ms_median']:>11}{r['frames_per_s']:>8}"
        )
    table = "\n".join(lines)
    with open(os.path.join("results", "benchmark.txt"), "w",
              encoding="utf-8") as f:
        f.write(table + "\n")

    print(table)


if __name__ == "__main__":
    main()

