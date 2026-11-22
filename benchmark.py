"""
Offline performance benchmark (handbook ch. 14.7).

Renders the same level offscreen at every internal resolution preset, measures
render time and triangle count, and writes a table to ``results/benchmark.txt``
plus machine-readable ``results/benchmark.json`` for the technical report.

Run from project root:
    .venv/Scripts/python.exe benchmark.py
"""

import json
import os
import statistics
import time

import numpy as np
import pygame

from maze import COLS, ROWS, cell_center_3d, START_CELL, FINISH_CELL
from game import build_level, apple_world_pos
from geometry import ground_mesh, cube_mesh
from materials import Material, make_grass_texture, make_stone_texture
from renderer import Renderer
from transforms import model_matrix
from main import (
    QUALITY_PRESETS, fit_buffer, make_camera, build_wall_meshes, IDENTITY,
    WINDOW_W, WINDOW_H,
)

FRAMES_PER_PRESET = 5


def render_once(state, wall_meshes, ground, cam, mats, t):
    renderer = Renderer(cam.width, cam.height)
    grass_mat, stone_mat, water_mat, flag_mat, red_mat, gold_mat, bird_mat, worm_mat = mats

    renderer.draw_mesh(cam, ground, IDENTITY, grass_mat)
    for wall in wall_meshes:
        renderer.draw_mesh(cam, wall, IDENTITY, stone_mat)

    for hole in state["water_holes"]:
        pad = cube_mesh((0.6, 0.06, 0.6), (hole[0], 0.03, hole[1]))
        renderer.draw_mesh(cam, pad, IDENTITY, water_mat)

    start_flag = cube_mesh((0.2, 0.8, 0.2),
                           cell_center_3d(START_CELL[0], START_CELL[1], 0.4))
    finish_flag = cube_mesh((0.2, 0.8, 0.2),
                            cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.4))
    renderer.draw_mesh(cam, start_flag, IDENTITY, flag_mat)
    renderer.draw_mesh(cam, finish_flag, IDENTITY, red_mat)

    for apple in state["apples"]:
        if apple["collected"]:
            continue
        mat = gold_mat if apple["value"] == 100 else red_mat
        mesh = cube_mesh((0.22, 0.22, 0.22), apple_world_pos(apple, t))
        renderer.draw_mesh(cam, mesh, IDENTITY, mat)

    for bird in state["birds"]:
        mesh = cube_mesh((0.3, 0.18, 0.3), bird["pos"])
        renderer.draw_mesh(cam, mesh, IDENTITY, bird_mat)

    worm_mesh = cube_mesh((0.34, 0.34, 0.34), state["position"])
    renderer.draw_mesh(cam, worm_mesh, IDENTITY, worm_mat)

    triangle_count = (
        len(ground.triangles)
        + len(wall_meshes) * 12
        + 4 + len(state["apples"]) + len(state["birds"]) + 1
    )
    return triangle_count


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  # offscreen; no real window needed

    mats = (
        Material(base_color=(35, 105, 48), texture=make_grass_texture(64)),
        Material(base_color=(112, 116, 119), texture=make_stone_texture(64)),
        Material(base_color=(10, 90, 155)),
        Material(base_color=(40, 220, 75)),
        Material(base_color=(220, 35, 40)),
        Material(base_color=(255, 190, 20)),
        Material(base_color=(15, 15, 22)),
        Material(base_color=(75, 195, 65)),
    )

    state = build_level(1, seed=12)
    wall_meshes = build_wall_meshes(state)
    ground = ground_mesh(COLS, ROWS)
    t = 1.0

    results = []
    for preset_id in sorted(QUALITY_PRESETS):
        label, target_h = QUALITY_PRESETS[preset_id]
        buf_w, buf_h = fit_buffer(WINDOW_W, WINDOW_H, target_h)
        cam = make_camera(0, state["position"], "perspective", buf_w, buf_h)

        # warm up
        render_once(state, wall_meshes, ground, cam, mats, t)

        times = []
        tris = 0
        for _ in range(FRAMES_PER_PRESET):
            t0 = time.perf_counter()
            tris = render_once(state, wall_meshes, ground, cam, mats, t)
            times.append((time.perf_counter() - t0) * 1000.0)

        results.append({
            "preset": label,
            "resolution": f"{buf_w}x{buf_h}",
            "triangles": tris,
            "render_ms_mean": round(statistics.mean(times), 2),
            "render_ms_median": round(statistics.median(times), 2),
            "render_ms_min": round(min(times), 2),
            "render_ms_max": round(max(times), 2),
            "frames_per_s": round(1000.0 / statistics.median(times), 1),
        })

    pygame.quit()

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = (f"{'PRESET':<11}{'RESOLUTION':<12}{'TRIS':>7}"
              f"{'MEAN ms':>10}{'MEDIAN ms':>11}{'FPS':>8}")
    lines = ["Performance benchmark (offscreen, perspective, overview view)",
             f"frames/preset = {FRAMES_PER_PRESET}", "", header]
    for r in results:
        lines.append(
            f"{r['preset']:<11}{r['resolution']:<12}{r['triangles']:>7}"
            f"{r['render_ms_mean']:>10}{r['render_ms_median']:>11}{r['frames_per_s']:>8}"
        )
    table = "\n".join(lines)
    with open(os.path.join("results", "benchmark.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(table)


if __name__ == "__main__":
    main()
