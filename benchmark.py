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
from geometry import (
    ground_mesh, cube_mesh, disc_mesh, worm_parts, apple_parts,
    bird_parts, flag_mesh,
)
from materials import Material, make_grass_texture, make_stone_texture
from renderer import Renderer
from transforms import model_matrix
from main import (
    QUALITY_PRESETS, fit_buffer, make_camera, build_wall_meshes, IDENTITY,
    WINDOW_W, WINDOW_H,
)

FRAMES_PER_PRESET = 5


def render_once(state, wall_meshes, ground, cam, mats, t, static=None):
    renderer = Renderer(cam.width, cam.height)
    grass_mat, stone_mat, water_mat, flag_mat, red_mat, gold_mat, bird_mat, worm_mat = mats
    stem_mat = Material(base_color=(96, 60, 32))
    beak_mat = Material(base_color=(255, 184, 64))
    leaf_mat = Material(base_color=(58, 148, 58))
    eye_mat = Material(base_color=(12, 12, 14))
    belly_mat = Material(base_color=(205, 200, 188))
    walltop_mat = Material(base_color=(92, 152, 80))

    renderer.draw_mesh(cam, ground, IDENTITY, grass_mat)
    for side, top in wall_meshes:
        renderer.draw_mesh(cam, side, IDENTITY, stone_mat)
        renderer.draw_mesh(cam, top, IDENTITY, walltop_mat)

    tri_total = len(ground.triangles) + sum(
        len(s.triangles) + len(t.triangles) for s, t in wall_meshes)

    if static is None:
        pads = [disc_mesh(0.26, (hole[0], 0.03, hole[1]))
                for hole in state["water_holes"]]
        start_flag = flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0))
        finish_flag = flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0))
    else:
        pads, start_flag, finish_flag = static
    for pad in pads:
        tri_total += len(pad.triangles)
        renderer.draw_mesh(cam, pad, IDENTITY, water_mat)

    tri_total += len(start_flag.triangles) + len(finish_flag.triangles)
    renderer.draw_mesh(cam, start_flag, IDENTITY, flag_mat)
    renderer.draw_mesh(cam, finish_flag, IDENTITY, red_mat)

    for apple in state["apples"]:
        if apple["collected"]:
            continue
        fruit_mat = gold_mat if apple["value"] == 100 else red_mat
        role_mat = {"fruit": fruit_mat, "stem": stem_mat, "leaf": leaf_mat}
        apple_model = model_matrix((1.0, 1.0, 1.0), 0.0,
                                   apple_world_pos(apple, t))
        for mesh, role in apple_parts():
            tri_total += len(mesh.triangles)
            renderer.draw_mesh(cam, mesh, apple_model, role_mat[role])

    bird_role_mat = {"bird": bird_mat, "belly": belly_mat,
                     "beak": beak_mat, "eye": eye_mat}
    for bird in state["birds"]:
        bird_model = model_matrix((1.0, 1.0, 1.0), bird.get("yaw", 0.0),
                                  bird["pos"])
        for mesh, role in bird_parts():
            tri_total += len(mesh.triangles)
            renderer.draw_mesh(cam, mesh, bird_model, bird_role_mat[role])

    worm_role_mat = {"worm": worm_mat, "eye": eye_mat}
    worm_model = model_matrix((1.0, 1.0, 1.0), 0.0, state["position"])
    for mesh, role in worm_parts():
        tri_total += len(mesh.triangles)
        renderer.draw_mesh(cam, mesh, worm_model, worm_role_mat[role])

    return tri_total


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  # offscreen; no real window needed

    mats = (
        Material(base_color=(54, 118, 70), texture=make_grass_texture(128)),
        Material(base_color=(172, 166, 154), texture=make_stone_texture(128)),
        Material(base_color=(24, 134, 192)),
        Material(base_color=(46, 226, 122)),
        Material(base_color=(212, 40, 52)),
        Material(base_color=(250, 186, 34)),
        Material(base_color=(58, 50, 84)),
        Material(base_color=(86, 206, 118)),
    )

    state = build_level(1, seed=12)
    wall_meshes = build_wall_meshes(state)
    ground = ground_mesh(COLS, ROWS)
    pads = [disc_mesh(0.26, (hole[0], 0.03, hole[1]))
            for hole in state["water_holes"]]
    start_flag = flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0))
    finish_flag = flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0))
    static = (pads, start_flag, finish_flag)
    t = 1.0

    results = []
    for preset_id in sorted(QUALITY_PRESETS):
        label, target_h = QUALITY_PRESETS[preset_id]
        buf_w, buf_h = fit_buffer(WINDOW_W, WINDOW_H, target_h)
        cam = make_camera(0, state["position"], "perspective", buf_w, buf_h)

        # warm up
        render_once(state, wall_meshes, ground, cam, mats, t, static)

        times = []
        tris = 0
        for _ in range(FRAMES_PER_PRESET):
            t0 = time.perf_counter()
            tris = render_once(state, wall_meshes, ground, cam, mats, t, static)
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
