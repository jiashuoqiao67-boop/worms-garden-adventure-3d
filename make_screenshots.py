"""
Offscreen screenshot generator (handbook 16.3 / 14.8 evidence).

Renders the same scenes without an interactive window and saves labelled PNGs
into ``screenshots/``:

  - perspective / orthographic from a comparable camera
  - solid geometry with correct occlusion
  - textured, illuminated maze with worm, apples, birds and water
  - low vs high internal resolution
  - Part B before/after: fog off/on, nearest/bilinear filtering

Run from project root:
    .venv/Scripts/python.exe make_screenshots.py
"""

import os

import numpy as np
import pygame
from PIL import Image

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
    make_camera, build_wall_meshes, fit_buffer, IDENTITY, FOG_CONFIG,
    WINDOW_W, WINDOW_H,
)

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "screenshots")


def _save(renderer, name):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    surface = pygame.surfarray.make_surface(
        np.ascontiguousarray(renderer.surface()).transpose(1, 0, 2))
    path = os.path.join(SCREENSHOT_DIR, name)
    pygame.image.save(surface, path)
    print("saved", name, f"({renderer.width}x{renderer.height})")


def _materials(filter_mode="nearest"):
    return {
        "grass": Material(base_color=(35, 105, 48),
                          texture=make_grass_texture(64),
                          filter_mode=filter_mode),
        "stone": Material(base_color=(112, 116, 119),
                          texture=make_stone_texture(64),
                          filter_mode=filter_mode),
        "water": Material(base_color=(10, 90, 155)),
        "flag": Material(base_color=(40, 220, 75)),
        "red": Material(base_color=(220, 35, 40)),
        "gold": Material(base_color=(255, 190, 20)),
        "bird": Material(base_color=(55, 58, 72)),
        "belly": Material(base_color=(205, 200, 188)),
        "worm": Material(base_color=(75, 195, 65)),
        "stem": Material(base_color=(96, 60, 32)),
        "beak": Material(base_color=(230, 150, 45)),
        "leaf": Material(base_color=(58, 148, 58)),
        "eye": Material(base_color=(12, 12, 14)),
    }


def render_state(state, view_mode=0, projection="perspective",
                 width=1120, height=700, fog=None, filter_mode="nearest",
                 t=1.0):
    mats = _materials(filter_mode)
    renderer = Renderer(width, height, fog=fog)
    cam = make_camera(view_mode, state["position"], projection, width, height)
    wall_meshes = build_wall_meshes(state)
    ground = ground_mesh(COLS, ROWS)

    renderer.draw_mesh(cam, ground, IDENTITY, mats["grass"])
    for wall in wall_meshes:
        renderer.draw_mesh(cam, wall, IDENTITY, mats["stone"])

    for hole in state["water_holes"]:
        pad = disc_mesh(0.26, (hole[0], 0.03, hole[1]))
        renderer.draw_mesh(cam, pad, IDENTITY, mats["water"])

    renderer.draw_mesh(
        cam, flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0)),
        IDENTITY, mats["flag"])
    renderer.draw_mesh(
        cam, flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0)),
        IDENTITY, mats["red"])

    for apple in state["apples"]:
        if apple["collected"]:
            continue
        fruit = mats["gold"] if apple["value"] == 100 else mats["red"]
        role_mat = {"fruit": fruit, "stem": mats["stem"], "leaf": mats["leaf"]}
        for mesh, role in apple_parts(apple_world_pos(apple, t)):
            renderer.draw_mesh(cam, mesh, IDENTITY, role_mat[role])

    bird_role_mat = {"bird": mats["bird"], "belly": mats["belly"],
                     "beak": mats["beak"], "eye": mats["eye"]}
    for bird in state["birds"]:
        for mesh, role in bird_parts(bird["pos"]):
            renderer.draw_mesh(cam, mesh, IDENTITY, bird_role_mat[role])

    worm_role_mat = {"worm": mats["worm"], "eye": mats["eye"]}
    for mesh, role in worm_parts(state["position"]):
        renderer.draw_mesh(cam, mesh, IDENTITY, worm_role_mat[role])
    return renderer


def occlusion_scene(width=640, height=480):
    """Blue cube in front of a red cube; red must be hidden (16.2 occlusion)."""
    renderer = Renderer(width, height)
    eye = np.array([0.0, 0.0, 6.0])
    target = np.array([0.0, 0.0, 0.0])
    from camera import Camera
    cam = Camera(eye=eye, target=target, projection="perspective",
                 width=width, height=height)
    blue = Material(base_color=(40, 90, 220))
    red = Material(base_color=(220, 35, 40))
    # far red cube, then near blue cube (near wins the Z-buffer)
    renderer.draw_mesh(cam, cube_mesh((1.4, 1.4, 1.4), (0.0, 0.0, -1.5)),
                       IDENTITY, red)
    renderer.draw_mesh(cam, cube_mesh((1.4, 1.4, 1.4), (0.0, 0.0, 1.0)),
                       IDENTITY, blue)
    return renderer


def wireframe_cube():
    """Appendix E wireframe unit cube (Stage 1: project edges, Pygame lines)."""
    from geometry import unit_cube
    from camera import Camera
    cube = unit_cube()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
             (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    w, h = 640, 480
    surface = pygame.Surface((w, h))
    surface.fill((20, 24, 32))
    cam = Camera(eye=(3.0, 2.0, 3.0), target=(0, 0, 0),
                 projection="perspective", width=w, height=h)
    projected = [cam.project(v) for v in cube.vertices]
    for i, j in edges:
        if projected[i] is not None and projected[j] is not None:
            pygame.draw.line(surface, (155, 210, 155),
                             projected[i], projected[j], 2)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    pygame.image.save(surface, os.path.join(SCREENSHOT_DIR,
                                             "01_wireframe_cube.png"))
    print("saved 01 wireframe cube (640x480)")


def texture_zoom_comparison():
    """2D image-processing comparison: magnify the grass texture with
    nearest neighbour (hard blocks) vs bilinear filtering (smooth)."""
    grass = Image.fromarray(make_grass_texture(64))
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    grass.resize((256, 256), Image.NEAREST).save(
        os.path.join(SCREENSHOT_DIR, "13_tex_nearest_zoom.png"))
    grass.resize((256, 256), Image.BILINEAR).save(
        os.path.join(SCREENSHOT_DIR, "14_tex_bilinear_zoom.png"))
    print("saved 13/14 texture zoom comparison (256x256)")


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  # offscreen

    state = build_level(1, seed=12)

    # Appendix E wireframe unit cube.
    wireframe_cube()

    # 16.3 perspective vs orthographic, comparable camera (overview)
    _save(render_state(state, 0, "perspective"), "02_perspective.png")
    _save(render_state(state, 0, "orthographic"), "03_orthographic.png")

    # Occlusion: near blue covers far red.
    _save(occlusion_scene(), "04_occlusion.png")

    # Textured illuminated maze + gameplay objects.
    _save(render_state(state, 0, "perspective"), "05_textured_scene.png")
    _save(render_state(state, 2, "perspective"), "06_gameplay_follow.png")

    # 14.8 low vs high internal resolution, fixed camera/scene.
    _save(render_state(state, 0, "perspective", width=672, height=420),
          "07_low_res_672x420.png")
    _save(render_state(state, 0, "perspective", width=1600, height=1000),
          "08_high_res_1600x1000.png")

    # Part B extension 1: fog off vs on (same overview camera).
    _save(render_state(state, 0, "perspective", fog=None),
          "09_fog_off.png")
    _save(render_state(state, 0, "perspective", fog=FOG_CONFIG),
          "10_fog_on.png")

    # Part B extension 2: nearest vs bilinear on a textured ground patch.
    near = build_level(1, seed=12)
    _save(render_state(near, 0, "perspective", filter_mode="nearest"),
          "11_filter_nearest.png")
    _save(render_state(near, 0, "perspective", filter_mode="bilinear"),
          "12_filter_bilinear.png")

    # Pure 2D magnification comparison (handbook Experiment C).
    texture_zoom_comparison()

    pygame.quit()
    print("All screenshots written to", SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
