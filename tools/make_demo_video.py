"""
Offscreen demo video generator (handbook 17.3 demonstration script).

Renders a scripted sequence of camera/state keyframes with our software renderer
and writes an MP4 to ``report/demo_video.mp4``. Because the renderer is pure
Python/NumPy, frames are rendered offscreen at the FAST internal resolution;
the sequence covers: elevated overview, both projections, side and follow
views, a scripted move, a jump, an apple, a bird/particle burst, and both
Part B extensions.

Run from the project root:
    .venv/Scripts/python.exe tools/make_demo_video.py
"""

import os
import sys

import numpy as np
import pygame
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze import COLS, ROWS, cell_center_3d, START_CELL, FINISH_CELL
from game import build_level, apple_world_pos
from geometry import (
    ground_mesh, cube_mesh, disc_mesh, worm_parts, apple_parts,
    bird_parts, flag_mesh,
)
from materials import Material, make_grass_texture, make_stone_texture
from renderer import Renderer
from transforms import model_matrix
from particles import spawn_firework, update_particles
from main import make_camera, build_wall_meshes, IDENTITY, FOG_CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "demo_video.mp4")
# 672x432: both dimensions divisible by the h264 macro block (16) to avoid
# encoder resizing and aspect distortion.
W, H = 672, 432
# 540 frames at 4.5 fps gives a 120 s (2 minute) demonstration clip.
FPS = 4.5


def materials(filter_mode="nearest"):
    return {
        "grass": Material(base_color=(35, 105, 48),
                          texture=make_grass_texture(64), filter_mode=filter_mode),
        "stone": Material(base_color=(112, 116, 119),
                          texture=make_stone_texture(64), filter_mode=filter_mode),
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


def render_frame(state, worm_pos, view_mode, projection, fog, filter_mode, t):
    mats = materials(filter_mode)
    renderer = Renderer(W, H, fog=FOG_CONFIG if fog else None)
    cam = make_camera(view_mode, worm_pos, projection, W, H)
    wall_meshes = build_wall_meshes(state)
    ground = ground_mesh(COLS, ROWS)

    renderer.draw_mesh(cam, ground, IDENTITY, mats["grass"])
    for wall in wall_meshes:
        renderer.draw_mesh(cam, wall, IDENTITY, mats["stone"])
    for hole in state["water_holes"]:
        pad = disc_mesh(0.26, (hole[0], 0.03, hole[1]))
        renderer.draw_mesh(cam, pad, IDENTITY, mats["water"])
    renderer.draw_mesh(cam, flag_mesh(
        cell_center_3d(START_CELL[0], START_CELL[1], 0.0)),
        IDENTITY, mats["flag"])
    renderer.draw_mesh(cam, flag_mesh(
        cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0)),
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
    for mesh, role in worm_parts(worm_pos):
        renderer.draw_mesh(cam, mesh, IDENTITY, worm_role_mat[role])
    return renderer.surface()


def add_caption(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H - 26, W, H], fill=(0, 0, 0))
    draw.text((8, H - 22), text, fill=(255, 255, 255))
    return np.asarray(img)


def lerp(a, b, k):
    return a + (b - a) * k


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))

    state = build_level(1, seed=12)
    start = cell_center_3d(START_CELL[0], START_CELL[1], 0.22)
    # A waypoint down the first corridor for the follow/move segment.
    waypoint = cell_center_3d(3, START_CELL[1], 0.22)

    # Script: (frames, view, projection, fog, filter, worm_from, worm_to, caption)
    script = [
        (48, 0, "perspective", False, "nearest", start, start,
         "Elevated overview - perspective projection"),
        (48, 0, "orthographic", False, "nearest", start, start,
         "Orthographic projection - size independent of depth"),
        (48, 1, "perspective", False, "nearest", start, start,
         "Side camera view"),
        (60, 2, "perspective", False, "nearest", start, waypoint,
         "Follow camera - worm moves along the corridor"),
        (48, 2, "perspective", False, "nearest", waypoint, waypoint,
         "Vertical jump (real world Y, gravity)"),
        (48, 0, "perspective", False, "nearest", waypoint, waypoint,
         "Birds patrol corridors; apples bob above the ground"),
        (48, 0, "perspective", False, "nearest", waypoint, waypoint,
         "Firework particle burst (3D particles)"),
        (48, 0, "perspective", True, "nearest", waypoint, waypoint,
         "Extension 1: atmospheric depth fog ON"),
        (48, 0, "perspective", False, "nearest", waypoint, waypoint,
         "Extension 1: fog OFF again"),
        (48, 0, "perspective", False, "nearest", waypoint, waypoint,
         "Extension 2: nearest-neighbour texture filtering"),
        (48, 0, "perspective", False, "bilinear", waypoint, waypoint,
         "Extension 2: bilinear texture filtering"),
    ]

    frames = []
    particles = []
    frame_index = 0
    for seg in script:
        n, view, proj, fog, filt, wfrom, wto, caption = seg
        for f in range(n):
            k = f / max(1, n - 1)
            worm = lerp(wfrom, wto, k)
            # Jump parabola on the dedicated jump segment.
            if caption.startswith("Vertical jump"):
                worm = worm.copy()
                worm[1] = 0.22 + 0.9 * np.sin(np.pi * k)
            if caption.startswith("Firework") and f == 0:
                spawn_firework(particles, worm)
            update_particles(particles, 1.0 / FPS)

            surface = render_frame(state, worm, view, proj, fog, filt,
                                   t=frame_index * 0.1)
            frame = add_caption(surface, caption)
            frames.append(frame)
            frame_index += 1

    pygame.quit()

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=FPS, quality=8)
    print(f"Demo video written to {OUT} ({len(frames)} frames, {FPS} fps)")


if __name__ == "__main__":
    main()
