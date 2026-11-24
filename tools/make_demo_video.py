"""
Offscreen demo video generator (handbook 17.3 demonstration script).

Renders a scripted but genuine demonstration with our own software renderer and
writes an MP4 to ``report/demo_video.mp4``:

  * title card
  * elevated overview, slow orbit, perspective
  * projection toggle to orthographic (same camera) and back
  * side and third-person follow cameras
  * a real walk through the maze along a BFS corridor path, with a jump
  * an apple collected with the real gameplay functions
  * 3D sparks and a 3D firework burst
  * extension 1: fog OFF -> LIGHT -> HEAVY
  * extension 2: nearest vs bilinear texture sampling
  * F1 wireframe and F3 depth debug views
  * end card with the controls

Frames are streamed to the encoder (never all held in memory).

Run from the project root:
    .venv/Scripts/python.exe tools/make_demo_video.py
"""

import collections
import math
import os
import sys

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze import (COLS, ROWS, START_CELL, FINISH_CELL, cell_center_3d,
                  get_available_neighbors)
from game import build_level, apple_world_pos, collect_apple
from materials import Material
from camera import Camera
from particles import spawn_firework, spawn_spark, update_particles
from renderer import Renderer
from transforms import model_matrix
from main import (
    make_camera, build_materials, build_static_scene, draw_static,
    draw_apples, draw_birds, draw_worm, reset_trail, update_trail,
    fog_for_camera, WORM_SCALE, FOG_MODES, SKY_BG, SKY_GRAD,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "demo_video.mp4")
W, H = 1280, 720
FPS = 24
FONT = "C:/Windows/Fonts/arial.ttf"


def _font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold
                                  else FONT, size)
    except Exception:
        return ImageFont.load_default()


def caption(frame, text, sub=None):
    """Draw the section caption and a small permanent title onto a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, H - 54, W, H], fill=(8, 16, 22, 235))
    draw.text((18, H - 44), text, fill=(240, 246, 242), font=_font(22))
    if sub:
        draw.text((18, H - 20), sub, fill=(150, 200, 170), font=_font(15))
    draw.text((18, 14), "Worm's Garden Adventure 3D  -  own NumPy/Pillow/Pygame "
                        "software renderer",
              fill=(255, 255, 255, 210), font=_font(16))
    return np.asarray(img)


def card(lines, note=None):
    """A plain title / end card."""
    img = Image.new("RGB", (W, H), (14, 22, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H // 2 - 2, W, H // 2], fill=(60, 160, 110))
    for i, (text, size, color) in enumerate(lines):
        font = _font(size, bold=size >= 34)
        width = draw.textlength(text, font=font)
        draw.text(((W - width) / 2, H // 2 - 130 + i * 70), text,
                  fill=color, font=font)
    if note:
        font = _font(18)
        width = draw.textlength(note, font=font)
        draw.text(((W - width) / 2, H // 2 + 130), note,
                  fill=(150, 190, 170), font=font)
    return np.asarray(img)


def bfs_path(maze, start, goal, cols=COLS, rows=ROWS):
    """Shortest corridor path between two cells (demo walking route)."""
    queue = collections.deque([start])
    prev = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for nxt in get_available_neighbors(maze, cell, cols, rows):
            if nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    if goal not in prev:
        return [start]
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = prev[cell]
    return list(reversed(path))


class Demo:
    def __init__(self):
        self.state = build_level(1, seed=12)
        self.state["position"] = cell_center_3d(*START_CELL, 0.22)
        reset_trail(self.state)
        self.scene = build_static_scene(self.state)
        self.mats = build_materials()
        self.renderer = Renderer(W, H, background=SKY_BG,
                                 sky_grad=SKY_GRAD)
        self.particles = []
        self.frames = 0

    # -- rendering -----------------------------------------------------
    def render(self, camera, fog_mode=0, wireframe=False):
        r = self.renderer
        r.fog = (fog_for_camera(camera,
                                density=(1.0 if fog_mode == 1 else 0.72))
                 if fog_mode else None)
        r.wireframe = wireframe
        r.clear()
        draw_static(r, camera, self.scene, self.mats)
        draw_apples(r, camera, self.state, self.mats, self.time)
        draw_birds(r, camera, self.state, self.mats, self.time)
        draw_worm(r, camera, self.state, self.mats, self.time,
                  self.worm_yaw)
        return r

    def hud_frame(self, camera, text, sub=None, fog_mode=0,
                  wireframe=False, depth=False, particles=True):
        r = self.render(camera, fog_mode, wireframe)
        src = r.depth_image() if depth else r.surface()
        if particles and not depth:
            sx = W / r.width
            sy = H / r.height
            img = Image.fromarray(src)
            draw = ImageDraw.Draw(img)
            for p in self.particles:
                c = camera.to_camera_space(p["pos"])
                if c[2] <= camera.near:
                    continue
                sp = camera.project_cam(c)
                if sp is None:
                    continue
                px, py = int(sp[0] * sx), int(sp[1] * sy)
                if not (0 <= px < W and 0 <= py < H):
                    continue
                if c[2] > r.depthbuf[int(sp[1]), int(sp[0])] + 0.02:
                    continue
                size = max(1, int(round(p["size"] * 3.2 / max(0.7, c[2]))))
                draw.ellipse((px - size, py - size, px + size, py + size),
                             fill=tuple(int(v) for v in p["color"]))
            src = np.asarray(img)
        return caption(src, text, sub)


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = imageio.get_writer(OUT, fps=FPS, quality=5,
                                macro_block_size=16)
    demo = Demo()
    demo.time = 0.0
    demo.worm_yaw = 0.0
    dt = 1.0 / FPS

    def emit(frame, seconds=1):
        for _ in range(int(round(seconds * FPS))):
            writer.append_data(frame)

    # ---------------------------------------------------------- title
    emit(card([
        ("Worm's Garden Adventure 3D", 42, (235, 245, 240)),
        ("GEIAL31L-B2a - Labs 10 & 11", 24, (170, 210, 190)),
        ("2D-to-3D conversion with a NumPy software renderer", 22,
         (200, 220, 210)),
    ], note="perspective + orthographic projection, Z-buffer, lighting, "
            "Pillow textures, 3D gameplay and two extensions"), 3.0)

    demo.state["position"] = cell_center_3d(*START_CELL, 0.22)
    reset_trail(demo.state)

    # ------------------------------------------- overview, slow orbit
    for i in range(int(14.0 * FPS)):
        demo.time += dt
        angle = 0.55 * math.sin(i / (14.0 * FPS) * math.pi)
        cam = make_camera(0, demo.state["position"], "perspective", W, H,
                          orbit=angle, zoom=1.0)
        frame = demo.hud_frame(
            cam, "Elevated overview, perspective projection",
            "the whole 23x15 maze is 5256 static triangles in six merged meshes")
        writer.append_data(frame)
        demo.frames += 1

    # --------------------------------------- orthographic same camera
    for i in range(int(11.0 * FPS)):
        demo.time += dt
        cam = make_camera(0, demo.state["position"], "orthographic", W, H)
        frame = demo.hud_frame(
            cam, "Orthographic projection (P): parallel edges stay parallel",
            "same world data and target; only the projection changed")
        writer.append_data(frame)

    # ------------------------------------------------- side view orbit
    for i in range(int(9.0 * FPS)):
        demo.time += dt
        angle = -0.45 + 0.9 * (i / (9.0 * FPS))
        cam = make_camera(1, demo.state["position"], "perspective", W, H,
                          orbit=angle, zoom=1.0)
        frame = demo.hud_frame(
            cam, "Side camera preset with Q/E orbit",
            "camera presets 1 overview / 2 side / 3 third-person follow")
        writer.append_data(frame)

    # --------------------------------- walking along a real BFS path
    apple = demo.state["apples"][0]
    path_cells = bfs_path(demo.state["maze"], START_CELL, apple["cell"])
    waypoints = [cell_center_3d(c, r, 0.22) for c, r in path_cells]
    path_length = sum(float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
                      for i in range(len(waypoints) - 1))
    seg = 0
    pos = waypoints[0]
    speed = 4.6  # world units / second, the real worm speed
    walk_seconds = min(30.0, max(8.0, path_length / speed))
    for i in range(int(walk_seconds * FPS)):
        demo.time += dt
        travel = speed * dt
        while travel > 0 and seg < len(waypoints) - 1:
            step = waypoints[seg + 1] - pos
            remaining = float(np.linalg.norm(step))
            if remaining < 1e-6:
                seg += 1
                continue
            if travel < remaining:
                pos = pos + step / remaining * travel
                travel = 0.0
            else:
                travel -= remaining
                pos = waypoints[seg + 1].copy()
                seg += 1
        demo.state["position"] = pos.copy()
        update_trail(demo.state)
        update_particles(demo.particles, dt)
        if seg < len(waypoints) - 1:
            step = waypoints[seg + 1] - waypoints[seg]
            demo.worm_yaw = math.atan2(step[0], step[2])
        cam = make_camera(2, pos, "perspective", W, H)
        frame = demo.hud_frame(
            cam, "Third-person follow camera: the worm walks the corridors",
            "movement is circle-vs-AABB against the same wall boxes that are "
            "rendered")
        writer.append_data(frame)
        if i % 12 == 0:
            spawn_spark(demo.particles, pos)

    # ------------------------------------------------------ the jump
    for i in range(int(8.0 * FPS)):
        demo.time += dt
        k = i / (8.0 * FPS)
        jump_pos = pos.copy()
        jump_pos[1] = 0.22 + 0.512 * math.sin(min(1.0, k / 0.64) * math.pi)
        demo.state["position"] = jump_pos
        update_trail(demo.state)
        cam = make_camera(2, jump_pos, "perspective", W, H)
        frame = demo.hud_frame(
            cam, "Vertical jump: gravity 10 units/s^2, peak rise 0.512 units",
            "world Y is animated, not a screen offset; water is cleared "
            "while airborne")
        writer.append_data(frame)
    demo.state["position"][1] = 0.22

    # ------------------------------------- apple collection + fireworks
    for i in range(int(10.0 * FPS)):
        demo.time += dt
        if i == 6:
            collect_apple(demo.state, apple)
            spawn_firework(demo.particles, apple_world_pos(apple, demo.time))
        update_particles(demo.particles, dt)
        cam = make_camera(2, demo.state["position"], "perspective", W, H)
        frame = demo.hud_frame(
            cam, "Apples are collected by 3D distance (real scoring rules)",
            "a click casts a ray onto the ground plane and bursts a 3D firework")
        writer.append_data(frame)

    # ----------------------------------------- extension 1: fog modes
    cam = make_camera(0, demo.state["position"], "perspective", W, H)
    for mode in (0, 1, 2):
        for i in range(int(7.0 * FPS)):
            demo.time += dt
            update_particles(demo.particles, dt)
            frame = demo.hud_frame(
                cam, f"Extension 1 - depth fog: {FOG_MODES[mode]}",
                "C = C_frag*(1-t) + C_fog*t, t = clamp((z-near)/(far-near)); "
                "F cycles OFF / LIGHT / HEAVY", fog_mode=mode)
            writer.append_data(frame)

    # ------------------------------------------ extension 2: filtering
    close = Camera(eye=(9.3, 2.1, 10.6), target=(11.4, 0.35, 7.4),
                   projection="perspective", width=W, height=H)
    for mode, name in (("nearest", "nearest-neighbour"), ("bilinear", "bilinear")):
        for key in ("grass", "lawn", "stone"):
            demo.mats[key].filter_mode = mode
        for i in range(int(8.0 * FPS)):
            demo.time += dt
            frame = demo.hud_frame(
                close, f"Extension 2 - {name} texture sampling",
                "T toggles the sampler: one texel versus a blend of four "
                "neighbours (perspective-correct UVs)")
            writer.append_data(frame)
    for key in ("grass", "lawn", "stone"):
        demo.mats[key].filter_mode = "nearest"

    # ---------------------------------------------- debug views
    for i in range(int(7.0 * FPS)):
        demo.time += dt
        cam = make_camera(1, demo.state["position"], "perspective", W, H)
        frame = demo.hud_frame(
            cam, "F1 debug view: projected triangle edges",
            "same geometry and camera, edges drawn after the depth-tested fill",
            wireframe=True)
        writer.append_data(frame)
    for i in range(int(7.0 * FPS)):
        demo.time += dt
        cam = make_camera(1, demo.state["position"], "perspective", W, H)
        frame = demo.hud_frame(
            cam, "F3 debug view: banded depth buffer",
            "near surfaces blue, far surfaces orange - the Z-buffer decides "
            "visibility", depth=True)
        writer.append_data(frame)

    # --------------------------------------------------------- end card
    emit(card([
        ("Controls", 34, (235, 245, 240)),
        ("WASD move   SPACE jump   P projection   1/2/3 camera", 20,
         (200, 220, 210)),
        ("Q/E orbit   wheel zoom   F fog   T filter   F5 quality", 20,
         (200, 220, 210)),
        ("F1 wireframe   F3 depth   R restart", 20, (200, 220, 210)),
    ], note="technical report: report/Worms_Garden_Adventure_3D_Report.docx  -  "
            "evidence: screenshots/"), 4.0)

    writer.close()
    pygame.quit()
    print(f"Demo video written to {OUT} ({demo.frames + 1} rendered frames, "
          f"{FPS} fps)")


if __name__ == "__main__":
    main()




