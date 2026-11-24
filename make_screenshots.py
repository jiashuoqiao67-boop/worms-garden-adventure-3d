"""
Offscreen / windowed screenshot generator (handbook 16.3, 14.8 evidence).

Every image is produced by the real renderer, captioned with its internal
framebuffer size (and the scene facts that matter), and written to
``screenshots/``:

  01  Appendix E wireframe cube (Stage 1 projection)
  02  perspective overview, full HUD
  03  orthographic overview, comparable camera
  04  Z-buffer occlusion test (near cube hides far cube)
  05  textured + illuminated close view
  06  gameplay chase view (worm, apples, birds, water)
  07  FAST internal resolution (self-labelled in the HUD)
  08  HIGH internal resolution, same camera
  09  fog off          10  fog on (Part B extension 1)
  11  nearest filtering 12  bilinear filtering (Part B extension 2)
  13  texture magnification: nearest / 14: bilinear (2D image processing)
  15  F1 wireframe debug view
  16  F3 depth-buffer debug view
  17  wide window resize test
  18  narrow window resize test

Run from the project root with the project interpreter:
    .venv/Scripts/python.exe make_screenshots.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

from maze import COLS, ROWS, cell_center_3d, START_CELL, FINISH_CELL
from game import build_level
from geometry import (
    ground_mesh, cube_mesh, disc_mesh, worm_parts, apple_parts,
    bird_parts, flag_mesh, maze_wall_meshes,
)
from materials import Material, make_grass_texture, make_stone_texture
from renderer import Renderer
from transforms import model_matrix
from main import (
    make_camera, build_materials, IDENTITY, SKY_BG, SKY_GRAD,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(ROOT, "screenshots")
FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:  # pragma: no cover - fallback for non-Windows machines
        return ImageFont.load_default()


def save_captioned(image, name, caption):
    """Save a PNG with a wrapped caption strip listing the evidence shown."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    img = image.convert("RGB")
    font = _font(18)
    pad = 10
    max_w = img.width - 2 * pad
    draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    # Wrap the caption on the "|" separators, then on spaces if needed.
    chunks = [c.strip() for c in caption.split("   |   ")]
    lines, current = [], ""
    for chunk in chunks:
        trial = (current + "   |   " + chunk) if current else chunk
        if draw_probe.textlength(trial, font=font) <= max_w:
            current = trial
            continue
        if current:
            lines.append(current)
        words, current = chunk.split(" "), ""
        for word in words:
            trial = (current + " " + word) if current else word
            if draw_probe.textlength(trial, font=font) <= max_w:
                current = trial
            else:
                lines.append(current)
                current = word
    if current:
        lines.append(current)
    bar_h = 26 * len(lines) + 10
    canvas = Image.new("RGB", (img.width, img.height + bar_h), (16, 22, 28))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((pad, img.height + 6 + i * 26), line,
                  fill=(225, 235, 230), font=font)
    path = os.path.join(SCREENSHOT_DIR, name)
    canvas.save(path)
    print("saved", name, f"({img.width}x{img.height})")


def _caption_line(*parts):
    return "   |   ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Windowed captures (real HUD, real Pygame path)
# ---------------------------------------------------------------------------

def hud_shot(name, caption, env):
    """Run main.py for a few frames offscreen and capture the whole window."""
    shot = os.path.join(tempfile.gettempdir(), "wg3d_" + name)
    child_env = dict(os.environ)
    child_env.update({
        "SDL_VIDEODRIVER": "dummy",
        "MAIN_SHOT": shot,
        "MAIN_WORM": "11,7",
        "MAIN_VIEW": "2",
        "MAIN_QUALITY": "2",
        "MAIN_PROJECTION": "perspective",
        "MAIN_FOG": "0",
        "MAIN_WIRE": "0",
        "MAIN_DEPTH": "0",
        "MAIN_BILINEAR": "0",
    })
    child_env.update({k: str(v) for k, v in env.items()})
    subprocess.run([sys.executable, "-c",
                    "import main; main.main(max_frames=6)"],
                   cwd=ROOT, env=child_env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with Image.open(shot) as img:
        save_captioned(img.copy(), name, caption)
    os.remove(shot)


# ---------------------------------------------------------------------------
# Offscreen scenes with an explicit camera
# ---------------------------------------------------------------------------

def _materials(filter_mode="nearest"):
    mats = build_materials()
    for key in ("grass", "lawn", "stone"):
        mats[key].filter_mode = filter_mode
    return mats


def render_state(state, camera, width, height, mats, t=1.0, fog=None):
    renderer = Renderer(width, height, fog=fog, background=SKY_BG,
                        sky_grad=SKY_GRAD)
    renderer.clear()
    from main import build_static_scene, draw_static, draw_apples
    from main import draw_birds, draw_worm
    scene = build_static_scene(state)
    draw_static(renderer, camera, scene, mats)
    draw_apples(renderer, camera, state, mats, t)
    draw_birds(renderer, camera, state, mats, t)
    draw_worm(renderer, camera, state, mats, t, 0.0)
    return renderer


def occlusion_scene(width=960, height=600):
    """Near blue cube in front of a far red cube (handbook 16.2 occlusion)."""
    from camera import Camera
    renderer = Renderer(width, height, background=(24, 30, 38))
    cam = Camera(eye=(3.6, 2.6, 4.8), target=(0.0, 0.15, -0.6),
                 projection="perspective", width=width, height=height)
    ground = ground_mesh(12, 12, y=-0.6, uv_per_unit=0.5)
    grass = Material(base_color=(228, 236, 226),
                     texture=make_grass_texture(96))
    blue = Material(base_color=(52, 110, 226))
    red = Material(base_color=(224, 52, 60))
    stone = Material(base_color=(255, 250, 242),
                     texture=make_stone_texture(64))
    renderer.draw_mesh(cam, ground, model_matrix((1, 1, 1), 0.0,
                                                 (-3.0, 0.0, -3.0)),
                       grass)
    # A wall slab behind both cubes shows that the Z-buffer works on walls too.
    renderer.draw_mesh(cam, cube_mesh((5.0, 1.2, 0.16), (0.6, -0.1, -2.6)),
                       IDENTITY, stone)
    renderer.draw_mesh(cam, cube_mesh((1.0, 1.0, 1.0), (-0.9, -0.1, -1.9)),
                       IDENTITY, red)
    renderer.draw_mesh(cam, cube_mesh((1.0, 1.0, 1.0), (0.15, -0.05, 0.0)),
                       IDENTITY, blue)
    return renderer


def wireframe_cube():
    """Appendix E unit cube: project the 12 edges and draw them with Pygame."""
    from geometry import unit_cube
    from camera import Camera
    cube = unit_cube()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
             (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    w, h = 720, 480
    surface = pygame.Surface((w, h))
    surface.fill((18, 24, 32))
    cam = Camera(eye=(2.6, 1.9, 2.9), target=(0, 0, 0),
                 projection="perspective", width=w, height=h)
    projected = [cam.project(v) for v in cube.vertices]
    for i, j in edges:
        if projected[i] is not None and projected[j] is not None:
            pygame.draw.line(surface, (170, 235, 180),
                             projected[i], projected[j], 2)
    img = Image.frombytes("RGB", (w, h),
                          pygame.image.tostring(surface, "RGB"))
    save_captioned(img, "01_wireframe_cube.png",
                   _caption_line("Appendix E wireframe unit cube",
                                 "own projection + pygame.draw.line",
                                 "720x480"))


def texture_zoom_comparison():
    """2D image-processing comparison: magnify the grass texture."""
    grass = Image.fromarray(make_grass_texture(96))
    for name, resample, label in (
            ("13_tex_nearest_zoom.png", Image.NEAREST, "nearest"),
            ("14_tex_bilinear_zoom.png", Image.BILINEAR, "bilinear")):
        img = grass.resize((360, 360), resample)
        save_captioned(img, name,
                       _caption_line("Pillow 4x magnification of the 96x96 "
                                     "procedural grass texture", label,
                                     "360x360"))


def _ground_closeup(filter_mode):
    """Close view of the textured floor + wall for the filtering comparison."""
    from camera import Camera
    width, height = 900, 600
    mats = _materials(filter_mode)
    state = build_level(1, seed=12)
    renderer = Renderer(width, height, background=SKY_BG, sky_grad=SKY_GRAD)
    renderer.clear()
    cam = Camera(eye=(4.6, 1.15, 5.6), target=(5.2, 0.05, 3.2),
                 projection="perspective", width=width, height=height)
    ground = ground_mesh(COLS, ROWS, uv_per_unit=0.5)
    sides, tops = maze_wall_meshes(state["maze"], COLS, ROWS)
    renderer.draw_mesh(cam, sides, IDENTITY, mats["stone"])
    renderer.draw_mesh(cam, tops, IDENTITY, mats["walltop"])
    renderer.draw_mesh(cam, ground, IDENTITY, mats["grass"])
    return renderer


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))

    # --- Appendix E wireframe cube -------------------------------------
    wireframe_cube()

    # --- Perspective vs orthographic from a comparable camera ----------
    state = build_level(1, seed=12)
    state["position"] = cell_center_3d(11, 7, 0.22)
    mats = _materials("nearest")
    for name, projection, label in (
            ("02_perspective.png", "perspective", "perspective"),
            ("03_orthographic.png", "orthographic", "orthographic")):
        w, h = 1280, 800
        cam = make_camera(0, state["position"], projection, w, h)
        r = render_state(state, cam, w, h, mats)
        save_captioned(Image.fromarray(r.surface()), name,
                       _caption_line(
                           f"{label} projection, elevated overview",
                           f"internal {w}x{h}",
                           f"{r.stats['submitted']} triangles submitted",
                           "same world scene / same target"))

    # --- Occlusion ------------------------------------------------------
    r = occlusion_scene()
    save_captioned(Image.fromarray(r.surface()), "04_occlusion.png",
                   _caption_line(
                       "Z-buffer occlusion: near blue cube hides the far red "
                       "cube and the wall slab behind it",
                       "960x600 framebuffer, own software rasterizer",
                       "depth test: nearer fragment wins"))

    # --- Textured / illuminated close view ------------------------------
    from camera import Camera
    w, h = 1280, 800
    close_cam = Camera(eye=(9.3, 2.1, 10.6), target=(11.4, 0.35, 7.4),
                       projection="perspective", width=w, height=h)
    r = render_state(state, close_cam, w, h, mats)
    save_captioned(Image.fromarray(r.surface()), "05_textured_scene.png",
                   _caption_line("Pillow grass + stone textures, ambient + "
                                 "diffuse flat shading",
                                 f"internal {w}x{h}",
                                 "nearest-neighbour sampling (Part A default)"))

    # --- Gameplay views (real window + HUD) -----------------------------
    hud_shot("06_gameplay_follow.png",
             _caption_line("Gameplay chase camera: worm, apples, birds, water "
                           "hazards, HUD and minimap",
                           "internal 1024x640 in a 1280x800 window",
                           "F1/F3 debug views available"),
             {"MAIN_VIEW": 2, "MAIN_QUALITY": 2})

    # --- Internal resolution comparison (same camera, same scene) -------
    for name, quality, label, window in (
            ("07_low_res_672x420.png", 1, "FAST", "1280x800"),
            ("08_high_res_1600x1000.png", 3, "HIGH", "1600x1000")):
        hud_shot(name,
                 _caption_line(f"{label} internal resolution",
                               "same camera / same maze seed",
                               "HUD line bottom-left reports the real "
                               "framebuffer size and render ms"),
                 {"MAIN_VIEW": 0, "MAIN_QUALITY": quality,
                  "MAIN_WINDOW": window})

    # --- Part B extension 1: fog off / on -------------------------------
    hud_shot("09_fog_off.png",
             _caption_line("Part B extension 1: depth fog OFF",
                           "F cycles OFF -> LIGHT -> HEAVY"),
             {"MAIN_VIEW": 0, "MAIN_FOG": 0})
    hud_shot("10_fog_on.png",
             _caption_line("Part B extension 1: depth fog HEAVY",
                           "per-fragment blend toward the sky colour by "
                           "camera depth"),
             {"MAIN_VIEW": 0, "MAIN_FOG": 2})

    # --- Part B extension 2: nearest vs bilinear ------------------------
    for name, mode, label in (
            ("11_filter_nearest.png", "nearest", "nearest-neighbour"),
            ("12_filter_bilinear.png", "bilinear", "bilinear")):
        r = _ground_closeup(mode)
        save_captioned(Image.fromarray(r.surface()), name,
                       _caption_line(
                           f"Part B extension 2: {label} texture sampling",
                           "same close camera, magnified texels",
                           "900x600"))

    texture_zoom_comparison()

    # --- Debug views (F1 / F3) ------------------------------------------
    hud_shot("15_wireframe_view.png",
             _caption_line("F1 debug view: projected triangle edges",
                           "same scene, same camera"),
             {"MAIN_VIEW": 1, "MAIN_WIRE": 1})
    hud_shot("16_depth_view.png",
             _caption_line("F3 debug view: banded depth buffer "
                           "(near = blue, far = orange)",
                           "shows which surfaces are nearest"),
             {"MAIN_VIEW": 1, "MAIN_DEPTH": 1})

    # --- Window resize test ---------------------------------------------
    hud_shot("17_resize_wide.png",
             _caption_line("Resize test: 1280x800 window (16:10)",
                           "internal buffer keeps the window aspect ratio",
                           "maze not stretched"),
             {"MAIN_VIEW": 1, "MAIN_WINDOW": "1280x800"})
    hud_shot("18_resize_narrow.png",
             _caption_line("Resize test: 840x760 window (1.1:1)",
                           "projection and buffer recomputed on resize",
                           "maze still square, HUD readable"),
             {"MAIN_VIEW": 1, "MAIN_WINDOW": "840x760"})

    pygame.quit()
    print("All screenshots written to", SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
