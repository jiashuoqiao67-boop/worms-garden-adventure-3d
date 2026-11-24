"""
Worm's Garden Adventure 3D - main loop (handbook Appendix F2 / ch. 14).

Stage 2 pipeline: update world -> render 3D scene into a NumPy framebuffer
(own software rasterizer + Z-buffer) -> blit to the Pygame window -> draw the
2D HUD, minimap and debug overlays in screen space.

Run:
    .venv/Scripts/python.exe main.py
"""

import math
import os
import time

import numpy as np
import pygame

from maze import (
    COLS, ROWS, WORM_RADIUS, WORM_SPEED,
    WATER_HOLE_RADIUS, START_CELL, FINISH_CELL, cell_center_3d,
)
from game import (
    build_level, update_birds, apple_world_pos, collect_apple,
    check_finish, reset_player, complete_level, restart_run, APPLE_COUNT,
)
from geometry import (
    ground_mesh, lawn_mesh, maze_wall_meshes, disc_mesh, flag_mesh,
    worm_head_mesh, worm_segment_mesh, WORM_SEGMENT_COUNT,
    WORM_SEGMENT_SPACING, apple_parts, bird_parts, bird_wing_meshes,
    merge_meshes,
)
from camera import Camera
from renderer import Renderer
from materials import Material, make_grass_texture, make_stone_texture
from physics import (
    move_xz, JumpState, request_jump, update_jump,
    hits_water, hits_bird, collects_apple,
)
from particles import spawn_spark, spawn_firework, update_particles
from transforms import model_matrix, rotation_z

WINDOW_W, WINDOW_H = 1280, 800

# Internal render-resolution presets (handbook 14.5); target internal height.
QUALITY_PRESETS = {
    1: ("FAST", 420),
    2: ("BALANCED", 640),
    3: ("HIGH", 1000),
    4: ("NATIVE", None),
}

IDENTITY = model_matrix((1, 1, 1), 0.0, (0.0, 0.0, 0.0))

# Part B extension 1: atmospheric depth fog configuration. The range is tied to
# the camera distance so the near corridor stays crisp while the far side of
# the maze fades toward the sky colour.
FOG_COLOR = np.array([184, 214, 228])
MAZE_CENTER = np.array([COLS / 2.0, 0.0, ROWS / 2.0])

# Three-stop clear-sky backdrop (top -> mid -> low), matching the reference
# garden look. SKY_BG keeps the low stop as the plain-colour fallback.
SKY_TOP = (26, 62, 112)
SKY_MID = (92, 160, 212)
SKY_LOW = (184, 214, 228)
SKY_BG = SKY_LOW
SKY_GRAD = (SKY_TOP, SKY_MID, SKY_LOW)

# Track the worm's recent head positions so the body segments can follow an
# older sample of that path (handbook 12.2 - a 3D animation).
TRAIL_SPACING = 0.045
WORM_SCALE = 1.25      # visual scale so the body matches its 0.22 collider
TRAIL_MAX = 80


def fit_buffer(window_w, window_h, target_h):
    if target_h is None:
        return window_w, window_h
    w = round(window_w * target_h / window_h)
    return w, target_h


FOG_MODES = ("OFF", "LIGHT", "HEAVY")
FOG_DENSITY = (1.0, 1.0, 0.72)


def fog_for_camera(camera, density=1.0):
    """Depth-fog range derived from the camera distance (extension 1)."""
    distance = float(np.linalg.norm(camera.eye - MAZE_CENTER))
    far = min(48.0, max(10.0, 1.6 * distance)) * density
    return {"color": FOG_COLOR, "near": 0.45 * far, "far": far}


def _orbit_eye(eye, target, yaw, zoom):
    """Rotate the eye about the world Y axis around the target and dolly."""
    offset = np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)
    if abs(yaw) > 1e-6:
        c, s = math.cos(yaw), math.sin(yaw)
        x, z = offset[0], offset[2]
        offset = np.array([c * x + s * z, offset[1], -s * x + c * z])
    offset = offset * zoom
    eye = np.asarray(target, dtype=float) + offset
    # Never let the camera drop through the garden floor.
    if eye[1] < 0.45:
        eye[1] = 0.45
    return eye


def make_camera(view_mode, worm_pos, projection, width, height,
                orbit=0.0, zoom=1.0):
    """Camera presets (handbook 7.5): 1 overview, 2 side, 3 follow."""
    if view_mode == 1:      # elevated side view from the east edge
        eye = np.array([COLS + 8.0, 6.5, ROWS * 0.75])
        target = np.array([COLS * 0.45, 0.0, ROWS * 0.5])
    elif view_mode == 2:    # third-person chase cam behind the worm
        back, cam_height, look = 2.1, 3.3, 1.1
        eye = np.asarray(worm_pos, dtype=float) + np.array([0.0, cam_height,
                                                            back])
        target = np.asarray(worm_pos, dtype=float) + np.array([0.0, 0.35,
                                                               -look])
    else:                   # elevated overview of the whole maze
        eye = np.array([COLS / 2.0, 12.0, ROWS + 4.0])
        target = np.array([COLS / 2.0, 0.0, ROWS / 2.0])
    eye = _orbit_eye(eye, target, orbit, zoom)
    if projection == "orthographic":
        # Parallel rays start on a plane through the eye: if that plane dips
        # below the floor, the bottom of the orthographic image samples below
        # the ground and shows the sky. Lifting the eye keeps the same aim
        # (target) while making the screen plane steeper than the floor.
        eye = eye.copy()
        eye[1] = max(eye[1] + 6.0, 9.0)
        pixels_per_unit = max(24.0, min(width / (COLS + 4.0),
                                        height / (ROWS + 4.0)))
    else:
        pixels_per_unit = 45.0
    return Camera(eye=eye, target=target, projection=projection,
                  width=width, height=height,
                  pixels_per_unit=pixels_per_unit)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def build_materials(bilinear=True):
    """Materials used by the game.

    Textures are generated at a higher resolution than the first version so
    that the floor and walls stay crisp at native framebuffer resolution, and
    the default sampler is bilinear so magnified texels do not look like
    blocks (T switches to the nearest-neighbour sampler).
    """
    mode = "bilinear" if bilinear else "nearest"
    return {
        "grass": Material(base_color=(236, 242, 234),
                          texture=make_grass_texture(192),
                          filter_mode=mode),
        "lawn": Material(base_color=(228, 236, 226),
                         texture=make_grass_texture(192),
                         filter_mode=mode),
        "stone": Material(base_color=(255, 250, 242),
                          texture=make_stone_texture(128),
                          filter_mode=mode),
        "walltop": Material(base_color=(104, 168, 92)),
        "water": Material(base_color=(30, 140, 198)),
        "water_rim": Material(base_color=(18, 66, 120)),
        "flag": Material(base_color=(46, 226, 122)),
        "red": Material(base_color=(214, 44, 54)),
        "gold": Material(base_color=(250, 186, 34)),
        "bird": Material(base_color=(66, 56, 96)),
        "belly": Material(base_color=(205, 200, 188)),
        "worm": Material(base_color=(150, 236, 118)),
        "stem": Material(base_color=(96, 60, 32)),
        "beak": Material(base_color=(255, 184, 64)),
        "leaf": Material(base_color=(58, 148, 58)),
        "eye": Material(base_color=(12, 12, 14)),
    }


def build_static_scene(state):
    """Merge the immovable level geometry into a few big meshes.

    Ground + surrounding lawn, all wall faces, all wall caps, the water pads
    and the two flags are rebuilt only when a level is generated, so a frame
    needs six draw calls for the whole static world.
    """
    ground = ground_mesh(COLS, ROWS, uv_per_unit=0.5)
    lawn = lawn_mesh(COLS, ROWS, margin=60.0, step=8.0, uv_per_unit=0.5)
    wall_sides, wall_tops = maze_wall_meshes(state["maze"], COLS, ROWS)
    # Water hazards: a dark rim plus a bright blue disc so the trap that
    # resets the worm is unmistakable on the ground (the collision radius is
    # worm radius + hole radius, i.e. exactly the rim).
    pads = merge_meshes([
        disc_mesh(WATER_HOLE_RADIUS + 0.07, (hole[0], 0.012, hole[1]))
        for hole in state["water_holes"]
    ])
    water = merge_meshes([
        disc_mesh(WATER_HOLE_RADIUS, (hole[0], 0.03, hole[1]))
        for hole in state["water_holes"]
    ])
    start_flag = flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0))
    finish_flag = flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0))
    return {
        "ground": ground,
        "lawn": lawn,
        "wall_sides": wall_sides,
        "wall_tops": wall_tops,
        "pads": pads,
        "water": water,
        "start_flag": start_flag,
        "finish_flag": finish_flag,
        "triangles": (len(ground.triangles) + len(lawn.triangles)
                      + len(wall_sides.triangles)
                      + len(wall_tops.triangles) + len(pads.triangles)
                      + len(water.triangles)
                      + len(start_flag.triangles)
                      + len(finish_flag.triangles)),
    }


def draw_static(renderer, cam, scene, mats):
    # Draw the nearest, most occluding surfaces first: later fragments then
    # fail the Z-buffer test before their texture is sampled (early-Z).
    renderer.draw_mesh(cam, scene["wall_sides"], IDENTITY, mats["stone"])
    renderer.draw_mesh(cam, scene["wall_tops"], IDENTITY, mats["walltop"])
    renderer.draw_mesh(cam, scene["ground"], IDENTITY, mats["grass"])
    # Water hazards are drawn after the floor they sit on (rim, then water).
    renderer.draw_mesh(cam, scene["pads"], IDENTITY, mats["water_rim"])
    renderer.draw_mesh(cam, scene["water"], IDENTITY, mats["water"])
    renderer.draw_mesh(cam, scene["start_flag"], IDENTITY, mats["flag"])
    renderer.draw_mesh(cam, scene["finish_flag"], IDENTITY, mats["red"])
    renderer.draw_mesh(cam, scene["lawn"], IDENTITY, mats["lawn"])


def update_trail(state):
    """Push the newest head position and keep only a short history."""
    trail = state.setdefault("trail", [])
    head = np.asarray(state["position"], dtype=float)
    if not trail or np.linalg.norm(head - trail[0]) >= TRAIL_SPACING * 0.5:
        trail.insert(0, head.copy())
        del trail[TRAIL_MAX:]


def reset_trail(state):
    """Drop the recorded path (after a respawn or a new level)."""
    state["trail"] = [np.asarray(state["position"], dtype=float).copy()]


def trail_point(trail, distance, fallback_dir):
    """Point `distance` behind the head along the recorded path."""
    if not trail:
        return np.asarray(fallback_dir, dtype=float)
    acc = 0.0
    prev = trail[0]
    for point in trail[1:]:
        segment = point - prev
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-6:
            continue
        if acc + seg_len >= distance:
            f = (distance - acc) / seg_len
            return prev + segment * f
        acc += seg_len
        prev = point
    # Not enough history yet: extrapolate behind the oldest sample.
    if len(trail) >= 2:
        direction = trail[-2] - trail[-1]
        n = np.linalg.norm(direction)
        if n > 1e-6:
            return trail[-1] + direction / n * max(0.0, distance - acc)
    return trail[-1].copy()


def draw_worm(renderer, cam, state, mats, t, worm_yaw):
    """Head plus body segments trailing along the recorded path (12.2)."""
    trail = state.get("trail") or [np.asarray(state["position"], dtype=float)]
    forward = np.array([math.sin(worm_yaw), 0.0, math.cos(worm_yaw)])
    head_pos = np.asarray(state["position"], dtype=float)
    bob = 0.006 * math.sin(7.0 * t)
    geo = cached_geometry()
    renderer.draw_mesh(cam, geo["worm_head"],
                       model_matrix((WORM_SCALE,) * 3, worm_yaw,
                                    head_pos + np.array([0.0, bob, 0.0])),
                       mats["worm"])
    for i in range(WORM_SEGMENT_COUNT):
        dist = WORM_SEGMENT_SPACING * (i + 1)
        pos = trail_point(trail, dist, head_pos - forward * dist)
        ahead = trail_point(trail, max(0.0, dist - 0.08), pos)
        behind = trail_point(trail, dist + 0.08, pos)
        tangent = ahead - behind
        yaw = math.atan2(tangent[0], tangent[2]) if np.linalg.norm(
            tangent) > 1e-6 else worm_yaw
        pos = pos + np.array([0.0, 0.004 * math.sin(7.0 * t - i), 0.0])
        renderer.draw_mesh(cam, geo["worm_segments"][i],
                           model_matrix((WORM_SCALE,) * 3, yaw, pos),
                           mats["worm"])


_GEOM_CACHE = {}


def cached_geometry():
    """Build the low-poly character meshes once (level-independent, local)."""
    if not _GEOM_CACHE:
        _GEOM_CACHE["apple"] = apple_parts()
        _GEOM_CACHE["bird"] = bird_parts()
        _GEOM_CACHE["wings"] = bird_wing_meshes()
        _GEOM_CACHE["worm_head"] = worm_head_mesh()
        _GEOM_CACHE["worm_segments"] = [worm_segment_mesh(i)
                                        for i in range(WORM_SEGMENT_COUNT)]
    return _GEOM_CACHE


def draw_apples(renderer, cam, state, mats, t):
    parts = cached_geometry()["apple"]
    for apple in state["apples"]:
        if apple["collected"]:
            continue
        fruit_mat = mats["gold"] if apple["value"] == 100 else mats["red"]
        role_mat = {"fruit": fruit_mat, "stem": mats["stem"],
                    "leaf": mats["leaf"]}
        model = model_matrix((1.0, 1.0, 1.0), 0.0,
                             apple_world_pos(apple, t))
        for mesh, role in parts:
            renderer.draw_mesh(cam, mesh, model, role_mat[role])


def draw_birds(renderer, cam, state, mats, t):
    parts = cached_geometry()["bird"]
    wing_left, wing_right = cached_geometry()["wings"]
    for bird in state["birds"]:
        model = model_matrix((1.0, 1.0, 1.0), bird["yaw"], bird["pos"])
        for mesh, role in parts:
            renderer.draw_mesh(cam, mesh, model, mats[role])
        # Wing flap: rotate each wing about the body axis with a sine wave.
        flap = 0.85 * math.sin(9.0 * t + bird.get("phase", 0.0))
        renderer.draw_mesh(cam, wing_left, model @ rotation_z(flap),
                           mats["bird"])
        renderer.draw_mesh(cam, wing_right, model @ rotation_z(-flap),
                           mats["bird"])


def particle_dots(renderer, cam, particles):
    """Project particles and hide the ones behind opaque geometry (13.1)."""
    dots = []
    for p in particles:
        depth = float(cam.to_camera_space(p["pos"])[2])
        if depth <= cam.near:
            continue
        screen = cam.project_cam(cam.to_camera_space(p["pos"]))
        if screen is None:
            continue
        px, py = int(screen[0]), int(screen[1])
        if not (0 <= px < renderer.width and 0 <= py < renderer.height):
            continue
        if depth > renderer.depthbuf[py, px] + 0.02:
            continue        # occluded by a wall / the ground
        size = max(1, int(round(p["size"] * 3.2 / max(0.7, depth))))
        dots.append((p["color"], size, screen))
    return dots


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_minimap(screen, state, window_w, window_h, font, bottom_bar=44):
    """Bottom-right corridor map: walls, apples, birds, worm, start and exit.

    The handbook camera basis maps world +X onto screen left, so the map is
    drawn with the same handedness as the 3D view (the worm dot then moves in
    the same direction as the keys). The wall layer is rasterised once per
    level into a cached surface: 382 wall rectangles per frame cost more than
    the whole 3D HUD.
    """
    mh = min(172, window_h // 3)
    mw = min(250, window_w // 2)
    mx, my = window_w - mw - 22, window_h - mh - bottom_bar - 30

    layer = state.get("_minimap_layer")
    if layer is None or layer.get_size() != (mw, mh):
        layer = pygame.Surface((mw, mh), pygame.SRCALPHA)
        pygame.draw.rect(layer, (8, 16, 24, 170), layer.get_rect(),
                         border_radius=12)
        pygame.draw.rect(layer, (120, 190, 150, 90), layer.get_rect(), 1,
                         border_radius=12)
        ox, oy = 12, 20
        sx = (mw - 24) / COLS
        sy = (mh - 34) / ROWS

        def mp_static(x, z):
            return (ox + (COLS - x) * sx, oy + z * sy)

        for center, size in state["wall_boxes"]:
            x0 = center[0] - size[0] / 2.0
            z0 = center[2] - size[2] / 2.0
            x1 = center[0] + size[0] / 2.0
            z1 = center[2] + size[2] / 2.0
            p0, p1 = mp_static(x1, z0), mp_static(x0, z1)
            pygame.draw.rect(layer, (86, 118, 96),
                             pygame.Rect(p0[0], p0[1],
                                         max(1, p1[0] - p0[0]),
                                         max(1, p1[1] - p0[1])))
        layer.blit(font.render("MAP", True, (170, 220, 190)), (14, 5))
        state["_minimap_layer"] = layer
    screen.blit(layer, (mx, my))

    ox, oy = mx + 12, my + 20
    sx = (mw - 24) / COLS
    sy = (mh - 34) / ROWS

    def mp(x, z):
        return (ox + (COLS - x) * sx, oy + z * sy)

    for apple in state["apples"]:
        if apple["collected"]:
            continue
        p = mp(apple["cell"][0] + 0.5, apple["cell"][1] + 0.5)
        color = (250, 186, 34) if apple["value"] == 100 else (212, 40, 52)
        pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), 3)
    for bird in state["birds"]:
        p = mp(bird["pos"][0], bird["pos"][2])
        pygame.draw.circle(screen, (150, 130, 190),
                           (int(p[0]), int(p[1])), 3)
    pf = mp(FINISH_CELL[0] + 0.5, FINISH_CELL[1] + 0.5)
    pygame.draw.rect(screen, (255, 240, 240),
                     pygame.Rect(int(pf[0] - 3), int(pf[1] - 3), 6, 6))
    ps = mp(START_CELL[0] + 0.5, START_CELL[1] + 0.5)
    pygame.draw.circle(screen, (46, 226, 122), (int(ps[0]), int(ps[1])), 3)
    pp = mp(state["position"][0], state["position"][2])
    pygame.draw.circle(screen, (120, 255, 150), (int(pp[0]), int(pp[1])), 3)
    pygame.draw.circle(screen, (12, 20, 28), (int(pp[0]), int(pp[1])), 4, 1)


def main(max_frames=None):
    pygame.init()
    window_env = os.environ.get("MAIN_WINDOW")
    if window_env:
        win_w, win_h = (int(v) for v in window_env.lower().split("x"))
    else:
        win_w, win_h = WINDOW_W, WINDOW_H
    window_w, window_h = win_w, win_h
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Worm's Garden Adventure 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)
    tiny_font = pygame.font.SysFont("Arial", 13)

    # Quality defaults: render 1:1 with the window (no upscale blur) and use
    # the bilinear sampler, which is what makes texture close-ups look smooth.
    bilinear = os.environ.get("MAIN_BILINEAR", "0") == "1"   # nearest by default
    mats = build_materials(bilinear)
    state = build_level(1, seed=12)
    reset_trail(state)
    scene = build_static_scene(state)
    jump = JumpState()
    particles = []

    view_mode = int(os.environ.get("MAIN_VIEW", "2"))
    projection = os.environ.get("MAIN_PROJECTION", "perspective")
    worm_yaw = 0.0
    quality = int(os.environ.get("MAIN_QUALITY", "4"))
    fog_mode = int(os.environ.get("MAIN_FOG", "0")) % 3
    fog_enabled = fog_mode != 0
    wireframe = os.environ.get("MAIN_WIRE", "0") == "1"
    depth_view = os.environ.get("MAIN_DEPTH", "0") == "1"
    # Screenshot / demo hook: start the worm on a chosen cell.
    start_cell_env = os.environ.get("MAIN_WORM")
    if start_cell_env:
        col, row = (int(v) for v in start_cell_env.split(","))
        state["position"] = cell_center_3d(col, row, jump.y)
        reset_trail(state)
    orbit_yaw = 0.0
    zoom = 1.0
    spark_timer = 0.0
    start_time = time.time()
    render_ms = 0.0
    triangle_count = 0
    renderer = None
    cam = None
    shot_path = os.environ.get("MAIN_SHOT")
    # Capture a little after start-up so the HUD in the screenshot shows the
    # steady-state render time / FPS instead of the first-frame warm-up.
    shot_frame = int(os.environ.get("MAIN_SHOT_FRAME", "36"))
    # Smoothed chase-camera pose (eye, target); None until the follow view
    # first activates. Exponential damping gives the reference soft follow.
    cam_smooth = None

    running = True
    frames = 0
    hud_cache = {}
    splash_msg = None
    splash_until = 0.0
    while running:
        dt = min(clock.tick(120) / 1000.0, 0.1)
        frames += 1

        # ---------------- Events / input ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Keep the internal framebuffer aspect ratio tied to the
                # window; the projection uses the new buffer size.
                window_w, window_h = max(640, event.w), max(400, event.h)
                screen = pygame.display.set_mode((window_w, window_h),
                                                 pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    projection = ("orthographic" if projection == "perspective"
                                  else "perspective")
                elif event.key == pygame.K_1:
                    view_mode = 0
                elif event.key == pygame.K_2:
                    view_mode = 1
                elif event.key == pygame.K_3:
                    view_mode = 2
                elif event.key == pygame.K_F1:
                    wireframe = not wireframe
                elif event.key == pygame.K_F3:
                    depth_view = not depth_view
                elif event.key == pygame.K_F5:
                    quality = quality % 4 + 1
                elif event.key == pygame.K_r:
                    state = restart_run(state)
                    reset_trail(state)
                    scene = build_static_scene(state)
                    jump = JumpState()
                    particles.clear()
                    cam_smooth = None
                    start_time = time.time()
                elif event.key == pygame.K_SPACE:
                    request_jump(jump)
                elif event.key == pygame.K_f:
                    fog_mode = (fog_mode + 1) % 3
                    fog_enabled = fog_mode != 0
                elif event.key == pygame.K_t:
                    bilinear = not bilinear
                    mode = "bilinear" if bilinear else "nearest"
                    mats["grass"].filter_mode = mode
                    mats["lawn"].filter_mode = mode
                    mats["stone"].filter_mode = mode
            elif event.type == pygame.MOUSEWHEEL:
                zoom = min(1.8, max(0.55, zoom * math.exp(-event.y * 0.1)))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 3D click rule (handbook 13.2): cast the pixel ray onto the
                # ground plane and burst there.
                if cam is not None and renderer is not None:
                    mx, my = event.pos
                    gx = mx * renderer.width / max(1, window_w)
                    gy = my * renderer.height / max(1, window_h)
                    hit = cam.ground_hit(gx, gy, ground_y=0.05)
                    if hit is not None:
                        spawn_firework(particles, hit)

        keys = pygame.key.get_pressed()
        # Continuous camera orbit (Q / E) and dolly are applied to every
        # preset; the zoom is clamped so the camera cannot enter the geometry.
        if keys[pygame.K_q]:
            orbit_yaw -= 1.6 * dt
        if keys[pygame.K_e]:
            orbit_yaw += 1.6 * dt
        direction = np.array([0.0, 0.0, 0.0])
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            direction[2] -= 1.0   # N = -Z
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            direction[2] += 1.0   # S = +Z
        # The camera basis puts world +X on the screen left (handbook 7.2),
        # so A/D are mapped to the world direction that matches the screen.
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            direction[0] += 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            direction[0] -= 1.0

        # ---------------- Update world ----------------
        if np.linalg.norm(direction) > 0:
            worm_yaw = math.atan2(direction[0], direction[2])
            move_xz(state["position"], direction, WORM_SPEED, dt,
                    state["wall_boxes"], WORM_RADIUS)
            spark_timer += dt
            while spark_timer >= 0.15:
                spawn_spark(particles, state["position"])
                spark_timer -= 0.15

        update_jump(jump, dt)
        state["position"][1] = jump.y
        update_trail(state)

        # Collisions. A reset also shows a short message in the HUD, so the
        # player learns *why* the worm jumped back to the start.
        if hits_water(state["position"], state["water_holes"],
                      WORM_RADIUS, WATER_HOLE_RADIUS):
            reset_player(state)
            reset_trail(state)
            jump = JumpState()
            splash_msg = "SPLASH! A water hole sent the worm back to the start"
            splash_until = time.time() + 2.5

        for bird in state["birds"]:
            if hits_bird(state["position"], bird["pos"]):
                reset_player(state)
                reset_trail(state)
                jump = JumpState()
                splash_msg = "A bird caught the worm - back to the start"
                splash_until = time.time() + 2.5
                break

        now = time.time()
        for apple in state["apples"]:
            if not apple["collected"] and collects_apple(
                    state["position"], apple_world_pos(apple, now)):
                collect_apple(state, apple)

        update_birds(state, dt)
        update_particles(particles, dt)

        if check_finish(state):
            state = complete_level(state)
            reset_trail(state)
            scene = build_static_scene(state)
            jump = JumpState()
            particles.clear()

        # ---------------- Render 3D scene ----------------
        label, target_h = QUALITY_PRESETS[quality]
        buf_w, buf_h = fit_buffer(window_w, window_h, target_h)
        fog_cfg = None
        if renderer is None or (renderer.width, renderer.height) != (buf_w,
                                                                     buf_h):
            renderer = Renderer(buf_w, buf_h, background=SKY_BG,
                                sky_grad=SKY_GRAD)
        if view_mode == 2:
            # Third-person chase cam with exponential smoothing.
            target_pose = make_camera(2, state["position"], projection,
                                      buf_w, buf_h, orbit_yaw, zoom)
            if cam_smooth is None:
                s_eye, s_target = target_pose.eye.copy(), target_pose.target.copy()
            else:
                k = 1.0 - math.exp(-dt * 6.5)
                s_eye = cam_smooth[0] + (target_pose.eye - cam_smooth[0]) * k
                s_target = (cam_smooth[1]
                            + (target_pose.target - cam_smooth[1]) * k)
            cam_smooth = (s_eye, s_target)
            cam = Camera(eye=s_eye, target=s_target, projection=projection,
                         width=buf_w, height=buf_h)
        else:
            cam_smooth = None
            cam = make_camera(view_mode, state["position"], projection,
                              buf_w, buf_h, orbit_yaw, zoom)

        renderer.fog = (fog_for_camera(cam, FOG_DENSITY[fog_mode])
                        if fog_mode else None)
        renderer.wireframe = wireframe
        renderer.clear()

        t0 = time.perf_counter()
        draw_static(renderer, cam, scene, mats)
        draw_apples(renderer, cam, state, mats, now)
        draw_birds(renderer, cam, state, mats, now)
        draw_worm(renderer, cam, state, mats, now, worm_yaw)
        dots = particle_dots(renderer, cam, particles)
        render_ms = (time.perf_counter() - t0) * 1000.0
        triangle_count = renderer.stats["submitted"]

        # ---------------- Blit framebuffer + HUD ----------------
        if depth_view:
            frame_src = renderer.depth_image()
        else:
            frame_src = renderer.surface()
        # frombuffer + blit is ~4x faster than surfarray.make_surface followed
        # by an unconditional smoothscale; at NATIVE quality the sizes already
        # match, so no scaling pass is needed at all.
        rgb = np.ascontiguousarray(frame_src)
        frame = pygame.image.frombuffer(rgb.tobytes(), (buf_w, buf_h), "RGB")
        if (buf_w, buf_h) != (window_w, window_h):
            frame = pygame.transform.smoothscale(frame, (window_w, window_h))
        screen.blit(frame, (0, 0))

        sx = window_w / buf_w
        sy = window_h / buf_h
        for color, size, sp in dots:
            pygame.draw.circle(screen, color, (int(sp[0] * sx),
                                               int(sp[1] * sy)), size)

        elapsed = time.time() - start_time
        mm, ss = int(elapsed // 60), int(elapsed % 60)
        fps = 1.0 / max(dt, 1e-4)

        # ---- Top information panel (dark translucent + green divider) ----
        # Panels and fixed labels are cached per window size: re-creating
        # alpha surfaces and re-rendering static text every frame cost more
        # than a millisecond at 1280x800.
        hud = hud_cache.get((window_w, window_h))
        if hud is None:
            panel = pygame.Surface((window_w - 40, 62), pygame.SRCALPHA)
            pygame.draw.rect(panel, (10, 20, 30, 176), panel.get_rect(),
                             border_radius=12)
            bar = pygame.Surface((window_w - 40, 3), pygame.SRCALPHA)
            pygame.draw.rect(bar, (120, 220, 150, 120), bar.get_rect())
            bottom = pygame.Surface((window_w, 44), pygame.SRCALPHA)
            pygame.draw.rect(bottom, (10, 26, 18, 180), bottom.get_rect())
            hud = {
                "panel": panel, "bar": bar, "bottom": bottom,
                "line1": tiny_font.render(
                    "WASD MOVE   SPACE JUMP   CLICK FIREWORK   P PROJECTION   "
                    "1/2/3 CAMERA   Q/E ORBIT   WHEEL ZOOM", True,
                    (220, 238, 228)),
                "line2": tiny_font.render(
                    "F FOG   T FILTER   F1 WIREFRAME   F3 DEPTH   F5 QUALITY   "
                    "R RESTART   ESC QUIT", True, (196, 214, 204)),
            }
            hud_cache[(window_w, window_h)] = hud
        screen.blit(hud["panel"], (20, 12))
        screen.blit(hud["bar"], (20, 73))

        def _text(txt, color, pos, fnt=font):
            screen.blit(fnt.render(txt, True, color), pos)

        _text(f"LEVEL {state['level']}", (245, 248, 255), (40, 26))
        _text(f"SCORE {state['score']}", (255, 200, 60),
              (int(window_w * 0.16), 26))
        _text(f"APPLES {state['collected']}/{APPLE_COUNT}",
              (255, 140, 140), (int(window_w * 0.30), 26))
        _text(f"TIME {mm:02d}:{ss:02d}", (245, 248, 255),
              (int(window_w * 0.46), 26))
        fps_col = (150, 240, 170) if dt < 0.03 else (255, 170, 120)
        _text(f"{int(fps)} FPS", fps_col, (int(window_w * 0.62), 29),
              small_font)

        if state["collected"] < APPLE_COUNT:
            hint = f"Collect all {APPLE_COUNT} apples to unlock the checkered exit"
            hint_col = (255, 200, 60)
        else:
            hint = "EXIT UNLOCKED - reach the checkered flag!"
            hint_col = (120, 255, 160)
        ht = small_font.render(hint, True, hint_col)
        screen.blit(ht, (window_w // 2 - ht.get_width() // 2, 82))

        # Hazards explain themselves for a couple of seconds after a reset.
        if time.time() < splash_until and splash_msg:
            sht = font.render(splash_msg, True, (255, 140, 90))
            screen.blit(sht, (window_w // 2 - sht.get_width() // 2, 112))

        # ---- Bottom control bar (two compact lines, resize friendly) ----
        bar_h = 44
        screen.blit(hud["bottom"], (0, window_h - bar_h))
        screen.blit(hud["line1"], (24, window_h - 39))
        screen.blit(hud["line2"], (24, window_h - 21))

        # ---- Renderer debug line (projection, view, buffer, cost, eye) ----
        mode = "WIREFRAME" if wireframe else (
            "DEPTH" if depth_view else "SHADED")
        _text(f"{projection.upper()}  |  VIEW {view_mode + 1}  |  "
              f"{label} {buf_w}x{buf_h}  |  {render_ms:.1f} ms  |  "
              f"{triangle_count} tris  |  "
              f"EYE ({cam.eye[0]:.1f}, {cam.eye[1]:.1f}, {cam.eye[2]:.1f})  |  "
              f"FOG {FOG_MODES[fog_mode]}  |  "
              f"FILTER {'BILINEAR' if bilinear else 'NEAREST'}  |  {mode}",
              (170, 190, 180), (15, window_h - bar_h - 20), tiny_font)

        draw_minimap(screen, state, window_w, window_h, tiny_font,
                     bottom_bar=bar_h)

        pygame.display.flip()

        if shot_path and frames >= shot_frame:
            pygame.image.save(screen, shot_path)
            shot_path = None
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()









