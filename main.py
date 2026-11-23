"""
Worm's Garden Adventure 3D - main loop (handbook Appendix F2 / ch. 14).

Stage 2 pipeline: update world -> render 3D scene into a NumPy framebuffer
-> blit to the Pygame window -> draw 2D HUD in screen space.

Run:
    .venv/Scripts/python.exe main.py
"""

import math
import os
import time

import pygame
import numpy as np

from maze import (
    COLS, ROWS, WORM_RADIUS, WORM_SPEED,
    WATER_HOLE_RADIUS, START_CELL, FINISH_CELL, cell_center_3d,
)
from game import (
    build_level, update_birds, apple_world_pos, collect_apple,
    check_finish, reset_player, complete_level, restart_run, APPLE_COUNT,
)
from geometry import (
    ground_mesh, cube_mesh, wall_box_meshes, disc_mesh, worm_parts,
    apple_parts, bird_parts, flag_mesh,
)
from camera import Camera
from renderer import Renderer
from materials import Material, make_grass_texture, make_stone_texture
from physics import (
    move_xz, JumpState, request_jump, update_jump,
    hits_water, hits_bird, collects_apple,
)
from particles import spawn_spark, spawn_firework, update_particles
from transforms import model_matrix

WINDOW_W, WINDOW_H = 1280, 800

# Internal render-resolution presets (handbook 14.5); target internal height.
QUALITY_PRESETS = {
    1: ("FAST", 512),
    2: ("BALANCED", 640),
    3: ("HIGH", 800),
    4: ("NATIVE", None),
}

IDENTITY = model_matrix((1, 1, 1), 0.0, (0.0, 0.0, 0.0))

# Part B extension 1: atmospheric depth fog configuration.
FOG_CONFIG = {"color": np.array([182, 212, 226]), "near": 5.0, "far": 20.0}
# Three-stop clear-sky backdrop (top -> mid -> low), matching the reference
# garden look. SKY_BG keeps the low stop as the plain-colour fallback.
SKY_TOP = (26, 62, 112)
SKY_MID = (92, 160, 212)
SKY_LOW = (184, 214, 228)
SKY_BG = SKY_LOW
SKY_GRAD = (SKY_TOP, SKY_MID, SKY_LOW)


def fit_buffer(window_w, window_h, target_h):
    if target_h is None:
        return window_w, window_h
    w = round(window_w * target_h / window_h)
    return w, target_h


def make_camera(view_mode, worm_pos, projection, width, height, yaw=0.0):
    if view_mode == 1:      # side
        eye = np.array([COLS + 6.0, 4.0, ROWS / 2.0])
        target = np.array([COLS / 2.0, 0.0, ROWS / 2.0])
    elif view_mode == 2:    # third-person chase cam: fixed north-up heading,
        # camera sits behind (+Z) and above the worm, looking ahead (-Z). The
        # maze never spins, matching the reference follow view.
        back, cam_height, look = 3.8, 3.6, 2.0
        eye = worm_pos + np.array([0.0, cam_height, back])
        target = worm_pos + np.array([0.0, 0.3, -look])
        return Camera(eye=eye, target=target, projection=projection,
                      width=width, height=height)
    else:                   # elevated overview
        eye = np.array([COLS / 2.0, 13.0, ROWS + 6.0])
        target = np.array([COLS / 2.0, 0.0, ROWS / 2.0])
    return Camera(eye=eye, target=target, projection=projection,
                  width=width, height=height)


def build_wall_meshes(state):
    """Return [(side_mesh, top_mesh), ...]; side carries the grey wall face,
    top carries the green wall-cap material."""
    return [wall_box_meshes(center, size) for center, size in state["wall_boxes"]]


def build_static_meshes(state):
    """Static placements that do not move during a run: water-hole discs and
    the two flags. Built once per level instead of every frame."""
    pads = [disc_mesh(WATER_HOLE_RADIUS, (hole[0], 0.03, hole[1]))
            for hole in state["water_holes"]]
    start_flag = flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0))
    finish_flag = flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0))
    return pads, start_flag, finish_flag


def draw_minimap(screen, state, window_w, window_h, tiny_font):
    """Bottom-right corridor map: wall blocks, apples, birds, worm and exit."""
    mw, mh = 246, 168
    mx, my = window_w - mw - 22, window_h - mh - 22
    panel = pygame.Surface((mw, mh), pygame.SRCALPHA)
    pygame.draw.rect(panel, (8, 16, 24, 168), panel.get_rect(),
                     border_radius=12)
    pygame.draw.rect(panel, (120, 190, 150, 90), panel.get_rect(), 1,
                     border_radius=12)
    screen.blit(panel, (mx, my))

    ox, oy = mx + 12, my + 18
    sx = (mw - 24) / COLS
    sy = (mh - 30) / ROWS

    def mp(x, z):
        return (ox + x * sx, oy + z * sy)

    # Wall blocks from the cached wall boxes (centre + size -> min/max).
    for center, size in state["wall_boxes"]:
        x0, z0 = center[0] - size[0] / 2.0, center[2] - size[2] / 2.0
        x1, z1 = center[0] + size[0] / 2.0, center[2] + size[2] / 2.0
        p0, p1 = mp(x0, z0), mp(x1, z1)
        pygame.draw.rect(screen, (86, 118, 96),
                         pygame.Rect(p0[0], p0[1],
                                     max(1, p1[0] - p0[0]),
                                     max(1, p1[1] - p0[1])))
    # Apples: red ruby / gold medal.
    for apple in state["apples"]:
        if apple["collected"]:
            continue
        c = apple["cell"]
        p = mp(c[0] + 0.5, c[1] + 0.5)
        col = (250, 186, 34) if apple["value"] == 100 else (212, 40, 52)
        pygame.draw.circle(screen, col, (int(p[0]), int(p[1])), 3)
    # Birds (dark violet).
    for bird in state["birds"]:
        p = mp(bird["pos"][0], bird["pos"][2])
        pygame.draw.circle(screen, (150, 130, 190),
                           (int(p[0]), int(p[1])), 3)
    # Exit marker (checkered finish) and start flag.
    pf = mp(FINISH_CELL[0] + 0.5, FINISH_CELL[1] + 0.5)
    pygame.draw.rect(screen, (255, 240, 240),
                     pygame.Rect(int(pf[0] - 3), int(pf[1] - 3), 6, 6))
    ps = mp(START_CELL[0] + 0.5, START_CELL[1] + 0.5)
    pygame.draw.circle(screen, (46, 226, 122), (int(ps[0]), int(ps[1])), 3)
    # Player worm.
    pp = mp(state["position"][0], state["position"][2])
    pygame.draw.circle(screen, (120, 255, 150), (int(pp[0]), int(pp[1])), 3)
    pygame.draw.circle(screen, (12, 20, 28), (int(pp[0]), int(pp[1])), 4, 1)
    screen.blit(tiny_font.render("MAP", True, (170, 220, 190)),
                (mx + 14, my + 5))


def main(max_frames=None):
    pygame.init()
    window_w, window_h = WINDOW_W, WINDOW_H
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Worm's Garden Adventure 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)
    tiny_font = pygame.font.SysFont("Arial", 13)

    # Materials (palette matched to the garden-ruin reference: mossy green
    # wall tops over warm weathered stone, bright green worm, dark violet bird).
    grass_mat = Material(base_color=(54, 118, 70),
                         texture=make_grass_texture(64))
    stone_mat = Material(base_color=(172, 166, 154),
                         texture=make_stone_texture(64))
    walltop_mat = Material(base_color=(92, 152, 80))
    worm_mat = Material(base_color=(86, 206, 118))
    red_mat = Material(base_color=(212, 40, 52))
    gold_mat = Material(base_color=(250, 186, 34))
    bird_mat = Material(base_color=(58, 50, 84))
    belly_mat = Material(base_color=(205, 200, 188))
    water_mat = Material(base_color=(24, 134, 192))
    flag_mat = Material(base_color=(46, 226, 122))
    stem_mat = Material(base_color=(96, 60, 32))
    beak_mat = Material(base_color=(255, 184, 64))
    leaf_mat = Material(base_color=(58, 148, 58))
    eye_mat = Material(base_color=(12, 12, 14))

    state = build_level(1, seed=12)
    wall_meshes = build_wall_meshes(state)
    water_pads, start_flag, finish_flag = build_static_meshes(state)
    ground = ground_mesh(COLS, ROWS)
    jump = JumpState()
    particles = []

    view_mode = 2
    projection = "perspective"
    worm_yaw = 0.0
    quality = 2
    fog_enabled = False
    bilinear = True
    spark_timer = 0.0
    start_time = time.time()
    render_ms = 0.0
    triangle_count = 0
    renderer = None
    # Smoothed chase-camera pose (eye, target); None until the follow view
    # first activates. Exponential damping gives the reference's soft follow.
    cam_smooth = None

    running = True
    frames = 0
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)
        frames += 1

        # ---------------- Events / input ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                # Keep the internal framebuffer aspect ratio tied to the window;
                # the camera projection uses the new buffer width/height.
                window_w, window_h = event.w, event.h
                screen = pygame.display.set_mode((window_w, window_h),
                                                 pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    projection = ("orthographic"
                                  if projection == "perspective" else "perspective")
                elif event.key == pygame.K_1:
                    view_mode = 0
                elif event.key == pygame.K_2:
                    view_mode = 1
                elif event.key == pygame.K_3:
                    view_mode = 2
                elif event.key == pygame.K_F5:
                    quality = quality % 4 + 1
                elif event.key == pygame.K_r:
                    state = restart_run(state)
                    wall_meshes = build_wall_meshes(state)
                    water_pads, start_flag, finish_flag = build_static_meshes(state)
                    jump = JumpState()
                    particles.clear()
                    start_time = time.time()
                elif event.key == pygame.K_SPACE:
                    request_jump(jump)
                elif event.key == pygame.K_f:
                    fog_enabled = not fog_enabled
                elif event.key == pygame.K_t:
                    bilinear = not bilinear
                    mode = "bilinear" if bilinear else "nearest"
                    grass_mat.filter_mode = mode
                    stone_mat.filter_mode = mode
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                spawn_firework(particles, state["position"])

        keys = pygame.key.get_pressed()
        direction = np.array([0.0, 0.0, 0.0])
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            direction[2] -= 1.0   # N = -Z
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            direction[2] += 1.0   # S = +Z
        # Under the handbook camera basis the overview/follow view puts world
        # -X on the screen right, so swap the X mapping to match screen intuition.
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            direction[0] += 1.0   # screen left
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            direction[0] -= 1.0   # screen right

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

        # Collisions
        if hits_water(state["position"], state["water_holes"],
                      WORM_RADIUS, WATER_HOLE_RADIUS):
            reset_player(state)
            jump = JumpState()

        for bird in state["birds"]:
            if hits_bird(state["position"], bird["pos"]):
                reset_player(state)
                jump = JumpState()
                break

        for apple in state["apples"]:
            if not apple["collected"] and collects_apple(
                    state["position"], apple_world_pos(apple, time.time())):
                collect_apple(state, apple)

        update_birds(state, dt)
        update_particles(particles, dt)

        if check_finish(state):
            state = complete_level(state)
            wall_meshes = build_wall_meshes(state)
            water_pads, start_flag, finish_flag = build_static_meshes(state)
            jump = JumpState()
            particles.clear()

        # ---------------- Render 3D scene ----------------
        label, target_h = QUALITY_PRESETS[quality]
        buf_w, buf_h = fit_buffer(window_w, window_h, target_h)
        fog_cfg = FOG_CONFIG if fog_enabled else None
        if renderer is None or (renderer.width, renderer.height) != (buf_w, buf_h):
            renderer = Renderer(buf_w, buf_h, fog=fog_cfg, background=SKY_BG,
                                sky_grad=SKY_GRAD)
        else:
            renderer.fog = fog_cfg
        # Range cull only for the chase camera; overview/side keep every wall.
        renderer.cull_distance = (view_mode == 2)
        renderer.clear()
        if view_mode == 2:
            # Third-person chase cam with exponential smoothing. The target
            # pose follows the worm behind and above; the stored pose eases
            # toward it each frame (reference-style soft follow).
            wp = state["position"]
            tgt_eye = wp + np.array([0.0, 3.6, 3.8])
            tgt_target = wp + np.array([0.0, 0.3, -2.0])
            if cam_smooth is None:
                s_eye, s_target = tgt_eye.copy(), tgt_target.copy()
            else:
                k = 1.0 - math.exp(-dt * 6.5)
                s_eye = cam_smooth[0] + (tgt_eye - cam_smooth[0]) * k
                s_target = cam_smooth[1] + (tgt_target - cam_smooth[1]) * k
            cam_smooth = (s_eye, s_target)
            cam = Camera(eye=s_eye, target=s_target, projection=projection,
                         width=buf_w, height=buf_h)
        else:
            cam_smooth = None
            cam = make_camera(view_mode, state["position"], projection,
                              buf_w, buf_h, yaw=worm_yaw)

        t0 = time.perf_counter()
        tri_total = len(ground.triangles) + sum(
            len(s.triangles) + len(t.triangles) for s, t in wall_meshes)
        renderer.draw_mesh(cam, ground, IDENTITY, grass_mat)
        for side, top in wall_meshes:
            renderer.draw_mesh(cam, side, IDENTITY, stone_mat)
            renderer.draw_mesh(cam, top, IDENTITY, walltop_mat)

        # Water holes (flat blue discs on the ground) and flags: cached once.
        for pad in water_pads:
            tri_total += len(pad.triangles)
            renderer.draw_mesh(cam, pad, IDENTITY, water_mat)

        tri_total += len(start_flag.triangles) + len(finish_flag.triangles)
        renderer.draw_mesh(cam, start_flag, IDENTITY, flag_mat)
        renderer.draw_mesh(cam, finish_flag, IDENTITY, red_mat)

        # Apples
        t = time.time()
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

        # Birds
        bird_role_mat = {"bird": bird_mat, "belly": belly_mat,
                         "beak": beak_mat, "eye": eye_mat}
        for bird in state["birds"]:
            bird_model = model_matrix((1.0, 1.0, 1.0), bird["yaw"],
                                       bird["pos"])
            for mesh, role in bird_parts():
                tri_total += len(mesh.triangles)
                renderer.draw_mesh(cam, mesh, bird_model, bird_role_mat[role])

        # Worm
        worm_role_mat = {"worm": worm_mat, "eye": eye_mat}
        worm_model = model_matrix((1.0, 1.0, 1.0), worm_yaw,
                                  state["position"])
        for mesh, role in worm_parts():
            tri_total += len(mesh.triangles)
            renderer.draw_mesh(cam, mesh, worm_model, worm_role_mat[role])

        # Project particles; draw them after the framebuffer is blitted.
        particle_dots = []
        for p in particles:
            sp = cam.project(p["pos"])
            if sp is not None:
                particle_dots.append((p["color"], p["size"], sp))
        render_ms = (time.perf_counter() - t0) * 1000.0

        triangle_count = tri_total

        # ---------------- Blit framebuffer + HUD ----------------
        surface = renderer.surface()
        frame = pygame.surfarray.make_surface(
            np.ascontiguousarray(surface).transpose(1, 0, 2))
        frame = pygame.transform.smoothscale(frame, (window_w, window_h))
        screen.blit(frame, (0, 0))

        for color, size, sp in particle_dots:
            pygame.draw.circle(screen, color,
                               (int(sp[0] * window_w / buf_w),
                                int(sp[1] * window_h / buf_h)), size)

        elapsed = time.time() - start_time
        mm, ss = int(elapsed // 60), int(elapsed % 60)
        fps = 1.0 / max(dt, 1e-4)

        # ---- Top information panel (dark translucent + green divider) ----
        panel = pygame.Surface((window_w - 40, 62), pygame.SRCALPHA)
        pygame.draw.rect(panel, (10, 20, 30, 176), panel.get_rect(),
                         border_radius=12)
        screen.blit(panel, (20, 12))
        bar = pygame.Surface((window_w - 40, 3), pygame.SRCALPHA)
        pygame.draw.rect(bar, (120, 220, 150, 120), bar.get_rect())
        screen.blit(bar, (20, 73))

        def _text(txt, color, pos, fnt=font):
            sh = fnt.render(txt, True, (8, 14, 20))
            screen.blit(sh, (pos[0] + 1, pos[1] + 1))
            screen.blit(fnt.render(txt, True, color), pos)

        _text(f"LEVEL {state['level']}", (245, 248, 255), (40, 26))
        _text(f"SCORE {state['score']}", (255, 200, 60), (190, 26))
        _text(f"APPLES {state['collected']}/{APPLE_COUNT}",
              (255, 140, 140), (350, 26))
        _text(f"TIME {mm:02d}:{ss:02d}", (245, 248, 255), (550, 26))
        fps_col = (150, 240, 170) if dt < 0.021 else (255, 170, 120)
        _text(f"{int(fps)} FPS", fps_col, (760, 29), small_font)

        # ---- Central unlock hint ----
        if state["collected"] < APPLE_COUNT:
            hint = f"Collect all {APPLE_COUNT} apples to unlock the checkered exit"
            hint_col = (255, 200, 60)
        else:
            hint = "EXIT UNLOCKED - reach the checkered flag!"
            hint_col = (120, 255, 160)
        ht = small_font.render(hint, True, hint_col)
        screen.blit(small_font.render(hint, True, (8, 14, 20)),
                    (window_w // 2 - ht.get_width() // 2 + 1, 83))
        screen.blit(ht, (window_w // 2 - ht.get_width() // 2, 82))

        # ---- Bottom control bar ----
        bottom = pygame.Surface((window_w, 30), pygame.SRCALPHA)
        pygame.draw.rect(bottom, (10, 26, 18, 170), bottom.get_rect())
        screen.blit(bottom, (0, window_h - 30))
        _text("WASD / ARROWS  MOVE     SPACE  JUMP     CLICK  FIREWORKS     "
              "P  PROJECTION     R  RESTART",
              (220, 238, 228), (40, window_h - 26), small_font)

        # ---- Renderer debug line (small, bottom-left above the bar) ----
        _text(f"{projection.upper()}  |  VIEW {view_mode + 1}  |  "
              f"{label} {buf_w}x{buf_h}  |  {render_ms:.1f} ms  |  "
              f"{triangle_count} tris  |  FOG {'ON' if fog_enabled else 'OFF'}  |  "
              f"FILTER {'BILINEAR' if bilinear else 'NEAREST'}",
              (170, 190, 180), (15, window_h - 48), tiny_font)

        draw_minimap(screen, state, window_w, window_h, tiny_font)

        pygame.display.flip()

        shot = os.environ.get("MAIN_SHOT")
        if shot:
            pygame.image.save(screen, shot)
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
