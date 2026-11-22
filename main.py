"""
Worm's Garden Adventure 3D - main loop (handbook Appendix F2 / ch. 14).

Stage 2 pipeline: update world -> render 3D scene into a NumPy framebuffer
-> blit to the Pygame window -> draw 2D HUD in screen space.

Run:
    .venv/Scripts/python.exe main.py
"""

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
    ground_mesh, cube_mesh, disc_mesh, worm_parts, apple_parts,
    bird_parts, flag_mesh,
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
    1: ("FAST", 420),
    2: ("BALANCED", 700),
    3: ("HIGH", 1000),
    4: ("NATIVE", None),
}

IDENTITY = model_matrix((1, 1, 1), 0.0, (0.0, 0.0, 0.0))

# Part B extension 1: atmospheric depth fog configuration.
FOG_CONFIG = {"color": np.array([176, 196, 214]), "near": 8.0, "far": 26.0}


def fit_buffer(window_w, window_h, target_h):
    if target_h is None:
        return window_w, window_h
    w = round(window_w * target_h / window_h)
    return w, target_h


def make_camera(view_mode, worm_pos, projection, width, height):
    if view_mode == 1:      # side
        eye = np.array([COLS + 6.0, 4.0, ROWS / 2.0])
        target = np.array([COLS / 2.0, 0.0, ROWS / 2.0])
    elif view_mode == 2:    # follow: directly overhead, bird's-eye view
        eye = worm_pos + np.array([0.0, 7.5, 0.0])
        target = worm_pos.copy()
        # Up vector chosen so the camera right matches the overview view
        # (screen right = world -X); avoids degeneracy when looking straight
        # down and keeps the A/D mapping consistent.
        return Camera(eye=eye, target=target, projection=projection,
                      width=width, height=height, up=(0.0, 0.0, -1.0))
    else:                   # elevated overview
        eye = np.array([COLS / 2.0, 13.0, ROWS + 6.0])
        target = np.array([COLS / 2.0, 0.0, ROWS / 2.0])
    return Camera(eye=eye, target=target, projection=projection,
                  width=width, height=height)


def build_wall_meshes(state):
    return [cube_mesh(size, center) for center, size in state["wall_boxes"]]


def build_static_meshes(state):
    """Static placements that do not move during a run: water-hole discs and
    the two flags. Built once per level instead of every frame."""
    pads = [disc_mesh(WATER_HOLE_RADIUS, (hole[0], 0.03, hole[1]))
            for hole in state["water_holes"]]
    start_flag = flag_mesh(cell_center_3d(START_CELL[0], START_CELL[1], 0.0))
    finish_flag = flag_mesh(cell_center_3d(FINISH_CELL[0], FINISH_CELL[1], 0.0))
    return pads, start_flag, finish_flag


def main():
    pygame.init()
    window_w, window_h = WINDOW_W, WINDOW_H
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Worm's Garden Adventure 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)

    # Materials
    grass_mat = Material(base_color=(35, 105, 48),
                         texture=make_grass_texture(64))
    stone_mat = Material(base_color=(112, 116, 119),
                         texture=make_stone_texture(64))
    worm_mat = Material(base_color=(75, 195, 65))
    red_mat = Material(base_color=(220, 35, 40))
    gold_mat = Material(base_color=(255, 190, 20))
    bird_mat = Material(base_color=(55, 58, 72))
    belly_mat = Material(base_color=(205, 200, 188))
    water_mat = Material(base_color=(10, 90, 155))
    flag_mat = Material(base_color=(40, 220, 75))
    stem_mat = Material(base_color=(96, 60, 32))
    beak_mat = Material(base_color=(230, 150, 45))
    leaf_mat = Material(base_color=(58, 148, 58))
    eye_mat = Material(base_color=(12, 12, 14))

    state = build_level(1, seed=12)
    wall_meshes = build_wall_meshes(state)
    water_pads, start_flag, finish_flag = build_static_meshes(state)
    ground = ground_mesh(COLS, ROWS)
    jump = JumpState()
    particles = []

    view_mode = 0
    projection = "perspective"
    quality = 1
    fog_enabled = False
    bilinear = False
    spark_timer = 0.0
    start_time = time.time()
    render_ms = 0.0
    triangle_count = 0
    renderer = None

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)

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
            renderer = Renderer(buf_w, buf_h, fog=fog_cfg)
        else:
            renderer.fog = fog_cfg
            renderer.clear()
        cam = make_camera(view_mode, state["position"], projection,
                          buf_w, buf_h)

        t0 = time.perf_counter()
        tri_total = len(ground.triangles) + sum(len(w.triangles) for w in wall_meshes)
        renderer.draw_mesh(cam, ground, IDENTITY, grass_mat)
        for wall in wall_meshes:
            renderer.draw_mesh(cam, wall, IDENTITY, stone_mat)

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
            for mesh, role in apple_parts(apple_world_pos(apple, t)):
                tri_total += len(mesh.triangles)
                renderer.draw_mesh(cam, mesh, IDENTITY, role_mat[role])

        # Birds
        bird_role_mat = {"bird": bird_mat, "belly": belly_mat,
                         "beak": beak_mat, "eye": eye_mat}
        for bird in state["birds"]:
            for mesh, role in bird_parts(bird["pos"]):
                tri_total += len(mesh.triangles)
                renderer.draw_mesh(cam, mesh, IDENTITY, bird_role_mat[role])

        # Worm
        worm_role_mat = {"worm": worm_mat, "eye": eye_mat}
        for mesh, role in worm_parts(state["position"]):
            tri_total += len(mesh.triangles)
            renderer.draw_mesh(cam, mesh, IDENTITY, worm_role_mat[role])

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
        screen.blit(font.render(f"LEVEL: {state['level']}", True, (255, 255, 255)), (15, 12))
        screen.blit(font.render(f"SCORE: {state['score']}", True, (255, 215, 60)), (160, 12))
        screen.blit(font.render(f"APPLES: {state['collected']}/{APPLE_COUNT}", True, (255, 130, 130)), (330, 12))
        screen.blit(font.render(f"TIME: {mm:02d}:{ss:02d}", True, (255, 255, 255)), (520, 12))
        screen.blit(small_font.render(
            f"{projection.upper()}  |  VIEW {view_mode + 1}  |  {label} {buf_w}x{buf_h}  |  "
            f"{render_ms:.1f} ms  |  {triangle_count} tris  |  "
            f"FOG {'ON' if fog_enabled else 'OFF'}  |  FILTER {'BILINEAR' if bilinear else 'NEAREST'}",
            True, (200, 200, 200)), (15, 44))
        screen.blit(small_font.render(
            "WASD move  SPACE jump  P projection  1/2/3 view  F5 quality  F fog  T filter  R restart  click firework",
            True, (180, 180, 180)), (15, window_h - 26))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
