"""
Movement, wall collision, jump physics and height-aware collisions
(handbook ch. 11).

Horizontal navigation stays on the X/Z ground plane; Y carries the vertical
jump. Wall colliders use the same footprint as the 3D wall meshes.
"""

import math
from dataclasses import dataclass

import numpy as np

from maze import WORM_RADIUS, BIRD_RADIUS, WALL_THICKNESS_WORLD
from geometry import WALL_HEIGHT as WALL_HEIGHT_WORLD

GRAVITY = 10.0          # world units / s^2
JUMP_SPEED = 3.2        # world units / s (handbook 11.3)
JUMP_COOLDOWN = 0.65
GROUND_Y = WORM_RADIUS  # sphere centre rests one radius above the ground

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Wall collision: circle vs AABB footprint on X/Z
# ---------------------------------------------------------------------------

def circle_hits_box_xz(px, pz, radius, center, size):
    """Circle (px, pz) against an axis-aligned box footprint on X/Z."""
    cx, _, cz = center
    sx, _, sz = size
    nearest_x = max(cx - sx / 2.0, min(px, cx + sx / 2.0))
    nearest_z = max(cz - sz / 2.0, min(pz, cz + sz / 2.0))
    dx = px - nearest_x
    dz = pz - nearest_z
    return dx * dx + dz * dz < radius * radius


def clears_wall(pos, radius, wall_height=WALL_HEIGHT_WORLD):
    """True when the worm's feet are above the wall top (handbook 11.4).

    The jump peak is about 0.51 world units while the walls are 0.75 high, so
    in practice the worm can never hop over a wall; the rule is implemented
    and tested anyway so the height logic is explicit and not an accident.
    """
    return float(pos[1]) - radius >= wall_height


def hits_any_wall(pos, boxes, radius, wall_height=WALL_HEIGHT_WORLD):
    if clears_wall(pos, radius, wall_height):
        return False
    for center, size in boxes:
        if circle_hits_box_xz(pos[0], pos[2], radius, center, size):
            return True
    return False


def move_xz(pos, direction, speed, dt, boxes, radius):
    """Move one axis at a time, sliding along walls without penetration.

    The whole frame step is subdivided into small sub-steps (as in the
    supplied 2D reference) so a large low-frame-rate step cannot skip past a
    wall face; each sub-step slides independently along the wall, which keeps
    corners passable instead of locking both axes.
    """
    d = np.asarray(direction, dtype=float)
    length = np.linalg.norm(d)
    if length > _EPS:
        d /= length
    movement = d * speed * dt

    max_step = max(1e-3, WALL_THICKNESS_WORLD / 2.0)
    steps = max(1, math.ceil(max(abs(movement[0]), abs(movement[2])) / max_step))
    step_x = movement[0] / steps
    step_z = movement[2] / steps

    for _ in range(steps):
        new = pos.copy()
        new[0] += step_x
        if not hits_any_wall(new, boxes, radius):
            pos[0] = new[0]

        new = pos.copy()
        new[2] += step_z
        if not hits_any_wall(new, boxes, radius):
            pos[2] = new[2]


# ---------------------------------------------------------------------------
# Vertical jump (real gravity in world Y)
# ---------------------------------------------------------------------------

@dataclass
class JumpState:
    y: float = GROUND_Y
    vy: float = 0.0
    cooldown: float = 0.0
    grounded: bool = True


def request_jump(state: JumpState):
    if state.grounded and state.cooldown <= 0.0:
        state.vy = JUMP_SPEED
        state.grounded = False
        state.cooldown = JUMP_COOLDOWN


def update_jump(state: JumpState, dt):
    state.vy -= GRAVITY * dt
    state.y += state.vy * dt
    if state.cooldown > 0.0:
        state.cooldown -= dt
    if state.y <= GROUND_Y:
        state.y = GROUND_Y
        state.vy = 0.0
        state.grounded = True


# ---------------------------------------------------------------------------
# Height-aware collisions
# ---------------------------------------------------------------------------

def hits_water(pos, holes, radius, hole_radius, clearance=0.05):
    """Fall into water only when the worm is near ground level.

    A genuine jump over the hole clears it (handbook 11.4).
    """
    if pos[1] > GROUND_Y + clearance:
        return False
    for hole in holes:
        dx = pos[0] - hole[0]
        dz = pos[2] - hole[1]
        if dx * dx + dz * dz < (radius + hole_radius) ** 2:
            return True
    return False


def hits_bird(pos, bird_pos, worm_radius=WORM_RADIUS, bird_radius=BIRD_RADIUS):
    """3D sphere distance against a bird (declared as 3D ball test)."""
    return np.linalg.norm(pos - np.asarray(bird_pos, dtype=float)) < \
        worm_radius + bird_radius


def collects_apple(pos, apple_pos, worm_radius=WORM_RADIUS, apple_radius=0.18):
    return np.linalg.norm(pos - np.asarray(apple_pos, dtype=float)) < \
        worm_radius + apple_radius
