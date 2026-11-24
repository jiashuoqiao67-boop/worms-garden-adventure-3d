"""
3D particle system (handbook ch. 13).

Every particle stores a 3-component position and velocity in world space.
Gravity and damping are optional; particles have a bounded lifetime and count.
"""

import random

import numpy as np

MAX_PARTICLES = 250

FIREWORK_COLORS = [
    (40, 140, 255),
    (255, 80, 150),
    (170, 70, 255),
]


def _add(particles, particle):
    particles.append(particle)
    if len(particles) > MAX_PARTICLES:
        del particles[:len(particles) - MAX_PARTICLES]


def spawn_spark(particles, pos, rng=None):
    """A small orange spark emitted while moving."""
    rng = rng or random.Random()
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(0.6, 3.0)  # world units / s (px/s / 50)
    velocity = np.array([
        np.cos(angle) * speed,
        rng.uniform(0.2, 1.0),
        np.sin(angle) * speed,
    ])
    life = rng.uniform(0.4, 1.0)
    _add(particles, {
        "pos": np.asarray(pos, dtype=float).copy(),
        "vel": velocity,
        "life": life,
        "max_life": life,
        "size": rng.randint(2, 4),
        "color": (255, 180, 30),
    })


def spawn_firework(particles, pos, rng=None):
    """A burst of coloured particles in 3D."""
    rng = rng or random.Random()
    for _ in range(rng.randint(70, 110)):
        angle = rng.uniform(0, 2 * np.pi)
        elevation = rng.uniform(-1, 1)
        speed = rng.uniform(2.0, 7.0)  # world units / s
        horizontal = np.sqrt(max(0.0, 1.0 - elevation ** 2))
        velocity = np.array([
            np.cos(angle) * horizontal * speed,
            elevation * speed,
            np.sin(angle) * horizontal * speed,
        ])
        life = rng.uniform(0.8, 1.6)
        _add(particles, {
            "pos": np.asarray(pos, dtype=float).copy(),
            "vel": velocity,
            "life": life,
            "max_life": life,
            "size": rng.randint(2, 5),
            "color": rng.choice(FIREWORK_COLORS),
        })


def update_particles(particles, dt, gravity=4.0, damping=0.97):
    """Integrate motion, apply gravity/damping, drop dead particles."""
    alive = []
    damp = damping ** (dt * 60)
    for particle in particles:
        particle["vel"][1] -= gravity * dt
        particle["pos"] += particle["vel"] * dt
        particle["vel"] *= damp
        particle["life"] -= dt
        if particle["life"] > 0:
            alive.append(particle)
    particles[:] = alive
