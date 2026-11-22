r"""
M11 self-check: 3D particle spawn, lifetime, gravity and cap.

Run from the project root:
    .venv/Scripts/python.exe tests/test_particles.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particles import (
    spawn_spark,
    spawn_firework,
    update_particles,
    MAX_PARTICLES,
    FIREWORK_COLORS,
)


def test_spark_is_3d():
    particles = []
    spawn_spark(particles, np.array([1.0, 0.3, 2.0]))
    p = particles[0]
    assert p["pos"].shape == (3,)
    assert p["vel"].shape == (3,)
    assert p["life"] > 0


def test_firework_colors_and_count():
    particles = []
    spawn_firework(particles, np.array([1.0, 0.5, 1.0]))
    assert 70 <= len(particles) <= 110
    assert all(p["color"] in FIREWORK_COLORS for p in particles)


def test_dead_particles_are_removed():
    particles = []
    spawn_spark(particles, np.array([0.0, 0.3, 0.0]))
    particles[0]["life"] = 0.01
    update_particles(particles, dt=0.1)
    assert len(particles) == 0


def test_gravity_reduces_vertical_velocity():
    particles = []
    spawn_spark(particles, np.array([0.0, 0.3, 0.0]))
    vy_before = particles[0]["vel"][1]
    update_particles(particles, dt=0.05)
    assert particles[0]["vel"][1] < vy_before


def test_particle_cap():
    particles = []
    for _ in range(300):
        spawn_spark(particles, np.array([0.0, 0.3, 0.0]))
    assert len(particles) <= MAX_PARTICLES


if __name__ == "__main__":
    test_spark_is_3d()
    test_firework_colors_and_count()
    test_dead_particles_are_removed()
    test_gravity_reduces_vertical_velocity()
    test_particle_cap()
    print("M11 particle tests OK")
