"""
First runnable 3D scene (handbook Appendix E, Stage 1).

An orbiting camera looks at a wireframe unit cube. Press P to toggle
perspective / orthographic projection. This uses our camera + geometry
modules; solid rasterization and the Z-buffer come later (M5).

Run:
    .venv/Scripts/python.exe first_cube.py
"""

import math

import pygame
import numpy as np

from geometry import unit_cube
from camera import Camera

WIDTH, HEIGHT = 800, 600

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("First cube - wireframe (M3/M4)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    cube = unit_cube()
    angle = 0.0
    running = True
    cam = Camera(eye=(3.0, 2.0, 3.0), target=(0, 0, 0),
                 width=WIDTH, height=HEIGHT)

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        angle += 0.5 * dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                cam.toggle_projection()

        # Orbiting elevated camera looking at the origin
        eye = np.array([3.0 * math.sin(angle), 2.0,
                        3.0 * math.cos(angle)], dtype=float)
        cam.eye = eye

        screen.fill((20, 24, 32))

        projected = [cam.project(v) for v in cube.vertices]

        for i, j in CUBE_EDGES:
            if projected[i] is not None and projected[j] is not None:
                pygame.draw.line(screen, (155, 210, 155),
                                 projected[i], projected[j], 2)

        # Debug overlay in screen space
        mode = "PERSPECTIVE" if cam.projection == "perspective" else "ORTHOGRAPHIC"
        screen.blit(font.render(f"Projection: {mode}  (P to toggle)",
                                True, (230, 230, 230)), (12, 10))
        screen.blit(font.render(f"Eye: ({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f})",
                                True, (180, 180, 180)), (12, 34))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
