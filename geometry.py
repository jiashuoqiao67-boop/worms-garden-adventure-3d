"""
3D mesh construction: cube, ground plane and wall boxes (handbook ch. 6).

A Mesh holds world/local vertex positions and triangle index lists. Meshes are
built from low-poly cubes and planes; placement (model matrix) is applied
later by the renderer.
"""

from dataclasses import dataclass

import numpy as np

from maze import WALL_THICKNESS_WORLD

# Wall dimensions in world units (handbook 6.2)
WALL_HEIGHT = 0.75
WALL_THICKNESS = WALL_THICKNESS_WORLD   # 0.16


@dataclass
class Mesh:
    vertices: np.ndarray      # (N, 3) float
    triangles: np.ndarray     # (M, 3) int


# ---------------------------------------------------------------------------
# Unit cube, centred at the origin, side length 1 (handbook 6.1)
# ---------------------------------------------------------------------------

CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
    [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
    [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
], dtype=float)

CUBE_TRIANGLES = np.array([
    (0, 2, 1), (0, 3, 2),
    (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4),
    (1, 2, 6), (1, 6, 5),
    (3, 7, 6), (3, 6, 2),
    (0, 4, 7), (0, 7, 3),
], dtype=int)


def unit_cube() -> Mesh:
    return Mesh(vertices=CUBE_VERTICES.copy(), triangles=CUBE_TRIANGLES.copy())


def cube_mesh(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> Mesh:
    """A cube scaled to `size` and translated to `center`."""
    sx, sy, sz = size
    cx, cy, cz = center
    vertices = CUBE_VERTICES * np.array([sx, sy, sz]) + np.array([cx, cy, cz])
    return Mesh(vertices=vertices, triangles=CUBE_TRIANGLES.copy())


# ---------------------------------------------------------------------------
# Ground plane: a cols x rows grid on Y = 0, upward-facing (+Y) triangles
# ---------------------------------------------------------------------------

def ground_mesh(cols=3, rows=3, y=0.0) -> Mesh:
    vertices = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([float(c), y, float(r)])
    vertices = np.array(vertices, dtype=float)

    triangles = []
    stride = cols + 1
    for r in range(rows):
        for c in range(cols):
            v00 = r * stride + c
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            # Winding chosen so the face normal points up (+Y)
            triangles.append((v00, v01, v11))
            triangles.append((v00, v11, v10))

    return Mesh(vertices=vertices, triangles=np.array(triangles, dtype=int))


# ---------------------------------------------------------------------------
# Wall box: centre and size for one maze wall (handbook 6.2)
# N/S walls run along X; E/W walls run along Z.
# ---------------------------------------------------------------------------

def wall_box(col, row, direction):
    """Return (center_xyz, size_xyz) for the wall on cell (col, row)."""
    h = WALL_HEIGHT / 2.0
    t = WALL_THICKNESS

    if direction == "N":
        center = (col + 0.5, h, row)
        size = (1.0, WALL_HEIGHT, t)
    elif direction == "S":
        center = (col + 0.5, h, row + 1)
        size = (1.0, WALL_HEIGHT, t)
    elif direction == "E":
        center = (col + 1, h, row + 0.5)
        size = (t, WALL_HEIGHT, 1.0)
    elif direction == "W":
        center = (col, h, row + 0.5)
        size = (t, WALL_HEIGHT, 1.0)
    else:
        raise ValueError(f"Unknown wall direction: {direction}")

    return np.array(center, dtype=float), np.array(size, dtype=float)


def maze_wall_boxes(maze, cols, rows):
    """Convert maze wall flags into 3D wall boxes (handbook Appendix F1).

    Duplicate walls are avoided exactly as in the 2D draw strategy: emit N/W
    walls for every cell; emit E walls only on the last column and S walls
    only on the last row. Entrance/exit openings keep their cleared flags.
    """
    boxes = []
    for row in range(rows):
        for col in range(cols):
            walls = maze[row][col]["walls"]
            if walls["N"]:
                boxes.append(wall_box(col, row, "N"))
            if walls["W"]:
                boxes.append(wall_box(col, row, "W"))
            if col == cols - 1 and walls["E"]:
                boxes.append(wall_box(col, row, "E"))
            if row == rows - 1 and walls["S"]:
                boxes.append(wall_box(col, row, "S"))
    return boxes
