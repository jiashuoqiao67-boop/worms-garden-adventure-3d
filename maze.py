"""
Maze topology, neighbours and 2D -> 3D world coordinate conversion.

Coordinate convention (handbook 4.1, 4.3):
    X = east, Y = up (height), Z = south.
    The ground plane is X/Z; Y only carries height above the ground.
    One maze cell = 1 world unit. Original pixel lengths/speeds are / 50.

The maze generation, wall flags and neighbour rules are preserved from the
supplied 2D reference (reference/Start2Dto3D.py). Only the position model
changes: game data stores world positions, not screen pixels.
"""

import random

import numpy as np

# ---------------------------------------------------------------------------
# Original 2D source settings (kept for reference and pixel conversion)
# ---------------------------------------------------------------------------

CELL_PX = 50          # original cell size in pixels -> 1 world unit
ORIGIN_X = 40         # original maze origin in screen pixels
ORIGIN_Y = 95

COLS = 23
ROWS = 15
START_CELL = (0, ROWS // 2)
FINISH_CELL = (COLS - 1, ROWS // 2)

# ---------------------------------------------------------------------------
# World-unit physical constants (pixel value / CELL_PX)
# ---------------------------------------------------------------------------

WALL_THICKNESS_WORLD = 8 / CELL_PX     # 0.16
WORM_RADIUS = 11 / CELL_PX            # 0.22
WORM_SPEED = 230 / CELL_PX            # 4.6 world units / s
BIRD_RADIUS = 10 / CELL_PX            # 0.20
BIRD_SPEED = 75 / CELL_PX             # 1.5 world units / s
WATER_HOLE_RADIUS = 13 / CELL_PX      # 0.26

# Direction: name -> (col delta, row delta, opposite direction name)
DIRECTIONS = {
    "N": (0, -1, "S"),
    "E": (1, 0, "W"),
    "S": (0, 1, "N"),
    "W": (-1, 0, "E"),
}


# ---------------------------------------------------------------------------
# Maze generation (depth-first recursive backtracking, preserved from 2D)
# ---------------------------------------------------------------------------

def generate_maze(cols=COLS, rows=ROWS, start_cell=START_CELL,
                  finish_cell=FINISH_CELL, seed=None):
    """Return a maze as maze[row][col] = {"walls": {...}, "visited": bool}.

    Wall flags N/E/S/W keep their original meaning. The start W wall and the
    finish E wall are opened exactly as in the 2D source.
    """
    rng = random.Random(seed)

    maze = [
        [
            {"walls": {"N": True, "E": True, "S": True, "W": True},
             "visited": False}
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]

    stack = [start_cell]
    sx, sy = start_cell
    maze[sy][sx]["visited"] = True

    while stack:
        x, y = stack[-1]
        neighbours = []

        for direction, (dx, dy, opposite) in DIRECTIONS.items():
            nx = x + dx
            ny = y + dy
            if (0 <= nx < cols and 0 <= ny < rows
                    and not maze[ny][nx]["visited"]):
                neighbours.append((nx, ny, direction, opposite))

        if neighbours:
            nx, ny, direction, opposite = rng.choice(neighbours)
            maze[y][x]["walls"][direction] = False
            maze[ny][nx]["walls"][opposite] = False
            maze[ny][nx]["visited"] = True
            stack.append((nx, ny))
        else:
            stack.pop()

    # Entrance / exit openings (unchanged from the 2D source)
    maze[start_cell[1]][start_cell[0]]["walls"]["W"] = False
    maze[finish_cell[1]][finish_cell[0]]["walls"]["E"] = False

    return maze


def get_available_neighbors(maze, cell, cols=COLS, rows=ROWS):
    """Return reachable neighbour cells along open corridors."""
    x, y = cell
    neighbours = []

    for direction, (dx, dy, _opposite) in DIRECTIONS.items():
        if not maze[y][x]["walls"][direction]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                neighbours.append((nx, ny))

    return neighbours


# ---------------------------------------------------------------------------
# Coordinate conversion: grid / old screen pixels -> world space
# ---------------------------------------------------------------------------

def cell_center_3d(col, row, height=0.0):
    """World-space centre of a ground cell.

    X = col + 0.5, Y = height, Z = row + 0.5  (handbook 4.1).
    """
    return np.array([col + 0.5, height, row + 0.5], dtype=float)


def old_screen_to_world(x_old, y_old, height=0.0):
    """Convert a legacy 2D screen pixel position to world coordinates.

    P_world = [(x_old - ORIGIN_X)/CELL_PX, height, (y_old - ORIGIN_Y)/CELL_PX]
    (handbook 4.4). Note the old row becomes world Z, not world Y.
    """
    return np.array([
        (x_old - ORIGIN_X) / CELL_PX,
        height,
        (y_old - ORIGIN_Y) / CELL_PX,
    ], dtype=float)
