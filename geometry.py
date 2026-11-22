"""
3D mesh construction: cube, ground plane and wall boxes (handbook ch. 6).

A Mesh holds world/local vertex positions and triangle index lists. Meshes are
built from low-poly cubes and planes; placement (model matrix) is applied
later by the renderer.
"""

import math
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
    uvs: np.ndarray = None    # (N, 2) float, optional


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


# Cube triangle indices: the last two in the top (+Y) face, the rest form
# the four side walls plus the bottom face. Used to give wall tops a separate
# material (green cap) from the grey side faces.
WALL_SIDE_TRIS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11], dtype=int)
WALL_TOP_TRIS = np.array([8, 9], dtype=int)


def unit_cube() -> Mesh:
    return Mesh(vertices=CUBE_VERTICES.copy(), triangles=CUBE_TRIANGLES.copy())


def wall_box_meshes(center, size):
    """Split a wall box into a side mesh and a separate top-cap mesh so the
    top can carry its own material. Returns (side_mesh, top_mesh)."""
    sx, sy, sz = size
    verts = CUBE_VERTICES * np.array([sx, sy, sz]) + np.asarray(center, dtype=float)
    side = Mesh(vertices=verts, triangles=CUBE_TRIANGLES[WALL_SIDE_TRIS])
    top = Mesh(vertices=verts, triangles=CUBE_TRIANGLES[WALL_TOP_TRIS])
    return side, top


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
    uvs = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([float(c), y, float(r)])
            uvs.append([c / cols, r / rows])
    vertices = np.array(vertices, dtype=float)
    uvs = np.array(uvs, dtype=float)

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

    return Mesh(vertices=vertices, triangles=np.array(triangles, dtype=int),
                uvs=uvs)


# ---------------------------------------------------------------------------
# Wall box: centre and size for one maze wall (handbook 6.2)
# N/S walls run along X; E/W walls run along Z.
# ---------------------------------------------------------------------------

def wall_box(col, row, direction):
    """Return (center_xyz, size_xyz) for the wall on cell (col, row)."""
    h = WALL_HEIGHT / 2.0
    t = WALL_THICKNESS

    # Walls extend half a thickness past each cell edge, matching the 2D
    # reference rectangles (CELL + thickness), so corner seams stay sealed.
    span = 1.0 + t
    if direction == "N":
        center = (col + 0.5, h, row)
        size = (span, WALL_HEIGHT, t)
    elif direction == "S":
        center = (col + 0.5, h, row + 1)
        size = (span, WALL_HEIGHT, t)
    elif direction == "E":
        center = (col + 1, h, row + 0.5)
        size = (t, WALL_HEIGHT, span)
    elif direction == "W":
        center = (col, h, row + 0.5)
        size = (t, WALL_HEIGHT, span)
    else:
        raise ValueError(f"Unknown wall direction: {direction}")

    return np.array(center, dtype=float), np.array(size, dtype=float)


# ---------------------------------------------------------------------------
# Parametric round solids: sphere / ellipsoid, ground disc, mesh merge
# ---------------------------------------------------------------------------

def _merge_meshes(meshes):
    """Combine several Meshes into one, offsetting triangle indices."""
    verts, tris, uvs = [], [], []
    v_off = 0
    for m in meshes:
        verts.append(m.vertices)
        tris.append(m.triangles + v_off)
        if m.uvs is not None:
            uvs.append(m.uvs)
        v_off += len(m.vertices)
    vertices = np.vstack(verts)
    triangles = np.vstack(tris)
    uv = np.vstack(uvs) if uvs else None
    return Mesh(vertices=vertices, triangles=triangles, uvs=uv)


def sphere_mesh(radius=1.0, center=(0.0, 0.0, 0.0), rings=5, sectors=8,
                scale=(1.0, 1.0, 1.0)) -> Mesh:
    """Lat-long sphere (ellipsoid when scale != 1), centred at `center`."""
    cx, cy, cz = center
    sx, sy, sz = scale
    vertices, uvs = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings          # 0 = north pole (+Y)
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(sectors + 1):
            theta = 2.0 * math.pi * j / sectors
            x = radius * sp * math.cos(theta) * sx
            y = radius * cp * sy
            z = radius * sp * math.sin(theta) * sz
            vertices.append([x + cx, y + cy, z + cz])
            uvs.append([j / sectors, i / rings])

    triangles = []
    stride = sectors + 1
    for i in range(rings):
        for j in range(sectors):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            triangles.append((a, d, c))
            triangles.append((a, b, d))

    return Mesh(vertices=np.array(vertices, dtype=float),
                triangles=np.array(triangles, dtype=int),
                uvs=np.array(uvs, dtype=float))


def disc_mesh(radius=0.26, center=(0.0, 0.0, 0.0), segments=12) -> Mesh:
    """Flat upward-facing disc on the ground (water-hole pad)."""
    cx, cy, cz = center
    vertices = [[cx, cy, cz]]
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        vertices.append([cx + radius * math.cos(theta), cy,
                         cz + radius * math.sin(theta)])
    triangles = []
    for j in range(segments):
        a = 1 + j
        b = 1 + (j + 1) % segments
        triangles.append((0, a, b))   # winding -> normal +Y
    return Mesh(vertices=np.array(vertices, dtype=float),
                triangles=np.array(triangles, dtype=int))


def _sph(center, scale, radius=1.0, rings=7, sectors=10) -> Mesh:
    """Convenience ellipsoid at a centre with per-axis scale."""
    return sphere_mesh(radius, center, rings=rings, sectors=sectors,
                       scale=scale)


def _place(mesh, center):
    """Translate a local-origin mesh to a world centre (compat wrapper)."""
    mesh.vertices = mesh.vertices + np.asarray(center, dtype=float)
    return mesh


def worm_parts(center=(0.0, 0.0, 0.0)):
    """Caterpillar: tapered segment chain + round head, split into a body
    part and an eye part. Vertices are local to the origin (head faces +Z);
    the renderer positions/rotates them with a model matrix.
    The `center` argument is kept for signature compatibility and is not
    baked into the local vertices."""
    body = []
    seg_z = [-0.21, -0.12, -0.03, 0.06]
    for i, z in enumerate(seg_z):
        r = 0.105 + 0.02 * (i / 3.0)
        body.append(_sph((0.0, 0.0, z), (r, r, 0.075),
                         rings=4, sectors=6))
    body.append(_sph((0.0, 0.01, 0.21),
                      (0.14, 0.135, 0.125), rings=5, sectors=8))
    # Eyes sit proud of the front/top of the head, not buried inside it.
    eyes = [
        _sph((-0.06, 0.075, 0.305),
             (0.026, 0.026, 0.026), rings=3, sectors=5),
        _sph((0.06, 0.075, 0.305),
             (0.026, 0.026, 0.026), rings=3, sectors=5),
    ]
    return [(_merge_meshes(body), "worm"),
            (_merge_meshes(eyes), "eye")]


def worm_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    parts = worm_parts()
    for m, _ in parts:
        _place(m, center)
    return _merge_meshes([m for m, _ in parts])


def apple_parts(center=(0.0, 0.0, 0.0)):
    """Apple: flattened fruit, short stem and one leaf (local origin)."""
    fruit = [_sph((0.0, 0.0, 0.0), (0.15, 0.135, 0.15), rings=5, sectors=8)]
    stem = [cube_mesh((0.024, 0.10, 0.024), (0.0, 0.17, 0.0))]
    leaf = [_sph((0.06, 0.19, 0.0),
                 (0.06, 0.014, 0.032), rings=3, sectors=5)]
    return [(_merge_meshes(fruit), "fruit"),
            (_merge_meshes(stem), "stem"),
            (_merge_meshes(leaf), "leaf")]


def apple_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    parts = apple_parts()
    for m, _ in parts:
        _place(m, center)
    return _merge_meshes([m for m, _ in parts])


def bird_parts(center=(0.0, 0.0, 0.0)):
    """Bird: body, head, folded wings and tail, plus a beak and eyes
    (local origin, head/beak face +Z)."""
    body = [
        _sph((0.0, 0.0, 0.0), (0.15, 0.11, 0.19), rings=4, sectors=6),
        _sph((0.0, 0.05, 0.17),
             (0.09, 0.085, 0.09), rings=4, sectors=6),
        _sph((-0.14, 0.02, -0.04),
             (0.11, 0.016, 0.10), rings=3, sectors=5),
        _sph((0.14, 0.02, -0.04),
             (0.11, 0.016, 0.10), rings=3, sectors=5),
        _sph((0.0, 0.02, -0.21),
             (0.05, 0.02, 0.085), rings=3, sectors=5),
    ]
    belly = [_sph((0.0, -0.02, 0.05),
                  (0.11, 0.08, 0.13), rings=3, sectors=5)]
    beak = [cube_mesh((0.03, 0.028, 0.08),
                      (0.0, 0.03, 0.30))]
    eyes = [
        _sph((-0.085, 0.06, 0.18),
             (0.02, 0.02, 0.02), rings=3, sectors=5),
        _sph((0.085, 0.06, 0.18),
             (0.02, 0.02, 0.02), rings=3, sectors=5),
    ]
    return [(_merge_meshes(body), "bird"),
            (_merge_meshes(belly), "belly"),
            (_merge_meshes(beak), "beak"),
            (_merge_meshes(eyes), "eye")]


def bird_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    parts = bird_parts()
    for m, _ in parts:
        _place(m, center)
    return _merge_meshes([m for m, _ in parts])


def flag_mesh(base=(0.0, 0.0, 0.0), height=0.8) -> Mesh:
    """Flag: thin pole with a triangular pennant pointing +X."""
    bx, by, bz = base
    pole = cube_mesh((0.03, height, 0.03), (bx, by + height / 2.0, bz))
    # Pennant triangle in the X-Y plane, attached to the pole top.
    top = [bx, by + height, bz]
    mid = [bx, by + height - 0.18, bz]
    tip = [bx + 0.32, by + height - 0.09, bz]
    flag = Mesh(vertices=np.array([top, mid, tip], dtype=float),
                triangles=np.array([(0, 1, 2)], dtype=int))
    return _merge_meshes([pole, flag])


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
