"""
3D mesh construction: cube, ground plane, wall boxes and low-poly characters
(handbook ch. 6).

A Mesh holds local/world vertex positions, triangle index lists and optional
per-vertex UV coordinates. Static level geometry (ground, walls, water pads and
flags) is merged into a few large meshes once per level so the renderer can
process whole batches instead of hundreds of tiny meshes.
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


def cube_mesh(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> Mesh:
    """A cube scaled to `size` and translated to `center`."""
    sx, sy, sz = size
    cx, cy, cz = center
    vertices = CUBE_VERTICES * np.array([sx, sy, sz]) + np.array([cx, cy, cz])
    return Mesh(vertices=vertices, triangles=CUBE_TRIANGLES.copy())


# ---------------------------------------------------------------------------
# UV-mapped box: every face owns its four vertices so each face can carry a
# 0..1 UV rectangle (texture atlas style). Outward winding: cross(v1-v0,
# v2-v0) points away from the box.
# ---------------------------------------------------------------------------

# name, four corners (counter-clockwise seen from outside)
_BOX_FACES = (
    ("+X", [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5),
            (0.5, -0.5, 0.5)]),
    ("-X", [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5),
            (-0.5, 0.5, -0.5)]),
    ("+Y", [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5)]),
    ("-Y", [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
            (-0.5, -0.5, 0.5)]),
    ("+Z", [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5)]),
    ("-Z", [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5)]),
)


def box_mesh_uv(center, size, uv_per_unit=1.0, faces=None) -> Mesh:
    """Axis-aligned box with per-face UVs (texture repeats once per world unit).

    ``faces`` selects which faces to emit, e.g. ``("+X", "-X", "+Y", "+Z",
    "-Z")`` for a wall that never shows its bottom face.
    """
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    vertices, uvs, triangles = [], [], []
    for name, corners in _BOX_FACES:
        if faces is not None and name not in faces:
            continue
        base = len(vertices)
        pts = [np.array(c, dtype=float) * size + center for c in corners]
        # Corner order is (0,0) (0,1) (1,1) (1,0): u runs along c0->c3 and
        # v along c0->c1, so the texture keeps its aspect on every face.
        du = float(np.linalg.norm(pts[3] - pts[0])) * uv_per_unit
        dv = float(np.linalg.norm(pts[1] - pts[0])) * uv_per_unit
        vertices.extend(pts)
        uvs.extend([(0.0, 0.0), (0.0, dv), (du, dv), (du, 0.0)])
        triangles.extend([(base, base + 1, base + 2),
                          (base, base + 2, base + 3)])
    return Mesh(vertices=np.array(vertices, dtype=float),
                triangles=np.array(triangles, dtype=int),
                uvs=np.array(uvs, dtype=float))


def wall_box_meshes(center, size):
    """Split one wall box into (side faces, green top cap).

    The bottom face is never visible, so it is not emitted. Both parts carry
    UVs, so the stone texture is mapped onto the wall sides.
    """
    sides = box_mesh_uv(center, size, uv_per_unit=1.0,
                        faces=("+X", "-X", "+Z", "-Z"))
    top = box_mesh_uv(center, size, uv_per_unit=1.0, faces=("+Y",))
    return sides, top


# ---------------------------------------------------------------------------
# Ground plane and surrounding lawn (Y = 0 grid, upward-facing triangles)
# ---------------------------------------------------------------------------

def ground_mesh(cols=3, rows=3, y=0.0, uv_per_unit=1.0) -> Mesh:
    vertices = []
    uvs = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            vertices.append([float(c), y, float(r)])
            uvs.append([c * uv_per_unit, r * uv_per_unit])
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


def lawn_mesh(cols, rows, margin=16.0, y=-0.01, step=8.0,
              uv_per_unit=0.5) -> Mesh:
    """The lawn *around* the maze (a ring, not a full plane).

    The maze floor is drawn separately, so the lawn skips the area under the
    maze: those pixels would only lose the depth test and cost fill time.
    """
    x0, z0 = -margin, -margin
    x1, z1 = cols + margin, rows + margin
    # (xa, xb, za, zb) rectangles covering everything outside the maze.
    rects = (
        (x0, x1, z0, 0.0),
        (x0, x1, rows, z1),
        (x0, 0.0, 0.0, rows),
        (cols, x1, 0.0, rows),
    )
    vertices, uvs, triangles = [], [], []
    for (xa, xb, za, zb) in rects:
        nx = max(1, int(math.ceil((xb - xa) / step)))
        nz = max(1, int(math.ceil((zb - za) / step)))
        base = len(vertices)
        for iz in range(nz + 1):
            z = za + (zb - za) * iz / nz
            for ix in range(nx + 1):
                x = xa + (xb - xa) * ix / nx
                vertices.append([x, y, z])
                uvs.append([x * uv_per_unit, z * uv_per_unit])
        stride = nx + 1
        for iz in range(nz):
            for ix in range(nx):
                v00 = base + iz * stride + ix
                v10 = v00 + 1
                v01 = v00 + stride
                v11 = v01 + 1
                triangles.append((v00, v01, v11))
                triangles.append((v00, v11, v10))
    return Mesh(vertices=np.array(vertices, dtype=float),
                triangles=np.array(triangles, dtype=int),
                uvs=np.array(uvs, dtype=float))


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

def merge_meshes(meshes):
    """Combine several Meshes into one, offsetting triangle indices.

    UVs survive only when every input mesh has them; mixing textured and
    untextured parts in one mesh would silently mis-map the texture.
    """
    meshes = [m for m in meshes
              if m is not None and len(np.asarray(m.vertices))]
    verts, tris = [], []
    v_off = 0
    for m in meshes:
        verts.append(np.asarray(m.vertices, dtype=float))
        if len(m.triangles):
            tris.append(np.asarray(m.triangles, dtype=int) + v_off)
        v_off += len(m.vertices)
    if not verts:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    vertices = np.vstack(verts)
    triangles = np.vstack(tris) if tris else np.zeros((0, 3), dtype=int)
    all_uv = all(m.uvs is not None for m in meshes)
    uv = np.vstack([np.asarray(m.uvs, dtype=float) for m in meshes]) \
        if all_uv else None
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
    """Flat upward-facing disc on the ground (water-hole pad).

    The fan is wound (centre, next, current) so the face normal points +Y;
    the opposite winding faces down and would be removed by back-face culling.
    """
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
        triangles.append((0, b, a))   # winding -> normal +Y
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


# ---------------------------------------------------------------------------
# Worm: a head plus four body segments. Each segment is its own mesh so the
# gameplay code can place it on an older sample of the head path (12.2).
# ---------------------------------------------------------------------------

WORM_SEGMENT_COUNT = 4          # body segments behind the head
WORM_SEGMENT_SPACING = 0.155    # world units between segment centres


def worm_head_mesh() -> Mesh:
    """Head (with eyes) centred on the local origin, facing +Z."""
    head = _sph((0.0, 0.01, 0.0), (0.145, 0.135, 0.155), rings=5, sectors=8)
    eyes = [
        _sph((-0.062, 0.085, 0.10), (0.026, 0.026, 0.026), rings=3, sectors=5),
        _sph((0.062, 0.085, 0.10), (0.026, 0.026, 0.026), rings=3, sectors=5),
    ]
    return merge_meshes([head] + eyes)


def worm_segment_mesh(index) -> Mesh:
    """Body segment `index` (0 = closest to the head, larger radius)."""
    t = index / max(1, WORM_SEGMENT_COUNT - 1)
    r = 0.115 - 0.045 * t
    return _sph((0.0, 0.0, 0.0), (r, r * 0.92, 0.085),
                rings=4, sectors=7)


def worm_parts(center=(0.0, 0.0, 0.0)):
    """Legacy single-mesh worm (kept for the offscreen tools/tests).

    Returns [(body_mesh, "worm"), (eyes_mesh, "eye")] with the worm local to
    the origin and facing +Z.
    """
    segs = [worm_segment_mesh(i) for i in range(WORM_SEGMENT_COUNT)]
    for i, seg in enumerate(segs):
        z = -WORM_SEGMENT_SPACING * (i + 1)
        seg.vertices[:, 2] += z
    body = merge_meshes(segs + [worm_head_mesh()])
    eyes = merge_meshes([
        _sph((-0.062, 0.095, 0.10), (0.026, 0.026, 0.026),
             rings=3, sectors=5),
        _sph((0.062, 0.095, 0.10), (0.026, 0.026, 0.026),
             rings=3, sectors=5),
    ])
    return [(body, "worm"), (eyes, "eye")]


def worm_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    body, eyes = worm_parts(center)
    for m in (body, eyes):
        _place(m, center)
    return merge_meshes([body, eyes])


# ---------------------------------------------------------------------------
# Collectibles and birds
# ---------------------------------------------------------------------------

def apple_parts(center=(0.0, 0.0, 0.0)):
    """Apple: flattened fruit, short stem and one leaf (local origin)."""
    fruit = [_sph((0.0, 0.0, 0.0), (0.15, 0.135, 0.15), rings=5, sectors=8)]
    stem = [cube_mesh((0.024, 0.10, 0.024), (0.0, 0.17, 0.0))]
    leaf = [_sph((0.06, 0.19, 0.0),
                 (0.06, 0.014, 0.032), rings=3, sectors=5)]
    return [(merge_meshes(fruit), "fruit"),
            (merge_meshes(stem), "stem"),
            (merge_meshes(leaf), "leaf")]


def apple_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    parts = apple_parts()
    for m, _ in parts:
        _place(m, center)
    return merge_meshes([m for m, _ in parts])


def bird_parts(center=(0.0, 0.0, 0.0)):
    """Bird body, belly, beak and eyes (local origin, head/beak face +Z).

    The two wings are separate meshes (``bird_wing_meshes``) so they can flap
    by rotating about the bird's forward axis (12.2).
    """
    body = [
        _sph((0.0, 0.0, 0.0), (0.15, 0.11, 0.19), rings=4, sectors=6),
        _sph((0.0, 0.05, 0.17),
             (0.09, 0.085, 0.09), rings=4, sectors=6),
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
    return [(merge_meshes(body), "bird"),
            (merge_meshes(belly), "belly"),
            (merge_meshes(beak), "beak"),
            (merge_meshes(eyes), "eye")]


def bird_wing_meshes():
    """Left and right wing as separate meshes hinged at the body axis.

    The wings are built around the local origin so a rotation about Z through
    the origin swings the wing tip up and down.
    """
    left = _sph((-0.14, 0.02, -0.04), (0.11, 0.016, 0.10),
                rings=3, sectors=5)
    right = _sph((0.14, 0.02, -0.04), (0.11, 0.016, 0.10),
                 rings=3, sectors=5)
    return left, right


def bird_mesh(center=(0.0, 0.0, 0.0)) -> Mesh:
    parts = bird_parts()
    left, right = bird_wing_meshes()
    for m, _ in parts:
        _place(m, center)
    return merge_meshes([m for m, _ in parts] + [left, right])


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
    return merge_meshes([pole, flag])


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


def maze_wall_meshes(maze, cols, rows):
    """Merged wall geometry: (all side faces, all green top caps).

    One mesh per material keeps the per-frame draw-call count tiny; the
    colliders still come from ``maze_wall_boxes`` (same centre/size data).
    """
    sides, tops = [], []
    for center, size in maze_wall_boxes(maze, cols, rows):
        s, t = wall_box_meshes(center, size)
        sides.append(s)
        tops.append(t)
    return merge_meshes(sides), merge_meshes(tops)
