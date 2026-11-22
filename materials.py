"""
Materials, procedural Pillow textures, nearest-neighbour sampling and the
ambient + diffuse lighting model (handbook ch. 9-10).

Lighting (flat shading, one colour per triangle):
    I = ka + kd * max(0, n . L)
with ka = 0.25, kd = 0.75; L points from the surface toward the light.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from transforms import normalized, dot

AMBIENT = 0.25
DIFFUSE = 0.75


@dataclass
class Material:
    base_color: tuple = (128, 128, 128)
    texture: np.ndarray = None      # HxWx3 RGB uint8, or None
    uvs: np.ndarray = None           # per-vertex UV, shape (N, 2), or None
    filter_mode: str = "nearest"    # "nearest" (Part A) or "bilinear" (Part B)


# ---------------------------------------------------------------------------
# Procedural textures (handbook 10.1)
# ---------------------------------------------------------------------------

def make_grass_texture(size=128, seed=12):
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (35, 105, 48))
    draw = ImageDraw.Draw(img)
    tones = [(30, 74, 34), (54, 130, 48), (77, 146, 72), (49, 125, 57)]
    for _ in range(900):
        x, y = rng.randrange(size), rng.randrange(size)
        tone = rng.choice(tones)
        draw.line((x, y, x + 1, y - 3), fill=tone)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def make_stone_texture(size=128, seed=7):
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (112, 116, 119))
    draw = ImageDraw.Draw(img)
    for _ in range(500):
        x, y = rng.randrange(size), rng.randrange(size)
        tone = rng.choice([(55, 59, 62), (112, 116, 119), (175, 178, 178)])
        draw.rectangle((x, y, x + rng.randrange(2, 6), y + rng.randrange(2, 6)),
                       fill=tone)
    return np.array(img.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Texture sampling
# ---------------------------------------------------------------------------

def sample_nearest(tex, u, v):
    """Nearest-neighbour texel lookup with horizontal/vertical wrap (10.2)."""
    h, w, _ = tex.shape
    u = u % 1.0
    v = v % 1.0
    tx = min(int(u * w), w - 1)
    ty = min(int(v * h), h - 1)
    return tex[ty, tx]


def sample_bilinear(tex, u, v):
    """Bilinear texel interpolation with wrap (Part B extension 2).

    Four neighbouring texels are blended by the fractional parts of the
    texture coordinates, so magnified textures soften instead of showing
    hard nearest-neighbour blocks. Works on scalar ``u``, ``v``.
    """
    h, w, _ = tex.shape
    u = u % 1.0
    v = v % 1.0
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0

    x0m = x0 % w
    x1m = (x0 + 1) % w
    y0m = y0 % h
    y1m = (y0 + 1) % h

    c00 = tex[y0m, x0m].astype(float)
    c10 = tex[y0m, x1m].astype(float)
    c01 = tex[y1m, x0m].astype(float)
    c11 = tex[y1m, x1m].astype(float)

    top = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bottom * fy


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def face_normal(a, b, c):
    """Unit face normal from world-space triangle vertices (scalar math to
    avoid per-triangle NumPy array allocation).

    Degenerate triangles (zero area) fall back to an upward normal; the
    rasterizer drops them later by its edge-area test.
    """
    ax, ay, az = a[0], a[1], a[2]
    ux, uy, uz = b[0] - ax, b[1] - ay, b[2] - az
    vx, vy, vz = c[0] - ax, c[1] - ay, c[2] - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-10:
        return (0.0, 1.0, 0.0)
    return (nx / length, ny / length, nz / length)


def shade(base_rgb, normal, light_dir, ambient=AMBIENT, diffuse=DIFFUSE,
          intensity=None):
    """Ambient + Lambertian diffuse shading applied per RGB channel.

    ``normal`` and ``light_dir`` are expected to be unit vectors; an optional
    precomputed ``intensity`` avoids re-running the diffuse dot product.
    """
    if intensity is None:
        intensity = ambient + diffuse * max(0.0, dot(normal, light_dir))
    return [min(255.0, max(0.0, float(base_rgb[i]) * intensity))
            for i in range(3)]
