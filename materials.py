"""
Materials, procedural Pillow textures, nearest-neighbour sampling and the
ambient + diffuse lighting model (handbook ch. 9-10).

Lighting (flat shading, one colour per triangle):
    I = ka + kd * max(0, n . L)
with ka = 0.25, kd = 0.75; L points from the surface toward the light.
"""

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


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def face_normal(a, b, c):
    """Unit face normal from world-space triangle vertices."""
    n = np.cross(np.asarray(b, float) - np.asarray(a, float),
                 np.asarray(c, float) - np.asarray(a, float))
    return normalized(n)


def shade(base_rgb, normal, light_dir, ambient=AMBIENT, diffuse=DIFFUSE):
    """Ambient + Lambertian diffuse shading applied per RGB channel."""
    n = normalized(normal)
    light = normalized(light_dir)
    intensity = ambient + diffuse * max(0.0, dot(n, light))
    return np.clip(np.asarray(base_rgb, dtype=float) * intensity,
                   0, 255).astype(np.uint8)
