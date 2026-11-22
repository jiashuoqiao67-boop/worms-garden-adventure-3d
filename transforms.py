"""
Vectors and 4x4 homogeneous transforms (handbook chapter 5).

Conventions:
    - Column vectors; a transform matrix is applied as  M @ v.
    - A position uses homogeneous weight w = 1; a direction uses w = 0, so
      translation never moves a direction vector.
    - Combined model matrix:  M = T @ R @ S  (scale first, then rotate,
      then translate).

Handedness: right-handed, X = east, Y = up, Z = south.
"""

import numpy as np

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def normalized(v):
    """Return v scaled to unit length. Raises on a zero-length vector."""
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length < _EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / length


def vector_length(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def dot(a, b):
    return float(np.dot(np.asarray(a, dtype=float),
                        np.asarray(b, dtype=float)))


def cross(a, b):
    return np.cross(np.asarray(a, dtype=float),
                    np.asarray(b, dtype=float))


# ---------------------------------------------------------------------------
# 4x4 matrices (column-vector convention)
# ---------------------------------------------------------------------------

def translation(tx, ty, tz):
    M = np.eye(4)
    M[0:3, 3] = [tx, ty, tz]
    return M


def rotation_y(angle_rad):
    """Rotate about the world Y axis (right-handed)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0]).astype(float)


def model_matrix(scale_vec, rot_y_angle, translation_vec):
    """Build M = T @ R @ S (scale, then rotate, then translate)."""
    sx, sy, sz = scale_vec
    tx, ty, tz = translation_vec
    return translation(tx, ty, tz) @ rotation_y(rot_y_angle) @ scale(sx, sy, sz)


# ---------------------------------------------------------------------------
# Applying a transform
# ---------------------------------------------------------------------------

def transform_point(M, xyz):
    """Transform a position (homogeneous weight w = 1)."""
    v = np.append(np.asarray(xyz, dtype=float), 1.0)
    return (M @ v)[:3]


def transform_direction(M, xyz):
    """Transform a direction (w = 0); translation has no effect."""
    v = np.append(np.asarray(xyz, dtype=float), 0.0)
    return (M @ v)[:3]


def transform_normal(M, xyz):
    """Transform a surface normal with the inverse-transpose 3x3, then
    normalize. Required under non-uniform scaling (handbook 5.3)."""
    R = M[0:3, 0:3]
    normal = np.linalg.inv(R).T @ np.asarray(xyz, dtype=float)
    return normalized(normal)
