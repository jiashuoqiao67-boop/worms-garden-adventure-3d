"""
Camera basis, perspective / orthographic projection and near-plane clipping
(handbook ch. 7).

Pipeline: model -> world -> camera space -> screen pixels.
Camera-space convention here: +Z points in front of the camera.
"""

import math

import numpy as np

from transforms import normalized, dot, cross


class Camera:
    def __init__(self, eye, target, up=(0.0, 1.0, 0.0),
                 projection="perspective", fov_y=70.0, near=0.1,
                 width=640, height=480, pixels_per_unit=45.0):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.up0 = np.asarray(up, dtype=float)
        self.projection = projection
        self.fov_y = fov_y
        self.near = near
        self.width = width
        self.height = height
        self.pixels_per_unit = pixels_per_unit

    def basis(self):
        """Return forward, right, up unit vectors (handbook 7.2)."""
        forward = normalized(self.target - self.eye)
        right = normalized(cross(self.up0, forward))
        up = cross(forward, right)
        return forward, right, up

    def to_camera_space(self, point):
        """Transform a world point into camera-space coordinates."""
        point = np.asarray(point, dtype=float)
        forward, right, up = self.basis()
        d = point - self.eye
        return np.array([dot(d, right), dot(d, up), dot(d, forward)])

    def project(self, point):
        """Project a world point to screen pixels; None if behind the near plane."""
        c = self.to_camera_space(point)
        if self.projection == "orthographic":
            return orthographic_project(c, self.width, self.height,
                                        self.pixels_per_unit, self.near)
        return perspective_project(c, self.width, self.height,
                                   self.fov_y, self.near)

    def toggle_projection(self):
        self.projection = ("orthographic"
                           if self.projection == "perspective"
                           else "perspective")


def perspective_project(c, width=640, height=480, fov_y=70.0, near=0.1):
    """Perspective projection; distant objects shrink with depth (7.3)."""
    x, y, depth = c
    if depth <= near:
        return None
    focal = (height / 2.0) / math.tan(math.radians(fov_y) / 2.0)
    sx = width / 2.0 + focal * x / depth
    sy = height / 2.0 - focal * y / depth
    return np.array([sx, sy])


def orthographic_project(c, width=640, height=480, pixels_per_unit=45.0,
                         near=0.1):
    """Orthographic projection; size is independent of depth (7.4)."""
    x, y, depth = c
    if depth <= near:
        return None
    sx = width / 2.0 + pixels_per_unit * x
    sy = height / 2.0 - pixels_per_unit * y
    return np.array([sx, sy])


def clip_triangle_near(triangle, near=0.1):
    """Sutherland-Hodgman clip against the near plane in camera space (7.6).

    Returns a list of 3 or 4 visible vertices, or None if the whole triangle
    is behind the plane. The resulting polygon is triangulated as a fan later.
    """
    pts = [np.asarray(v, dtype=float) for v in triangle]
    visible = []

    for start, end in zip(pts, pts[1:] + pts[:1]):
        a_visible = start[2] >= near
        b_visible = end[2] >= near

        if a_visible != b_visible:
            t = (near - start[2]) / (end[2] - start[2])
            intersection = start + t * (end - start)
            visible.append(intersection)
        if b_visible:
            visible.append(end)

    if len(visible) < 3:
        return None
    return visible
