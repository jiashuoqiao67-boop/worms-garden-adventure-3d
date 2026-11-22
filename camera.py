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
        # Perspective focal length is constant for the camera; cache it so the
        # per-vertex projection does not recompute tan(fov/2) every point.
        self._focal = (height / 2.0) / math.tan(math.radians(fov_y) / 2.0)
        # Cache the camera basis once per instance; the per-vertex transform
        # below must not recompute the cross products / normalization every time.
        f, r, u = self.basis()
        # Store plain Python tuples so the per-vertex camera-space dot products
        # do not pay NumPy scalar indexing overhead.
        self._forward = (float(f[0]), float(f[1]), float(f[2]))
        self._right = (float(r[0]), float(r[1]), float(r[2]))
        self._up = (float(u[0]), float(u[1]), float(u[2]))

    def basis(self):
        """Return forward, right, up unit vectors (handbook 7.2)."""
        forward = normalized(self.target - self.eye)
        right = normalized(cross(self.up0, forward))
        up = cross(forward, right)
        return forward, right, up

    def to_camera_space(self, point):
        """Transform a world point into camera-space coordinates (scalar dot
        products, returned as a tuple to avoid per-vertex array allocation)."""
        ex, ey, ez = self.eye
        dx = point[0] - ex
        dy = point[1] - ey
        dz = point[2] - ez
        f, r, u = self._forward, self._right, self._up
        return (
            dx * r[0] + dy * r[1] + dz * r[2],
            dx * u[0] + dy * u[1] + dz * u[2],
            dx * f[0] + dy * f[1] + dz * f[2],
        )

    def project(self, point):
        """Project a world point to screen pixels; None if behind the near plane."""
        return self.project_cam(self.to_camera_space(point))

    def project_cam(self, c):
        """Project an already camera-space point using cached camera constants."""
        x, y, depth = c
        if depth <= self.near:
            return None
        if self.projection == "orthographic":
            sx = self.width / 2.0 + self.pixels_per_unit * x
            sy = self.height / 2.0 - self.pixels_per_unit * y
        else:
            sx = self.width / 2.0 + self._focal * x / depth
            sy = self.height / 2.0 - self._focal * y / depth
        return (sx, sy)

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
    return (sx, sy)


def orthographic_project(c, width=640, height=480, pixels_per_unit=45.0,
                         near=0.1):
    """Orthographic projection; size is independent of depth (7.4)."""
    x, y, depth = c
    if depth <= near:
        return None
    sx = width / 2.0 + pixels_per_unit * x
    sy = height / 2.0 - pixels_per_unit * y
    return (sx, sy)


def clip_triangle_near(triangle, near=0.1):
    """Sutherland-Hodgman clip against the near plane in camera space (7.6).

    Returns a list of 3 or 4 visible vertices, or None if the whole triangle
    is behind the plane. The resulting polygon is triangulated as a fan later.
    """
    pts = [(v[0], v[1], v[2]) for v in triangle]
    # Fast path: the whole triangle is in front of the near plane.
    if all(v[2] >= near for v in pts):
        return pts
    visible = []

    for i in range(len(pts)):
        start = pts[i]
        end = pts[(i + 1) % len(pts)]
        a_visible = start[2] >= near
        b_visible = end[2] >= near

        if a_visible != b_visible:
            t = (near - start[2]) / (end[2] - start[2])
            intersection = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                near,
            )
            visible.append(intersection)
        if b_visible:
            visible.append(end)

    if len(visible) < 3:
        return None
    return visible
