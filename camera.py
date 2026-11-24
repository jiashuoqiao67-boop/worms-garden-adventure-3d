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
        """Project an already camera-space point using cached camera constants.

        A vertex that lies exactly on the near plane is still visible
        (handbook 7.6): clipping produces such vertices, so rejecting them
        here would silently drop every clipped triangle.
        """
        x, y, depth = c
        if depth < self.near - 1e-9:
            return None
        if self.projection == "orthographic":
            sx = self.width / 2.0 + self.pixels_per_unit * x
            sy = self.height / 2.0 - self.pixels_per_unit * y
        else:
            sx = self.width / 2.0 + self._focal * x / depth
            sy = self.height / 2.0 - self._focal * y / depth
        return (sx, sy)

    def screen_ray(self, sx, sy):
        """Camera ray through a screen pixel, in world space (handbook 13.2).

        Returns ``(origin, direction)`` with a unit direction. Used to turn a
        mouse click into a world-space firework position by intersecting the
        ray with the ground plane.
        """
        right = np.array(self._right)
        up = np.array(self._up)
        forward = np.array(self._forward)
        if self.projection == "orthographic":
            x = (sx - self.width / 2.0) / self.pixels_per_unit
            y = -(sy - self.height / 2.0) / self.pixels_per_unit
            return self.eye + right * x + up * y, forward
        x = (sx - self.width / 2.0) / self._focal
        y = -(sy - self.height / 2.0) / self._focal
        direction = right * x + up * y + forward
        return self.eye.copy(), normalized(direction)

    def ground_hit(self, sx, sy, ground_y=0.0):
        """Intersect the pixel ray with the horizontal plane y = ground_y.

        Returns a world-space point or ``None`` when the ray is parallel to the
        plane or hits it behind the camera.
        """
        origin, direction = self.screen_ray(sx, sy)
        if abs(direction[1]) < 1e-6:
            return None
        t = (ground_y - origin[1]) / direction[1]
        if t <= 0.0:
            return None
        return origin + direction * t

    def toggle_projection(self):
        self.projection = ("orthographic"
                           if self.projection == "perspective"
                           else "perspective")


def perspective_project(c, width=640, height=480, fov_y=70.0, near=0.1):
    """Perspective projection; distant objects shrink with depth (7.3).

    A point exactly on the near plane still projects (handbook 7.6).
    """
    x, y, depth = c
    if depth < near - 1e-9:
        return None
    focal = (height / 2.0) / math.tan(math.radians(fov_y) / 2.0)
    sx = width / 2.0 + focal * x / depth
    sy = height / 2.0 - focal * y / depth
    return (sx, sy)


def orthographic_project(c, width=640, height=480, pixels_per_unit=45.0,
                         near=0.1):
    """Orthographic projection; size is independent of depth (7.4)."""
    x, y, depth = c
    if depth < near - 1e-9:
        return None
    sx = width / 2.0 + pixels_per_unit * x
    sy = height / 2.0 - pixels_per_unit * y
    return (sx, sy)


def clip_triangle_near(triangle, near=0.1):
    """Sutherland-Hodgman clip against the near plane in camera space (7.6).

    Returns a list of 3 or 4 visible vertices, or None if the whole triangle
    is behind the plane. The resulting polygon is triangulated as a fan later.
    """
    poly, _ = clip_polygon_near(triangle, near)
    return poly


def clip_polygon_near(points, near=0.1, attributes=None):
    """Clip a convex polygon against the near plane, interpolating attributes.

    ``attributes`` may hold one tuple per vertex (UVs, colours, ...); each
    intersection vertex receives the same interpolation parameter ``t`` that
    the handbook (7.6) prescribes, so clipped textured triangles keep their
    texture instead of showing a seam.
    """
    pts = [(float(v[0]), float(v[1]), float(v[2])) for v in points]
    if attributes is None:
        attributes = [()] * len(pts)

    # Fast path: the whole polygon is in front of the near plane.
    if all(v[2] >= near for v in pts):
        return pts, (list(attributes) if attributes and attributes[0] else None)

    out_pts, out_attrs = [], []
    n = len(pts)
    for i in range(n):
        start = pts[i]
        end = pts[(i + 1) % n]
        a_visible = start[2] >= near
        b_visible = end[2] >= near

        if a_visible != b_visible:
            t = (near - start[2]) / (end[2] - start[2])
            out_pts.append((start[0] + t * (end[0] - start[0]),
                            start[1] + t * (end[1] - start[1]),
                            near))
            if attributes[0]:
                sa, ea = attributes[i], attributes[(i + 1) % n]
                out_attrs.append(tuple(s + t * (e - s)
                                       for s, e in zip(sa, ea)))
        if b_visible:
            out_pts.append(end)
            if attributes[0]:
                out_attrs.append(attributes[(i + 1) % n])

    if len(out_pts) < 3:
        return None, None
    return out_pts, (out_attrs if attributes[0] else None)


