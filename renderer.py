"""
Software renderer: triangle rasterization, barycentric coordinates and the
Z-buffer (handbook ch. 8).

Stage 2 pipeline per triangle:
    model -> world -> camera space -> near-plane clip -> project -> rasterize.

Lighting and texture sampling are added in M7. Colors are plain RGB here.
"""

import numpy as np

from transforms import transform_point
from camera import (
    clip_triangle_near,
    perspective_project,
    orthographic_project,
)

_EPS = 1e-9


def _edge(a, b, p):
    """Signed area edge function (handbook 8.3)."""
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def barycentric(p, a, b, c):
    """Return barycentric weights (wa, wb, wc), or None if outside."""
    area = _edge(a, b, c)
    if abs(area) < _EPS:
        return None
    wa = _edge(b, c, p) / area
    wb = _edge(c, a, p) / area
    wc = _edge(a, b, p) / area
    if min(wa, wb, wc) < -_EPS:
        return None
    return wa, wb, wc


class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.colorbuf = np.zeros((height, width, 3), dtype=np.uint8)
        self.depthbuf = np.full((height, width), np.inf)

    def clear(self):
        self.colorbuf.fill(0)
        self.depthbuf.fill(np.inf)

    def _project(self, camera, cam_point):
        if camera.projection == "orthographic":
            return orthographic_project(cam_point, camera.width, camera.height,
                                        camera.pixels_per_unit, camera.near)
        return perspective_project(cam_point, camera.width, camera.height,
                                  camera.fov_y, camera.near)

    def rasterize_triangle(self, screen, depths, rgb):
        """Fill one projected triangle with a Z-buffer test.

        screen: three (x, y) pixels; depths: three camera-space z values.
        """
        a, b, c = screen
        za, zb, zc = depths
        H, W = self.depthbuf.shape

        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))

        rgb = np.clip(np.asarray(rgb, dtype=float), 0, 255).astype(np.uint8)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                p = np.array([x + 0.5, y + 0.5])
                weights = barycentric(p, a, b, c)
                if weights is None:
                    continue
                wa, wb, wc = weights

                # Perspective-correct depth: 1/z = wa/za + wb/zb + wc/zc
                invz = wa / za + wb / zb + wc / zc
                if invz <= _EPS:
                    continue
                z = 1.0 / invz

                if z < self.depthbuf[y, x]:
                    self.depthbuf[y, x] = z
                    self.colorbuf[y, x] = rgb

    def draw_mesh(self, camera, mesh, model_matrix, color):
        """Transform, project and rasterize every triangle of a mesh."""
        for tri in mesh.triangles:
            world_pts = [transform_point(model_matrix, mesh.vertices[i])
                         for i in tri]
            cam_pts = [camera.to_camera_space(p) for p in world_pts]

            poly = clip_triangle_near(cam_pts, camera.near)
            if poly is None:
                continue

            # Fan triangulation of the clipped polygon
            for k in range(1, len(poly) - 1):
                cam_tri = [poly[0], poly[k], poly[k + 1]]
                screen = [self._project(camera, c) for c in cam_tri]
                if any(s is None for s in screen):
                    continue
                depths = [c[2] for c in cam_tri]
                self.rasterize_triangle(screen, depths, color)

    def surface(self):
        """Return the HxWx3 uint8 framebuffer for blitting."""
        return self.colorbuf
