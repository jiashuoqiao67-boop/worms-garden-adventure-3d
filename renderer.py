"""
Software renderer: triangle rasterization, barycentric coordinates, Z-buffer,
flat lighting and texture sampling (handbook ch. 8-10).

Pipeline per triangle:
    model -> world -> camera space -> near-plane clip -> project -> rasterize.

Lighting is flat (one intensity per triangle). Textures use nearest-neighbour
sampling with perspective-correct UV interpolation.
"""

import numpy as np

from transforms import transform_point, normalized, dot
from camera import (
    clip_triangle_near,
    perspective_project,
    orthographic_project,
)
from materials import face_normal, shade, sample_nearest, AMBIENT, DIFFUSE

_EPS = 1e-9
DEFAULT_LIGHT_DIR = (0.5, 1.0, 0.3)


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

    def rasterize_triangle(self, screen, depths, rgb, texture=None,
                           uvs=None, intensity=1.0):
        """Fill one projected triangle with a Z-buffer test.

        screen: three (x, y) pixels; depths: three camera-space z values.
        If texture + uvs are given, sample the texture with perspective-correct
        UV interpolation and modulate by the flat lighting intensity.
        """
        a, b, c = screen
        za, zb, zc = depths
        H, W = self.depthbuf.shape

        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))

        rgb = np.clip(np.asarray(rgb, dtype=float), 0, 255)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                p = np.array([x + 0.5, y + 0.5])
                weights = barycentric(p, a, b, c)
                if weights is None:
                    continue
                wa, wb, wc = weights

                invz = wa / za + wb / zb + wc / zc
                if invz <= _EPS:
                    continue
                z = 1.0 / invz

                if z >= self.depthbuf[y, x]:
                    continue

                if texture is not None and uvs is not None:
                    # Perspective-correct UV: interpolate u/z, v/z, 1/z (10.2)
                    u = (wa * uvs[0][0] / za + wb * uvs[1][0] / zb
                         + wc * uvs[2][0] / zc) / invz
                    v = (wa * uvs[0][1] / za + wb * uvs[1][1] / zb
                         + wc * uvs[2][1] / zc) / invz
                    texel = sample_nearest(texture, u, v).astype(float)
                    final = np.clip(texel * intensity, 0, 255)
                else:
                    final = rgb

                self.depthbuf[y, x] = z
                self.colorbuf[y, x] = final.astype(np.uint8)

    def draw_mesh(self, camera, mesh, model_matrix, material,
                  light_dir=DEFAULT_LIGHT_DIR):
        """Transform, light, project and rasterize every triangle of a mesh."""
        light = normalized(light_dir)

        for tri in mesh.triangles:
            world_pts = [transform_point(model_matrix, mesh.vertices[i])
                         for i in tri]
            cam_pts = [camera.to_camera_space(p) for p in world_pts]

            poly = clip_triangle_near(cam_pts, camera.near)
            if poly is None:
                continue

            # Flat shading: one face normal and intensity per triangle.
            normal = face_normal(world_pts[0], world_pts[1], world_pts[2])
            intensity = AMBIENT + DIFFUSE * max(0.0, dot(normal, light))
            shaded = shade(material.base_color, normal, light)

            textured = (material.texture is not None
                        and mesh.uvs is not None and len(poly) == 3)
            uvs_tri = mesh.uvs[tri] if textured else None
            tex = material.texture if textured else None

            for k in range(1, len(poly) - 1):
                cam_tri = [poly[0], poly[k], poly[k + 1]]
                screen = [self._project(camera, c) for c in cam_tri]
                if any(s is None for s in screen):
                    continue
                depths = [c[2] for c in cam_tri]
                self.rasterize_triangle(screen, depths, shaded, texture=tex,
                                        uvs=uvs_tri, intensity=intensity)

    def surface(self):
        """Return the HxWx3 uint8 framebuffer for blitting."""
        return self.colorbuf
