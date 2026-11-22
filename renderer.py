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
from materials import (
    face_normal, shade, sample_nearest, sample_bilinear, AMBIENT, DIFFUSE,
)

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
    def __init__(self, width, height, fog=None):
        self.width = width
        self.height = height
        self.colorbuf = np.zeros((height, width, 3), dtype=np.uint8)
        self.depthbuf = np.full((height, width), np.inf)
        # Part B extension 1: atmospheric depth fog.
        # fog = {"color": np.ndarray(3), "near": float, "far": float} or None.
        self.fog = fog

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
                           uvs=None, intensity=1.0, filter_mode="nearest"):
        """Fill one projected triangle with a Z-buffer test.

        screen: three (x, y) pixels; depths: three camera-space z values.
        If texture + uvs are given, sample the texture with perspective-correct
        UV interpolation and modulate by the flat lighting intensity.

        The bounding box is scanned with NumPy arrays; the per-pixel math
        (barycentric weights, perspective-correct depth/UV, depth test) is the
        same as the handbook description, only vectorized for speed.
        """
        a, b, c = screen
        za, zb, zc = depths
        H, W = self.depthbuf.shape

        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            return

        rgb = np.clip(np.asarray(rgb, dtype=float), 0, 255)

        xs = np.arange(x0, x1 + 1, dtype=float) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=float) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        def edge(p, q):
            return (gx - p[0]) * (q[1] - p[1]) - (gy - p[1]) * (q[0] - p[0])

        area = _edge(a, b, c)
        if abs(area) < _EPS:
            return
        wa = edge(b, c) / area
        wb = edge(c, a) / area
        wc = edge(a, b) / area

        mask = (wa >= -_EPS) & (wb >= -_EPS) & (wc >= -_EPS)
        invz = wa / za + wb / zb + wc / zc
        mask &= invz > _EPS
        if not mask.any():
            return
        z = 1.0 / invz

        sub_depth = self.depthbuf[y0:y1 + 1, x0:x1 + 1]
        mask &= z < sub_depth
        if not mask.any():
            return

        if texture is not None and uvs is not None:
            # Perspective-correct UV: interpolate u/z, v/z, 1/z (10.2)
            u = (wa * uvs[0][0] / za + wb * uvs[1][0] / zb
                 + wc * uvs[2][0] / zc) / invz
            v = (wa * uvs[0][1] / za + wb * uvs[1][1] / zb
                 + wc * uvs[2][1] / zc) / invz
            th, tw, _ = texture.shape
            u = np.mod(u, 1.0)
            v = np.mod(v, 1.0)

            if filter_mode == "bilinear":
                # Part B extension 2: interpolate the four neighbouring texels.
                final = np.zeros(gx.shape + (3,), dtype=float)
                hit_y, hit_x = np.where(mask)
                for yy, xx in zip(hit_y, hit_x):
                    texel = sample_bilinear(texture, u[yy, xx], v[yy, xx])
                    final[yy, xx] = np.clip(texel * intensity, 0, 255)
            else:
                tx = np.minimum((u * tw).astype(int), tw - 1)
                ty = np.minimum((v * th).astype(int), th - 1)
                texel = texture[ty, tx].astype(float)
                final = np.clip(texel * intensity, 0, 255)
        else:
            final = np.broadcast_to(rgb, gx.shape + (3,))

        # Part B extension 1: blend distant fragments toward the fog colour.
        if self.fog is not None:
            fog = self.fog
            fog_t = np.clip((z - fog["near"]) / (fog["far"] - fog["near"]),
                            0.0, 1.0)
            fog_color = np.asarray(fog["color"], dtype=float)
            final = final * (1.0 - fog_t[..., None]) + fog_color * fog_t[..., None]

        sub_depth[mask] = z[mask]
        sub_color = self.colorbuf[y0:y1 + 1, x0:x1 + 1]
        sub_color[mask] = np.clip(final[mask], 0, 255).astype(np.uint8)

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
                                        uvs=uvs_tri, intensity=intensity,
                                        filter_mode=material.filter_mode)

    def surface(self):
        """Return the HxWx3 uint8 framebuffer for blitting."""
        return self.colorbuf
