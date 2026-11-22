"""
Software renderer: triangle rasterization, barycentric coordinates, Z-buffer,
flat lighting and texture sampling (handbook ch. 8-10).

Pipeline per triangle:
    model -> world -> camera space -> near-plane clip -> project -> rasterize.

Lighting is flat (one intensity per triangle). Textures use nearest-neighbour
sampling with perspective-correct UV interpolation.
"""

import numpy as np

from transforms import transform_point, normalized
from camera import clip_triangle_near
from materials import (
    face_normal, shade, sample_nearest, sample_bilinear, AMBIENT, DIFFUSE,
)
from renderer_jit import HAS_NUMBA, rasterize_njit

_EPS = 1e-9
DEFAULT_LIGHT_DIR = (0.5, 1.0, 0.3)
# Pre-normalized once; the light direction is constant across the scene.
_LIGHT_NORM = normalized(DEFAULT_LIGHT_DIR)


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
    def __init__(self, width, height, fog=None, background=(0, 0, 0),
                 sky_grad=None):
        self.width = width
        self.height = height
        self.background = np.asarray(background, dtype=np.uint8)
        self.colorbuf = np.zeros((height, width, 3), dtype=np.uint8)
        self.depthbuf = np.full((height, width), np.inf)
        # Optional vertical three-stop sky gradient: (top, mid, low) RGB.
        # Baked once into a background buffer; it only fills pixels where no
        # geometry is drawn, so it does not affect the Z-buffer pipeline.
        self._sky = None
        if sky_grad is not None:
            top, mid, low = (np.asarray(c, dtype=float) for c in sky_grad)
            rows = np.zeros((height, 3), dtype=float)
            for y in range(height):
                t = y / max(1, height - 1)
                if t < 0.55:
                    k = t / 0.55
                    rows[y] = top + (mid - top) * k
                else:
                    k = (t - 0.55) / 0.45
                    rows[y] = mid + (low - mid) * k
            self._sky = np.clip(rows, 0, 255).astype(np.uint8)[:, None, :]
            self.colorbuf[:] = self._sky
        # Part B extension 1: atmospheric depth fog.
        # fog = {"color": np.ndarray(3), "near": float, "far": float} or None.
        self.fog = fog
        # Back-face culling in camera space: skip triangles whose face normal
        # points away from the camera (+Z is forward here). Flat horizontal
        # faces whose normal is nearly parallel to the image plane are kept.
        self.backface_cull = True
        # Scene-level distance/range cull, enabled only for the chase camera
        # (where most walls lie behind or beyond the fog). Off for overview /
        # side views so the whole maze stays on screen with no extra per-mesh
        # centre work.
        self.cull_distance = False
        # Cache for the identity-model-matrix check (see draw_mesh).
        self._last_model = None
        self._last_is_identity = False

    def clear(self):
        if self._sky is not None:
            self.colorbuf[:] = self._sky
        else:
            self.colorbuf[:] = self.background
        self.depthbuf.fill(np.inf)

    def _project(self, camera, cam_point):
        return camera.project_cam(cam_point)

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

        if HAS_NUMBA:
            sx = np.asarray([a[0], b[0], c[0]], dtype=float)
            sy = np.asarray([a[1], b[1], c[1]], dtype=float)
            sz = np.asarray([za, zb, zc], dtype=float)
            base = np.asarray(rgb, dtype=float)
            use_tex = texture is not None and uvs is not None
            tex = texture if use_tex else np.empty((0, 0, 3), dtype=np.uint8)
            uv = np.asarray(uvs, dtype=float) if use_tex else np.empty((0, 2))
            if self.fog is not None:
                fog_color = np.asarray(self.fog["color"], dtype=float)
                fog_near, fog_far = self.fog["near"], self.fog["far"]
                use_fog = True
            else:
                fog_color = np.zeros(3)
                fog_near, fog_far, use_fog = 0.0, 1.0, False
            rasterize_njit(self.colorbuf, self.depthbuf, sx, sy, sz, base,
                           tex, uv, float(intensity), use_tex,
                           filter_mode == "bilinear", use_fog,
                           fog_color, fog_near, fog_far)
            return

        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            return

        rgb = np.asarray(rgb, dtype=float)

        xs = np.arange(x0, x1 + 1, dtype=float) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=float) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        area = _edge(a, b, c)
        if abs(area) < _EPS:
            return
        wa = ((gx - b[0]) * (c[1] - b[1]) - (gy - b[1]) * (c[0] - b[0])) / area
        wb = ((gx - c[0]) * (a[1] - c[1]) - (gy - c[1]) * (a[0] - c[0])) / area
        wc = ((gx - a[0]) * (b[1] - a[1]) - (gy - a[1]) * (b[0] - a[0])) / area

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
        light = _LIGHT_NORM if light_dir is DEFAULT_LIGHT_DIR else normalized(light_dir)
        # Most meshes already hold world-space vertices under an identity
        # model matrix; skip the 4x4 multiply in that common case. Cache the
        # identity check by matrix object identity (the shared IDENTITY matrix
        # is passed for every static mesh).
        if self._last_model is model_matrix:
            is_identity = self._last_is_identity
        else:
            is_identity = np.allclose(model_matrix, np.eye(4))
            self._last_model = model_matrix
            self._last_is_identity = is_identity

        # Bounding-centre quick reject (scene-level cull, mirroring the
        # reference range cull): skip a whole mesh that sits behind the near
        # plane or beyond the fog far plane, where it already blends into the
        # backdrop. The centre is cached per mesh so it is computed once.
        if self.cull_distance:
            far = self.fog["far"] if self.fog is not None else 60.0
            center = getattr(mesh, "_cull_center", None)
            if center is None:
                center = mesh.vertices.mean(axis=0)
                mesh._cull_center = center
            if not is_identity:
                center = transform_point(model_matrix, center)
            cc = camera.to_camera_space(center)
            if cc[2] < camera.near or cc[2] > far:
                return

        for tri in mesh.triangles:
            if is_identity:
                world_pts = [mesh.vertices[i] for i in tri]
            else:
                world_pts = [transform_point(model_matrix, mesh.vertices[i])
                             for i in tri]
            cam_pts = [camera.to_camera_space(p) for p in world_pts]

            # Back-face culling (camera space, +Z forward): a face pointing
            # away from the viewer has a positive camera-space normal Z.
            if self.backface_cull:
                ca, cb, cc = cam_pts
                cux = cb[0] - ca[0]; cuy = cb[1] - ca[1]; cuz = cb[2] - ca[2]
                cvx = cc[0] - ca[0]; cvy = cc[1] - ca[1]; cvz = cc[2] - ca[2]
                if cux * cvy - cuy * cvx > 1e-6:
                    continue

            poly = clip_triangle_near(cam_pts, camera.near)
            if poly is None:
                continue

            # Flat shading: one face normal and intensity per triangle.
            normal = face_normal(world_pts[0], world_pts[1], world_pts[2])
            nd = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]
            intensity = AMBIENT + DIFFUSE * (nd if nd > 0.0 else 0.0)
            bc = material.base_color
            shaded = [min(255.0, max(0.0, float(bc[i]) * intensity))
                      for i in range(3)]

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
