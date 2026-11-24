"""
Software renderer: triangle rasterization, barycentric coordinates, Z-buffer,
flat lighting and texture sampling (handbook ch. 8-10).

Pipeline per mesh:
    model -> world -> camera space -> near-plane clip -> project -> rasterize.

All triangles that are fully in front of the near plane are transformed and
projected with NumPy array operations and rasterized in one batched pass
(Numba when available, pure NumPy otherwise). Only the rare triangles that
cross the near plane take the per-triangle Sutherland-Hodgman path.

Lighting is flat (one intensity per triangle). Textures use nearest-neighbour
sampling by default and perspective-correct UV interpolation; bilinear
filtering and depth fog are Part B extensions.
"""

import math

import numpy as np

from transforms import normalized
from camera import clip_polygon_near
from materials import (
    face_normal, sample_nearest, sample_bilinear, AMBIENT, DIFFUSE,
)
from renderer_jit import HAS_NUMBA, rasterize_batch, draw_edges_batch

_EPS = 1e-9
DEFAULT_LIGHT_DIR = (0.35, 0.80, 0.50)
# Pre-normalized once; the light direction is constant across the scene.
_LIGHT_NORM = normalized(DEFAULT_LIGHT_DIR)
WIRE_COLOR = (140, 255, 180)


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
        # Depth is stored as float32: half the memory traffic of float64 for
        # the same visibility result at the depth ranges this game uses.
        self.depthbuf = np.full((height, width), np.inf, dtype=np.float32)
        # Optional vertical three-stop sky gradient: (top, mid, low) RGB.
        # Baked once into a full background image; it only fills pixels where
        # no geometry is drawn, so it does not affect the Z-buffer pipeline.
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
            column = np.clip(rows, 0, 255).astype(np.uint8)[:, None, :]
            # One contiguous (H, W, 3) image clears ~3x faster than
            # broadcasting a column every frame.
            self._sky = np.ascontiguousarray(
                np.broadcast_to(column, (height, width, 3)))
            self.colorbuf[:] = self._sky
        # Part B extension 1: atmospheric depth fog.
        # fog = {"color": np.ndarray(3), "near": float, "far": float} or None.
        self.fog = fog
        # Back-face culling in world space: skip triangles whose outward
        # normal points away from the eye (handbook 8.4).
        self.backface_cull = True
        # F1 debugging view: fill flat + draw projected triangle edges.
        self.wireframe = False
        # Cache for the identity-model-matrix check (see draw_mesh).
        self._last_model = None
        self._last_is_identity = False
        # Per-frame counters, reported in the HUD / benchmark.
        self.stats = {"submitted": 0, "culled": 0, "clipped": 0,
                      "rasterized": 0}

    def clear(self):
        if self._sky is not None:
            np.copyto(self.colorbuf, self._sky)
        else:
            self.colorbuf[:] = self.background
        self.depthbuf.fill(np.inf)
        for key in self.stats:
            self.stats[key] = 0

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_arrays(self, cam):
        """Vectorized camera-space -> screen-pixel projection."""
        depth = cam[:, 2]
        safe = np.where(depth > _EPS, depth, 1.0)
        if self._ortho:
            sx = self.width / 2.0 + self._ppu * cam[:, 0]
            sy = self.height / 2.0 - self._ppu * cam[:, 1]
        else:
            sx = self.width / 2.0 + self._focal * cam[:, 0] / safe
            sy = self.height / 2.0 - self._focal * cam[:, 1] / safe
        return sx, sy

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _prepare(self, camera):
        self._focal = camera._focal
        self._ppu = camera.pixels_per_unit
        self._ortho = camera.projection == "orthographic"
        self._near = camera.near

    def rasterize_triangle(self, screen, depths, rgb, texture=None,
                           uvs=None, intensity=1.0, filter_mode="nearest"):
        """Fill one projected triangle with a Z-buffer test.

        Public single-triangle entry point used by the numerical tests and by
        the near-plane clipping fallback. ``screen`` holds three (x, y)
        pixels, ``depths`` the three camera-space z values.
        """
        sx = np.array([[p[0] for p in screen]], dtype=float)
        sy = np.array([[p[1] for p in screen]], dtype=float)
        sz = np.array([depths], dtype=float)
        base = np.array([rgb], dtype=float)
        inten = np.array([float(intensity)], dtype=float)
        use_tex = texture is not None and uvs is not None
        uv = np.array([uvs], dtype=float) if use_tex else np.empty((0, 3, 2))
        tex = (np.asarray(texture, dtype=np.uint8) if use_tex
               else np.empty((0, 0, 3), dtype=np.uint8))
        self._submit(sx, sy, sz, base, inten, uv, tex,
                     texture is not None, filter_mode)
        self.stats["rasterized"] += 1

    def _submit(self, sx, sy, sz, base, inten, uv, tex, use_texture,
                filter_mode):
        """Rasterize one batch of triangles (same texture for all of them)."""
        if not len(sx):
            return
        if self.fog is not None:
            fog_color = np.asarray(self.fog["color"], dtype=float)
            fog_near = float(self.fog["near"])
            fog_far = float(self.fog["far"])
            use_fog = True
        else:
            fog_color = np.zeros(3)
            fog_near, fog_far, use_fog = 0.0, 1.0, False

        if HAS_NUMBA:
            sx_c = np.ascontiguousarray(sx, dtype=float)
            sy_c = np.ascontiguousarray(sy, dtype=float)
            sz_c = np.ascontiguousarray(sz, dtype=float)
            base_c = np.ascontiguousarray(base, dtype=float)
            inten_c = np.ascontiguousarray(inten, dtype=float)
            uv_c = np.ascontiguousarray(uv, dtype=float)
            rasterize_batch(self.colorbuf, self.depthbuf, sx_c, sy_c, sz_c,
                            base_c, inten_c, uv_c, tex, bool(use_texture),
                            filter_mode == "bilinear", use_fog, fog_color,
                            fog_near, fog_far)
        else:
            for t in range(len(sx)):
                self._rasterize_one_numpy(
                    sx[t], sy[t], sz[t], base[t], inten[t],
                    uv[t] if use_texture else None, tex, filter_mode,
                    fog_color if use_fog else None, fog_near, fog_far)

        if self.wireframe:
            line = np.array(WIRE_COLOR, dtype=np.uint8)
            if HAS_NUMBA:
                draw_edges_batch(self.colorbuf, sx, sy, line)
            else:
                self._draw_edges_numpy(sx, sy, line)

    def _rasterize_one_numpy(self, sx, sy, sz, base, inten, uv, tex,
                             filter_mode, fog_color, fog_near, fog_far):
        """Pure NumPy fallback for one triangle (no Numba installed)."""
        a = (sx[0], sy[0])
        b = (sx[1], sy[1])
        c = (sx[2], sy[2])
        za, zb, zc = sz
        H, W = self.depthbuf.shape
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            return

        xs = np.arange(x0, x1 + 1, dtype=float) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=float) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        area = _edge(a, b, c)
        if abs(area) < _EPS:
            return
        wa = ((gx - b[0]) * (c[1] - b[1])
              - (gy - b[1]) * (c[0] - b[0])) / area
        wb = ((gx - c[0]) * (a[1] - c[1])
              - (gy - c[1]) * (a[0] - c[0])) / area
        wc = 1.0 - wa - wb

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

        if uv is not None:
            u = (wa * uv[0][0] / za + wb * uv[1][0] / zb
                 + wc * uv[2][0] / zc) / invz
            v = (wa * uv[0][1] / za + wb * uv[1][1] / zb
                 + wc * uv[2][1] / zc) / invz
            u = np.mod(u, 1.0)
            v = np.mod(v, 1.0)
            th, tw, _ = tex.shape
            if filter_mode == "bilinear":
                final = np.zeros(gx.shape + (3,), dtype=float)
                hit_y, hit_x = np.where(mask)
                for yy, xx in zip(hit_y, hit_x):
                    texel = sample_bilinear(tex, u[yy, xx], v[yy, xx])
                    final[yy, xx] = texel * inten
            else:
                tx = np.minimum((u * tw).astype(int), tw - 1)
                ty = np.minimum((v * th).astype(int), th - 1)
                final = tex[ty, tx].astype(float) * inten
            final *= np.asarray(base, dtype=float) / 255.0
        else:
            final = np.broadcast_to(np.asarray(base, dtype=float) * inten,
                                    gx.shape + (3,))

        if fog_color is not None:
            ft = np.clip((z - fog_near) / (fog_far - fog_near), 0.0, 1.0)
            final = final * (1.0 - ft[..., None]) + fog_color * ft[..., None]

        sub_depth[mask] = z[mask]
        sub_color = self.colorbuf[y0:y1 + 1, x0:x1 + 1]
        sub_color[mask] = np.clip(final[mask], 0, 255).astype(np.uint8)

    def _draw_edges_numpy(self, sx, sy, rgb):
        H, W, _ = self.colorbuf.shape
        for t in range(len(sx)):
            for e in range(3):
                i, j = e, (e + 1) % 3
                x0, y0 = sx[t, i], sy[t, i]
                x1, y1 = sx[t, j], sy[t, j]
                steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
                for s in range(steps + 1):
                    f = s / steps
                    x = int(x0 + (x1 - x0) * f)
                    y = int(y0 + (y1 - y0) * f)
                    if 0 <= x < W and 0 <= y < H:
                        self.colorbuf[y, x] = rgb

    # ------------------------------------------------------------------
    # Mesh drawing
    # ------------------------------------------------------------------

    def draw_mesh(self, camera, mesh, model_matrix, material,
                  light_dir=DEFAULT_LIGHT_DIR):
        """Transform, light, project and rasterize every triangle of a mesh."""
        if not len(mesh.triangles):
            return
        self._prepare(camera)
        light = (_LIGHT_NORM if light_dir is DEFAULT_LIGHT_DIR
                 else normalized(light_dir))

        # Model -> world (handbook 5.2). The shared identity matrix used by
        # every static mesh skips the matrix product entirely.
        if self._last_model is model_matrix:
            is_identity = self._last_is_identity
        else:
            is_identity = np.allclose(model_matrix, np.eye(4))
            self._last_model = model_matrix
            self._last_is_identity = is_identity
        if is_identity:
            world = np.asarray(mesh.vertices, dtype=float)
        else:
            M = np.asarray(model_matrix, dtype=float)
            world = np.asarray(mesh.vertices, dtype=float) @ M[:3, :3].T \
                + M[:3, 3]

        # World -> camera space in one matrix product against the camera basis.
        basis = np.array([camera._right, camera._up, camera._forward],
                         dtype=float)
        cam = (world - camera.eye) @ basis.T

        tri = np.asarray(mesh.triangles, dtype=int)
        C = cam[tri]                # (T, 3, 3) camera-space corners
        self.stats["submitted"] += len(tri)

        # Static meshes (identity model, constant light) keep their world
        # corners, face normals, lighting intensities and per-triangle UVs in
        # a cache; re-deriving them every frame cost about 10 ms of the frame
        # at 1280x800. Animated meshes compute them per frame as before.
        use_cache = is_identity and light is _LIGHT_NORM
        if use_cache:
            prep = getattr(mesh, "_static_prep", None)
            if prep is None or prep["triangles"] is not tri:
                P = world[tri]
                e1 = P[:, 1] - P[:, 0]
                e2 = P[:, 2] - P[:, 0]
                n = np.cross(e1, e2)
                length = np.linalg.norm(n, axis=1)
                safe = length > 1e-12
                n[safe] /= length[safe, None]
                n[~safe] = (0.0, 1.0, 0.0)
                nd = n @ light
                prep_inten = AMBIENT + DIFFUSE * np.maximum(0.0, nd)
                prep_uv = (np.asarray(mesh.uvs, dtype=float)[tri]
                           if mesh.uvs is not None else None)
                prep = {"triangles": tri, "P": P, "normals": n,
                        "inten": prep_inten, "uv": prep_uv}
                mesh._static_prep = prep
            P = prep["P"]
            n = prep["normals"]
            inten = prep["inten"]
            uv_tri = prep["uv"]
        else:
            P = world[tri]          # (T, 3, 3) world-space triangle corners
            e1 = P[:, 1] - P[:, 0]
            e2 = P[:, 2] - P[:, 0]
            n = np.cross(e1, e2)
            length = np.linalg.norm(n, axis=1)
            safe = length > 1e-12
            n[safe] /= length[safe, None]
            n[~safe] = (0.0, 1.0, 0.0)
            nd = n @ light
            inten = AMBIENT + DIFFUSE * np.maximum(0.0, nd)
            uv_tri = (np.asarray(mesh.uvs, dtype=float)[tri]
                      if mesh.uvs is not None else None)
        base = np.array(material.base_color, dtype=float)

        depth = C[:, :, 2]
        dmin = depth.min(axis=1)
        dmax = depth.max(axis=1)
        behind = dmax <= self._near
        crossing = (dmin <= self._near) & ~behind

        keep = ~behind
        if self.backface_cull:
            # View vector from each triangle corner to the eye; a front face
            # has its outward normal pointing toward the camera (8.4).
            view = camera.eye[None, :] - P[:, 0, :]
            keep &= np.einsum("ij,ij->i", n, view) > 0.0
        self.stats["culled"] += int(np.count_nonzero(~keep & ~behind))

        solid = keep & ~crossing
        self.stats["clipped"] += int(np.count_nonzero(keep & crossing))

        if solid.any():
            sx, sy = self._project_arrays(C[solid].reshape(-1, 3))
            tri_count = int(np.count_nonzero(solid))
            sx = sx.reshape(tri_count, 3)
            sy = sy.reshape(tri_count, 3)
            sz = depth[solid]
            uvs = (uv_tri[solid]
                   if (material.texture is not None and uv_tri is not None)
                   else np.empty((0, 3, 2)))
            tex = (np.asarray(material.texture, dtype=np.uint8)
                   if len(uvs) else np.empty((0, 0, 3), dtype=np.uint8))
            self._submit(sx, sy, sz, np.repeat(base[None, :], tri_count, 0),
                         inten[solid], uvs, tex, len(uvs) > 0,
                         material.filter_mode)
            self.stats["rasterized"] += tri_count

        if crossing.any():
            self._draw_clipped(camera, mesh, tri, C, crossing, material,
                               inten, base)

    def _draw_clipped(self, camera, mesh, tri, C, crossing, material, inten,
                      base):
        """Near-plane path: clip each crossing triangle, then rasterize.

        The clipped fan triangles are collected and submitted as one batch so
        this path costs a single rasterizer call instead of one per triangle.
        Projection is inlined here (no per-vertex method calls) because this
        path runs for every triangle that straddles the near plane and was a
        measurable share of the follow-camera frame time.
        """
        tri_idx = np.nonzero(crossing)[0]
        has_uv = material.texture is not None and mesh.uvs is not None
        near = self._near
        focal = self._focal
        ppu = self._ppu
        ortho = self._ortho
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        sx_list, sy_list, depth_list, uv_list, inten_list = [], [], [], [], []
        mesh_uvs = mesh.uvs
        for idx in tri_idx:
            cam_tri = (tuple(C[idx, 0]), tuple(C[idx, 1]), tuple(C[idx, 2]))
            attrs = None
            if has_uv:
                attrs = (tuple(mesh_uvs[tri[idx, 0]]),
                         tuple(mesh_uvs[tri[idx, 1]]),
                         tuple(mesh_uvs[tri[idx, 2]]))
            poly, poly_attrs = clip_polygon_near(cam_tri, near, attrs)
            if poly is None:
                continue
            # Fan-triangulate the clipped polygon and project its vertices.
            for k in range(1, len(poly) - 1):
                tri_pts = (poly[0], poly[k], poly[k + 1])
                xs, ys, zs, ok = [], [], [], True
                for (x, y, z) in tri_pts:
                    if z < near - 1e-9:
                        ok = False
                        break
                    if ortho:
                        xs.append(half_w + ppu * x)
                        ys.append(half_h - ppu * y)
                    else:
                        inv = 1.0 / z
                        xs.append(half_w + focal * x * inv)
                        ys.append(half_h - focal * y * inv)
                    zs.append(z)
                if not ok:
                    continue
                sx_list.append(xs)
                sy_list.append(ys)
                depth_list.append(zs)
                inten_list.append(inten[idx])
                if poly_attrs is not None:
                    uv_list.append([poly_attrs[0], poly_attrs[k],
                                    poly_attrs[k + 1]])
        if not sx_list:
            return
        count = len(sx_list)
        uvs = (np.array(uv_list, dtype=float) if uv_list
               else np.empty((0, 3, 2)))
        tex = (np.asarray(material.texture, dtype=np.uint8) if len(uvs)
               else np.empty((0, 0, 3), dtype=np.uint8))
        self._submit(np.array(sx_list, dtype=float),
                     np.array(sy_list, dtype=float),
                     np.array(depth_list, dtype=float),
                     np.repeat(base[None, :], count, 0),
                     np.array(inten_list, dtype=float), uvs, tex, len(uvs) > 0,
                     material.filter_mode)
        self.stats["rasterized"] += count

    # ------------------------------------------------------------------
    # Framebuffer output
    # ------------------------------------------------------------------

    def surface(self):
        """Return the HxWx3 uint8 framebuffer for blitting."""
        return self.colorbuf

    def depth_image(self):
        """Banded false-colour depth buffer view for F3 (14.3).

        Depth is shown on a logarithmic scale (near depths get more of the
        ramp) and quantised into 16 bands with a thin dark edge, so surfaces
        that are close in depth are still distinguishable.
        """
        depth = self.depthbuf
        finite = np.isfinite(depth)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not finite.any():
            return img
        dmin = float(depth[finite].min())
        dmax = float(np.percentile(depth[finite], 99.9))
        dmax = max(dmax, dmin + 1e-3)
        safe = np.where(finite, np.maximum(depth, dmin), dmax)
        span = math.log(dmax) - math.log(max(dmin, 1e-3))
        t = (np.log(safe) - math.log(max(dmin, 1e-3))) / max(span, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        bands = 14.0
        q = np.floor(t * bands) / bands
        frac = t * bands - np.floor(t * bands)
        stops = np.array([[30, 80, 200], [40, 200, 150],
                          [250, 200, 60], [240, 90, 40]], dtype=float)
        pos = q * 3.0
        idx = np.clip(pos.astype(int), 0, 2)
        blend = np.clip(pos - idx, 0.0, 1.0)[..., None]
        c0 = stops[idx]
        c1 = stops[np.clip(idx + 1, 0, 3)]
        color = c0 + (c1 - c0) * blend
        edge = (frac < 0.12)[..., None]
        color = np.where(edge, color * 0.45, color)
        color[~finite] = 0.0
        return np.clip(color, 0, 255).astype(np.uint8)

