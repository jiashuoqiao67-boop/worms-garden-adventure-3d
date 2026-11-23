"""Optional Numba JIT rasterization path (handbook p.4 / p.30).

Numba JIT is an OPTIONAL CPU optimization, not a prerequisite for Part A.
If Numba is installed, the per-fragment scan below is compiled to machine
code; if it is missing or fails to import, ``HAS_NUMBA`` is False and
``renderer.py`` uses its pure Python/NumPy path unchanged.

The JIT function only does plain scalar/array arithmetic -- no Pygame, no
dictionaries, no object methods -- so it stays compatible with ``@njit``.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba is absent
    njit = None
    HAS_NUMBA = False

_EPS = 1e-9


if HAS_NUMBA:

    @njit(cache=True)
    def rasterize_njit(colorbuf, depthbuf, sx, sy, sz, base_rgb, tex, uvs,
                       intensity, use_texture, bilinear, use_fog,
                       fog_color, fog_near, fog_far):
        """Rasterize one projected triangle into the frame/depth buffers.

        Arrays:
            colorbuf  : H x W x 3 uint8 (written in place)
            depthbuf  : H x W float64 (written in place)
            sx, sy, sz: length-3 projected pixel x/y and camera-space z
            base_rgb  : length-3 float, flat colour when no texture
            tex       : th x tw x 3 uint8 texture, or empty (0,0,3)
            uvs       : 3 x 2 float UVs, or empty (0,2)
        """
        H, W = depthbuf.shape

        minx = min(sx[0], sx[1], sx[2])
        maxx = max(sx[0], sx[1], sx[2])
        miny = min(sy[0], sy[1], sy[2])
        maxy = max(sy[0], sy[1], sy[2])
        x0 = max(0, int(np.floor(minx)))
        x1 = min(W - 1, int(np.ceil(maxx)))
        y0 = max(0, int(np.floor(miny)))
        y1 = min(H - 1, int(np.ceil(maxy)))
        if x1 < x0 or y1 < y0:
            return

        area = ((sx[2] - sx[0]) * (sy[1] - sy[0])
                - (sy[2] - sy[0]) * (sx[1] - sx[0]))
        if abs(area) < _EPS:
            return
        inv_area = 1.0 / area

        th = tex.shape[0]
        tw = tex.shape[1]

        for y in range(y0, y1 + 1):
            gy = y + 0.5
            for x in range(x0, x1 + 1):
                gx = x + 0.5
                wa = ((gx - sx[1]) * (sy[2] - sy[1])
                      - (gy - sy[1]) * (sx[2] - sx[1])) * inv_area
                wb = ((gx - sx[2]) * (sy[0] - sy[2])
                      - (gy - sy[2]) * (sx[0] - sx[2])) * inv_area
                wc = 1.0 - wa - wb
                if wa < -_EPS or wb < -_EPS or wc < -_EPS:
                    continue

                invz = wa / sz[0] + wb / sz[1] + wc / sz[2]
                if invz <= _EPS:
                    continue
                z = 1.0 / invz
                if z >= depthbuf[y, x]:
                    continue

                if use_texture:
                    u = (wa * uvs[0, 0] / sz[0] + wb * uvs[1, 0] / sz[1]
                         + wc * uvs[2, 0] / sz[2]) / invz
                    v = (wa * uvs[0, 1] / sz[0] + wb * uvs[1, 1] / sz[1]
                         + wc * uvs[2, 1] / sz[2]) / invz
                    u = u % 1.0
                    v = v % 1.0

                    if bilinear:
                        xq = u * tw - 0.5
                        yq = v * th - 0.5
                        xi = int(np.floor(xq))
                        yi = int(np.floor(yq))
                        fx = xq - xi
                        fy = yq - yi
                        xa = xi % tw
                        xb = (xi + 1) % tw
                        ya = yi % th
                        yb = (yi + 1) % th
                        c00 = tex[ya, xa]
                        c10 = tex[ya, xb]
                        c01 = tex[yb, xa]
                        c11 = tex[yb, xb]
                        top_r = c00[0] * (1.0 - fx) + c10[0] * fx
                        top_g = c00[1] * (1.0 - fx) + c10[1] * fx
                        top_b = c00[2] * (1.0 - fx) + c10[2] * fx
                        bot_r = c01[0] * (1.0 - fx) + c11[0] * fx
                        bot_g = c01[1] * (1.0 - fx) + c11[1] * fx
                        bot_b = c01[2] * (1.0 - fx) + c11[2] * fx
                        r = (top_r * (1.0 - fy) + bot_r * fy) * intensity
                        g = (top_g * (1.0 - fy) + bot_g * fy) * intensity
                        b = (top_b * (1.0 - fy) + bot_b * fy) * intensity
                    else:
                        tx = min(int(u * tw), tw - 1)
                        ty = min(int(v * th), th - 1)
                        t = tex[ty, tx]
                        r = t[0] * intensity
                        g = t[1] * intensity
                        b = t[2] * intensity
                else:
                    r = base_rgb[0]
                    g = base_rgb[1]
                    b = base_rgb[2]

                if use_fog:
                    ft = (z - fog_near) / (fog_far - fog_near)
                    if ft < 0.0:
                        ft = 0.0
                    if ft > 1.0:
                        ft = 1.0
                    omt = 1.0 - ft
                    r = r * omt + fog_color[0] * ft
                    g = g * omt + fog_color[1] * ft
                    b = b * omt + fog_color[2] * ft

                depthbuf[y, x] = z
                colorbuf[y, x, 0] = min(255, max(0, r))
                colorbuf[y, x, 1] = min(255, max(0, g))
                colorbuf[y, x, 2] = min(255, max(0, b))

else:
    # Numba absent: keep the name importable as None. renderer.py gates the
    # JIT call on HAS_NUMBA and falls back to its pure Python/NumPy path.
    rasterize_njit = None
