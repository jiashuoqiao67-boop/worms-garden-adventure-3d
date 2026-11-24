"""Optional Numba JIT rasterization path (handbook p.4 / p.30).

Numba JIT is an OPTIONAL CPU optimization, not a prerequisite for Part A.
If Numba is installed, the per-fragment scan below is compiled to machine
code; if it is missing or fails to import, ``HAS_NUMBA`` is False and
``renderer.py`` uses its pure Python/NumPy path unchanged.

The JIT functions only do plain scalar/array arithmetic -- no Pygame, no
dictionaries, no object methods -- so they stay compatible with ``@njit``.
"""

import numpy as np

try:  # pragma: no cover - depends on the environment
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba is absent
    njit = None
    HAS_NUMBA = False

_EPS = 1e-9


if HAS_NUMBA:

    @njit(cache=True)
    def rasterize_batch(colorbuf, depthbuf, sx, sy, sz, base, inten, uv, tex,
                        use_texture, bilinear, use_fog, fog_color,
                        fog_near, fog_far):
        """Rasterize a batch of projected triangles with a Z-buffer test.

        Arrays:
            colorbuf  : H x W x 3 uint8 (written in place)
            depthbuf  : H x W float64 (written in place)
            sx, sy, sz: (T, 3) projected pixel x/y and camera-space depth
            base      : (T, 3) material colour (unlit)
            inten     : (T,)   flat lighting intensity per triangle
            uv        : (T, 3, 2) texture coordinates, or empty (0, 3, 2)
            tex       : th x tw x 3 uint8 texture, or empty (0, 0, 3)
        """
        H, W, _ = colorbuf.shape
        T = sx.shape[0]
        th = tex.shape[0]
        tw = tex.shape[1]

        for t in range(T):
            a_x = sx[t, 0]
            b_x = sx[t, 1]
            c_x = sx[t, 2]
            a_y = sy[t, 0]
            b_y = sy[t, 1]
            c_y = sy[t, 2]

            minx = min(min(a_x, b_x), c_x)
            maxx = max(max(a_x, b_x), c_x)
            miny = min(min(a_y, b_y), c_y)
            maxy = max(max(a_y, b_y), c_y)

            x0 = int(np.floor(minx))
            x1 = int(np.ceil(maxx))
            y0 = int(np.floor(miny))
            y1 = int(np.ceil(maxy))
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > W - 1:
                x1 = W - 1
            if y1 > H - 1:
                y1 = H - 1
            if x1 < x0 or y1 < y0:
                continue

            # Signed area uses the same edge-function convention as
            # renderer._edge(a, b, c), so the barycentric weights are
            # positive inside the triangle for either winding.
            area = ((c_x - a_x) * (b_y - a_y)
                    - (c_y - a_y) * (b_x - a_x))
            if abs(area) < _EPS:
                continue
            inv_area = 1.0 / area

            za = sz[t, 0]
            zb = sz[t, 1]
            zc = sz[t, 2]
            iza = 1.0 / za
            izb = 1.0 / zb
            izc = 1.0 / zc
            base_r = base[t, 0] * inten[t]
            base_g = base[t, 1] * inten[t]
            base_b = base[t, 2] * inten[t]
            # The material colour also tints a textured surface (base 255
            # leaves the texture unchanged).
            tint_r = base[t, 0] / 255.0
            tint_g = base[t, 1] / 255.0
            tint_b = base[t, 2] / 255.0

            # Edge-function planes (linear in screen x, y) so the barycentric
            # weights can be advanced with one addition per pixel instead of
            # recomputing two products for every fragment.
            e0x = c_y - b_y
            e0y = b_x - c_x
            e0c = b_y * (c_x - b_x) - b_x * (c_y - b_y)
            e1x = a_y - c_y
            e1y = c_x - a_x
            e1c = c_y * (a_x - c_x) - c_x * (a_y - c_y)
            dwa = e0x * inv_area
            dwb = e1x * inv_area
            dinvz = dwa * iza + dwb * izb - (dwa + dwb) * izc
            if use_texture:
                iua = uv[t, 0, 0] * iza
                iub = uv[t, 1, 0] * izb
                iuc = uv[t, 2, 0] * izc
                iva = uv[t, 0, 1] * iza
                ivb = uv[t, 1, 1] * izb
                ivc = uv[t, 2, 1] * izc
                dun = dwa * iua + dwb * iub - (dwa + dwb) * iuc
                dvn = dwa * iva + dwb * ivb - (dwa + dwb) * ivc
            else:
                iua = iub = iuc = iva = ivb = ivc = 0.0
                dun = dvn = 0.0

            for y in range(y0, y1 + 1):
                gy = y + 0.5
                gx = x0 + 0.5
                wa = (e0x * gx + e0y * gy + e0c) * inv_area
                wb = (e1x * gx + e1y * gy + e1c) * inv_area
                wc = 1.0 - wa - wb
                invz = wa * iza + wb * izb + wc * izc
                un = wa * iua + wb * iub + wc * iuc
                vn = wa * iva + wb * ivb + wc * ivc

                for x in range(x0, x1 + 1):
                    if wa < -_EPS or wb < -_EPS or wc < -_EPS:
                        wa += dwa
                        wb += dwb
                        wc -= (dwa + dwb)
                        invz += dinvz
                        un += dun
                        vn += dvn
                        continue

                    if invz <= _EPS:
                        z = np.inf
                    else:
                        z = 1.0 / invz
                    if z < depthbuf[y, x]:
                        r = base_r
                        g = base_g
                        b = base_b
                        if use_texture:
                            u = un * z
                            v = vn * z
                            # Wrap with a subtraction instead of a float
                            # modulo: u, v are non-negative here.
                            iu = int(u)
                            u = u - iu
                            iv = int(v)
                            v = v - iv
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
                                r = (top_r * (1.0 - fy) + bot_r * fy) \
                                    * inten[t] * tint_r
                                g = (top_g * (1.0 - fy) + bot_g * fy) \
                                    * inten[t] * tint_g
                                b = (top_b * (1.0 - fy) + bot_b * fy) \
                                    * inten[t] * tint_b
                            else:
                                tx = int(u * tw)
                                ty = int(v * th)
                                if tx > tw - 1:
                                    tx = tw - 1
                                if ty > th - 1:
                                    ty = th - 1
                                t_col = tex[ty, tx]
                                r = t_col[0] * inten[t] * tint_r
                                g = t_col[1] * inten[t] * tint_g
                                b = t_col[2] * inten[t] * tint_b

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
                        if r < 0.0:
                            r = 0.0
                        if g < 0.0:
                            g = 0.0
                        if b < 0.0:
                            b = 0.0
                        if r > 255.0:
                            r = 255.0
                        if g > 255.0:
                            g = 255.0
                        if b > 255.0:
                            b = 255.0
                        colorbuf[y, x, 0] = r
                        colorbuf[y, x, 1] = g
                        colorbuf[y, x, 2] = b

                    wa += dwa
                    wb += dwb
                    wc -= (dwa + dwb)
                    invz += dinvz
                    un += dun
                    vn += dvn

    @njit(cache=True)
    def draw_edges_batch(colorbuf, sx, sy, rgb):
        """Draw the three edges of every projected triangle (F1 wireframe)."""
        H, W, _ = colorbuf.shape
        T = sx.shape[0]
        for t in range(T):
            for e in range(3):
                i = e
                j = e + 1
                if j == 3:
                    j = 0
                x0 = sx[t, i]
                y0 = sy[t, i]
                x1 = sx[t, j]
                y1 = sy[t, j]
                dx = x1 - x0
                dy = y1 - y0
                steps = int(max(abs(dx), abs(dy))) + 1
                if steps > 4000:
                    steps = 4000
                for s in range(steps + 1):
                    f = s / steps
                    x = int(x0 + dx * f)
                    y = int(y0 + dy * f)
                    if 0 <= x < W and 0 <= y < H:
                        colorbuf[y, x, 0] = rgb[0]
                        colorbuf[y, x, 1] = rgb[1]
                        colorbuf[y, x, 2] = rgb[2]

else:  # pragma: no cover - exercised only without numba
    rasterize_batch = None
    draw_edges_batch = None

