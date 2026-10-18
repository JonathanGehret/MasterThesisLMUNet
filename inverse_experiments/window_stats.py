"""Label-invariant summary of a 5x5 moving window.

Every landscape-level metric used in the thesis (the 57 lsm_l_* metrics) can be
computed from this summary. So if two landscapes have identical summary maps,
they are guaranteed to have identical metric maps as well.

The summary contains, per class (without the class number itself):
  - class area
  - number of like (same-class) rook adjacencies
  - the list of its patches (8-neighbour patches, as in landscapemetrics), each
    described by area, perimeter, core area (4- and 8-neighbour rule),
    number of disjunct cores, and smallest circumscribing circle
  - its adjacency counts to every other class (paired with that class' summary)

Cells outside the landscape (moving window at the border, NA in R) are -1.
"""
import math
from functools import lru_cache

import numpy as np

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _components(cells, nbrs):
    cells = set(cells)
    comps = []
    while cells:
        start = cells.pop()
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in nbrs:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    comp.append(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def _circle_r2(points):
    """Radius^2 of the smallest enclosing circle (Welzl, iterative)."""
    pts = sorted(set(points))

    def circ2(a, b):
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return cx, cy, (a[0] - cx) ** 2 + (a[1] - cy) ** 2

    def circ3(a, b, c):
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            cands = [circ2(a, b), circ2(a, c), circ2(b, c)]
            return max(cands, key=lambda t: t[2])
        ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1])
              + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
        uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0])
              + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
        return ux, uy, (a[0] - ux) ** 2 + (a[1] - uy) ** 2

    def inside(cc, p):
        return (p[0] - cc[0]) ** 2 + (p[1] - cc[1]) ** 2 <= cc[2] + 1e-9

    cc = (pts[0][0], pts[0][1], 0.0)
    for i, p in enumerate(pts):
        if inside(cc, p):
            continue
        cc = (p[0], p[1], 0.0)
        for j in range(i):
            q = pts[j]
            if inside(cc, q):
                continue
            cc = circ2(p, q)
            for k in range(j):
                s = pts[k]
                if not inside(cc, s):
                    cc = circ3(p, q, s)
    return round(cc[2], 6)


def _patch_desc(patch, cls_of):
    pset = set(patch)
    area = len(patch)
    perim = 0
    for r, c in patch:
        for dr, dc in N4:
            if (r + dr, c + dc) not in pset:
                perim += 1  # edge to other class or to outside: both count
    # core cells: every in-landscape neighbour is the same class
    # (landscape boundary does not count as edge, consider_boundary = FALSE)
    def core(nbrs):
        out = []
        for r, c in patch:
            ok = True
            for dr, dc in nbrs:
                q = (r + dr, c + dc)
                v = cls_of.get(q, -1)
                if v != -1 and q not in pset:
                    ok = False
                    break
            if ok:
                out.append((r, c))
        return out
    c4, c8 = core(N4), core(N8)
    corners = [(r + a, c + b) for r, c in patch for a in (0, 1) for b in (0, 1)]
    return (area, perim, len(c4), len(c8),
            len(_components(c4, N4)), len(_components(c8, N8)),
            len(_components(c4, N8)), len(_components(c8, N4)),
            _circle_r2(corners))


@lru_cache(maxsize=2_000_000)
def window_summary(win_bytes, h, w):
    win = np.frombuffer(win_bytes, dtype=np.int8).reshape(h, w)
    cls_of = {(r, c): int(win[r, c]) for r in range(h) for c in range(w)}
    classes = sorted({v for v in cls_of.values() if v >= 0})
    if not classes:
        return ()
    adj = {}
    for (r, c), v in cls_of.items():
        if v < 0:
            continue
        for dr, dc in N4:
            u = cls_of.get((r + dr, c + dc), -1)
            if u >= 0:
                adj[(v, u)] = adj.get((v, u), 0) + 1  # double counted, like R
    base = {}
    for k in classes:
        cells = [p for p, v in cls_of.items() if v == k]
        patches = sorted(_patch_desc(p, cls_of) for p in _components(cells, N8))
        base[k] = (len(cells), adj.get((k, k), 0), tuple(patches))
    full = []
    for k in classes:
        row = tuple(sorted((base[j], adj.get((k, j), 0)) for j in classes if j != k))
        full.append((base[k], row))
    return tuple(sorted(full))


def summary_map(ls, win=5, fn=None):
    fn = fn or window_summary
    """Summary for the window centred on each pixel (NA outside the landscape)."""
    H, W = ls.shape
    h = win // 2
    pad = np.full((H + 2 * h, W + 2 * h), -1, dtype=np.int8)
    pad[h:h + H, h:h + W] = ls
    out = {}
    for i in range(H):
        for j in range(W):
            sub = np.ascontiguousarray(pad[i:i + win, j:j + win])
            out[(i, j)] = fn(sub.tobytes(), win, win)
    return out
