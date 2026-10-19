"""Exact finisher: fix the last wrong pixels by exhaustive local search.

For each cluster of windows whose metrics do not match the target, a set of
pixels around it is freed and a depth-first search tries values for them
(current value first). A window is checked against the exact metric formulas
as soon as all its free pixels are set. If no solution is found, the freed
area is enlarged. A result where every window matches is verifiably correct.
"""
import time

import numpy as np
from scipy import ndimage

from repair import _vec, _ok


def _bad_map(pad, t, H, W):
    return np.array([[not _ok(_vec(pad, i, j), t[:, i, j]) for j in range(W)] for i in range(H)])


def _solve_region(pad, t, H, W, free, node_limit):
    """DFS over the free pixels. Returns True if a fully consistent fill is found."""
    free = sorted(free)
    pos = {p: n for n, p in enumerate(free)}
    wins = {(i, j) for r, c in free for i in range(max(0, r - 2), min(H, r + 3))
            for j in range(max(0, c - 2), min(W, c + 3))}
    checks = [[] for _ in free]
    for (i, j) in wins:
        last = max(pos[(r, c)] for r in range(i - 2, i + 3) for c in range(j - 2, j + 3) if (r, c) in pos)
        checks[last].append((i, j))
    orig = [int(pad[r + 2, c + 2]) for r, c in free]
    nodes = [0]

    def rec(n):
        if n == len(free):
            return True
        nodes[0] += 1
        if nodes[0] > node_limit:
            return None
        r, c = free[n]
        for v in [orig[n]] + [x for x in range(5) if x != orig[n]]:
            pad[r + 2, c + 2] = v
            if all(_ok(_vec(pad, i, j), t[:, i, j]) for i, j in checks[n]):
                res = rec(n + 1)
                if res:
                    return True
                if res is None:
                    break
        else:
            pad[r + 2, c + 2] = orig[n]
            return False
        pad[r + 2, c + 2] = orig[n]
        return None

    ok = rec(0)
    if not ok:
        for (r, c), v in zip(free, orig):
            pad[r + 2, c + 2] = v
    return bool(ok)


def _patch_moves(pad, t, H, W, bad, max_rounds=5):
    """Try giving a whole patch (8-connected) another label; keep if fewer
    windows mismatch. Fixes the typical error: a patch that should share the
    label of a nearby patch (or must not) got the wrong one."""
    for _ in range(max_rounds):
        if not bad.any():
            return bad
        ls = pad[2:H + 2, 2:W + 2]
        near_bad = ndimage.binary_dilation(bad, np.ones((5, 5)))
        improved = False
        for v in range(5):
            lab, n = ndimage.label(ls == v, np.ones((3, 3)))
            for k in range(1, n + 1):
                cells = lab == k
                if not (cells & near_bad).any():
                    continue
                touched = ndimage.binary_dilation(cells, np.ones((5, 5)))
                wins = list(zip(*np.where(touched)))
                before = sum(bad[i, j] for i, j in wins)
                if before == 0:
                    continue
                best = (before, None, None)
                rr, cc = np.where(cells)
                for nv in range(5):
                    if nv == v:
                        continue
                    pad[rr + 2, cc + 2] = nv
                    nb = {(i, j): not _ok(_vec(pad, i, j), t[:, i, j]) for i, j in wins}
                    s = sum(nb.values())
                    if s < best[0]:
                        best = (s, nv, nb)
                pad[rr + 2, cc + 2] = v if best[1] is None else best[1]
                if best[1] is not None:
                    for (i, j), b in best[2].items():
                        bad[i, j] = b
                    improved = True
                    break  # patch structure changed; relabel from scratch
            if improved:
                break
        if not improved:
            return bad
    return bad


def _solve_patch_labels(pad, t, H, W, comp, node_limit=50000, max_patches=20):
    """Exact search over the labels of whole patches near a mismatch,
    keeping their shapes. Returns True if every affected window matches."""
    ls = pad[2:H + 2, 2:W + 2].copy()
    inner = ndimage.binary_dilation(comp, np.ones((5, 5)))   # pixels in bad windows
    outer = ndimage.binary_dilation(comp, np.ones((13, 13)))  # pixels sharing a window with those
    patches = []
    for v in range(5):
        lab, n = ndimage.label(ls == v, np.ones((3, 3)))
        for k in range(1, n + 1):
            cells = lab == k
            if (cells & outer).any():
                patches.append(((cells & inner).any(), -cells.sum(), v, cells))
    patches.sort(key=lambda x: (not x[0], -x[1]))  # in bad windows first, then small first
    patches = [(0, p[2], p[3]) for p in patches]
    patches = patches[:max_patches]
    cells = [np.where(p[2]) for p in patches]
    orig = [p[1] for p in patches]
    # windows touched by each patch; check a window after its last patch is set
    touch = [set(zip(*np.where(ndimage.binary_dilation(p[2], np.ones((5, 5)))))) for p in patches]
    seen, checks = set(), [[] for _ in patches]
    for n in range(len(patches) - 1, -1, -1):
        for w in touch[n]:
            if w not in seen:
                seen.add(w)
                checks[n].append(w)
    nodes = [0]

    def rec(n):
        if n == len(patches):
            return True
        nodes[0] += 1
        if nodes[0] > node_limit:
            return None
        rr, cc = cells[n]
        for v in [orig[n]] + [x for x in range(5) if x != orig[n]]:
            pad[rr + 2, cc + 2] = v
            if all(_ok(_vec(pad, i, j), t[:, i, j]) for i, j in checks[n]):
                res = rec(n + 1)
                if res:
                    return True
                if res is None:
                    return None
        return False

    ok = rec(0)
    if not ok:
        for (rr, cc), v in zip(cells, orig):
            pad[rr + 2, cc + 2] = v
    return bool(ok)


def relabel_global(pad, t, H, W, node_limit=500000):
    """Exact search over the labels of ALL patches (shapes kept), checking each
    window with the exact metric formulas as soon as all its patches have a
    label. Current labels are tried first. Fixes label errors that chain
    across the map. Returns True if every window matches."""
    import sys
    ls = pad[2:H + 2, 2:W + 2].copy()
    pid = -np.ones((H, W), int)
    labels, n = [], 0
    for v in range(5):
        lab, k = ndimage.label(ls == v, np.ones((3, 3)))
        for j in range(1, k + 1):
            pid[lab == j] = n
            labels.append(v)
            n += 1
    # order patches by first pixel in raster order
    first = {}
    for idx, p in enumerate(pid.flat):
        first.setdefault(p, idx)
    order = sorted(range(n), key=lambda p: first[p])
    rank = {p: r for r, p in enumerate(order)}
    cells = [np.where(pid == p) for p in order]
    orig = [labels[p] for p in order]
    checks = [[] for _ in order]
    for i in range(H):
        for j in range(W):
            ps = pid[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
            checks[max(rank[p] for p in np.unique(ps))].append((i, j))
    nodes = [0]
    sys.setrecursionlimit(max(10000, 4 * n))

    def rec(k):
        if k == n:
            return True
        nodes[0] += 1
        if nodes[0] > node_limit:
            return None
        rr, cc = cells[k]
        for v in [orig[k]] + [x for x in range(5) if x != orig[k]]:
            pad[rr + 2, cc + 2] = v
            if all(_ok(_vec(pad, i, j), t[:, i, j]) for i, j in checks[k]):
                res = rec(k + 1)
                if res:
                    return True
                if res is None:
                    return None
        return False

    ok = rec(0)
    if not ok:
        for (rr, cc), v in zip(cells, orig):
            pad[rr + 2, cc + 2] = v
    return bool(ok)


def finish(ls, target, node_limit=20000, time_limit=120):
    H, W = ls.shape
    t = target.astype(np.float32)
    pad = np.full((H + 4, W + 4), -1, dtype=np.int8)
    pad[2:H + 2, 2:W + 2] = ls
    t0 = time.time()
    with np.errstate(all="ignore"):
        bad = _bad_map(pad, t, H, W)
        bad = _patch_moves(pad, t, H, W, bad, max_rounds=50)
        if bad.any() and relabel_global(pad, t, H, W):
            return pad[2:H + 2, 2:W + 2].copy()
        # group bad windows that can influence each other (within 4 px)
        lab, n = ndimage.label(ndimage.binary_dilation(bad, iterations=2), np.ones((3, 3)))
        for k in range(1, n + 1):
            comp = (lab == k) & bad
            if not comp.any():
                continue
            if _solve_patch_labels(pad, t, H, W, comp):
                continue
            # how many bad windows cover each pixel
            cover = ndimage.convolve(comp.astype(int), np.ones((5, 5), int), mode="constant")
            order = np.argsort(-cover, axis=None)
            cand = [divmod(int(x), W) for x in order if cover.flat[x] > 0]
            for size in (1, 2, 4, 6, 9, 12, 16, 25, 36):
                if time.time() - t0 > time_limit:
                    break
                free = cand[:size]
                if _solve_region(pad, t, H, W, free, node_limit):
                    still = [(i, j) for i, j in zip(*np.where(comp))
                             if not _ok(_vec(pad, i, j), t[:, i, j])]
                    if not still:
                        break
                if size >= len(cand):
                    break
        # pixel fixes may have made a consistent labelling possible
        bad = _bad_map(pad, t, H, W)
        if bad.any():
            relabel_global(pad, t, H, W)
    return pad[2:H + 2, 2:W + 2].copy()
