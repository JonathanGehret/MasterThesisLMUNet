"""Enumerate ALL landscapes that produce the same moving-window summary maps.

Exhaustive depth-first search. Pixels are filled in a corner-growing order so
that a 5x5 window can be checked as soon as its last pixel is set. The first
class seen is called 0, the next new one 1, and so on, so that pure global
relabelings (5! = 120 of them) are counted once.

For every solution we also record its "local partition": for every pair of
pixels that share at least one window, whether they are the same class. Two
solutions with the same local partition differ only in labels of patches that
are far apart; different local partitions are different spatial layouts.
"""
import sys
import time

import numpy as np

from window_stats import window_summary, summary_map

WIN = 5
HW = WIN // 2


def canonical(ls):
    m, out = {}, np.empty_like(ls)
    for idx, v in enumerate(ls.flat):
        if v not in m:
            m[v] = len(m)
        out.flat[idx] = m[v]
    return out


def local_partition(ls):
    H, W = ls.shape
    key = []
    for di in range(0, WIN):
        for dj in range(-WIN + 1, WIN):
            if di == 0 and dj <= 0:
                continue
            c0, c1 = max(0, -dj), W - max(0, dj)
            a = ls[0:H - di, c0:c1]
            b = ls[di:H, c0 + dj:c1 + dj]
            key.append((a == b).tobytes())
    return tuple(key)


def corner_order(H, W):
    order = []
    for k in range(max(H, W)):
        if k < H:
            order += [(k, j) for j in range(min(k, W))]
        if k < W:
            order += [(i, k) for i in range(min(k + 1, H))]
    return order


def enumerate_preimages(target, H, W, n_classes=5, max_solutions=100000, time_limit=600,
                        fn=window_summary):
    order = corner_order(H, W)
    pos = {p: t for t, p in enumerate(order)}
    checks = [[] for _ in order]
    for (i, j) in target:
        cells = [(r, c) for r in range(i - HW, i + HW + 1) for c in range(j - HW, j + HW + 1)
                 if 0 <= r < H and 0 <= c < W]
        checks[max(pos[p] for p in cells)].append((i, j))

    ls = np.full((H, W), -1, dtype=np.int8)
    pad = np.full((H + 2 * HW, W + 2 * HW), -1, dtype=np.int8)
    sols = []
    t0 = time.time()
    stats = {"nodes": 0, "timeout": False}

    def ok(t):
        for (i, j) in checks[t]:
            sub = np.ascontiguousarray(pad[i:i + WIN, j:j + WIN])
            if fn(sub.tobytes(), WIN, WIN) != target[(i, j)]:
                return False
        return True

    def dfs(t, used):
        if len(sols) >= max_solutions or stats["timeout"]:
            return
        if t == len(order):
            sols.append(ls.copy())
            return
        stats["nodes"] += 1
        if stats["nodes"] % 20000 == 0 and time.time() - t0 > time_limit:
            stats["timeout"] = True
            return
        r, c = order[t]
        for v in range(min(used + 1, n_classes)):
            ls[r, c] = v
            pad[r + HW, c + HW] = v
            if ok(t):
                dfs(t + 1, max(used, v + 1))
        ls[r, c] = -1
        pad[r + HW, c + HW] = -1

    sys.setrecursionlimit(10000)
    dfs(0, 0)
    stats["seconds"] = round(time.time() - t0, 1)
    return sols, stats
