"""Repair a decoded landscape with the exact metric formulas.

Greedy local search: for pixels near windows whose metrics do not match the
target, try every class and keep the change if fewer windows mismatch.
Only the 25 windows that contain a pixel need recomputing.
"""
import numpy as np

from lsm_metrics import METRICS, compute_metrics


from functools import lru_cache


@lru_cache(maxsize=1_000_000)
def _vec_cached(b):
    d = compute_metrics(np.frombuffer(b, dtype=np.int8).reshape(5, 5))
    v = np.array([d[k] for k in METRICS], dtype=np.float64)
    v[np.isinf(v)] = np.nan
    return np.clip(v, -6e4, 6e4).astype(np.float16).astype(np.float32)


def _vec(pad, i, j):
    return _vec_cached(np.ascontiguousarray(pad[i:i + 5, j:j + 5]).tobytes())


def _ok(v, t):
    return bool((np.isclose(v, t, rtol=1e-3, atol=1e-3) | (np.isnan(v) & np.isnan(t))).all())


def repair(ls, target, k=5, max_rounds=4):
    """ls: (H,W) ints, target: (56,H,W) metric maps. Returns repaired landscape."""
    H, W = ls.shape
    t = target.astype(np.float32)
    pad = np.full((H + 4, W + 4), -1, dtype=np.int8)
    pad[2:H + 2, 2:W + 2] = ls
    with np.errstate(all="ignore"):
        bad = np.array([[not _ok(_vec(pad, i, j), t[:, i, j]) for j in range(W)] for i in range(H)])
        for _ in range(max_rounds):
            if not bad.any():
                break
            improved = False
            cand = {(r, c) for i, j in zip(*np.where(bad))
                    for r in range(max(0, i - 2), min(H, i + 3))
                    for c in range(max(0, j - 2), min(W, j + 3))}
            for r, c in sorted(cand):
                win = [(i, j) for i in range(max(0, r - 2), min(H, r + 3))
                       for j in range(max(0, c - 2), min(W, c + 3))]
                before = sum(bad[i, j] for i, j in win)
                if before == 0:
                    continue
                orig = pad[r + 2, c + 2]
                best, best_val, best_bad = before, orig, None
                for v in range(k):
                    if v == orig:
                        continue
                    pad[r + 2, c + 2] = v
                    nb = {(i, j): not _ok(_vec(pad, i, j), t[:, i, j]) for i, j in win}
                    s = sum(nb.values())
                    if s < best:
                        best, best_val, best_bad = s, v, nb
                pad[r + 2, c + 2] = best_val
                if best_bad is not None:
                    for (i, j), b in best_bad.items():
                        bad[i, j] = b
                    improved = True
            if not improved:
                break
    return pad[2:H + 2, 2:W + 2].copy()
