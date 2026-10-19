"""How many distinct 5x5 windows (up to relabelling) occur? If few, exact
metrics can be looked up in a table instead of computed or predicted."""
import numpy as np


def canonical_windows(L, win=5):
    """All windows of landscapes L (N,H,W) as canonical byte keys (labels renamed
    by first appearance; outside = -1)."""
    N, H, W = L.shape
    h = win // 2
    pad = np.full((N, H + 2 * h, W + 2 * h), -1, np.int8)
    pad[:, h:h + H, h:h + W] = L
    s = np.lib.stride_tricks.sliding_window_view(pad, (win, win), axis=(1, 2)).reshape(-1, win * win)
    out = np.full_like(s, -1)
    nxt = np.zeros(len(s), np.int8)
    mapping = np.full((len(s), 5), -1, np.int8)
    rows = np.arange(len(s))
    for k in range(win * win):  # vectorised relabel by first appearance
        v = s[:, k]
        valid = v >= 0
        vv = np.where(valid, v, 0)
        m = mapping[rows, vv]
        new = valid & (m < 0)
        mapping[rows[new], vv[new]] = nxt[new]
        nxt[new] += 1
        out[:, k] = np.where(valid, mapping[rows, vv], -1)
    return out


if __name__ == "__main__":
    import time
    tr = np.load("data/train32.npz")["landscapes"]
    t = time.time()
    keys = canonical_windows(tr)
    u = np.unique(keys, axis=0)
    print(f"train windows {len(keys):,}, distinct {len(u):,} ({time.time() - t:.0f}s)")
    te = np.load("data/test128.npz")["landscapes"]
    kt = canonical_windows(te)
    seen = set(map(bytes, u))
    cov = np.mean([bytes(k) in seen for k in kt])
    print(f"128x128 test windows already in table: {cov:.4f}")
