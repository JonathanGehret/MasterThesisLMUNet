"""Generate random-cluster landscapes and their 5x5 moving-window metric maps.

Usage: python make_dataset.py N SIZE SEED OUT.npz
Saves landscapes (N,H,W) int8 and metrics (N,56,H,W) float16 (NaN kept).
"""
import sys
from multiprocessing import Pool

import numpy as np

from lsm_metrics import METRICS, compute_metrics
from run_experiment import random_cluster
from enumerate_preimages import canonical


def metric_maps(ls):
    H, W = ls.shape
    pad = np.full((H + 4, W + 4), -1, dtype=np.int8)
    pad[2:H + 2, 2:W + 2] = ls
    out = np.empty((len(METRICS), H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            d = compute_metrics(pad[i:i + 5, j:j + 5])
            out[:, i, j] = [d[k] for k in METRICS]
    return out


def one(args):
    seed, size = args
    rng = np.random.default_rng(seed)
    ls = canonical(random_cluster(size, rng.uniform(0.1, 0.6), rng.dirichlet(np.ones(5)), rng))
    with np.errstate(all="ignore"):
        m = metric_maps(ls)
    m[np.isinf(m)] = np.nan
    return ls, np.clip(m, -6e4, 6e4).astype(np.float16)


if __name__ == "__main__":
    n, size, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with Pool(4) as p:
        res = p.map(one, [(seed * 1_000_000 + i, size) for i in range(n)], chunksize=16)
    np.savez(out, landscapes=np.stack([r[0] for r in res]), metrics=np.stack([r[1] for r in res]))
    print("saved", out)
