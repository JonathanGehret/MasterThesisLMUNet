"""How many landscapes give exactly the same moving-window metric maps?

Usage: python run_experiment.py SIZE N_LANDSCAPES [TIME_LIMIT_S]
"""
import json
import sys

import numpy as np
from scipy import ndimage

from window_stats import summary_map, window_summary
from lsm_metrics import metric_summary
from enumerate_preimages import enumerate_preimages, canonical, local_partition


def random_cluster(n, p, ai, rng):
    """Python version of NLMR::nlm_randomcluster (Saura & Martinez-Millan 2000)."""
    perc = rng.random((n, n)) < p
    lab, k = ndimage.label(perc)  # 4-neighbour clusters
    ls = np.full((n, n), -1, dtype=np.int8)
    cls = rng.choice(len(ai), size=k + 1, p=ai)
    ls[perc] = cls[lab[perc]]
    if (ls < 0).all():
        ls[rng.integers(n), rng.integers(n)] = rng.choice(len(ai), p=ai)
    while (ls < 0).any():  # fill the rest by majority of labelled neighbours
        new = ls.copy()
        for r, c in zip(*np.where(ls < 0)):
            nb = ls[max(0, r - 1):r + 2, max(0, c - 1):c + 2].ravel()
            nb = nb[nb >= 0]
            if len(nb):
                vals, cnt = np.unique(nb, return_counts=True)
                new[r, c] = rng.choice(vals[cnt == cnt.max()])
        ls = new
    return ls


def main():
    size, n_ls = int(sys.argv[1]), int(sys.argv[2])
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 600
    mode = sys.argv[4] if len(sys.argv) > 4 else "summary"
    fn = metric_summary if mode == "metrics" else window_summary
    rng = np.random.default_rng(42)
    results = []
    for k in range(n_ls):
        p = rng.uniform(0.1, 0.6)
        ai = rng.dirichlet(np.ones(5))
        ls = canonical(random_cluster(size, p, ai, rng))
        target = summary_map(ls, fn=fn)
        sols, st = enumerate_preimages(target, size, size, time_limit=tl, fn=fn)
        parts = {local_partition(s) for s in sols}
        n_diff_pixels = [int((canonical(s) != ls).sum()) for s in sols]
        res = dict(mode=mode, size=size, idx=k, p=round(p, 2), n_classes=int(ls.max() + 1),
                   n_patches_8=int(sum(ndimage.label(ls == v, np.ones((3, 3)))[1]
                                       for v in range(ls.max() + 1))),
                   solutions=len(sols), local_partitions=len(parts),
                   max_pixels_changed=max(n_diff_pixels) if sols else None, **st)
        results.append(res)
        print(json.dumps(res), flush=True)
        if len(parts) > 1:
            np.save(f"out/ambiguous_{mode}_{size}_{k}.npy", np.stack(sols[:50]))
            np.save(f"out/original_{mode}_{size}_{k}.npy", ls)


if __name__ == "__main__":
    main()
