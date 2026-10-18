"""Sanity check: with only a few metrics (np, pr, shdi, contag) the search
should find many different layouts, showing it can detect ambiguity."""
import json
from functools import lru_cache
import numpy as np
from lsm_metrics import compute_metrics, _r
from window_stats import summary_map
from enumerate_preimages import enumerate_preimages, canonical, local_partition
from run_experiment import random_cluster

@lru_cache(maxsize=None)
def weak(b, h, w):
    d = compute_metrics(np.frombuffer(b, dtype=np.int8).reshape(h, w))
    return tuple(_r(d[k]) for k in ("np", "pr", "shdi", "contag")) if d else ()

rng = np.random.default_rng(42)
for i in range(3):
    ls = canonical(random_cluster(8, rng.uniform(0.1, 0.6), rng.dirichlet(np.ones(5)), rng))
    sols, st = enumerate_preimages(summary_map(ls, fn=weak), 8, 8, time_limit=60, max_solutions=20000, fn=weak)
    print(json.dumps(dict(solutions=len(sols), local_partitions=len({local_partition(s) for s in sols}), **st)))
