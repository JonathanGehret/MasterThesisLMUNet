"""Worst case: pure random noise (every pixel an independent class)."""
import json, sys
import numpy as np
from window_stats import summary_map
from lsm_metrics import metric_summary
from enumerate_preimages import enumerate_preimages, canonical, local_partition

size, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
for i in range(n):
    ls = canonical(rng.integers(0, k, (size, size)).astype(np.int8))
    t = summary_map(ls, fn=metric_summary)
    sols, st = enumerate_preimages(t, size, size, time_limit=300, fn=metric_summary)
    print(json.dumps(dict(noise_classes=k, size=size, solutions=len(sols),
          local_partitions=len({local_partition(s) for s in sols}),
          max_pixels_changed=max(int((canonical(s) != ls).sum()) for s in sols) if sols else None, **st)), flush=True)
