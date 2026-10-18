"""Evaluate reconstructions: recompute exact metric maps of the decoded
landscape and compare them with the target metric maps."""
import numpy as np

from make_dataset import metric_maps
from enumerate_preimages import local_partition


def metric_match(ls_rec, target_m, rtol=1e-3):
    with np.errstate(all="ignore"):
        m = metric_maps(ls_rec.astype(np.int8))
    m[np.isinf(m)] = np.nan
    m = np.clip(m, -6e4, 6e4).astype(np.float16).astype(np.float32)
    t = target_m.astype(np.float32)
    both_nan = np.isnan(m) & np.isnan(t)
    close = np.isclose(m, t, rtol=rtol, atol=1e-3) | both_nan
    window_ok = close.all(axis=0)  # all metrics right in this window
    return window_ok.mean(), close.mean()


def summarize(landscapes, targets, recs):
    rows = []
    for ls, t, rec in zip(landscapes, targets, recs):
        w, c = metric_match(rec, t)
        rows.append(dict(
            exact_layout=local_partition(rec) == local_partition(ls),
            windows_all_metrics_exact=w,
            metric_values_exact=c,
        ))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return agg, rows
