"""Evaluate the inverse on 128x128 landscapes.

Usage: python eval128.py INVERSE.pt N [FORWARD.pt]

exact input:     metric maps computed with the exact formulas (as R would)
predicted input: metric maps predicted by the forward model (like the thesis
                 U-Net). The result is always checked against the exact maps.
Repair + finisher need exact target metrics, so they are only run on exact input.
"""
import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from inverse_model import InverseNet
from decode import decode_avg
from evaluate import metric_match
from repair import repair
from finisher import finish
from enumerate_preimages import local_partition


def score(x, ls, m):
    w, c = metric_match(x, m)
    return dict(exact_layout=local_partition(x) == local_partition(ls),
                windows_all_metrics_exact=w, metric_values_exact=c)


FAST = False


def work(a):
    ls, m, p_exact, p_pred = a
    out = {}
    rec = decode_avg(p_exact)[0][0]
    out["exact_input/network"] = score(rec, ls, m)
    if FAST:
        if p_pred is not None:
            out["predicted_input/network"] = score(decode_avg(p_pred)[0][0], ls, m)
        return out
    rep = repair(rec, m)
    out["exact_input/+repair"] = score(rep, ls, m)
    fin = finish(rep, m, time_limit=600)
    out["exact_input/+repair+finisher"] = score(fin, ls, m)
    if p_pred is not None:
        out["predicted_input/network"] = score(decode_avg(p_pred)[0][0], ls, m)
    return out


if __name__ == "__main__":
    inv_path, n = sys.argv[1], int(sys.argv[2])
    fwd_path = sys.argv[3] if len(sys.argv) > 3 else None
    FAST = len(sys.argv) > 4 and sys.argv[4] == "fast"
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    inv = InverseNet(); inv.load_state_dict(torch.load(inv_path)); inv.eval()
    d = np.load("data/test128.npz")
    L, M = d["landscapes"][:n], d["metrics"][:n]
    with torch.no_grad():
        Pe = [torch.sigmoid(inv(torch.from_numpy(norm(M[i:i + 1]))))[0].numpy() for i in range(n)]
        Pp = [None] * n
        if fwd_path:
            from forward import ForwardNet, predict_inverse_input
            fwd = ForwardNet(); fwd.load_state_dict(torch.load(fwd_path)); fwd.eval()
            Pp = [torch.sigmoid(inv(predict_inverse_input(fwd, L[i:i + 1])))[0].numpy() for i in range(n)]
    torch.set_num_threads(1)
    res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(work, zip(L, M, Pe, Pp))):
            res.append(r)
            print(f"{i + 1}/{n}", json.dumps(r), flush=True)
    agg = {k: {m: float(np.mean([r[k][m] for r in res])) for m in res[0][k]} for k in res[0]}
    print("SUMMARY", json.dumps(agg), flush=True)
