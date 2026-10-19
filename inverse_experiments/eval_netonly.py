"""Network-only evaluation (no repair, no finisher): inverse network + decoding.

Usage: python eval_netonly.py INVERSE.pt FORWARD.pt TEST.npz N
Input is either the exact metric maps or the forward model's predictions;
the result is always checked against the exact maps.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

from inverse_model import InverseNet, Normalizer  # noqa: F401
from forward import load_forward, predict_inverse_input
from decode import decode_avg
from evaluate import metric_match
from enumerate_preimages import local_partition


def work(a):
    ls, m, pe, pp = a
    out = {}
    for name, p in (("exact_input", pe), ("predicted_input", pp)):
        t = time.time()
        x = decode_avg(p)[0][0]
        dt = time.time() - t
        w, c = metric_match(x, m)
        out[name] = dict(exact_layout=local_partition(x) == local_partition(ls),
                         windows_all_metrics_exact=w, decode_seconds=dt)
    return out


if __name__ == "__main__":
    inv_path, fwd_path, test_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    inv = InverseNet(); inv.load_state_dict(torch.load(inv_path)); inv.eval()
    fwd = load_forward(fwd_path)
    d = np.load(test_path); L, M = d["landscapes"][:n], d["metrics"][:n]
    Pe, Pp, tn = [], [], 0.0
    with torch.no_grad():
        for i in range(n):
            t = time.time()
            Pe.append(torch.sigmoid(inv(torch.from_numpy(norm(M[i:i + 1]))))[0].numpy())
            tn += time.time() - t
            Pp.append(torch.sigmoid(inv(predict_inverse_input(fwd, L[i:i + 1])))[0].numpy())
    torch.set_num_threads(1)
    with Pool(4) as pool:
        res = pool.map(work, zip(L, M, Pe, Pp))
    agg = {k: {m: float(np.mean([r[k][m] for r in res])) for m in res[0][k]} for k in res[0]}
    agg.update(test=test_path, n=n, inverse=inv_path, forward=fwd_path,
               network_seconds_per_map=tn / n)
    print(json.dumps(agg), flush=True)
