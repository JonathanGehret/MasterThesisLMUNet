"""Inverse on forward-model-predicted metrics (32x32 test set), network only.
Usage: python eval_predicted32.py INVERSE.pt N"""
import json, sys
from multiprocessing import Pool
import numpy as np, torch
from inverse_model import InverseNet
from forward import ForwardNet, predict_inverse_input
from decode import decode_avg
from evaluate import metric_match
from enumerate_preimages import local_partition


def work(a):
    ls, m, p = a
    x = decode_avg(p)[0][0]
    w, c = metric_match(x, m)
    return local_partition(x) == local_partition(ls), w, c


if __name__ == "__main__":
    inv_path, n = sys.argv[1], int(sys.argv[2])
    inv = InverseNet(); inv.load_state_dict(torch.load(inv_path)); inv.eval()
    fwd = ForwardNet(); fwd.load_state_dict(torch.load("data/forward.pt")); fwd.eval()
    d = np.load("data/test32.npz"); L, M = d["landscapes"][:n], d["metrics"][:n]
    with torch.no_grad():
        P = torch.sigmoid(inv(predict_inverse_input(fwd, L))).numpy()
    torch.set_num_threads(1)
    with Pool(4) as pool:
        r = pool.map(work, zip(L, M, P))
    print(json.dumps(dict(model=inv_path, input="forward-predicted metrics", n=n,
                          exact_layout=float(np.mean([x[0] for x in r])),
                          windows_all_metrics_exact=float(np.mean([x[1] for x in r])),
                          metric_values_exact=float(np.mean([x[2] for x in r])))))
