"""Network-only comparison of the two decoders. Usage: python compare_decoders.py MODEL TEST N"""
import json, pickle, sys
from multiprocessing import Pool
import numpy as np, torch
from inverse_model import InverseNet
from decode import decode, decode_avg
from evaluate import metric_match
from enumerate_preimages import local_partition


def work(a):
    ls, m, p = a
    out = {}
    for name, f in (("union", decode), ("avg_linkage", decode_avg)):
        x = f(p)[0][0]
        w, _ = metric_match(x, m)
        out[name] = (local_partition(x) == local_partition(ls), w)
    return out


if __name__ == "__main__":
    mp, tp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    model = InverseNet(); model.load_state_dict(torch.load(mp)); model.eval()
    d = np.load(tp); L, M = d["landscapes"][:n], d["metrics"][:n]
    with torch.no_grad():
        X = torch.from_numpy(norm(M))
        P = torch.sigmoid(torch.cat([model(X[i:i + 20]) for i in range(0, n, 20)])).numpy()
    torch.set_num_threads(1)
    with Pool(4) as pool:
        res = pool.map(work, zip(L, M, P))
    for k in ("union", "avg_linkage"):
        print(k, json.dumps(dict(exact_layout=float(np.mean([r[k][0] for r in res])),
                                 windows_all_metrics_exact=float(np.mean([r[k][1] for r in res])))))
