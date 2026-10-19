"""Parallel evaluation with progress. Usage: python eval_parallel.py MODEL TEST N"""
import json, pickle, sys
from multiprocessing import Pool
import numpy as np, torch
from inverse_model import InverseNet
from decode import decode_avg as decode, colourings
from evaluate import metric_match
from repair import repair
from finisher import finish
from enumerate_preimages import local_partition


def work(a):
    ls, m, p = a
    sols, node, edges = decode(p)
    rec = sols[0]
    rep = repair(rec, m)
    fin = finish(rep, m)
    out = {}
    for name, x in (("network", rec), ("network+repair", rep), ("network+repair+finisher", fin)):
        w, c = metric_match(x, m)
        out[name] = dict(exact_layout=local_partition(x) == local_partition(ls),
                         windows_all_metrics_exact=w, metric_values_exact=c)
    out["labellings"] = len(colourings(node.max() + 1, edges, limit=10000))
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    mp, tp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    model = InverseNet(); model.load_state_dict(torch.load(mp)); model.eval()
    d = np.load(tp); L, M = d["landscapes"][:n], d["metrics"][:n]
    with torch.no_grad():
        X = torch.from_numpy(norm(M))
        P = torch.sigmoid(torch.cat([model(X[i:i + 20]) for i in range(0, n, 20)])).numpy()
    torch.set_num_threads(1)
    res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(work, zip(L, M, P))):
            res.append(r)
            print(f"{i + 1}/{n}", json.dumps(r), flush=True)
    agg = {k: {m: float(np.mean([r[k][m] for r in res])) for m in res[0][k]} for k in ("network", "network+repair", "network+repair+finisher")}
    lab = np.array([r["labellings"] for r in res])
    agg.update(test=tp, n=n, median_labellings=float(np.median(lab)), share_10000_or_more=float((lab >= 10000).mean()))
    print("SUMMARY", json.dumps(agg), flush=True)
