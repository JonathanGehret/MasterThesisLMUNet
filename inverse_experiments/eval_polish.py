"""Network-only + NN polish evaluation. Usage: python eval_polish.py INVERSE FORWARD TEST N"""
import json, pickle, sys, time
from multiprocessing import Pool
import numpy as np, torch
from inverse_model import InverseNet, Normalizer  # noqa: F401
from forward import load_forward, predict_inverse_input
from decode import decode_avg
from nn_polish import decode_and_polish
from evaluate import metric_match
from enumerate_preimages import local_partition

FWD = None

def init(path):
    global FWD
    torch.set_num_threads(1)
    FWD = load_forward(path)

def work(a):
    ls, m, pe, xe, pp, xp = a
    out = {}
    for name, p, x in (("exact_input", pe, xe), ("predicted_input", pp, xp)):
        r = decode_avg(p)[0][0]
        t = time.time(); q = decode_and_polish(p, FWD, x); dt = time.time() - t
        out[name] = {}
        for tag, y in (("network", r), ("network+nn_polish", q)):
            w, _ = metric_match(y, m)
            out[name][tag] = dict(exact_layout=local_partition(y) == local_partition(ls), windows_all_metrics_exact=w)
        out[name]["polish_seconds_incl_decode"] = dt
    return out

if __name__ == "__main__":
    inv_path, fwd_path, test_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    inv = InverseNet(); inv.load_state_dict(torch.load(inv_path)); inv.eval()
    fwd = load_forward(fwd_path)
    d = np.load(test_path); L, M = d["landscapes"][:n], d["metrics"][:n]
    args = []
    with torch.no_grad():
        for i in range(n):
            xe = torch.from_numpy(norm(M[i:i + 1]))
            xp = predict_inverse_input(fwd, L[i:i + 1])
            args.append((L[i], M[i], torch.sigmoid(inv(xe))[0].numpy(), xe[0], torch.sigmoid(inv(xp))[0].numpy(), xp[0]))
    with Pool(4, initializer=init, initargs=(fwd_path,)) as pool:
        res = pool.map(work, args)
    agg = {}
    for k in ("exact_input", "predicted_input"):
        agg[k] = {t: {m: float(np.mean([r[k][t][m] for r in res])) for m in res[0][k][t]} for t in ("network", "network+nn_polish")}
        agg[k]["polish_seconds_incl_decode"] = float(np.mean([r[k]["polish_seconds_incl_decode"] for r in res]))
    agg.update(test=test_path, n=n)
    print(json.dumps(agg), flush=True)
