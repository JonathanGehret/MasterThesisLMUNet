"""Re-run the finisher with larger search limits on the landscapes it did not solve."""
import json, pickle, sys
from multiprocessing import Pool
import numpy as np, torch
import finisher
from inverse_model import InverseNet
from decode import decode_avg
from repair import repair
from evaluate import metric_match
from enumerate_preimages import local_partition

IDX = [int(x) for x in sys.argv[1].split(",")]

def work(a):
    ls, m, p = a
    x = repair(decode_avg(p)[0][0], m)
    for _ in range(3):  # repeated passes with bigger limits
        x = finisher.finish(x, m, node_limit=200000, time_limit=900)
    w, _ = metric_match(x, m)
    return dict(exact_layout=local_partition(x) == local_partition(ls), windows_all_metrics_exact=w)

if __name__ == "__main__":
    finisher._solve_patch_labels.__defaults__ = (300000, 30)
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    inv = InverseNet(); inv.load_state_dict(torch.load("data/inverse_noise0.3.pt")); inv.eval()
    d = np.load("data/test32.npz"); L, M = d["landscapes"][IDX], d["metrics"][IDX]
    with torch.no_grad():
        P = torch.sigmoid(inv(torch.from_numpy(norm(M)))).numpy()
    torch.set_num_threads(1)
    with Pool(3) as pool:
        for i, r in zip(IDX, pool.imap(work, zip(L, M, P))):
            print(i, json.dumps(r), flush=True)
