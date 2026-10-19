"""For maps whose patch shapes are right: which patch pairs got the wrong
same/different label, and what did the network predict for them?"""
import pickle, sys
import numpy as np, torch
from inverse_model import InverseNet, Normalizer, OFFSETS  # noqa: F401
from decode import decode_avg
from error_types import patch_ids


def pair_stats(pid, prob):
    H, W = pid.shape
    s, c = {}, {}
    for k, (di, dj) in enumerate(OFFSETS):
        r1, c0, c1 = H - di, max(0, -dj), W - max(0, dj)
        a = pid[:r1, c0:c1].ravel(); b = pid[di:, c0 + dj:c1 + dj].ravel(); p = prob[k, :r1, c0:c1].ravel()
        for x, y, v in zip(a.tolist(), b.tolist(), p.tolist()):
            if x != y:
                key = (min(x, y), max(x, y)); s[key] = s.get(key, 0) + v; c[key] = c.get(key, 0) + 1
    return {k: (s[k] / c[k], c[k]) for k in s}


norm = pickle.load(open("data/normalizer.pkl", "rb"))
inv = InverseNet(); inv.load_state_dict(torch.load(sys.argv[1])); inv.eval()
d = np.load("data/test128.npz")
for i in [int(x) for x in sys.argv[2].split(",")]:
    L, M = d["landscapes"][i], d["metrics"][i]
    with torch.no_grad():
        P = torch.sigmoid(inv(torch.from_numpy(norm(M[None]))))[0].numpy()
    x = decode_avg(P)[0][0]
    pid = patch_ids(L)  # shapes are right, so true patches = decoded patches
    lab_t = {q: int(L[pid == q][0]) for q in np.unique(pid)}
    lab_x = {q: int(x[pid == q][0]) for q in np.unique(pid)}
    stats = pair_stats(pid, P)
    wrong = [(k, v) for k, v in stats.items() if (lab_t[k[0]] == lab_t[k[1]]) != (lab_x[k[0]] == lab_x[k[1]])]
    print(f"map {i}: {len(stats)} nearby patch pairs, {len(wrong)} with wrong same/different")
    for (a, b), (mp, n) in wrong[:8]:
        print(f"   truth {'same' if lab_t[a] == lab_t[b] else 'diff'}  decoded {'same' if lab_x[a] == lab_x[b] else 'diff'}"
              f"  network avg p(same) {mp:.2f} over {n} pixel pairs")
