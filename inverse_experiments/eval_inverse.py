"""Full evaluation of metrics -> landscape.

Usage: python eval_inverse.py MODEL.pt TEST.npz N [NOISE_SD ...]
"""
import json
import pickle
import sys

import numpy as np
import torch

from inverse_model import InverseNet
from decode import decode_avg as decode, colourings
from evaluate import summarize
from repair import repair

model_path, test_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
noises = [float(s) for s in sys.argv[4:]] or [0.0]
norm = pickle.load(open("data/normalizer.pkl", "rb"))
model = InverseNet()
model.load_state_dict(torch.load(model_path))
model.eval()
d = np.load(test_path)
L, M = d["landscapes"][:n], d["metrics"][:n]
X = torch.from_numpy(norm(M))
torch.manual_seed(1)

for sd in noises:
    x = X.clone()
    if sd:
        x[:, :56] += sd * torch.randn_like(x[:, :56])
    with torch.no_grad():
        P = torch.sigmoid(torch.cat([model(x[i:i + 25]) for i in range(0, len(x), 25)])).numpy()
    recs, reps, n_label = [], [], []
    for p, m in zip(P, M):
        sols, node, edges = decode(p)
        recs.append(sols[0])
        if not sd:  # repair needs the true metrics; not fair under noise
            reps.append(repair(sols[0], m))
        n_label.append(len(colourings(node.max() + 1, edges, limit=10000)))
    agg, _ = summarize(L, M, recs)
    if reps:
        agg2, _ = summarize(L, M, reps)
        agg.update({k + "_after_repair": v for k, v in agg2.items()})
    agg.update(test=test_path, n=n, input_noise_sd=sd,
               median_valid_labellings=float(np.median(n_label)),
               share_capped_at_10000=float(np.mean(np.array(n_label) >= 10000)))
    print(json.dumps(agg), flush=True)
