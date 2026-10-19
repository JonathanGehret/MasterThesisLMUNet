"""Inverse network v2: trained on 32x32 and 64x64 maps, half of every batch
with metrics predicted by the forward model (realistic errors), the other half
exact metrics with random noise.

Usage: python train2.py EPOCHS INIT_WEIGHTS FORWARD_WEIGHTS FWD_WIDTH FWD_LAYERS LR OUT
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from inverse_model import InverseNet, Normalizer, affinity_targets  # noqa: F401
from forward import ForwardNet, predict_inverse_input

torch.manual_seed(0)
epochs, init, fwd_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
fw, fl, lr, out = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
torch.set_num_threads(4)
norm = pickle.load(open("data/normalizer.pkl", "rb"))
fwd = ForwardNet(c=fw, n_layers=fl); fwd.load_state_dict(torch.load(fwd_path)); fwd.eval()
sets = []
for p, bs in (("data/train32.npz", 32), ("data/train64.npz", 8)):
    d = np.load(p)
    sets.append((d["landscapes"], d["metrics"], bs))
te = np.load("data/test32.npz")
Lte, Mte = te["landscapes"][:200], te["metrics"][:200]

model = InverseNet(); model.load_state_dict(torch.load(init))
batches = [(s, b) for s, (L, _, bs) in enumerate(sets) for b in range(len(L) // bs)]
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(batches))


def targets(L):
    t = [affinity_targets(l) for l in L]
    return torch.from_numpy(np.stack([a for a, _ in t])), torch.from_numpy(np.stack([b for _, b in t]))


for ep in range(epochs):
    model.train()
    order = np.random.permutation(len(batches))
    perms = [np.random.permutation(len(L)) for L, _, _ in sets]
    t0, tot = time.time(), 0.0
    for n, bi in enumerate(order):
        s, b = batches[bi]
        L, M, bs = sets[s]
        idx = np.sort(perms[s][b * bs:(b + 1) * bs])
        x = torch.from_numpy(norm(M[idx]))
        x[:, :56] += 0.3 * torch.randn_like(x[:, :56]) * torch.rand(len(idx), 1, 1, 1)
        h = len(idx) // 2
        x[:h] = predict_inverse_input(fwd, L[idx[:h]])
        T, K = targets(L[idx])
        loss = F.binary_cross_entropy_with_logits(model(x)[K], T[K])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    model.eval()
    with torch.no_grad():
        T, K = targets(Lte)
        pe = model(torch.from_numpy(norm(Mte)))
        pp = model(predict_inverse_input(fwd, Lte))
        acc_e = ((pe[K] > 0) == (T[K] > 0.5)).float().mean().item()
        acc_p = ((pp[K] > 0) == (T[K] > 0.5)).float().mean().item()
    print(f"epoch {ep + 1}/{epochs} loss {tot / (n + 1):.4f} pair_acc exact_input {acc_e:.5f} "
          f"predicted_input {acc_p:.5f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out)
