"""Train the inverse network. Usage: python train.py EPOCHS [NOISE_SD]

NOISE_SD > 0 adds Gaussian noise to the normalised metric inputs during
training, to mimic imperfect (e.g. U-Net predicted) metrics.
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from inverse_model import InverseNet, Normalizer, affinity_targets

torch.manual_seed(0)
epochs = int(sys.argv[1])
noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
tag = f"noise{noise}" if noise else "clean"

tr = np.load("data/train32.npz")
te = np.load("data/test32.npz")
Mtr, Ltr = tr["metrics"], tr["landscapes"]
Mte, Lte = te["metrics"][:200], te["landscapes"][:200]
norm = Normalizer().fit(Mtr)
pickle.dump(norm, open("data/normalizer.pkl", "wb"))


def targets(L):
    t = [affinity_targets(l) for l in L]
    return (torch.from_numpy(np.stack([a for a, _ in t])),
            torch.from_numpy(np.stack([b for _, b in t])))


Ttr, Ktr = targets(Ltr)
Tte, Kte = targets(Lte)
Xte = torch.from_numpy(norm(Mte))

model = InverseNet()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
bs = 32
steps = epochs * (len(Ltr) // bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)


def masked_bce(logit, t, k):
    return F.binary_cross_entropy_with_logits(logit[k], t[k])


for ep in range(epochs):
    model.train()
    perm = np.random.permutation(len(Ltr))
    t0, tot = time.time(), 0.0
    for b in range(len(Ltr) // bs):
        idx = np.sort(perm[b * bs:(b + 1) * bs])
        x = torch.from_numpy(norm(Mtr[idx]))
        if noise:
            x[:, :56] += noise * torch.randn_like(x[:, :56]) * torch.rand(len(idx), 1, 1, 1)
        loss = masked_bce(model(x), Ttr[idx], Ktr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(Xte[i:i + 50]) for i in range(0, len(Xte), 50)])
        acc = ((p[Kte] > 0) == (Tte[Kte] > 0.5)).float().mean().item()
        perfect = np.mean([(((p[i][Kte[i]] > 0) == (Tte[i][Kte[i]] > 0.5)).all()).item()
                           for i in range(len(p))])
    print(f"epoch {ep + 1}/{epochs} loss {tot / (b + 1):.4f} pair_acc {acc:.5f} "
          f"landscapes_all_pairs_right {perfect:.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), f"data/inverse_{tag}.pt")
