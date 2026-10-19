"""Forward model (landscape -> 56 window metric maps), a stand-in for the
thesis U-Net, so the inverse can be tested on *predicted* metrics.

Usage: python forward.py EPOCHS

Input: one-hot landscape (5 channels), class labels randomly permuted during
training (landscape-level metrics do not depend on labels).
Output: 56 normalised metric values + 56 NaN logits per pixel.
A 5x5 first layer sees exactly one moving window; the rest is a per-pixel MLP.
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inverse_model import Normalizer  # noqa: F401  (needed to unpickle)


class ForwardNet(nn.Module):
    def __init__(self, c=192, n_layers=4, n_metrics=56):
        super().__init__()
        self.inp = nn.Conv2d(5, c, 5, padding=2)
        self.layers = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(n_layers)])
        self.out = nn.Conv2d(c, 2 * n_metrics, 1)
        self.n = n_metrics

    def forward(self, onehot):
        h = torch.relu(self.inp(onehot))
        for layer in self.layers:
            h = torch.relu(h + layer(h))
        o = self.out(h)
        return o[:, :self.n], o[:, self.n:]  # values, NaN logits


def onehot(L, perm=False):
    L = torch.as_tensor(L, dtype=torch.long)
    if perm:
        p = torch.stack([torch.randperm(5) for _ in range(len(L))])
        L = torch.gather(p, 1, L.flatten(1)).view_as(L)
    return F.one_hot(L, 5).permute(0, 3, 1, 2).float()


def targets(norm, M):
    x = norm(M)  # (N,112): normalised values (NaN->0) + NaN mask
    return torch.from_numpy(x[:, :56]), torch.from_numpy(x[:, 56:])


def predict_inverse_input(model, L):
    """Forward predictions in the inverse network's input format."""
    with torch.no_grad():
        v, nl = model(onehot(L))
        nan = (nl > 0).float()
        return torch.cat([v * (1 - nan), nan], 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    epochs = int(sys.argv[1])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    tr, te = np.load("data/train32.npz"), np.load("data/test32.npz")
    Ltr, Mtr = tr["landscapes"], tr["metrics"]
    Lte, Mte = te["landscapes"][:200], te["metrics"][:200]
    Vte, Nte = targets(norm, Mte)
    model = ForwardNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(Ltr) // bs))
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Ltr))
        t0, tot = time.time(), 0.0
        for b in range(len(Ltr) // bs):
            idx = np.sort(perm[b * bs:(b + 1) * bs])
            V, N = targets(norm, Mtr[idx])
            v, nl = model(onehot(Ltr[idx], perm=True))
            mask = N < 0.5
            loss = F.huber_loss(v[mask], V[mask]) + F.binary_cross_entropy_with_logits(nl, N)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            v, nl = model(onehot(Lte))
            mask = Nte < 0.5
            r2 = []
            for k in range(56):
                m = mask[:, k]
                if m.sum() > 10:
                    y, yh = Vte[:, k][m], v[:, k][m]
                    r2.append(1 - ((y - yh) ** 2).sum().item() / (((y - y.mean()) ** 2).sum().item() + 1e-12))
            nan_acc = ((nl > 0).float() == Nte).float().mean().item()
        print(f"epoch {ep + 1}/{epochs} loss {tot / (b + 1):.4f} mean_R2 {np.mean(r2):.4f} "
              f"median_R2 {np.median(r2):.4f} nan_acc {nan_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), "data/forward.pt")


def load_forward(path):
    """Load a ForwardNet, reading its width and depth from the saved weights."""
    sd = torch.load(path)
    c = sd["inp.weight"].shape[0]
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("layers.")})
    model = ForwardNet(c=c, n_layers=n_layers)
    model.load_state_dict(sd)
    model.eval()
    return model
