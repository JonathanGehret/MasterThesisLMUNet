"""Forward model v2, trained on distinct 5x5 windows.

Each output pixel depends only on its own 5x5 window, so instead of whole maps
(where most windows repeat, e.g. uniform areas) we train on every distinct
window (up to relabelling) exactly once, with random rotations/flips.
The network is the same shape as ForwardNet (5x5 conv + per-pixel MLP), so the
trained weights run on full maps of any size.

Usage: python forward2.py EPOCHS WIDTH LAYERS [RESUME_WEIGHTS LR]
With RESUME_WEIGHTS, training continues from those weights with a cosine
decaying learning rate starting at LR (no warm-up).
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from forward import ForwardNet, onehot
from inverse_model import Normalizer  # noqa: F401
from window_table import canonical_windows


def _dedup(k, m):
    v = np.ascontiguousarray(k).view(np.dtype((np.void, k.shape[1]))).ravel()
    _, idx = np.unique(v, return_index=True)
    return k[idx], m[idx]


def unique_windows(paths, chunk=400):
    """Distinct windows, computed in chunks to keep memory low."""
    keys, targets = [], []
    for p in paths:
        d = np.load(p, mmap_mode="r")
        L, M = d["landscapes"], d["metrics"]
        for s in range(0, len(L), chunk):
            k = canonical_windows(np.asarray(L[s:s + chunk]))
            m = np.asarray(M[s:s + chunk]).transpose(0, 2, 3, 1).reshape(-1, M.shape[1])
            k, m = _dedup(k, m)
            keys.append(k); targets.append(m)
        k, m = _dedup(np.concatenate(keys), np.concatenate(targets))
        keys, targets = [k], [m]
    return keys[0], targets[0]


def windows_onehot(k):
    """(B,25) labels (-1 outside) -> (B,5,5,5) one-hot, outside all zero."""
    k = torch.as_tensor(k, dtype=torch.long).view(-1, 5, 5)
    oh = F.one_hot(k.clamp(min=0), 5).permute(0, 3, 1, 2).float()
    return oh * (k >= 0).unsqueeze(1).float()


def forward_windows(model, x):
    """Run ForwardNet on (B,5,5,5) windows -> centre outputs (B,56)."""
    h = torch.relu(F.conv2d(x, model.inp.weight, model.inp.bias))  # (B,C,1,1)
    for layer in model.layers:
        h = torch.relu(h + layer(h))
    o = model.out(h)[:, :, 0, 0]
    return o[:, :model.n], o[:, model.n:]


if __name__ == "__main__":
    torch.manual_seed(0)
    epochs, width, layers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    torch.set_num_threads(4)
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    t0 = time.time()
    K, Mw = unique_windows(["data/train32.npz", "data/train64.npz"])
    X = norm(Mw[:, :, None, None])[:, :, 0, 0]               # (U,112)
    V, N = torch.from_numpy(X[:, :56]), torch.from_numpy(X[:, 56:])
    print(f"distinct windows: {len(K):,} ({time.time() - t0:.0f}s)", flush=True)
    te = np.load("data/test32.npz")
    Lte, Mte = te["landscapes"][:200], te["metrics"][:200]
    Xte = torch.from_numpy(norm(Mte)); Vte, Nte = Xte[:, :56], Xte[:, 56:]

    model = ForwardNet(c=width, n_layers=layers)
    bs = 1024
    steps = epochs * (len(K) // bs)
    if len(sys.argv) > 4:
        model.load_state_dict(torch.load(sys.argv[4]))
        lr = float(sys.argv[5])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(K))
        t1, tot = time.time(), 0.0
        for b in range(len(K) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            x = windows_onehot(K[idx.numpy()])
            r = np.random.randint(4)                         # rotation/flip: metrics unchanged
            x = torch.rot90(x, r, (2, 3))
            if np.random.rand() < 0.5:
                x = torch.flip(x, (3,))
            x = x[:, torch.randperm(5)]                      # relabel: metrics unchanged
            v, nl = forward_windows(model, x)
            mask = N[idx] < 0.5
            loss = F.huber_loss(v[mask], V[idx][mask], delta=0.1) + \
                F.binary_cross_entropy_with_logits(nl, N[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            v, nl = model(onehot(Lte))
            mask = Nte < 0.5
            r2, close = [], []
            for k in range(56):
                m = mask[:, k]
                if m.sum() > 10:
                    y, yh = Vte[:, k][m], v[:, k][m]
                    r2.append(1 - ((y - yh) ** 2).sum().item() / (((y - y.mean()) ** 2).sum().item() + 1e-12))
            err = (v - Vte).abs() * mask                    # normalised units
            win_ok = ((err < 0.01).all(1) & ((nl > 0).float() == Nte).all(1)).float().mean().item()
        print(f"epoch {ep + 1}/{epochs} loss {tot / (b + 1):.5f} mean_R2 {np.mean(r2):.4f} "
              f"median_R2 {np.median(r2):.4f} min_R2 {np.min(r2):.4f} "
              f"windows_all_metrics_within_0.01sd {win_ok:.4f} ({time.time() - t1:.0f}s)", flush=True)
        torch.save(model.state_dict(), f"data/forward2_w{width}_l{layers}.pt")
