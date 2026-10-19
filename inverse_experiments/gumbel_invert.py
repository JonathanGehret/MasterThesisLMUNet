"""The thesis method with the softmax fix: optimise a landscape through the
frozen forward model.

Each pixel has 5 scores (logits). Forward pass: hard one-hot of the best class
(straight-through Gumbel-softmax), so the model always sees a discrete
landscape. Backward pass: gradients through the soft probabilities.

Usage: python gumbel_invert.py FORWARD.pt TEST.npz N STEPS [RESTARTS]
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

from forward import ForwardNet, onehot, load_forward
from inverse_model import Normalizer  # noqa: F401
from evaluate import metric_match
from enumerate_preimages import local_partition


def loss_fn(model, x, V, N):
    v, nl = model(x)
    mask = N < 0.5
    per = (F.huber_loss(v, V, reduction="none") * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp(min=1)
    per = per + F.binary_cross_entropy_with_logits(nl, N, reduction="none").mean((1, 2, 3))
    return per  # one value per landscape


CFG = dict(lr=0.1, tau0=1.0, tau1=0.05, noise=1.0, soft_frac=0.0, init=None)


def invert(model, V, N, steps, seed=0, init_ls=None):
    """soft_frac: first part of the run feeds soft probabilities (no hard
    rounding) so gradients are informative, then switches to straight-through."""
    torch.manual_seed(seed)
    B, _, H, W = V.shape
    logits = 0.01 * torch.randn(B, 5, H, W)
    if init_ls is not None:  # start from a given landscape (e.g. the inverse network's answer)
        logits = logits + 3.0 * onehot(init_ls)
    logits.requires_grad_()
    opt = torch.optim.Adam([logits], lr=CFG["lr"])
    best = torch.full((B,), float("inf"))
    best_ls = torch.zeros(B, H, W, dtype=torch.long)
    for s in range(steps):
        frac = s / steps
        tau = CFG["tau0"] * (CFG["tau1"] / CFG["tau0"]) ** frac
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)))
        soft = torch.softmax((logits + CFG["noise"] * (1 - frac) * g) / tau, dim=1)
        if frac < CFG["soft_frac"]:
            x = soft
        else:
            hard = F.one_hot(soft.argmax(1), 5).permute(0, 3, 1, 2).float()
            x = hard + soft - soft.detach()          # straight-through
        per = loss_fn(model, x, V, N)
        opt.zero_grad()
        per.sum().backward()
        opt.step()
        with torch.no_grad():                        # keep the best discrete landscape
            ls = logits.argmax(1)
            l = loss_fn(model, onehot(ls), V, N)
            better = l < best
            best[better] = l[better]
            best_ls[better] = ls[better]
    return best_ls.numpy(), best.numpy()


if __name__ == "__main__":
    fwd_path, test_path, n, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    restarts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    if len(sys.argv) > 6:
        CFG.update(json.loads(sys.argv[6]))
    torch.set_num_threads(4)
    model = load_forward(fwd_path)
    for p in model.parameters():
        p.requires_grad_(False)
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    d = np.load(test_path); L, M = d["landscapes"][:n], d["metrics"][:n]
    x = torch.from_numpy(norm(M)); V, N = x[:, :56], x[:, 56:]
    with torch.no_grad():
        floor = loss_fn(model, onehot(L), V, N).numpy()  # loss of the true landscapes
    init_ls = None
    if CFG["init"]:
        from inverse_model import InverseNet
        from decode import decode_avg
        inv = InverseNet(); inv.load_state_dict(torch.load(CFG["init"])); inv.eval()
        with torch.no_grad():
            P = torch.sigmoid(inv(x)).numpy()
        init_ls = np.stack([decode_avg(p)[0][0] for p in P])
        win0 = [metric_match(a.astype(np.int8), m)[0] for a, m in zip(init_ls, M)]
        print("start (inverse network alone): windows exact", float(np.mean(win0)), flush=True)
    best_ls, best_l = None, np.full(n, np.inf)
    for r in range(restarts):
        ls, l = invert(model, V, N, steps, seed=r, init_ls=init_ls)
        if best_ls is None:
            best_ls = ls
        upd = l < best_l
        best_ls[upd], best_l[upd] = ls[upd], l[upd]
    exact = [local_partition(a.astype(np.int8)) == local_partition(b) for a, b in zip(best_ls, L)]
    win = [metric_match(a.astype(np.int8), m)[0] for a, m in zip(best_ls, M)]
    print(json.dumps(dict(method="optimise through frozen forward model (Gumbel straight-through)",
                          forward=fwd_path, cfg=CFG, n=n, steps=steps, restarts=restarts,
                          loss_true_landscape=float(floor.mean()), loss_found=float(best_l.mean()),
                          found_lower_loss_than_truth=float(np.mean(best_l < floor)),
                          exact_layout=float(np.mean(exact)),
                          windows_all_metrics_exact=float(np.mean(win)))), flush=True)
