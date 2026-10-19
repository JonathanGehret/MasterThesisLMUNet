"""Network-only polish: fix weakly supported label decisions using the forward
model as the judge (no exact metric formulas).

After decoding, pairs of nearby label groups that the inverse network separated
on little evidence (few pixel pairs) are candidates. For each, we try giving
one group the other's label (if that breaks no strong constraint) and keep the
change if the forward model's predicted metrics fit the target better in the
surrounding windows.
"""
import numpy as np
import torch
import torch.nn.functional as F

from forward import onehot
from decode import build_constraints_avg, colourings, tabu_colouring


def _local_loss(model, ls, V, N, r0, r1, c0, c1):
    """Forward-model loss over windows centred in [r0:r1, c0:c1]."""
    H, W = ls.shape
    R0, R1, C0, C1 = max(0, r0 - 2), min(H, r1 + 2), max(0, c0 - 2), min(W, c1 + 2)
    crop = ls[R0:R1, C0:C1]
    with torch.no_grad():
        v, nl = model(onehot(crop[None]))
    # the crop's own border windows see a fake edge; only score inner windows
    a0, a1, b0, b1 = r0 - R0, r1 - R0, c0 - C0, c1 - C0
    v, nl = v[0, :, a0:a1, b0:b1], nl[0, :, a0:a1, b0:b1]
    tv, tn = V[:, r0:r1, c0:c1], N[:, r0:r1, c0:c1]
    mask = tn < 0.5
    return (F.huber_loss(v, tv, reduction="none", delta=0.1) * mask).sum().item() + \
        F.binary_cross_entropy_with_logits(nl, tn, reduction="sum").item()


def decode_and_polish(prob, model, X, k=5, max_evidence=6.0, rounds=2):
    """prob: inverse network output (40,H,W); X: target in inverse-input format
    (112,H,W tensor: normalised values + NaN mask). Returns the landscape."""
    node, diff = build_constraints_avg(prob)
    edges = sorted(diff, key=lambda e: -diff[e])
    n = node.max() + 1
    sols = colourings(n, edges, k)
    col = np.array(sols[0] if sols else tabu_colouring(n, edges, diff, k))
    V, N = X[:56], X[56:]
    H, W = node.shape
    nb = [dict() for _ in range(n)]
    for (a, b), w in diff.items():
        nb[a][b] = w; nb[b][a] = w
    rows, cols = np.indices((H, W))
    boxes = {}
    for g in range(n):
        m = node == g
        boxes[g] = (rows[m].min(), rows[m].max() + 1, cols[m].min(), cols[m].max() + 1)
    weak = [e for e in edges if diff[e] <= max_evidence]
    for _ in range(rounds):
        changed = False
        for a, b in weak:
            if col[a] == col[b]:
                continue
            for src, dst in ((a, b), (b, a)):        # give dst the label of src
                new = col[src]
                if any(col[u] == new and w > max_evidence for u, w in nb[dst].items() if u != src):
                    continue                          # would break a strong constraint
                r0, r1, c0, c1 = boxes[dst]
                r0, r1, c0, c1 = max(0, r0 - 2), min(H, r1 + 2), max(0, c0 - 2), min(W, c1 + 2)
                ls = col[node]
                before = _local_loss(model, ls, V, N, r0, r1, c0, c1)
                old = col[dst]; col[dst] = new
                after = _local_loss(model, col[node], V, N, r0, r1, c0, c1)
                if after < before:
                    changed = True
                    break
                col[dst] = old
        if not changed:
            break
    return col[node]
