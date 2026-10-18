"""Direct inverse network: metric maps -> "same class?" for nearby pixel pairs.

For every pixel and every offset (di, dj) within one 5x5 window distance
(|di|, |dj| <= 4, half plane, 40 offsets) the network predicts the probability
that the two pixels have the same class. This target is label-free, so the
arbitrary class numbers never enter training.
"""
import numpy as np
import torch
import torch.nn as nn

OFFSETS = [(0, dj) for dj in range(1, 5)] + [(di, dj) for di in range(1, 5) for dj in range(-4, 5)]
N8_IDX = [OFFSETS.index(o) for o in [(0, 1), (1, -1), (1, 0), (1, 1)]]


def affinity_targets(ls):
    """ls: (H,W) ints -> target (40,H,W) float, mask (40,H,W) bool."""
    H, W = ls.shape
    tgt = np.zeros((len(OFFSETS), H, W), np.float32)
    msk = np.zeros((len(OFFSETS), H, W), bool)
    for k, (di, dj) in enumerate(OFFSETS):
        r1, c0, c1 = H - di, max(0, -dj), W - max(0, dj)
        tgt[k, :r1, c0:c1] = ls[:r1, c0:c1] == ls[di:, c0 + dj:c1 + dj]
        msk[k, :r1, c0:c1] = True
    return tgt, msk


class Normalizer:
    """Signed log transform + per-metric standardisation, NaN -> 0 + NaN-mask channel."""

    def fit(self, m):  # m: (N,C,H,W) float16 with NaN
        x = self._log(m[: min(len(m), 2000)].astype(np.float32))
        self.mu = np.nanmean(x, axis=(0, 2, 3))
        self.sd = np.nanstd(x, axis=(0, 2, 3)) + 1e-6
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.nan_to_num(self.sd, nan=1.0)
        return self

    @staticmethod
    def _log(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def __call__(self, m):
        x = self._log(m.astype(np.float32))
        nan = np.isnan(x)
        x = (x - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        x[nan] = 0
        return np.concatenate([x, nan.astype(np.float32)], axis=1)


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class InverseNet(nn.Module):
    def __init__(self, c_in=112, c=64, dil=(1, 2, 4, 1, 2, 4, 1, 1)):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(c_in, c, 1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Conv2d(c, len(OFFSETS), 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))
