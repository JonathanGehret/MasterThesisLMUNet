"""Python re-implementation of the thesis' 57 landscape-level metrics (lsm_l_*)
on one moving window, following the landscapemetrics formulas.

Cell size is 1 (areas in cells, edges in cell sides); the unit scaling used in R
(hectares, metres) does not change which windows have equal metrics.
Cells outside the landscape are -1 (NA in R).
"""
import math
from functools import lru_cache

import numpy as np

from window_stats import _components, _circle_r2, N4, N8

METRICS = [
    "ai", "area_cv", "area_mn", "area_sd", "cai_cv", "cai_mn", "cai_sd", "circle_cv",
    "circle_mn", "circle_sd", "cohesion", "condent", "contag", "core_cv", "core_mn",
    "core_sd", "dcad", "dcore_cv", "dcore_mn", "dcore_sd", "division", "ed", "ent",
    "frac_mn", "frac_sd", "iji", "joinent", "lpi", "lsi", "mesh", "msidi", "msiei",
    "mutinf", "ndca", "np", "pafrac", "para_cv", "para_mn", "para_sd", "pd", "pladj",
    "pr", "prd", "relmutinf", "rpr", "shape_cv", "shape_mn", "shape_sd", "shdi", "shei",
    "sidi", "siei", "split", "ta", "tca", "te",
]
NAN = float("nan")


def _mn_sd_cv(x):
    x = [v for v in x if not math.isnan(v)]
    if not x:
        return NAN, NAN, NAN
    mn = sum(x) / len(x)
    if len(x) < 2:
        return mn, NAN, NAN
    sd = math.sqrt(sum((v - mn) ** 2 for v in x) / (len(x) - 1))
    return mn, sd, (sd / mn * 100 if mn != 0 else NAN)


def _min_perim(a):
    n = math.floor(math.sqrt(a))
    m = a - n * n
    return 4 * n if m == 0 else (4 * n + 2 if a <= n * (n + 1) else 4 * n + 4)


def _max_like_adj(a):
    n = math.floor(math.sqrt(a))
    m = a - n * n
    if m == 0:
        return 2 * n * (n - 1)
    return 2 * n * (n - 1) + 2 * m - (1 if m <= n else 2)


def _ent(p, base=2.0):
    return -sum(v * math.log(v, base) for v in p if v > 0)


def compute_metrics(win):
    h, w = win.shape
    cls_of = {(r, c): int(win[r, c]) for r in range(h) for c in range(w) if win[r, c] >= 0}
    if not cls_of:
        return {}
    classes = sorted(set(cls_of.values()))
    m = len(classes)
    A = len(cls_of)
    area = {k: sum(1 for v in cls_of.values() if v == k) for k in classes}
    P = {k: area[k] / A for k in classes}

    adj = {}  # rook, double counted
    boundary = 0
    for (r, c), v in cls_of.items():
        for dr, dc in N4:
            u = cls_of.get((r + dr, c + dc), -1)
            if u >= 0:
                adj[(v, u)] = adj.get((v, u), 0) + 1
            else:
                boundary += 1
    g = lambda i, k: adj.get((i, k), 0)
    tot_adj = sum(adj.values())
    te = sum(cnt for (i, k), cnt in adj.items() if i != k) / 2

    # patches
    pa, pp, pcore, pdcore, pcirc = [], [], [], [], []
    for k in classes:
        cells = [p for p, v in cls_of.items() if v == k]
        for patch in _components(cells, N8):
            s = set(patch)
            a = len(patch)
            per = sum(1 for (r, c) in patch for dr, dc in N4 if (r + dr, c + dc) not in s)
            core = [(r, c) for (r, c) in patch
                    if all(cls_of.get((r + dr, c + dc), -1) in (-1, k) for dr, dc in N8)]
            pa.append(a); pp.append(per); pcore.append(len(core))
            pdcore.append(len(_components(core, N8)))
            corners = [(r + x, c + y) for r, c in patch for x in (0, 1) for y in (0, 1)]
            pcirc.append(1 - a / (math.pi * _circle_r2(corners)))
    npch = len(pa)
    out = {}
    out["ta"] = A
    out["np"] = npch
    out["pd"] = npch / A
    out["pr"] = m
    out["prd"] = m / A
    out["rpr"] = NAN
    out["area_mn"], out["area_sd"], out["area_cv"] = _mn_sd_cv(pa)
    out["lpi"] = max(pa) / A * 100
    out["te"] = te
    out["ed"] = te / A
    out["lsi"] = (te + boundary) / _min_perim(A)
    # ai
    ai = 0.0
    for k in classes:
        mx = _max_like_adj(area[k])
        ai += (g(k, k) / 2 / mx * 100 if mx > 0 else NAN) * P[k]
    out["ai"] = ai
    out["pladj"] = sum(g(k, k) for k in classes) / tot_adj * 100 if tot_adj else NAN
    if m >= 2 and tot_adj:
        s = 0.0
        for i in classes:
            row = sum(g(i, k) for k in classes)
            for k in classes:
                if row and g(i, k):
                    q = P[i] * g(i, k) / row
                    s += q * math.log(q)
        out["contag"] = (1 + s / (2 * math.log(m))) * 100
    else:
        out["contag"] = NAN
    # information theory metrics (Nowosad & Stepinski), co-occurrence matrix
    if tot_adj:
        pij = [cnt / tot_adj for cnt in adj.values()]
        pi = [sum(g(i, k) for k in classes) / tot_adj for i in classes]
        joinent = _ent(pij)
        ent = _ent(pi)
        condent = joinent - ent
        mutinf = ent - condent
        out.update(joinent=joinent, ent=ent, condent=condent, mutinf=mutinf,
                   relmutinf=mutinf / ent if ent > 0 else NAN)
    else:
        out.update(joinent=NAN, ent=NAN, condent=NAN, mutinf=NAN, relmutinf=NAN)
    shdi = -sum(p * math.log(p) for p in P.values())
    sidi = 1 - sum(p * p for p in P.values())
    msidi = -math.log(sum(p * p for p in P.values()))
    out["shdi"], out["sidi"], out["msidi"] = shdi, sidi, msidi
    out["shei"] = shdi / math.log(m) if m > 1 else NAN
    out["siei"] = sidi / (1 - 1 / m) if m > 1 else NAN
    out["msiei"] = msidi / math.log(m) if m > 1 else NAN
    if m >= 3 and te:
        E = te
        s = 0.0
        for a_i in range(m):
            for b_i in range(a_i + 1, m):
                e = g(classes[a_i], classes[b_i])
                if e:
                    s += (e / E) * math.log(e / E)
        out["iji"] = -s / math.log(0.5 * m * (m - 1)) * 100
    else:
        out["iji"] = NAN
    sp = sum(pp)
    spa = sum(p * math.sqrt(a) for p, a in zip(pp, pa))
    out["cohesion"] = (1 - sp / spa) / (1 - 1 / math.sqrt(A)) * 100 if A > 1 else NAN
    sa2 = sum(a * a for a in pa)
    out["division"] = 1 - sa2 / A ** 2
    out["mesh"] = sa2 / A
    out["split"] = A ** 2 / sa2
    out["shape_mn"], out["shape_sd"], out["shape_cv"] = _mn_sd_cv(
        [p / _min_perim(a) for p, a in zip(pp, pa)])
    out["para_mn"], out["para_sd"], out["para_cv"] = _mn_sd_cv([p / a for p, a in zip(pp, pa)])
    frac = [2 * math.log(0.25 * p) / math.log(a) if a > 1 else NAN for p, a in zip(pp, pa)]
    out["frac_mn"], out["frac_sd"], _ = _mn_sd_cv(frac)
    out["circle_mn"], out["circle_sd"], out["circle_cv"] = _mn_sd_cv(pcirc)
    out["core_mn"], out["core_sd"], out["core_cv"] = _mn_sd_cv(pcore)
    out["cai_mn"], out["cai_sd"], out["cai_cv"] = _mn_sd_cv(
        [c / a * 100 for c, a in zip(pcore, pa)])
    out["dcore_mn"], out["dcore_sd"], out["dcore_cv"] = _mn_sd_cv(pdcore)
    out["tca"] = sum(pcore)
    out["ndca"] = sum(pdcore)
    out["dcad"] = sum(pdcore) / A * 100
    if npch >= 10:
        x = np.log(pp); y = np.log(pa)
        slope = np.polyfit(x, y, 1)[0]
        out["pafrac"] = 2 / slope if slope else NAN
    else:
        out["pafrac"] = NAN
    return out


def _r(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return float(f"{v:.9g}")


@lru_cache(maxsize=2_000_000)
def metric_summary(win_bytes, h, w):
    win = np.frombuffer(win_bytes, dtype=np.int8).reshape(h, w)
    d = compute_metrics(win)
    return tuple(_r(d[k]) for k in METRICS) if d else ()
