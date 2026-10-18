"""Turn predicted pairwise "same class" probabilities into landscapes.

1. Patches: union neighbouring pixels (8-neighbourhood) predicted "same".
2. Between patches, average all predicted pairs (up to 4 px apart):
   mostly "same"  -> the two patches must get the same label
   mostly "diff"  -> they must get different labels
   no pairs       -> free (too far apart to be seen by any window)
3. Labels: a backtracking colouring with at most 5 labels. It can return one
   labelling, or count / list all valid ones.
"""
import numpy as np

from inverse_model import OFFSETS


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def build_constraints(prob, thr=0.5):
    """prob: (40,H,W). Returns node id per pixel, list of must-differ edges."""
    K, H, W = prob.shape
    idx = lambda r, c: r * W + c
    dsu = DSU(H * W)
    for k, (di, dj) in enumerate(OFFSETS):
        if max(abs(di), abs(dj)) > 1:
            continue
        for r in range(H - di):
            for c in range(max(0, -dj), W - max(0, dj)):
                if prob[k, r, c] > thr:
                    dsu.union(idx(r, c), idx(r + di, c + dj))
    patch = np.array([dsu.find(i) for i in range(H * W)]).reshape(H, W)
    # aggregate evidence between patches
    s, n = {}, {}
    for k, (di, dj) in enumerate(OFFSETS):
        r1, c0, c1 = H - di, max(0, -dj), W - max(0, dj)
        a = patch[:r1, c0:c1].ravel()
        b = patch[di:, c0 + dj:c1 + dj].ravel()
        p = prob[k, :r1, c0:c1].ravel()
        for x, y, v in zip(a, b, p):
            if x == y:
                continue
            key = (min(x, y), max(x, y))
            s[key] = s.get(key, 0.0) + v
            n[key] = n.get(key, 0) + 1
    # merge must-be-same patches into groups, then must-differ edges between groups
    ids = {v: i for i, v in enumerate(np.unique(patch))}
    g = DSU(len(ids))
    for key in s:
        if s[key] / n[key] > thr:
            g.union(ids[key[0]], ids[key[1]])
    group = np.vectorize(lambda v: g.find(ids[v]))(patch)
    gid = {v: i for i, v in enumerate(np.unique(group))}
    node = np.vectorize(gid.get)(group)
    diff = {}
    for key in s:
        if s[key] / n[key] <= thr:
            a, b = gid[g.find(ids[key[0]])], gid[g.find(ids[key[1]])]
            if a != b:
                e = (min(a, b), max(a, b))
                diff[e] = max(diff.get(e, 0.0), 1 - s[key] / n[key])
    return node, diff


def colourings(n_nodes, edges, k=5, limit=1, first_canonical=True):
    """Backtracking k-colouring (DSatur order). Returns up to `limit` solutions,
    counting global relabelings once."""
    nb = [set() for _ in range(n_nodes)]
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    order = sorted(range(n_nodes), key=lambda v: -len(nb[v]))
    col = [-1] * n_nodes
    sols = []

    def rec(i, used):
        if len(sols) >= limit:
            return
        if i == n_nodes:
            sols.append(col.copy())
            return
        v = order[i]
        bad = {col[u] for u in nb[v]}
        for c in range(min(used + 1, k) if first_canonical else k):
            if c not in bad:
                col[v] = c
                rec(i + 1, max(used, c + 1))
                col[v] = -1

    rec(0, 0)
    return sols


def decode(prob, k=5, limit=1):
    node, diff = build_constraints(prob)
    edges = sorted(diff, key=lambda e: -diff[e])
    sols = colourings(node.max() + 1, edges, k, limit)
    while not sols and edges:  # inconsistent prediction: drop least certain constraint
        edges = edges[:-1]
        sols = colourings(node.max() + 1, edges, k, limit)
    return [np.array(s)[node] for s in sols], node, edges


def build_constraints_avg(prob, thr=0.5):
    """Average-linkage version: start with single pixels and repeatedly merge
    the two groups whose pixel pairs are, on average, most confidently "same".
    Stops when no pair of groups averages above thr. One wrong link can no
    longer merge two whole patches."""
    import heapq
    K, H, W = prob.shape
    s, n = {}, {}
    nbr = [dict() for _ in range(H * W)]
    for k, (di, dj) in enumerate(OFFSETS):
        for r in range(H - di):
            for c in range(max(0, -dj), W - max(0, dj)):
                a, b = r * W + c, (r + di) * W + c + dj
                nbr[a][b] = [float(prob[k, r, c]), 1]
                nbr[b][a] = nbr[a][b]
    dsu = DSU(H * W)
    alive = set(range(H * W))
    heap = [(-v[0], a, b) for a in range(H * W) for b, v in nbr[a].items() if a < b and v[0] > thr]
    heapq.heapify(heap)
    while heap:
        negm, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive or b not in nbr[a]:
            continue
        sm, cnt = nbr[a][b]
        if abs(sm / cnt + negm) > 1e-12:
            continue  # stale entry
        # merge b into a
        alive.discard(b)
        dsu.p[b] = a
        del nbr[a][b]
        for x, (sx, cx) in nbr[b].items():
            if x == a:
                continue
            del nbr[x][b]
            if x in nbr[a]:
                e = nbr[a][x]
                e[0] += sx; e[1] += cx
            else:
                e = [sx, cx]
                nbr[a][x] = e
                nbr[x][a] = e
            if e[0] / e[1] > thr:
                heapq.heappush(heap, (-e[0] / e[1], min(a, x), max(a, x)))
        nbr[b] = {}
    root = np.array([dsu.find(i) for i in range(H * W)])
    ids = {v: i for i, v in enumerate(np.unique(root))}
    node = np.vectorize(ids.get)(root).reshape(H, W)
    diff = {}
    for a in alive:
        for x, (sm, cnt) in nbr[a].items():
            e = (min(ids[a], ids[x]), max(ids[a], ids[x]))
            diff[e] = 1 - sm / cnt
    return node, diff


def decode_avg(prob, k=5, limit=1):
    node, diff = build_constraints_avg(prob)
    edges = sorted(diff, key=lambda e: -diff[e])
    sols = colourings(node.max() + 1, edges, k, limit)
    while not sols and edges:
        edges = edges[:-1]
        sols = colourings(node.max() + 1, edges, k, limit)
    return [np.array(s)[node] for s in sols], node, edges
