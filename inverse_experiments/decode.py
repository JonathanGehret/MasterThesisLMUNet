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


def greedy_colouring(n_nodes, edges, weight, k=5):
    """Fallback when exact colouring is too hard: DSatur-style greedy that
    picks, for each node, the label violating the least total confidence."""
    nb = [[] for _ in range(n_nodes)]
    for a, b in edges:
        w = weight.get((a, b), 1.0)
        nb[a].append((b, w))
        nb[b].append((a, w))
    col = [-1] * n_nodes
    for v in sorted(range(n_nodes), key=lambda v: -len(nb[v])):
        cost = [0.0] * k
        for u, w in nb[v]:
            if col[u] >= 0:
                cost[col[u]] += w
        col[v] = int(np.argmin(cost))
    return col


def tabu_colouring(n_nodes, edges, weight, k=5, iters=30000, seed=0):
    """Weighted tabu search for a k-colouring: minimise the total confidence of
    must-differ constraints that end up with equal labels. Finds a conflict-free
    labelling when one exists (usually within a few hundred moves)."""
    rng = np.random.default_rng(seed)
    nb = [[] for _ in range(n_nodes)]
    for a, b in edges:
        w = weight.get((a, b), 1.0)
        nb[a].append((b, w)); nb[b].append((a, w))
    col = np.array(greedy_colouring(n_nodes, edges, weight, k))
    # cost[v, c] = total weight of v's neighbours that have colour c
    cost = np.zeros((n_nodes, k))
    for v in range(n_nodes):
        for u, w in nb[v]:
            cost[v, col[u]] += w
    total = sum(cost[v, col[v]] for v in range(n_nodes)) / 2
    best_total, best_col = total, col.copy()
    tabu = np.zeros((n_nodes, k), int)
    for it in range(iters):
        if best_total <= 1e-12:
            break
        conf = np.where(cost[np.arange(n_nodes), col] > 0)[0]
        if len(conf) == 0:
            break
        delta = cost[conf] - cost[conf, col[conf]][:, None]      # change in cost per move
        delta[np.arange(len(conf)), col[conf]] = np.inf
        allowed = tabu[conf] <= it
        cand = np.where(allowed | (total + delta < best_total - 1e-12), delta, np.inf)
        if not np.isfinite(cand).any():
            continue
        m = cand.min()
        options = np.argwhere(cand == m)
        i, c = options[rng.integers(len(options))]
        v, old = conf[i], col[conf[i]]
        for u, w in nb[v]:
            cost[u, old] -= w
            cost[u, c] += w
        col[v] = c
        total += m
        tabu[v, old] = it + 7 + rng.integers(10)
        if total < best_total - 1e-12:
            best_total, best_col = total, col.copy()
    return best_col.tolist()


def colourings(n_nodes, edges, k=5, limit=1, first_canonical=True, node_limit=200000):
    """Backtracking k-colouring (DSatur order). Returns up to `limit` solutions,
    counting global relabelings once. Gives up after node_limit steps."""
    nb = [set() for _ in range(n_nodes)]
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    order = sorted(range(n_nodes), key=lambda v: -len(nb[v]))
    col = [-1] * n_nodes
    sols = []
    steps = [0]

    def rec(i, used):
        if len(sols) >= limit or steps[0] > node_limit:
            return
        steps[0] += 1
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
    if not sols:  # too hard for exact search, or inconsistent: tabu search
        sols = [tabu_colouring(node.max() + 1, edges, diff, k)]
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
            # evidence that the two groups differ, summed over all their pixel
            # pairs: a long shared border counts much more than one corner pair
            diff[e] = cnt - sm
    return node, diff


def decode_avg(prob, k=5, limit=1):
    node, diff = build_constraints_avg(prob)
    edges = sorted(diff, key=lambda e: -diff[e])
    sols = colourings(node.max() + 1, edges, k, limit)
    if not sols:  # too hard for exact search, or inconsistent: tabu search
        sols = [tabu_colouring(node.max() + 1, edges, diff, k)]
    return [np.array(s)[node] for s in sols], node, edges


def decode_fast(prob, k=5, limit=1, sure=0.95):
    """Faster decode_avg: first merge neighbour pixels that are almost surely
    the same class (vectorised connected components), then run the same
    average-linkage merging on these groups instead of on single pixels."""
    import heapq
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    K, H, W = prob.shape
    ii, jj, pp = [], [], []
    idx = np.arange(H * W).reshape(H, W)
    for kk, (di, dj) in enumerate(OFFSETS):
        r1, c0, c1 = H - di, max(0, -dj), W - max(0, dj)
        ii.append(idx[:r1, c0:c1].ravel())
        jj.append(idx[di:, c0 + dj:c1 + dj].ravel())
        pp.append(prob[kk, :r1, c0:c1].ravel())
    ii, jj, pp = np.concatenate(ii), np.concatenate(jj), np.concatenate(pp).astype(np.float64)
    near = np.zeros(len(ii), bool)
    o = 0
    for kk, (di, dj) in enumerate(OFFSETS):
        n = (H - di) * (W - abs(dj))
        near[o:o + n] = max(abs(di), abs(dj)) == 1
        o += n
    m = near & (pp > sure)
    g = coo_matrix((np.ones(m.sum()), (ii[m], jj[m])), shape=(H * W, H * W))
    _, comp = connected_components(g, directed=False)
    # sum of probabilities and pair counts between groups (vectorised)
    a, b = comp[ii], comp[jj]
    keep = a != b
    a, b, p = np.minimum(a, b)[keep], np.maximum(a, b)[keep], pp[keep]
    key = a.astype(np.int64) * (H * W) + b
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=p)
    c = np.bincount(inv).astype(np.float64)
    ua, ub = uk // (H * W), uk % (H * W)
    nbr = {}
    for x, y, sx, cx in zip(ua.tolist(), ub.tolist(), s.tolist(), c.tolist()):
        e = [sx, cx]
        nbr.setdefault(x, {})[y] = e
        nbr.setdefault(y, {})[x] = e
    groups = np.unique(comp)
    for gg in groups.tolist():
        nbr.setdefault(gg, {})
    dsu = DSU(int(comp.max()) + 1)
    alive = set(groups.tolist())
    heap = [(-e[0] / e[1], x, y) for x, d in nbr.items() for y, e in d.items() if x < y and e[0] / e[1] > 0.5]
    heapq.heapify(heap)
    while heap:
        negm, x, y = heapq.heappop(heap)
        if x not in alive or y not in alive or y not in nbr[x]:
            continue
        sm, cnt = nbr[x][y]
        if abs(sm / cnt + negm) > 1e-12:
            continue
        alive.discard(y)
        dsu.p[y] = x
        del nbr[x][y]
        for z, (sz, cz) in nbr[y].items():
            if z == x:
                continue
            del nbr[z][y]
            if z in nbr[x]:
                e = nbr[x][z]; e[0] += sz; e[1] += cz
            else:
                e = [sz, cz]; nbr[x][z] = e; nbr[z][x] = e
            if e[0] / e[1] > 0.5:
                heapq.heappush(heap, (-e[0] / e[1], min(x, z), max(x, z)))
        nbr[y] = {}
    root = np.array([dsu.find(int(v)) for v in range(int(comp.max()) + 1)])[comp]
    ids = {v: i for i, v in enumerate(np.unique(root).tolist())}
    node = np.vectorize(ids.get)(root).reshape(H, W)
    diff = {}
    for x in alive:
        for z, (sm, cnt) in nbr[x].items():
            e = (min(ids[x], ids[z]), max(ids[x], ids[z]))
            diff[e] = 1 - sm / cnt
    edges = sorted(diff, key=lambda e: -diff[e])
    sols = colourings(node.max() + 1, edges, k, limit)
    if not sols:
        sols = [tabu_colouring(node.max() + 1, edges, diff, k)]
    return [np.array(s_)[node] for s_ in sols], node, edges
