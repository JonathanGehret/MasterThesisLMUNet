"""Split remaining errors into shape errors (patch outlines wrong) and label
errors (outlines right, but two patches wrongly share / don't share a label)."""
import json, pickle, sys
from multiprocessing import Pool
import numpy as np, torch
from scipy import ndimage
from inverse_model import InverseNet, Normalizer  # noqa: F401
from decode import decode_avg
from evaluate import metric_match


def patch_ids(ls):
    out = np.zeros(ls.shape, int); n = 0
    for v in range(5):
        lab, k = ndimage.label(ls == v, np.ones((3, 3)))
        out[lab > 0] = lab[lab > 0] + n; n += k
    return out


def same_patch_partition(a, b):
    pa, pb = patch_ids(a), patch_ids(b)
    pairs = set(zip(pa.ravel().tolist(), pb.ravel().tolist()))
    return len(pairs) == len(set(pa.ravel())) == len(set(pb.ravel())), pa, pb


def work(args):
    ls, m, p = args
    x = decode_avg(p)[0][0]
    w, _ = metric_match(x, m)
    shapes_ok, pa, pb = same_patch_partition(ls, x)
    # pixels whose patch outline differs: patch in x maps to >1 true patch or vice versa
    bad_px = 0
    if not shapes_ok:
        for q in np.unique(pb):
            if len(np.unique(pa[pb == q])) > 1:
                bad_px += int((pb == q).sum())
    return dict(windows_exact=w, patch_shapes_all_right=shapes_ok, pixels_in_wrong_shape_patches=bad_px)


if __name__ == "__main__":
    inv_path, test_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    norm = pickle.load(open("data/normalizer.pkl", "rb"))
    inv = InverseNet(); inv.load_state_dict(torch.load(inv_path)); inv.eval()
    d = np.load(test_path); L, M = d["landscapes"][:n], d["metrics"][:n]
    with torch.no_grad():
        P = [torch.sigmoid(inv(torch.from_numpy(norm(M[i:i + 1]))))[0].numpy() for i in range(n)]
    torch.set_num_threads(1)
    with Pool(4) as pool:
        for r in pool.map(work, zip(L, M, P)):
            print(json.dumps(r))
