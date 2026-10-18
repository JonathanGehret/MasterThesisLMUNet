"""Figure: original, reconstruction from metrics only, and alternative valid labellings."""
import pickle
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from inverse_model import InverseNet
from decode import decode_avg, colourings
from enumerate_preimages import local_partition
from evaluate import metric_match

norm = pickle.load(open("data/normalizer.pkl", "rb"))
model = InverseNet(); model.load_state_dict(torch.load("data/inverse_clean.pt")); model.eval()
d = np.load("data/test64.npz")
cmap = ListedColormap(["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#e6ab02"])
rng = np.random.default_rng(3)
picked = []
for i in range(len(d["landscapes"])):
    ls, m = d["landscapes"][i], d["metrics"][i]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(norm(m[None]))))[0].numpy()
    sols, node, edges = decode_avg(p)
    if local_partition(sols[0]) == local_partition(ls):
        picked.append((ls, m, node, edges))
    if len(picked) == 2:
        break
fig, ax = plt.subplots(2, 5, figsize=(15, 7.2))
for row, (ls, m, node, edges) in enumerate(picked):
    sols = colourings(node.max() + 1, edges, limit=5000)
    alts = [np.array(sols[j])[node] for j in rng.choice(len(sols), 3, replace=False)]
    panels = [(ls, "Original"), (np.array(sols[0])[node], "Rebuilt from metrics only")]
    panels += [(a, f"Alternative labelling {k + 1}") for k, a in enumerate(alts)]
    for k, (img, title) in enumerate(panels):
        w, _ = metric_match(img, m)
        ax[row, k].imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax[row, k].set_title(f"{title}\nwindows with identical metrics: {w:.0%}", fontsize=9)
        ax[row, k].axis("off")
fig.suptitle("64x64 test landscapes (never seen in training): one spatial layout, many valid labellings", fontsize=12)
plt.tight_layout()
plt.savefig("out/reconstruction_examples.png", dpi=110)
print("ok")
