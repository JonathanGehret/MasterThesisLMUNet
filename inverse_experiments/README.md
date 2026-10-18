# Is the inverse problem (metrics → landscape) unique?

Question: given the 5×5 moving-window maps of the thesis' landscape-level metrics,
how many landscapes produce exactly those maps?

## Method

- `run_experiment.py` generates random-cluster landscapes (Python port of
  `NLMR::nlm_randomcluster`, p in 0.1–0.6, 5 classes) and computes the window
  maps (NA outside the landscape, like `window_lsm` in R).
- `enumerate_preimages.py` then does an **exhaustive** search for every
  landscape that gives identical maps. Global relabelings (5! = 120) are counted
  once.
- Two kinds of maps were tested:
  - `metrics`: the 56 lsm_l_* metrics of `final_metrics.csv`, re-implemented in
    Python (`lsm_metrics.py`), because R packages could not be installed here.
  - `summary`: a per-window summary from which all of those metrics follow
    (`window_stats.py`).
- Each solution is reduced to its **local partition**: for every pixel pair
  that shares a window, are they the same class? Different local partitions
  mean different spatial layouts. Same local partition but different labels
  means only distant patches were relabeled.

## Results (`out/*.jsonl`, `noise_test.py`)

| Landscapes | Distinct spatial layouts | Label variants |
|---|---|---|
| 3 × 8×8 random cluster | 1 each | 1–3 |
| 10 × 16×16 random cluster | 1 each | 1–12 |
| 5 × 32×32 random cluster | 1 each | 192 to ≥100 000 |
| 6 × 12×12 pure noise (2 and 5 classes) | 1 each | 1 |

In every case the metric maps fix the spatial layout uniquely. The only
remaining freedom is which label patches get when they are too far apart to
share a window; that number grows quickly with landscape size.

Sanity check: with a summary carrying no information the search finds
thousands of layouts; with only "number of classes per window" it cannot finish,
so the search does detect ambiguity when it exists.

## Caveats

- Exact metric values were used. The U-Net's predicted metrics are noisy, so a
  practical inverse needs a learned model that is robust to that noise.
- Metrics are a Python re-implementation; small formula differences to R are
  possible but should not change which windows are distinguishable.
- Small landscapes only (exhaustive search); 128×128 was not enumerated.

## Part 2: solving it (metrics -> landscape)

Pipeline (all files in this folder):

1. `make_dataset.py`: 12 000 random-cluster landscapes (32x32) + their 56 window
   metric maps for training; separate 32x32 and 64x64 test sets.
2. `inverse_model.py` / `train.py`: a small CNN gets the metric maps and predicts,
   for every pixel and 40 nearby offsets (within one window), the probability
   that the two pixels have the same class. Label-free target, so the arbitrary
   class numbers never enter training. 15 epochs on CPU (~1 h).
3. `decode.py` (`decode_avg`): average-linkage clustering of pixels into
   "same class" groups, "must differ" constraints between groups, then a
   colouring solver assigns labels and can list all valid labellings.
4. `repair.py`: optional greedy fix-up using the exact metric formulas.
5. `eval_parallel.py` / `eval_inverse.py`: recompute exact metrics of the
   rebuilt landscape and compare with the target metric maps.

Results (`out/eval*_clean.log`, `out/eval32_noise.jsonl`):

| Test set | Exact layout | Windows with all 56 metrics exact |
|---|---|---|
| 32x32, network only | 68.5 % | 99.2 % |
| 32x32, network + repair | 78.5 % | 99.6 % |
| 64x64 (never trained on), network only | 30 % | 97.9 % |
| 64x64, network + repair | 43 % | 98.7 % |
| 32x32, input noise sd 0.05 (normalised units) | 42 % | 98.5 % |
| 32x32, input noise sd 0.1 | 8 % | 91 % |
| 32x32, input noise sd 0.2 | 0 % | 52 % |

The first decoder (`decode`, simple union of neighbours) reached only 44 % /
87 % on 32x32: one wrong link merged two whole patches.

`out/reconstruction_examples.png`: two 64x64 test landscapes, their
reconstruction from metrics only, and three alternative labellings; all have
identical metrics in 100 % of windows.

Limits: the network was trained on exact metrics only, so it is fragile to
noise (U-Net-predicted metrics with R2 ~0.95 are roughly the sd 0.2 case);
training with noise is the obvious next step. 128x128 not tested.

## Consequence for solving the inverse

1. Metrics → spatial layout (patch boundaries + same/different class for nearby
   pixels): well-posed, can be learned directly.
2. Layout → all landscapes: the valid labelings are the proper colourings of a
   graph of "must differ" patches, which a constraint solver can enumerate,
   count or sample.
