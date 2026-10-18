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

## Consequence for solving the inverse

1. Metrics → spatial layout (patch boundaries + same/different class for nearby
   pixels): well-posed, can be learned directly.
2. Layout → all landscapes: the valid labelings are the proper colourings of a
   graph of "must differ" patches, which a constraint solver can enumerate,
   count or sample.
