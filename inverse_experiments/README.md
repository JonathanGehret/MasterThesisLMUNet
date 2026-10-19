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

### Training with noise

`python train.py 12 0.3 data/inverse_clean.pt`: continue from the clean model,
adding Gaussian noise (sd drawn uniformly from 0 to 0.3, normalised units) to
each training example (`out/train_noise.log`, `out/eval32_noise_trained.jsonl`,
`out/eval32_noisemodel_clean.log`).

| 32x32 test, input noise sd | Clean-trained model: exact layout / windows exact | Noise-trained model |
|---|---|---|
| 0 (network only) | 68.5 % / 99.2 % | 70.5 % / 99.6 % |
| 0 (network + repair) | 78.5 % / 99.6 % | 80.5 % / 99.8 % |
| 0.05 | 42 % / 98.5 % | 74 % / 99.5 % |
| 0.1 | 8 % / 91 % | 78 % / 99.7 % |
| 0.2 | 0 % / 52 % | 65 % / 99.6 % |

(n = 100 for noisy rows, so differences of a few % are within chance.)

Limits: Gaussian noise is only a stand-in for real U-Net errors, which are
correlated in space and between metrics. 128x128 not tested.

## Part 3: exact finisher, realistic (U-Net-like) errors, 128x128

### Exact finisher (`finisher.py`)

After `repair`, any window whose metrics still do not match is fixed by exact
search with the real metric formulas:
1. whole-patch relabel moves,
2. an exact search over the labels of all patches near the mismatch,
3. an exact search over the labels of *all* patches on the map (shapes kept),
   for label errors that chain across the map,
4. an exhaustive search over small sets of pixels.

Because the layout is unique, "every window matches" means the answer is
right. In every test, all-windows-match coincided with the exact layout, and
every failure was flagged by remaining mismatched windows.

32x32 test set, noise-trained model, exact metrics (`out/eval32_finisher.log`,
`out/retry_failures.log`):

| Step | Exact layout |
|---|---|
| network | 70.5 % |
| + repair | 80.5 % |
| + finisher (without step 3) | 95.5 % |
| + step 3, on the 9 remaining failures | 6 of 9 fixed, so 197/200 = 98.5 % |

### Forward model as a stand-in for the thesis U-Net (`forward.py`)

Landscape -> 56 metric maps, trained on the same 12 000 landscapes (8 epochs):
median R2 0.94 across metrics, mean 0.85 (`out/train_forward.log`), roughly
the thesis U-Net's level. Its predictions are used as realistic, imperfect
inputs.

32x32, input = forward-model predictions, network only (`out/eval32_predicted.jsonl`):

| Inverse model | Exact layout | Windows with all metrics exact |
|---|---|---|
| trained with Gaussian noise | 3 % | 82.5 % |
| fine-tuned on forward predictions (`train.py 8 0.3 ... data/forward.pt`) | 27 % | 96.3 % |

Gaussian noise was a poor stand-in for real model errors; training on the
forward model's own predictions fixes most of the gap.

### 128x128 (`eval128.py`, `out/eval128_full.log`)

12 test landscapes at the thesis size, inverse model fine-tuned on forward
predictions (trained only on 32x32 maps):

| Input / step | Exact layout | Windows with all metrics exact |
|---|---|---|
| exact metrics, network | 2 of 12 | 98.5 % |
| exact metrics, + repair | 3 of 12 | 99.0 % |
| exact metrics, + repair + finisher | 6 of 12 | 99.1 % |
| forward-predicted metrics, network | 1 of 12 | 97.1 % |

One 128x128 map takes about 1 min to decode, 2.5 min to repair, and up to
~15 min in the finisher (pure Python). The unsolved maps still have mismatched
windows, so they are flagged rather than silently wrong.

## Part 4: networks only (no exact metric formulas at run time)

Goal: forward and backward with networks only, plus at most small, fast steps.

### Forward v2 (`forward2.py`)
Same shape as `forward.py` (5x5 conv + per-pixel MLP, width 384, 6 layers) but
trained on the 9.25 million *distinct* 5x5 windows of the 32x32 and 64x64
training maps (each window once, random rotations/flips/relabelling), 9 epochs
(5 + 4 resumed; `out/train_forward2*.log`).

| Forward model | mean R2 | median R2 | worst metric R2 |
|---|---|---|---|
| v1 (`forward.py`) | 0.85 | 0.94 | - |
| v2 (`forward2.py`) | 0.98 | 0.995 | 0.85 |

Note: only ~3.2 M distinct windows occur in 12 M windows of the 32x32 data; a
lookup table of exact values covers 80 % of windows in new 128x128 maps
(`window_table.py`), an exact alternative to a forward network.

### Thesis method with softmax fix (`gumbel_invert.py`)
Optimising 5 scores per pixel through the frozen forward model
(straight-through Gumbel-softmax), 32x32 test maps:

| Start | Forward model | Exact maps | Windows exact |
|---|---|---|---|
| random (best of 5 settings) | v1 | 0 / 8 | 30 % |
| random | v2 | 0 / 8 | 11 % |
| inverse network's answer (already exact) | v1 | stays exact | 100 % |
| inverse network's answer, 16 hard cases | v1 / v2 | no improvement | 97 % |

From random starts it gets stuck far from the truth (loss 0.3-1.1 vs 0.06/0.009
for the true maps). With v1, the wrong answer scored better than the truth in
4/16 hard cases, so the forward model could not even tell them apart.

### Inverse v2 (`train2.py`) + decoding fixes
Trained on 32x32 and 64x64 maps; half of each batch uses forward-v2 predicted
metrics. Decoding: labelling falls back to a weighted tabu search
(`tabu_colouring`) instead of a greedy guess, with constraints weighted by the
evidence behind them (summed over pixel pairs). The greedy fallback had merged
whole patches on 3 of 12 128x128 maps.

### NN polish (`nn_polish.py`)
For pairs of nearby label groups separated on weak evidence, try giving them the
same label and keep it if *forward v2* says the metrics fit better. No exact
formulas; ~0.5 s (32x32) to ~1 s (128x128) on top of decoding.

Results, network only vs network + NN polish (`out/eval_polish_v2.jsonl`):

| Map size | Input | Exact maps: network | + NN polish | Windows exact (+ polish) |
|---|---|---|---|---|
| 32x32 (200) | exact metrics | 78.5 % | 86.5 % | 99.8 % |
| 32x32 (200) | forward-v2 predicted | 74.5 % | 82.5 % | 99.8 % |
| 64x64 (60) | exact metrics | 40 % | 53 % | 99.8 % |
| 64x64 (60) | forward-v2 predicted | 42 % | 58 % | 99.8 % |
| 128x128 (12) | exact metrics | 2 / 12 | 3 / 12 | 99.75 % |
| 128x128 (12) | forward-v2 predicted | 2 / 12 | 3 / 12 | 99.7 % |

Timing per 128x128 map: networks ~0.1 s each, decoding ~10 s (pure Python).
Remaining errors are mostly label decisions between patches that only just
share a window (1-3 pixel pairs of evidence).

## Consequence for solving the inverse

1. Metrics → spatial layout (patch boundaries + same/different class for nearby
   pixels): well-posed, can be learned directly.
2. Layout → all landscapes: the valid labelings are the proper colourings of a
   graph of "must differ" patches, which a constraint solver can enumerate,
   count or sample.
