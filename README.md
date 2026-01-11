# LMUNet: Landscape Metrics U-Net

> Master's thesis project — Predicting Neutral Landscape Model (NLM) metrics using deep learning


## 📋 Quick summary

LMUNet investigates whether a convolutional model (U-Net) can learn to predict landscape metrics that are usually computed using dedicated R packages. The repository contains the R code used to generate Neutral Landscape Models (NLMs) and compute reference metrics, plus Python code and notebooks to train and evaluate models that predict those metrics from landscape images.


## 🚀 Highlights

- Two-part workflow: R-based NLM generation & metric calculation, and Python-based U-Net training & evaluation
- Custom NaN-aware loss functions to handle undefined metrics
- Multiple experiments and notebooks exploring single- and multi-metric training


## 📁 Repository structure (high-level)

- `Masterarbeit_R/` — R scripts and notes for generating NLMs and computing metrics (uses `NLMR`, `landscapemetrics`)
- `lmunet_downloadfolder_linuxmint/` — Python notebooks and modules (training, losses, graphs, optimizers, visualization)
- `initial--backup/` — project backups
- `older versions/` — legacy copies (ignored by `.gitignore`)


## 🧭 Getting started

### Recommended (short) checklist

1. Create a Python environment and install dependencies (see **Missing items** below). Example:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt` (not included — see **Missing items**)
2. If reproducing NLMs / metrics: install R and the required packages (`NLMR`, `landscapemetrics`, `raster`, `landscapetools`).
3. Run the R scripts in `Masterarbeit_R/` to generate landscapes and metrics, or use precomputed data (if available).
4. Open the main training notebook `lmunet_downloadfolder_linuxmint/LMUNet_nice_original_working.ipynb` and follow the cells for data loading, training, and evaluation.


## 🧾 Notable files

- `Masterarbeit_R/generate_landscapes.R` — NLM generation
- `Masterarbeit_R/landscapemetrics.R` — metric calculations (reference implementation)
- `lmunet_downloadfolder_linuxmint/LMUnet_Imports.py` — shared imports and helpers
- `lmunet_downloadfolder_linuxmint/LMUnet_custom_loss_functions.py` — NaN-aware loss functions
- `lmunet_downloadfolder_linuxmint/LMUnet_graph_operations.py` — visualization & helper functions
- `lmunet_downloadfolder_linuxmint/LMUnet_optimize_one_landscape.py` — optimization experiments
- Notebooks (`LMUNet_*.ipynb`) — various experiments and metric analyses


## 📚 Data

- The repository does not contain raw landscape datasets by default (to avoid large files).
- Expect training data to be folders of raster images and precomputed metric labels; add your dataset path in the notebooks or helper scripts when running experiments.


## ⚙️ Technical stack

- R: `NLMR`, `landscapemetrics`, `raster`, `landscapetools`
- Python: `pytorch`, `torcheval` (used for metrics), common utils (numpy, pandas, matplotlib)


## ✅ What I changed / added (README only)
- Rewrote README to provide clear Getting Started steps, file references, and a short checklist of missing items/next steps.


## ⚠️ Known missing items & recommendations

- Requirements: **`requirements.txt`** has been added — install with `pip install -r requirements.txt`. Note: install `torch` according to your CUDA/CPU setup (see PyTorch installation instructions).
- License: there is no license file. Add `LICENSE` (e.g., MIT, CC-BY) so others know how they can use the code.
- Tests / CI: add a small test suite and a basic CI (GitHub Actions) to run linters and tests on push.
- Data samples: include a small sample dataset and a script/notebook to show end-to-end reproducibility (generation → training → evaluation).
- File cleanup: there are many duplicated/legacy files (e.g., files with ` (1)` suffix and the `older versions/` folder). Consider a cleanup pass and canonical naming.


## 🛠️ Suggested next steps

1. (completed) `requirements.txt` has been added. Optional: create an `environment.yml` for conda and an R requirements script (e.g., `install_R_packages.R`) if you want a conda-based setup.
2. Add a `LICENSE` file and a brief `CONTRIBUTING.md` if you plan to accept external contributions.
3. Add a small sample dataset with a short `examples/` notebook that runs end-to-end quickly.
4. Introduce basic tests and a CI pipeline (lint, unit tests, notebook check).


## 🤝 Contributing

If you'd like help cleaning up the repo (renaming files, consolidating notebooks), I can assist with a focused cleanup plan and scripted refactors. I can also help scaffold `requirements.txt`, `environment.yml`, or a GitHub Actions workflow.


## 📬 Contact

Author: Jonathan Gehret


---

*If you'd like, I can now (pick one):*
- Add a `requirements.txt` with the packages inferred from the repository, or
- Create an `environment.yml` for conda use, or
- Add a simple `LICENSE` (MIT) and `CONTRIBUTING.md` template.

Tell me which of the above you'd like me to do next.