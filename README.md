# LMUNet — Landscape Metrics U‑Net

> Master's thesis project: predicting Neutral Landscape Model (NLM) metrics from landscape images using a U‑Net.

---

## Overview

LMUNet explores whether convolutional models (U‑Nets) can learn to predict landscape metrics that are commonly computed with R packages. The repository contains R scripts for generating Neutral Landscape Models and calculating reference metrics, alongside Python code and notebooks for training and evaluating models that estimate those metrics from image inputs.

## Quick start

1. Prerequisites
   - Python 3.8+ (virtual environment recommended)
   - R (required for NLM generation and metric calculation)
   - GPU or CPU: install PyTorch appropriate for your system (see https://pytorch.org for the correct install command)

2. Setup (example)
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements.txt

3. Data
   - Generate Neutral Landscape Models and metric labels using the R scripts in `Masterarbeit_R/`, or use your own dataset formatted as image folders with accompanying metric labels.
   - Update dataset paths in the notebooks or helper scripts before running experiments.

4. Run an experiment
   - Open `LMUnet_python/LMUNet_nice_original_working (1).ipynb` and follow the cells for data loading, training, and evaluation.

## Repository layout (selected)

- `Masterarbeit_R/` — R scripts for NLM generation and metric calculations
- `LMUnet_python/` — Python modules and notebooks (training, losses, optimization, visualization)
- `LMUnet_python/older versions/` — legacy copies and alternate experiment files

## Notable files

- `Masterarbeit_R/generate_landscapes_calculate_metrics.R` — generate NLMs and compute metrics
- `Masterarbeit_R/generate_landscapes_functions.R` — helper functions for NLM generation
- `LMUnet_python/LMUnet_Imports.py` — shared imports and helpers
- `LMUnet_python/LMUnet_custom_loss_functions.py` — NaN-aware loss functions
- `LMUnet_python/LMUnet_graph_operations (4).py` — visualization helpers
- `LMUnet_python/LMUnet_optimize_one_landscape (5).py` — optimization experiments
- `LMUnet_python/*.ipynb` — experiment notebooks (data management, metrics, inference, landscape generation)

## Data and reproducibility

- This repository does not include large raw datasets by default. Expect to supply or generate training data (images + metric labels).
- For a minimal reproduce run, generate a small set of landscapes via the `Masterarbeit_R/` scripts and point the notebooks at that data.

## Notes

- `requirements.txt` is included; install with `pip install -r requirements.txt` and install PyTorch separately to match your CUDA/CPU configuration.
- No license file is included in the repository.
- There are legacy/duplicate files stored in `older versions/` and similar backups.

## Contact

Author: Jonathan Gehret

---

(End of README)
