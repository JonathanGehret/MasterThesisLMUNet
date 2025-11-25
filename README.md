# LMUNet: Landscape Metrics U-Net

> **Master's Thesis Project** — Predicting Neutral Landscape Model (NLM) Metrics Using Deep Learning

## 📋 Overview

This repository contains the code and resources for my Master's thesis, which explores the intersection of landscape ecology and deep learning. The core idea is to use a **U-Net neural network** to predict landscape metrics that are traditionally computed using established R packages in landscape ecology.

### The Two-Part Approach

| Part | Language | Purpose |
|------|----------|---------|
| **1. NLM Generation & Metrics** | R | Generate Neutral Landscape Models (NLMs) using the `NLMR` package and compute landscape metrics using `landscapemetrics` |
| **2. U-Net Training** | Python (PyTorch) | Train a modified U-Net to predict those same landscape metrics directly from landscape images |

## 🎯 Research Goal

The traditional calculation of landscape metrics can be computationally expensive, especially at scale. This project investigates whether a deep learning model can learn to predict these metrics efficiently, potentially enabling:

- Faster metric computation for large-scale landscape analysis
- New insights into what visual features correlate with specific metrics
- A bridge between computer vision and landscape ecology

## 🗂️ Repository Structure

```
MasterThesisLMUNet/
├── Masterarbeit_R/                    # R-based NLM generation and metric calculation
│   └── LMUnet/
│       ├── generate_landscapes.R      # Generate NLMs using NLMR package
│       ├── landscapemetrics.R         # Calculate metrics using landscapemetrics
│       ├── NLMR_Overview_and_Tips.Rmd # Documentation for NLMR usage
│       └── ...
├── lmunet_downloadfolder_linuxmint/   # Python notebooks and modules
│   ├── LMUNet_nice_original_working.ipynb  # Main training notebook
│   ├── LMUnet_Imports.py              # Common imports
│   ├── LMUnet_custom_loss_functions.py # Custom loss functions (NaN handling)
│   ├── LMUnet_graph_operations.py     # Graph and visualization utilities
│   ├── LMUnet_optimize_one_landscape.py # Optimization scripts
│   └── LMUNet_metrics*.ipynb          # Various metric training experiments
├── initial--backup/                   # Backup of initial working versions
└── LMUNet_06 (1).ipynb               # Additional notebook version
```

## 🔧 Technical Stack

### R Components
- **NLMR**: Neutral Landscape Model generation
- **landscapemetrics**: Landscape metric calculation (FRAGSTATS-compatible)
- **raster**: Raster data handling
- **landscapetools**: Visualization utilities

### Python Components
- **PyTorch**: Deep learning framework
- **Custom U-Net**: Modified architecture for metric prediction (regression)
- **torcheval**: Evaluation metrics (MSE, R² Score)
- **Custom Loss Functions**: NaN-aware Huber and MSE losses

## ⚠️ Current State & Future Work

> **Note**: This repository is in an archival state from ~2 years ago. The code was originally run via SSH on an HPC cluster and may require adjustments to run on different hardware setups.

### Planned Improvements

- [ ] **Clean up repository**: Remove duplicates and organize files
- [ ] **Make reproducible**: Add proper requirements, environment files, and documentation
- [ ] **Update dependencies**: Ensure compatibility with current package versions
- [ ] **Add data pipeline**: Document the full data generation → training pipeline
- [ ] **Potential publication**: Prepare for academic publication

## 🖥️ Hardware Requirements

The original training was performed on:
- HPC cluster via SSH
- GPU acceleration (CUDA-enabled)
- Significant memory for large landscape datasets (100k+ samples)

## 📊 Key Concepts

### Landscape Metrics Predicted
The model predicts various landscape-level metrics including:
- Aggregation Index (AI)
- Diversity metrics (MSIDI, etc.)
- Core area metrics
- And 60+ other metrics in multi-channel configurations

### Model Architecture
A modified U-Net architecture adapted for:
- **Input**: Landscape raster images (128×128, multi-class categorical)
- **Output**: Metric maps or scalar metric values
- **Training**: MSE/Huber loss with NaN handling for undefined metrics

## 📝 License

*To be added*

## 👤 Author

Jonathan Gehret

---

*This README will be updated as the repository cleanup progresses.*
