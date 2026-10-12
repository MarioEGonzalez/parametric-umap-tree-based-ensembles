# Jacket Bolt-Loosening Classification: PCA and Parametric UMAP

Research code and experimental results supporting the manuscript *“Experiment-Wise Bolt-Loosening Classification in Jacket-Type Wind Turbine Structures Using Parametric UMAP and Tree-Based Ensembles.”*

This repository provides the computational workflow developed for vibration-based multiclass bolt-state classification in a laboratory-scale offshore wind turbine jacket structure.

## Research Overview

The study considers four structural conditions: Healthy, 6Nm, 9Nm, and NoBolt. The experimental dataset comprises 80 independent experiments, with vibration responses measured through 24 acceleration channels.

Vibration signals are segmented into overlapping windows and transformed into frequency-domain representations using the real Fast Fourier Transform (rFFT). The resulting 12,312-dimensional spectral features are standardized and reduced to 32 dimensions using two approaches:

- Principal Component Analysis (PCA).
- Supervised Parametric UMAP.

Five tree-based classifiers are evaluated: ExtraTrees, Random Forest, LightGBM, XGBoost, and CatBoost.

Model evaluation follows a five-fold experiment-wise cross-validation protocol, ensuring that windows originating from the same experiment remain within the same external partition.

## Repository Contents

- `scripts/` — Python scripts for signal preprocessing, FFT feature extraction, experiment-wise data partitioning, dimensionality reduction, classifier training, and result verification.
- `results/` — Selected audited results, including external out-of-fold (OOF) classification metrics, experimental partitions, experiment-level error analyses, and prediction comparisons.
- `verification_checks.json` — Verification records supporting the consistency and integrity of the supplied results.
- `requirements.txt` — Python package dependencies for the computational workflow.

The original experimental recordings are publicly available separately and are not redistributed in this repository.

## How to Run

Python 3.10 is recommended for compatibility with the research environment.

### 1. Prepare the environment

Create a Python 3.10 virtual environment and install the project dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
python -m pip install -r requirements.txt
```

The compatibility of TensorFlow and Parametric UMAP should be considered when configuring the environment.

### 2. Obtain and organize the experimental data

Download the original experimental vibration recordings from the research data repository:

https://doi.org/10.34810/data1011

Select the recordings corresponding to structural Level 1 and white-noise excitation amplitude 1.

Organize the extracted recordings under the `A_1/` directory at the project root, preserving the class subdirectories expected by the preprocessing script.

Expected input location:

```text
project_root/
├── A_1/
│   ├── Healthy/
│   ├── 6Nm/level_1/
│   ├── 9Nm/level_1/
│   └── NoBolt/level_1/
├── scripts/
├── results/
├── requirements.txt
└── README.md
```

The `A_1/` directory contains locally downloaded experimental recordings and is not distributed with this repository.

### 3. Prepare FFT features and experimental folds

Run:

```bash
python scripts/00_prepare_fft_and_group_folds.py
```

This stage prepares the spectral features and experiment-wise partitions required by the subsequent processing stages.

### 4. Generate dimensionality-reduced representations

Run the representation workflows:

```bash
python scripts/01_train_parametric_umap.py
python scripts/02_train_pca_baseline.py
```

These scripts generate the Parametric UMAP and PCA representations used for classifier evaluation.

### 5. Train and evaluate classifiers

Run:

```bash
python scripts/03_train_lightgbm_extratrees.py
```

Continue with the remaining classifier-training scripts provided in `scripts/`, using the generated upstream representations and fold definitions.

### 6. Verify and inspect results

Execute the relevant verification and analysis scripts after their required intermediate outputs have been generated.

Selected audited external OOF results and consistency checks are available in `results/` and `verification_checks.json`.

### Project directory configuration

The project root can be configured with the `WILEY_PROJECT_DIR` environment variable.

For example, on Windows PowerShell:

```powershell
$env:WILEY_PROJECT_DIR = (Get-Location).Path
```

On Linux or macOS:

```bash
export WILEY_PROJECT_DIR="$(pwd)"
```

Run the scripts from the repository root. Some training stages require substantial computational resources.

## Computational Workflow

The complete research methodology comprises five main stages:

1. **Data preparation:** Load vibration measurements, construct overlapping windows, extract rFFT magnitude features, and establish experiment-wise partitions.
2. **Representation learning:** Standardize spectral features and generate 32-dimensional PCA and supervised Parametric UMAP representations.
3. **Classifier training:** Train and tune ExtraTrees, Random Forest, LightGBM, XGBoost, and CatBoost using group-disjoint internal validation.
4. **External evaluation:** Aggregate held-out predictions from the five external folds and compute classification metrics.
5. **Verification and analysis:** Examine classification errors, compare predictions, and summarize the evaluated pipelines.

## Main Experimental Results

The evaluation comprises 3,840 held-out vibration windows originating from 80 independent experiments.

| Representation | Classifier | Accuracy | Macro-F1 | MCC | Errors |
|---|---|---:|---:|---:|---:|
| Parametric UMAP | ExtraTrees | 0.9870 | 0.9871 | 0.9828 | 50 |
| Parametric UMAP | CatBoost | 0.9870 | 0.9871 | 0.9828 | 50 |
| PCA | XGBoost | 0.9844 | 0.9842 | 0.9793 | 60 |

Parametric UMAP combined with ExtraTrees or CatBoost achieved the highest external OOF classification metrics among the evaluated pipelines. Both classifiers produced identical predicted labels across the 3,840 external test windows.

PCA combined with XGBoost provided the strongest PCA-based baseline.

The reported metrics are computed at the window level using predictions from held-out experimental groups. Multiple windows from the same experiment are correlated and should not be interpreted as independent experimental replicates.

## Reproducibility and Verification

The repository contains selected research scripts, experimental results, and verification records documenting the computational methodology.

The supplied resources describe the principal preprocessing procedures, representation configurations, classifier settings, experiment-wise validation protocol, and selected audited results reported in the manuscript.

Running individual analysis stages requires the original dataset, a compatible computational environment, and the intermediate artifacts specified by the corresponding scripts.

## Data Availability

The original vibration recordings are available through the public research data repository:

**Experimental dataset DOI:** https://doi.org/10.34810/data1011

The accompanying dataset publication is:

Valdez-Yepez, R., Tutivén, C., and Vidal, Y. (2024). *Structural Health Monitoring of Jacket-Type Support Structures in Offshore Wind Turbines: A Comprehensive Dataset for Bolt Loosening Detection through Vibrational Analysis*. Data in Brief, 53, 110222.

**Publication DOI:** https://doi.org/10.1016/j.dib.2024.110222

The original experimental data are not redistributed in this GitHub repository.

## Citation

If you use the research code or experimental results provided here, please cite the associated manuscript and the original dataset publication.
