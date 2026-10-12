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
- `requirements.txt` — Python package dependencies used in the computational workflow.

The original experimental recordings are publicly available through the dataset cited in the manuscript and are not redistributed in this repository.

## Computational Workflow

The analysis follows these main stages:

1. **Data preparation:** Load the vibration measurements, construct overlapping windows, extract rFFT magnitude features, and establish experiment-wise partitions.
2. **Representation learning:** Standardize the spectral features and generate 32-dimensional PCA and supervised Parametric UMAP representations.
3. **Classifier training:** Train and tune ExtraTrees, Random Forest, LightGBM, XGBoost, and CatBoost using group-disjoint internal validation.
4. **External evaluation:** Aggregate held-out predictions from the five external folds and compute classification performance metrics.
5. **Verification and analysis:** Examine classification errors, compare predictions, and summarize results across the evaluated pipelines.

Python 3.10 is recommended. The computational workflow uses the directory organization documented in the supplied scripts. The project root can be configured through the `WILEY_PROJECT_DIR` environment variable.

### Experimental Data Preparation

The original vibration measurements must be obtained separately from the publicly available dataset referenced in the Data Availability section.

The analysis uses the subset corresponding to structural Level 1 and a white-noise excitation amplitude of 1, comprising 20 independent experiments for each of the four structural conditions.

After obtaining the dataset, organize the recordings locally according to the input directory structure expected by `00_prepare_fft_and_group_folds.py`. The preprocessing script expects the corresponding experimental class directories under `data/A_1/` within the configured project root.

The local input directories and experimental recordings are not included in this repository.

## Main Experimental Results

The evaluation comprises 3,840 vibration windows originating from 80 independent experiments.

| Representation | Classifier | Accuracy | Macro-F1 | MCC | Errors |
|---|---|---:|---:|---:|---:|
| Parametric UMAP | ExtraTrees | 0.9870 | 0.9871 | 0.9828 | 50 |
| Parametric UMAP | CatBoost | 0.9870 | 0.9871 | 0.9828 | 50 |
| PCA | XGBoost | 0.9844 | 0.9842 | 0.9793 | 60 |

Parametric UMAP combined with ExtraTrees or CatBoost achieved the highest external OOF classification metrics among the evaluated pipelines. Both classifiers produced identical predicted labels across the 3,840 external test windows.

PCA combined with XGBoost provided the strongest PCA-based baseline.

These results correspond to window-level predictions obtained from held-out experimental groups. Windows originating from the same experiment are correlated and are not treated as independent experiments.

## Reproducibility and Verification

The repository includes research scripts, selected experimental results, and verification records documenting the computational methodology.

The supplied files preserve the principal preprocessing procedures, representation configurations, classifier settings, experiment-wise validation protocol, and selected audited results reported in the manuscript.

Reproducing an analysis requires the original dataset, the appropriate computational environment, and the intermediate artifacts specified by the corresponding scripts.

The verification records document the consistency of the reported experimental results and the integrity of the experiment-wise evaluation procedure.

## Data Availability

The original vibration dataset is publicly available through:

Valdez-Yepez, R., Tutivén, C., and Vidal, Y. (2024). *Structural Health Monitoring of Jacket-Type Support Structures in Offshore Wind Turbines: A Comprehensive Dataset for Bolt Loosening Detection through Vibrational Analysis*. Data in Brief, 53, 110222.

DOI: https://doi.org/10.1016/j.dib.2024.110222

## Citation

If you use the code or results provided in this repository, please cite the associated research manuscript and the original experimental dataset.

## License

Please refer to the repository license, if provided, for the applicable terms of code reuse.
