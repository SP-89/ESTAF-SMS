# ESTAF-SMS: EEG Spatiotemporal Attention Framework for Stroke Motor Score Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **ESTAF-SMS**, a deep learning framework for continuous motor function assessment from resting-state EEG in stroke patients. Published in *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*.

## Overview

ESTAF-SMS predicts clinical motor scores—ADL (0–100), FMA (0–100), and FMA-UE (0–66)—directly from multi-channel EEG signals. The framework integrates:

- **Band-power spectral feature extraction** with per-patient normalization
- **1D-CNN** for local spatiotemporal feature encoding
- **LSTM** for global temporal dependency modeling
- **Dual-stream aggregation**: temporal attention + global statistics pooling
- **SHAP-based interpretability** linking predictions to neurophysiological biomarkers

## Results (5-fold CV, mean ± std)

| Model | FMA-UE R | FMA R | ADL R |
|-------|----------|-------|-------|
| **ESTAF-SMS** | **0.80 ± 0.02** | **0.81 ± 0.03** | **0.68 ± 0.07** |
| Standard CNN | 0.73 | 0.64 | 0.39 |
| Standard LSTM | 0.71 | 0.52 | 0.28 |
| FBCNet (Mane 2021) | 0.06 | 0.04 | 0.08 |
| TSCeption (Ding 2022) | 0.27 | 0.23 | 0.07 |
| SVR | 0.26 | 0.09 | 0.17 |
| Random Forest | 0.18 | 0.12 | 0.15 |

## Installation

```bash
git clone https://github.com/yourusername/ESTAF-SMS.git
cd ESTAF-SMS
pip install -r requirements.txt
```

## Data Preparation

1. Place raw `.mat` EEG files in `mat_files/`
2. Place patient metadata Excel (`总表.xlsx`) in the project root
3. Convert MAT to H5 format:
```bash
python model/preprocess.py
```

**Data format**: 29-channel resting-state EEG, 1000 Hz sampling, PSD features extracted via Welch's method.

## Usage

### Train ESTAF-SMS
```bash
python model/train.py
```
Performs 5-fold stratified cross-validation with data augmentation.

### Compare with Published Baselines
```bash
python model/baselines.py
```
Runs FBCNet and TSCeption on the same data for fair comparison.

### SHAP Interpretability Analysis
```bash
python model/interpretability.py
```
Generates SHAP summary plots, spatial topomaps, and spectral importance visualizations.

### PSD Visualization
```bash
python model/visualize.py
```
Creates comprehensive PSD visualizations for each subject.

## Project Structure

```
ESTAF-SMS/
├── TNSRE.tex                 # LaTeX source for the paper
├── model/
│   ├── train.py               # Core ESTAF-SMS training pipeline
│   ├── baselines.py  # Published models comparison
│   ├── preprocess.py             # MAT to H5 preprocessing
│   ├── interpretability.py                  # Enhanced training with SHAP analysis
│   ├── visualize.py                  # PSD visualization tools
│   ├── legacy_baselines.py           # Original baseline comparison
│   ├── h5_files/             # Preprocessed H5 data (not tracked)
│   ├── mat_files/            # Raw MAT data (not tracked)
│   └── shap_results/         # SHAP output figures
├── figures/                  # Paper figures
├── requirements.txt
└── README.md
```

## Key Features

- **No data leakage**: Per-patient normalization; feature selection within each CV fold
- **Continuous regression**: Predicts scores across full clinical ranges (0-66, 0-100)
- **Interpretable**: SHAP values identify Beta/Gamma bands in frontal-temporal-central regions
- **Reproducible**: Fixed random seeds, 5-fold stratified CV

## Citation

```bibtex
@article{shi2025estaf,
  title={ESTAF-SMS: An EEG Spatiotemporal Attention Framework for Stroke Motor Score Prediction},
  author={Shi, Peng and Li, Wei and Kong, Yun and Cheng, Long and Mo, Linhong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Peng Shi: shipeng1@stu.ynu.edu.cn
- Wei Li (Corresponding): wei.li@ia.ac.cn
- Linhong Mo (Corresponding): molinhong@163.com
