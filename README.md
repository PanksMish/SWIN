# Swin-FANE + BiLSTM-ResNet: Emotion Recognition & Temporal Stress Estimation

> **Paper:** "A Two-Phase Swin–FANE and BiLSTM–ResNet Framework for Emotion Recognition and Temporal Stress Estimation"

---

## Overview

This repository provides the **full PyTorch implementation** of the two-phase spatial–temporal framework described in the paper:

| Phase | Component | Purpose |
|-------|-----------|---------|
| I | **Swin-FANE** (Swin Transformer + Facial Attention Network Embedding) | Frame-level emotion recognition |
| II | **BiLSTM-ResNet** (Bidirectional LSTM + Residual Refinement) | Temporal stress estimation |

The pipeline converts facial video frames → emotion probability vectors → stress index, without requiring explicit stress annotations.

---

## Repository Structure

```
swin_fane/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── data/
│   ├── fane_dataset.py          # FANE dataset loader (primary)
│   ├── fer2013_dataset.py       # FER2013 dataset loader (alternative)
│   ├── video_dataset.py         # Video sequence dataset loader
│   └── augmentations.py         # Custom augmentation pipeline
│
├── models/
│   ├── swin_transformer.py      # Swin Transformer backbone
│   ├── fane_module.py           # Facial Attention Network Embedding
│   ├── swin_fane.py             # Combined Swin-FANE spatial encoder
│   ├── bilstm_resnet.py         # BiLSTM + Residual temporal module
│   └── full_framework.py        # End-to-end two-phase framework
│
├── utils/
│   ├── metrics.py               # Evaluation metrics (ETSI, TVE, SCS, etc.)
│   ├── stress_formulation.py    # Deterministic stress index computation
│   ├── visualisation.py         # Grad-CAM, trajectory plots, confusion matrix
│   └── logger.py                # Training logger
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── infer_video.py           # Single video inference
│   └── run_ablation.py          # Ablation study runner
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/swin-fane-stress.git
cd swin-fane-stress
pip install -r requirements.txt
```

### 2. Dataset Preparation

**Option A — FANE Dataset (Primary)**
```
data/
└── FANE/
    ├── images/
    │   ├── anger/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happiness/
    │   ├── sadness/
    │   ├── surprise/
    │   └── neutral/
    └── masks/          # Expressive region masks (optional)
```

**Option B — FER2013 (Alternative)**
Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
```
data/
└── FER2013/
    ├── train/
    └── test/
```

### 3. Train

```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

### 5. Infer on Video

```bash
python scripts/infer_video.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth
```

---

## Results

| Model | Accuracy (%) | Precision (%) | F1-score (%) | ROC-AUC |
|-------|-------------|--------------|-------------|---------|
| Mujiyanto et al. | 88.4 | 87.9 | 88.1 | 0.924 |
| Xue et al. | 89.5 | 89.0 | 89.2 | 0.931 |
| Chen et al. | 90.1 | 90.0 | 90.0 | 0.938 |
| Bie et al. | 89.2 | 89.0 | 89.1 | 0.933 |
| **Swin-FANE (Ours)** | **90.7 ± 0.6** | **91.2** | **90.8** | **0.947** |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 32 |
| Epochs | 50 |
| Sequence Length | 16 frames |
| Stress α | 0.7 |
| Stress β | 0.3 |

---

## Citation

```bibtex
@article{swin_fane_2025,
  title={A Two-Phase Swin–FANE and BiLSTM–ResNet Framework for 
         Emotion Recognition and Temporal Stress Estimation},
  year={2025}
}
```

---

## License

MIT License
