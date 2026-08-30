# Semiconductor Wafer Defect Detection
### CNN-based classification of wafer map failure patterns — WM-811K Dataset

---

## Overview

This project builds a Convolutional Neural Network (CNN) to automatically classify semiconductor wafer maps into one of **9 failure pattern categories** using the WM-811K dataset (811,457 wafer entries). Model performance is evaluated using **Cohen's Kappa score**, which accounts for class imbalance — making it a more meaningful metric than raw accuracy for this dataset.

---

## Problem Statement

In semiconductor manufacturing, defects on a wafer surface leave distinct spatial signatures depending on their root cause. Identifying the failure type early in the production line allows engineers to trace and fix the source quickly, reducing waste and improving yield.

The 9 failure categories are:

| Class | Description |
|---|---|
| `none` | No defect pattern (healthy wafer) |
| `Center` | Clustered defects in the centre of the wafer |
| `Donut` | Ring-shaped defect band at mid-radius |
| `Edge-Ring` | Defects concentrated along the outer edge |
| `Edge-Loc` | Localised defects at one section of the edge |
| `Loc` | Localised blob of defects away from the edge |
| `Scratch` | Thin linear defect streak across the wafer |
| `Random` | Randomly scattered defects with no pattern |
| `Near-full` | Defects covering almost the entire wafer |

---

## Dataset

**WM-811K** — available on Kaggle:
[https://www.kaggle.com/code/paulbassaler/defect-detection-in-wafer-bin-maps/data](https://www.kaggle.com/code/paulbassaler/defect-detection-in-wafer-bin-maps/data)

The dataset is a pickled pandas DataFrame (`LSWMD.pkl`) with the following columns:

| Column | Description |
|---|---|
| `waferMap` | 2-D integer array. `0` = no die, `1` = normal die, `2` = defective die |
| `dieSize` | Total number of dies on the wafer (float) |
| `lotName` | Production lot identifier (string) |
| `waferIndex` | Position of the wafer within its lot (float) |
| `trianTestLabel` | Original train/test split label, stored as `[['Training']]` |
| `failureType` | Failure class label, stored as `[['Center']]`, or `[]` if unlabeled |

**Key dataset facts:**
- 811,457 total rows
- ~172,950 labeled wafers (rest are unlabeled — `failureType == []`)
- 46,293 distinct lots
- `none` makes up ~85% of labeled data — severely imbalanced

---

## Preprocessing Pipeline

Three non-obvious challenges required careful handling:

### 1. Nested Label Unwrapping
Labels arrive as `[['Center']]`, not `'Center'`. A custom `unwrap_label()` function extracts the plain string. Rows with `failureType == []` (unlabeled) are dropped entirely before training.

### 2. Nearest-Neighbor Resizing
Every wafer map has a different H×W shape depending on die size and lot. A CNN needs a fixed input shape, so all maps are resized to **32×32**. Standard bilinear/bicubic interpolation is intentionally avoided — pixel values `{0, 1, 2}` are **categories**, not intensities, so interpolation would invent meaningless fractional values like `1.4`. Nearest-neighbor index sampling only ever selects an existing integer value.

### 3. 3-Channel Categorical Encoding
Instead of feeding the raw `{0, 1, 2}` map as a single grayscale channel, each map is one-hot encoded into **3 binary channels**:
- Channel 0 → background (`pixel == 0`)
- Channel 1 → normal die (`pixel == 1`)
- Channel 2 → defective die (`pixel == 2`)

A single channel would imply an ordinal relationship ("2 is twice as large as 1"), which is false. Three binary channels let early conv filters learn *where the defects are* independently of *where the wafer edge is*.

### 4. Class Weighting
Inverse-frequency class weights are computed and passed to `model.fit()` so that errors on rare classes (Near-full, Donut) are penalised proportionally more. Without this, the model would simply predict `none` for everything — high accuracy, Kappa ≈ 0.

---

## Model Architecture

A 3-block CNN feature pyramid followed by a dense classification head.

```
Input (32, 32, 3)
│
├── Conv2D(32, 3×3, same, ReLU) → BatchNorm → MaxPool(2×2)   [→ 16×16]
├── Conv2D(64, 3×3, same, ReLU) → BatchNorm → MaxPool(2×2)   [→  8×8 ]
├── Conv2D(128, 3×3, same, ReLU) → BatchNorm → GlobalAvgPool [→  128  ]
│
├── Dense(64, ReLU)
├── Dropout(0.4)
└── Dense(9, Softmax)
```

**Key design decisions:**

- **3×3 kernels** — large enough to capture local defect clusters, small enough for a 32×32 input
- **`padding='same'`** — preserves border pixels, critical because Edge-Loc and Edge-Ring defects live at the wafer boundary
- **GlobalAveragePooling2D instead of Flatten** — collapses 8×8×128 to 128 values (~65× fewer parameters going into Dense layers), significantly reducing overfitting on rare classes
- **Dropout(0.4)** — strong regularisation for rare classes prone to memorisation
- **ReLU** activations in all hidden layers — avoids vanishing gradients
- **Softmax** output — mutually exclusive multi-class probabilities
- **Adam (lr=1e-3)** — adaptive per-parameter learning rates, robust to noisy gradients from class weighting
- **`sparse_categorical_crossentropy`** — used because labels are integer-encoded, not one-hot

**Total trainable parameters: 102,985**

---

## Training

- Epochs: up to **40** (upper bound)
- Batch size: 128
- Split: **70% train / 15% val / 15% test** (stratified)

Two callbacks control the actual stopping point:

| Callback | Setting | Purpose |
|---|---|---|
| `EarlyStopping` | `patience=8, restore_best_weights=True` | Halts training if val_loss doesn't improve for 8 epochs, rolls back to best weights |
| `ReduceLROnPlateau` | `factor=0.5, patience=4` | Halves learning rate on val_loss plateau, allowing finer convergence |

---

## Results

| Metric | Value |
|---|---|
| Cohen's Kappa (test set) | **0.996** |
| Epochs run | 29 of 40 |
| Best weights restored from | Epoch 21 |

A Kappa of 0.996 falls in the **"almost perfect agreement"** band (Landis & Koch scale: 0.81–1.00).

---

## Repository Structure

```
wafer-defect-detection/
│
├── wafer_defect_detection.py   # Full pipeline (load → preprocess → train → evaluate)
├── README.md                   # This file
└── wafer_defect_cnn.keras      # Saved model weights (generated after running)
```

> **Note:** `LSWMD.pkl` is not included in this repo due to file size (~3 GB). Download it from the Kaggle link above and update `DATA_PATH` in the script before running.

---

## How to Run

### On Google Colab (recommended)

1. Upload `LSWMD.pkl` to your Google Drive
2. Open `wafer_defect_detection.py` in Colab
3. Update `DATA_PATH` on line 43 to your file path
4. Uncomment the two Google Drive mount lines above it
5. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
6. Run all cells

### Local

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
python wafer_defect_detection.py
```

---

## Dependencies

```
numpy
pandas
tensorflow >= 2.x
scikit-learn
matplotlib
seaborn
```

---

## References

- WM-811K Dataset: [Kaggle — Defect Detection in Wafer Bin Maps](https://www.kaggle.com/code/paulbassaler/defect-detection-in-wafer-bin-maps/data)
- Cohen's Kappa interpretation: Landis, J.R. & Koch, G.G. (1977). *The measurement of observer agreement for categorical data.* Biometrics, 33(1), 159–174.