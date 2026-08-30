# =============================================================================
# Semiconductor Wafer Defect Detection — WM-811K
# CNN-based classification of wafer map failure patterns
# Problem Statement 4 (PS-4)
#
# Dataset: https://www.kaggle.com/code/paulbassaler/defect-detection-in-wafer-bin-maps/data
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
# Mount Google Drive and point DATA_PATH to where you saved LSWMD.pkl
#
# from google.colab import drive
# drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/LSWMD.pkl'

with open(DATA_PATH, 'rb') as f:
    df = pickle.load(f)

print("Raw dataset shape:", df.shape)
print(df.head())


# ── 3. LABEL CLEANING ─────────────────────────────────────────────────────────
# failureType and trianTestLabel are stored as doubly-nested lists,
# e.g. [['Center']] instead of plain strings.
# Rows where failureType == [] are unlabeled wafers — drop them entirely.

def unwrap_label(x):
    """Unwraps [['Center']] -> 'Center'. Returns None for [] (unlabeled)."""
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        inner = x[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            return inner[0]
        return inner
    return None

df['failureLabel'] = df['failureType'].apply(unwrap_label)
df['splitLabel']   = df['trianTestLabel'].apply(unwrap_label)

print("\nFailure label distribution (including unlabeled):")
print(df['failureLabel'].value_counts(dropna=False))

df_labeled = df.dropna(subset=['failureLabel']).reset_index(drop=True)
print("\nRows after dropping unlabeled wafers:", len(df_labeled))
print("\nClass distribution used for training:")
print(df_labeled['failureLabel'].value_counts())


# ── 4. VISUALISE CLASS DISTRIBUTION ───────────────────────────────────────────

plt.figure(figsize=(10, 4))
df_labeled['failureLabel'].value_counts().plot(kind='bar', color='#3b6ea5')
plt.yscale('log')
plt.ylabel('Count (log scale)')
plt.xlabel('Failure Type')
plt.title('Class Distribution Across Failure Types')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
plt.show()


# ── 5. VISUALISE SAMPLE WAFER MAPS ────────────────────────────────────────────

sample_classes = ['none', 'Center', 'Donut', 'Edge-Ring', 'Scratch']
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, cls in zip(axes, sample_classes):
    row = df_labeled[df_labeled['failureLabel'] == cls].iloc[0]
    ax.imshow(row['waferMap'], cmap='gray_r', interpolation='nearest')
    ax.set_title(cls, fontsize=10)
    ax.axis('off')
plt.suptitle('Sample Wafer Maps by Failure Type', fontsize=11)
plt.tight_layout()
plt.show()


# ── 6. PREPROCESSING ──────────────────────────────────────────────────────────

TARGET_SIZE = 32  # resize all maps to 32x32

def resize_wafer_map(wmap, target_size=TARGET_SIZE):
    """
    Resize a variable-sized wafer map to target_size x target_size using
    nearest-neighbor index sampling.

    IMPORTANT: pixel values {0, 1, 2} are CATEGORIES (background / normal die
    / defective die), NOT intensities. Standard bilinear or bicubic
    interpolation would invent meaningless fractional values (e.g. 1.4).
    Nearest-neighbor sampling only ever selects an existing integer value.
    """
    wmap = np.asarray(wmap, dtype=np.float32)
    h, w = wmap.shape
    row_idx = np.linspace(0, h - 1, target_size).astype(int)
    col_idx = np.linspace(0, w - 1, target_size).astype(int)
    return wmap[row_idx][:, col_idx]


def encode_channels(wmap_resized):
    """
    One-hot encode the 3 categorical pixel values into 3 binary channels:
        channel 0 = background  (pixel == 0)
        channel 1 = normal die  (pixel == 1)
        channel 2 = defective die (pixel == 2)

    A single grayscale channel would imply an ordinal relationship
    ('2 is twice as large as 1'), which is false here. Three binary channels
    let early conv filters learn 'where are the defects' independently of
    'where is the wafer edge'.
    """
    out = np.zeros((wmap_resized.shape[0], wmap_resized.shape[1], 3),
                   dtype=np.float32)
    out[..., 0] = (wmap_resized == 0).astype(np.float32)  # background
    out[..., 1] = (wmap_resized == 1).astype(np.float32)  # normal die
    out[..., 2] = (wmap_resized == 2).astype(np.float32)  # defective die
    return out


def preprocess_wafer(wmap):
    return encode_channels(resize_wafer_map(wmap))


# Visualise before vs after resizing
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for j, cls in enumerate(sample_classes):
    row  = df_labeled[df_labeled['failureLabel'] == cls].iloc[0]
    orig = np.array(row['waferMap'])
    res  = resize_wafer_map(orig)
    axes[0, j].imshow(orig, cmap='gray_r', interpolation='nearest')
    axes[0, j].set_title(f"{cls}\n{orig.shape}", fontsize=9)
    axes[0, j].axis('off')
    axes[1, j].imshow(res,  cmap='gray_r', interpolation='nearest')
    axes[1, j].set_title(f"{res.shape}",   fontsize=9)
    axes[1, j].axis('off')
plt.suptitle('Original (top) vs. Resized 32x32 Nearest-Neighbor (bottom)',
             fontsize=10)
plt.tight_layout()
plt.show()


# ── 7. BUILD X / y ARRAYS ─────────────────────────────────────────────────────
# Memory note: at float32, N x 32 x 32 x 3 bytes.
# For ~172K labeled wafers this is roughly 2 GB — manageable on Colab.
# If you run out of memory, uncomment the downsampling block below.

def build_arrays(dataframe, batch_size=5000):
    """Convert waferMap column to a stacked numpy array, in batches."""
    n     = len(dataframe)
    X     = np.zeros((n, TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
    wmaps = dataframe['waferMap'].values
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            X[i] = preprocess_wafer(wmaps[i])
        print(f"  Processed {end}/{n}")
    return X

print("\nBuilding feature array X ...")
X = build_arrays(df_labeled)
print("X shape:", X.shape, f"| Memory: {X.nbytes / 1e6:.1f} MB")

le = LabelEncoder()
y  = le.fit_transform(df_labeled['failureLabel'])
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))



# ── 8. TRAIN / VAL / TEST SPLIT ───────────────────────────────────────────────
# Stratified split ensures rare classes (Near-full, Donut) appear in all
# three sets proportionally. Without stratify=y, a random split could put
# zero examples of a rare class in the validation set.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


# ── 9. CLASS WEIGHTS ──────────────────────────────────────────────────────────
# 'none' makes up ~85% of labeled data. Without correction the model learns
# to predict 'none' for everything — high accuracy, Kappa ≈ 0.
# Balanced class weights penalise errors on rare classes proportionally more.

class_weights_arr  = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict  = {
    int(c): float(w)
    for c, w in zip(np.unique(y_train), class_weights_arr)
}
print("\nClass weights:")
for idx, w in class_weight_dict.items():
    print(f"  {le.classes_[idx]:12s}: {w:.3f}")


# ── 10. MODEL ─────────────────────────────────────────────────────────────────

def build_model(input_shape, num_classes):
    """
    3-block CNN feature pyramid:
      Block 1 — Conv2D(32)  + BN + MaxPool  → 16x16
      Block 2 — Conv2D(64)  + BN + MaxPool  →  8x8
      Block 3 — Conv2D(128) + BN + GAP      → 128-d vector
      Head    — Dense(64, ReLU) + Dropout(0.4) + Dense(9, Softmax)

    Design notes:
      - 3x3 kernels: large enough for local defect clusters, small enough
        for a 32x32 input.
      - padding='same': preserves border pixels — critical because Edge-Loc
        and Edge-Ring defects live at the wafer boundary.
      - ReLU: avoids vanishing gradients, computationally cheap.
      - GlobalAveragePooling2D instead of Flatten: 8x8x128 → 128 values,
        drastically fewer parameters, less overfitting on rare classes.
      - Dropout(0.4): strong regularisation for rare classes (Near-full,
        Donut) that are prone to memorisation.
      - Softmax output: mutually exclusive multi-class probabilities.
      - Loss: sparse_categorical_crossentropy (integer-encoded labels).
      - Optimizer: Adam lr=1e-3, adaptive per-parameter steps, robust to
        noisy gradients from class weighting.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # → 16x16

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # → 8x8

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)      # → 128-d

    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='wafer_defect_cnn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


model = build_model(X_train.shape[1:], num_classes)
model.summary()


# ── 11. TRAINING ──────────────────────────────────────────────────────────────
# Epochs capped at 40 (as per problem statement).
# EarlyStopping(patience=8) stops early and restores best weights if
# val_loss doesn't improve — prevents overfitting in later epochs.
# ReduceLROnPlateau(patience=4) halves LR on plateaus for finer convergence.

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# ── 12. TRAINING CURVES ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'],     label='Train Loss', marker='o', ms=3)
axes[0].plot(history.history['val_loss'], label='Val Loss',   marker='o', ms=3)
axes[0].set_title('Loss vs Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'],     label='Train Accuracy', marker='o', ms=3)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy',   marker='o', ms=3)
axes[1].set_title('Accuracy vs Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ── 13. EVALUATION ────────────────────────────────────────────────────────────

test_probs = model.predict(X_test, batch_size=256)
test_preds = test_probs.argmax(axis=1)

kappa = cohen_kappa_score(y_test, test_preds)
print(f"\nCohen's Kappa Score (test set): {kappa:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, test_preds,
    target_names=le.classes_,
    zero_division=0
))


# ── 14. CONFUSION MATRIX ──────────────────────────────────────────────────────

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cbar_kws={'shrink': 0.8}
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix — Test Set (Kappa = {kappa:.3f})')
plt.xticks(rotation=40, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ── 15. SAVE MODEL ────────────────────────────────────────────────────────────

model.save('wafer_defect_cnn.keras')
print("\nModel saved to wafer_defect_cnn.keras")

# To reload later:
# model = keras.models.load_model('wafer_defect_cnn.keras')
