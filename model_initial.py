# train_cnn_gru_knee_health.py
# -------------------------------------------------------
# 1D CNN + GRU for knee health classification
# - Time-series branch: 12 channels x 120 readings
# - Static branch: BMI, Age, activity_type_encoded
# - Labels: label_encoded (0=Bad, 1=Moderate, 2=Healthy)
# -------------------------------------------------------

import os
import ast
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input, Model


# -----------------------
# Config
# -----------------------
CSV_PATH = r"your_dataset_directory"  
SEQ_LEN = 120

# Time-series feature columns (pipe-separated strings)
# You listed Acceleration_lower_y twice; we include unique cols and try to add _z if present.
TS_CANDIDATES = [
    "Acceleration_upper_x", "Acceleration_upper_y", "Acceleration_upper_z",
    "Velocity_upper_x", "Velocity_upper_y", "Velocity_upper_z",
    "Velocity_lower_x", "Velocity_lower_y", "Velocity_lower_z",
    "Acceleration_lower_x", "Acceleration_lower_y", "Acceleration_lower_z"  # use if available
]

# Static numeric columns
STATIC_COLS = ["BMI", "Age", "activity_type_encoded"]

LABEL_COL = "label_encoded"  # 0=Bad, 1=Moderat, 2=Healthy


# -----------------------
# Utilities
# -----------------------
def parse_pipe_series_to_float_array(s: str) -> np.ndarray:
    """
    Converts a pipe-separated string 'a|b|c' to a float numpy array.
    Handles NAs and empty strings gracefully.
    """
    if pd.isna(s):
        return np.array([], dtype=np.float32)
    # Ensure string
    s = str(s).strip()
    if not s:
        return np.array([], dtype=np.float32)
    parts = [p.strip() for p in s.split("|") if p.strip() != ""]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            # Try to handle cases like brackets or commas accidentally present
            p2 = p.replace("[", "").replace("]", "").replace(",", "")
            try:
                vals.append(float(p2))
            except:
                # give up for that token
                continue
    return np.array(vals, dtype=np.float32)


def pad_or_truncate(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Pad with last value (or zeros if empty) or truncate to length seq_len.
    """
    if len(arr) == 0:
        return np.zeros(seq_len, dtype=np.float32)
    if len(arr) == seq_len:
        return arr
    if len(arr) > seq_len:
        return arr[:seq_len]
    # pad
    pad_val = arr[-1] if len(arr) > 0 else 0.0
    padded = np.pad(arr, (0, seq_len - len(arr)), mode="constant", constant_values=pad_val)
    return padded.astype(np.float32)


def build_time_series_tensor(df: pd.DataFrame, ts_cols: List[str], seq_len: int) -> np.ndarray:
    """
    Build a 3D tensor (N, seq_len, C) from pipe-separated time-series columns.
    """
    # Only keep columns that actually exist
    ts_cols = [c for c in ts_cols if c in df.columns]
    # Deduplicate if any accidental repeats
    ts_cols = list(dict.fromkeys(ts_cols))

    N = len(df)
    C = len(ts_cols)
    X_ts = np.zeros((N, seq_len, C), dtype=np.float32)

    for i, col in enumerate(ts_cols):
        series_arrays = df[col].apply(parse_pipe_series_to_float_array).apply(lambda a: pad_or_truncate(a, seq_len))
        stacked = np.stack(series_arrays.values)  # (N, seq_len)
        X_ts[:, :, i] = stacked

    return X_ts, ts_cols


def standardize_time_series_channels(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Channel-wise z-score over the entire dataset (flatten time+batch for each channel).
    Returns standardized X and per-channel (mean, std).
    """
    N, T, C = X.shape
    X_std = np.empty_like(X)
    means = np.zeros(C, dtype=np.float32)
    stds = np.ones(C, dtype=np.float32)

    for c in range(C):
        vec = X[:, :, c].reshape(-1)
        m = np.mean(vec)
        s = np.std(vec) + 1e-8
        means[c] = m
        stds[c] = s
        X_std[:, :, c] = (X[:, :, c] - m) / s

    return X_std, means, stds


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    Simple confusion matrix plot.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# -----------------------
# Build Model (CNN + GRU + Static)
# -----------------------
def build_cnn_gru_model(seq_len: int, n_channels: int, static_dim: int, n_classes: int = 3) -> Model:
    # Time-series branch
    ts_in = Input(shape=(seq_len, n_channels), name="ts_input")
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(ts_in)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    ts_out = layers.Dropout(0.25)(x)

    # Static branch
    st_in = Input(shape=(static_dim,), name="static_input")
    s = layers.Dense(32, activation='relu')(st_in)
    s = layers.Dropout(0.2)(s)

    # Fuse
    h = layers.Concatenate()([ts_out, s])
    h = layers.Dense(64, activation='relu')(h)
    h = layers.Dropout(0.25)(h)
    logits = layers.Dense(n_classes, activation='softmax')(h)

    model = Model(inputs=[ts_in, st_in], outputs=logits)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# -----------------------
# Main
# -----------------------
def main():
    # Load data
    df = pd.read_csv(CSV_PATH)

    # Validate required columns
    missing = [c for c in [LABEL_COL] + STATIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Build time-series tensor
    X_ts, used_ts_cols = build_time_series_tensor(df, TS_CANDIDATES, SEQ_LEN)
    print(f"Time-series channels used ({len(used_ts_cols)}): {used_ts_cols}")

    # Extract static features
    X_static = df[STATIC_COLS].copy()
    # Coerce to numeric
    for c in STATIC_COLS:
        X_static[c] = pd.to_numeric(X_static[c], errors="coerce")
    # Simple impute if needed
    X_static = X_static.fillna(X_static.median(numeric_only=True))
    # Scale static features
    static_scaler = StandardScaler()
    X_static = static_scaler.fit_transform(X_static.values)

    # Standardize time-series channels
    X_ts, ch_means, ch_stds = standardize_time_series_channels(X_ts)

    # Labels
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int).values

    # Train/Val/Test split (stratified)
    Xts_train, Xts_temp, Xst_train, Xst_temp, y_train, y_temp = train_test_split(
        X_ts, X_static, y, test_size=0.30, random_state=42, stratify=y
    )
    Xts_val, Xts_test, Xst_val, Xst_test, y_val, y_test = train_test_split(
        Xts_temp, Xst_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {Xts_train.shape}, Val: {Xts_val.shape}, Test: {Xts_test.shape}")

    # Build model
    model = build_cnn_gru_model(seq_len=SEQ_LEN, n_channels=X_ts.shape[2], static_dim=X_static.shape[1], n_classes=3)
    model.summary()

    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint("best_cnn_gru.keras", monitor='val_loss', save_best_only=True)
    ]

    # Train - FIXED: Pass inputs as list instead of dictionary
    history = model.fit(
        [Xts_train, Xst_train], y_train,
        validation_data=([Xts_val, Xst_val], y_val),
        epochs=50,
        batch_size=16,
        callbacks=cb,
        verbose=1
    )

    # Evaluate on test - FIXED: Pass inputs as list instead of dictionary
    y_prob = model.predict([Xts_test, Xst_test], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== Test Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec_w:.4f}")
    print(f"Recall    (weighted): {rec_w:.4f}")
    print(f"F1        (weighted): {f1_w:.4f}")
    print(f"Precision (macro)   : {prec_m:.4f}")
    print(f"Recall    (macro)   : {rec_m:.4f}")
    print(f"F1        (macro)   : {f1_m:.4f}\n")

    print("Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["Bad(0)", "Moderate(1)", "Healthy(2)"], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    plot_confusion_matrix(cm, classes=["Bad(0)", "Moderate(1)", "Healthy(2)"], normalize=False, title="Confusion Matrix")
    plot_confusion_matrix(cm, classes=["Bad(0)", "Moderate(1)", "Healthy(2)"], normalize=True, title="Confusion Matrix (Normalized)")


if __name__ == "__main__":
    # Make TF quiet
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Fix Unicode encoding issues on Windows
    import sys
    if sys.platform.startswith('win'):
        import locale
        import codecs
        # Set UTF-8 encoding for stdout
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    

    main()
