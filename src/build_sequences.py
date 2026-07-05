"""
build_sequences.py — clean.csv -> per-client padded transaction sequences (Task 2.1).

What it does (the "sequence view" of each transaction):
    For every transaction we build a CAUSAL sequence = that transaction plus up
    to the previous MAX_SEQ_LEN-1 transactions of the SAME client_id, ordered by
    TransactionDT. It never looks at future rows, so there is no leakage: at
    inference time a transaction only ever sees itself and its own past. Shorter
    histories are padded at the FRONT with zeros and flagged in a boolean mask.

    Each timestep is described by a compact, defensible feature vector:
      * a curated set of NUMERIC features, standardized with the SAME train-fit
        scaler used everywhere else (models/scaler.pkl) so the sequence and graph
        views share identical numbers (no leakage — scaler was fit on TRAIN only);
      * a curated set of CATEGORICAL features, FREQUENCY-encoded using TRAIN-row
        frequencies (value in [0,1], unseen categories -> 0.0). Frequency encoding
        keeps magnitudes comparable to the standardized numerics and is computed
        from train rows only, so it is also leakage-free.

    We deliberately use a SUBSET of features, not all 397 numeric columns: the
    output tensor is [N, MAX_SEQ_LEN, F]; with F=397 that would be ~19 GB. The
    curated ~32 features keep it to ~1.5 GB while preserving the fraud signal
    (amount, time, per-client behaviour, the C-count family, recency D-fields,
    product/card/email/device/OS/browser).

Inputs:
    data/processed/clean.csv   (rows in natural order 0..N-1)
    data/processed/splits.npz  (train_idx / val_idx / test_idx, same row order)
    models/scaler.pkl          ({"scaler": StandardScaler, "columns": [...]})

Outputs:
    data/processed/sequences.pt  a dict with:
        X_seq          float32 [N, MAX_SEQ_LEN, F]   left-padded sequences
        mask           bool    [N, MAX_SEQ_LEN]      True = REAL timestep, False = pad
        y              int8    [N]                   isFraud label
        transaction_id int64   [N]                   for cross-view alignment (Task 2.3)
        feature_names  list[str] length F            column order of the last axis
        max_seq_len    int                           = MAX_SEQ_LEN

Row order: X_seq[i] corresponds to clean.csv row i, so the indices in splits.npz
index X_seq/mask/y directly.

MASK CONVENTION: mask is a VALID mask (True where a real transaction sits). The
Transformer expects src_key_padding_mask where True = "ignore", so models should
pass `src_key_padding_mask = ~mask`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Curated feature set (kept compact so [N, 20, F] stays ~1.5 GB, not ~19 GB).
# --------------------------------------------------------------------------- #
# HIDDEN COUPLING (read before editing these lists): every column named below
# must SURVIVE preprocess.clean()'s >90%-missing column drop
# (preprocess.MISSING_DROP_THRESHOLD = 0.90). The sparse identity-derived fields
# here — dist1, id_30, id_31 — sit just under that threshold on IEEE-CIS (~60-76%
# missing) and are therefore kept; but if the threshold is lowered, or a different
# dataset slice makes one of these >90% missing, it gets dropped and _load_frame()
# will raise "clean.csv is missing expected columns". Keep this list and that
# threshold in sync. (build_graph.py reuses NUMERIC_FEATURES, so the same applies
# there.)
# Numeric features — standardized with the existing train-fit scaler.
NUMERIC_FEATURES = [
    # amount
    "TransactionAmt", "log_amt", "amt_vs_client_mean",
    # pseudo time-of-day / day-of-week
    "hour", "day",
    # per-client behaviour (causal stats from Task 1.3)
    "client_txn_count", "client_mean_amt", "client_std_amt",
    # distance
    "dist1",
    # the full C-count family (key IEEE-CIS fraud signals)
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    # recency / timedelta fields
    "D1", "D15",
]

# Categorical features — frequency-encoded from TRAIN rows only.
CATEGORICAL_FEATURES = [
    "ProductCD",        # product code
    "card4", "card6",   # card network / type
    "P_emaildomain",    # purchaser email domain
    "DeviceType",       # mobile / desktop
    "id_30",            # operating system
    "id_31",            # browser
]

# Columns we must load beyond the features themselves.
META_COLUMNS = ["client_id", "TransactionDT", "isFraud", "TransactionID"]


def _load_frame() -> pd.DataFrame:
    """Load only the columns we need from clean.csv (keeps memory modest)."""
    if not config.CLEAN_CSV.exists():
        raise ValueError(
            f"{config.CLEAN_CSV} not found — run preprocess (Tasks 1.1-1.3) first."
        )
    usecols = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + META_COLUMNS))
    df = pd.read_csv(config.CLEAN_CSV, usecols=usecols)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"clean.csv is missing expected columns: {missing}")
    return df


def _scaled_numeric(df: pd.DataFrame) -> np.ndarray:
    """Standardize the numeric subset using the saved train-fit StandardScaler.

    We apply the scaler column-by-column (using its stored mean_/scale_) instead
    of transforming all 397 fitted columns — same result, far less memory.
    """
    if not config.SCALER_PKL.exists():
        raise ValueError(
            f"{config.SCALER_PKL} not found — run split.make_splits() (Task 1.4)."
        )
    payload = joblib.load(config.SCALER_PKL)
    scaler, scaler_cols = payload["scaler"], payload["columns"]
    col_to_idx = {c: i for i, c in enumerate(scaler_cols)}

    out = np.empty((len(df), len(NUMERIC_FEATURES)), dtype=np.float32)
    for j, col in enumerate(NUMERIC_FEATURES):
        if col not in col_to_idx:
            raise ValueError(
                f"Numeric feature '{col}' was not in the fitted scaler columns."
            )
        idx = col_to_idx[col]
        mean = scaler.mean_[idx]
        scale = scaler.scale_[idx]
        out[:, j] = (df[col].to_numpy(dtype=np.float64) - mean) / scale
    return out


def _freq_encoded_categorical(df: pd.DataFrame, train_idx: np.ndarray) -> np.ndarray:
    """Frequency-encode categoricals using TRAIN-row frequencies (leakage-free).

    Each category -> its fraction of TRAIN rows; categories never seen in train
    (so unmeasurable) map to 0.0. Magnitudes land in [0,1], comparable to the
    standardized numerics.
    """
    out = np.empty((len(df), len(CATEGORICAL_FEATURES)), dtype=np.float32)
    for j, col in enumerate(CATEGORICAL_FEATURES):
        freq = df.loc[train_idx, col].value_counts(normalize=True)
        out[:, j] = df[col].map(freq).fillna(0.0).to_numpy(dtype=np.float32)
    return out


def _client_local_index(client_ordered: np.ndarray) -> np.ndarray:
    """Per-client 0-based position within a (client, time)-sorted array.

    `client_ordered` is already grouped (all rows of a client are contiguous and
    time-ordered), so a client's i-th transaction has local index i.
    """
    n = len(client_ordered)
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = client_ordered[1:] != client_ordered[:-1]
    # Start position (in the ordered array) of each row's client block.
    start_pos = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    return np.arange(n) - start_pos


def build_sequences(verbose: bool = True):
    """Run Task 2.1: build left-padded causal sequences + mask and save them."""
    splits = np.load(config.SPLITS_NPZ)
    train_idx = splits["train_idx"]

    if verbose:
        print(f"[seq] Loading {config.CLEAN_CSV.name} (selected columns only) ...")
    df = _load_frame()
    n = len(df)

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    f = len(feature_names)
    max_len = config.MAX_SEQ_LEN

    if verbose:
        print(f"[seq] N = {n:,} transactions | F = {f} features "
              f"({len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")
        print(f"[seq] MAX_SEQ_LEN = {max_len}  ->  X_seq target shape [{n}, {max_len}, {f}]")

    # --- Per-timestep feature matrix [N, F] (original row order) ----------- #
    num_mat = _scaled_numeric(df)
    cat_mat = _freq_encoded_categorical(df, train_idx)
    feat_matrix = np.concatenate([num_mat, cat_mat], axis=1).astype(np.float32)
    del num_mat, cat_mat

    # --- Causal ordering: sort by (client_id, TransactionDT), stable ------- #
    order = df.sort_values(["client_id", "TransactionDT"], kind="mergesort").index.to_numpy()
    client_ordered = df["client_id"].to_numpy()[order]
    local = _client_local_index(client_ordered)           # local position per ordered row
    ordered_feats = feat_matrix[order]                     # [N, F] in causal order

    # Keep small arrays, free the big frame before allocating X_seq.
    y = df["isFraud"].to_numpy(dtype=np.int8)
    tid = df["TransactionID"].to_numpy(dtype=np.int64)
    del df, feat_matrix

    # --- Build left-padded windows via a chunked gather -------------------- #
    # Slot k in [0, max_len): offset back o = (max_len-1) - k. Slot max_len-1 is
    # the current transaction (o=0); earlier slots are progressively older. A
    # slot is REAL only if the client actually has that many prior rows.
    X_seq = np.zeros((n, max_len, f), dtype=np.float32)    # zeros == padding
    mask = np.zeros((n, max_len), dtype=bool)              # True == real timestep
    offsets = (max_len - 1) - np.arange(max_len)           # [max_len], current-first=0 at end

    chunk = 20000
    for start in range(0, n, chunk):
        js = np.arange(start, min(start + chunk, n))       # ordered positions in this chunk
        valid = (local[js][:, None] - offsets[None, :]) >= 0       # [c, max_len]
        src = js[:, None] - offsets[None, :]                       # ordered source positions
        src_clamped = np.where(valid, src, 0)
        gathered = ordered_feats[src_clamped]              # [c, max_len, F]
        gathered[~valid] = 0.0                             # zero the padded slots
        orig = order[js]                                   # original clean.csv row ids
        X_seq[orig] = gathered
        mask[orig] = valid

    # --- Acceptance asserts (no silent failures) -------------------------- #
    # 1) Padded positions must be exactly the all-zero rows of X_seq.
    padded = ~mask
    assert not X_seq[padded].any(), "padded positions are not all zero"
    # 2) The current (last) slot is always real for every transaction.
    assert mask[:, -1].all(), "last timestep should never be padded"
    # 3) Number of real timesteps == min(history+1, max_len).
    expected_valid = np.minimum(local + 1, max_len)
    got_valid = np.empty(n, dtype=np.int64)
    got_valid[order] = mask[order].sum(axis=1)
    assert np.array_equal(got_valid[order], expected_valid), "valid-count mismatch"

    if verbose:
        lengths = mask.sum(axis=1)
        print(f"[seq] X_seq {X_seq.shape} ({X_seq.nbytes / 1e9:.2f} GB), "
              f"mask {mask.shape}, y {y.shape}")
        print(f"[seq] Sequence length (real timesteps): "
              f"min {lengths.min()}, mean {lengths.mean():.2f}, "
              f"max {lengths.max()}, % at full {max_len}: "
              f"{(lengths == max_len).mean():.2%}")
        print(f"[seq] Fraud rate: {y.mean():.4%}")

    # --- Save -------------------------------------------------------------- #
    payload = {
        "X_seq": torch.from_numpy(X_seq),                  # float32 [N, L, F]
        "mask": torch.from_numpy(mask),                    # bool    [N, L] True=real
        "y": torch.from_numpy(y),                          # int8    [N]
        "transaction_id": torch.from_numpy(tid),           # int64   [N]
        "feature_names": feature_names,                    # list[str] length F
        "max_seq_len": max_len,
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, config.SEQUENCES_PT)
    if verbose:
        size_mb = config.SEQUENCES_PT.stat().st_size / 1024 / 1024
        print(f"[seq] Saved -> {config.SEQUENCES_PT} ({size_mb:.0f} MB)")
        print(f"[seq] Mask convention: True = real timestep; "
              f"use src_key_padding_mask = ~mask in the Transformer.")

    return payload


if __name__ == "__main__":
    config.ensure_dirs()
    config.set_seed(config.SEED)
    build_sequences(verbose=True)
