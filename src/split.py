"""
split.py — time-aware train/val/test split + train-fit StandardScaler (Task 1.4).

What it does:
    1. Loads data/processed/clean.csv.
    2. Sorts rows by TransactionDT (a seconds offset, not a real timestamp) and
       cuts the timeline into the earliest 70% (train), next 15% (val), last 15%
       (test). A time-aware split avoids leakage: the model is always validated
       and tested on transactions that occur AFTER the ones it trained on.
       (Note: this means fraud ratios will differ slightly across splits — that
       is expected for a temporal split and worth noting in the report.)
    3. Saves the integer row-index arrays to data/processed/splits.npz.
    4. Fits a StandardScaler on the TRAIN rows' numeric feature columns ONLY,
       then saves it (with the fitted column list) to models/scaler.pkl so the
       same transform can be applied to val/test/inference without leakage.

Inputs:  data/processed/clean.csv
Outputs: data/processed/splits.npz   (train_idx, val_idx, test_idx)
         models/scaler.pkl           ({"scaler": StandardScaler, "columns": [...]})

The row indices refer to clean.csv's natural row order (0..N-1), so every
downstream view (sequences, graph) can map an index back to the same row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Columns that are numeric but must NOT be standardized:
#   isFraud        — the target label.
#   TransactionID  — an identifier, not a feature.
#   TransactionDT  — the raw time index used for ordering / temporal ranks;
#                    keep it in original seconds so downstream code can sort by
#                    it (a monotonic scale wouldn't break ordering, but leaving
#                    it raw keeps the value interpretable).
NON_FEATURE_NUMERIC = ["isFraud", "TransactionID", "TransactionDT"]


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric columns to standardize (all numeric minus excludes)."""
    num = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in num if c not in NON_FEATURE_NUMERIC]


def make_splits(verbose: bool = True):
    """Run Task 1.4: time-aware split + train-fit scaler. Returns split indices."""
    if not config.CLEAN_CSV.exists():
        raise ValueError(
            f"{config.CLEAN_CSV} not found — run preprocess.build_clean_csv() "
            f"(Tasks 1.1–1.3) first."
        )

    if verbose:
        print(f"[split] Loading {config.CLEAN_CSV.name} ...")
    df = pd.read_csv(config.CLEAN_CSV)
    n = len(df)
    if "TransactionDT" not in df.columns or "isFraud" not in df.columns:
        raise ValueError("clean.csv must contain 'TransactionDT' and 'isFraud'.")

    # --- Time-aware cut --------------------------------------------------- #
    # Stable sort by time; the sorted index values ARE the original row ids.
    sorted_idx = df.sort_values("TransactionDT", kind="mergesort").index.to_numpy()
    train_frac, val_frac, _ = config.SPLIT
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    train_idx = np.sort(sorted_idx[:n_train])
    val_idx = np.sort(sorted_idx[n_train:n_train + n_val])
    test_idx = np.sort(sorted_idx[n_train + n_val:])

    # --- Integrity asserts (no silent failures) --------------------------- #
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "split sizes != N"
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        "split indices overlap"
    assert s_tr | s_va | s_te == set(range(n)), "splits do not cover all rows"

    # Time-aware ⇒ train's max time <= val's min time <= test's min time.
    dt = df["TransactionDT"].to_numpy()
    assert dt[train_idx].max() <= dt[val_idx].min(), "train/val time overlap"
    assert dt[val_idx].max() <= dt[test_idx].min(), "val/test time overlap"

    np.savez(config.SPLITS_NPZ, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    if verbose:
        y = df["isFraud"].to_numpy()
        print(f"[split] N = {n:,}  ->  train {len(train_idx):,} / "
              f"val {len(val_idx):,} / test {len(test_idx):,}")
        print(f"[split] Fraud ratio  train: {y[train_idx].mean():.4%}  "
              f"val: {y[val_idx].mean():.4%}  test: {y[test_idx].mean():.4%}")
        print(f"[split] (ratios differ across splits — expected for a temporal "
              f"split; note in report)")
        print(f"[split] Saved indices -> {config.SPLITS_NPZ}")

    # --- Fit StandardScaler on TRAIN numeric features only ---------------- #
    num_cols = numeric_feature_columns(df)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_idx, num_cols])
    joblib.dump({"scaler": scaler, "columns": num_cols}, config.SCALER_PKL)

    if verbose:
        # Quick sanity: transformed TRAIN means ~0, std ~1.
        zt = scaler.transform(df.loc[train_idx, num_cols])
        print(f"[split] Fit StandardScaler on {len(num_cols)} numeric feature "
              f"cols (train only).")
        print(f"[split]   train mean(|mean|)~{np.abs(zt.mean(axis=0)).mean():.2e}, "
              f"mean(std)~{zt.std(axis=0).mean():.4f}")
        print(f"[split] Saved scaler -> {config.SCALER_PKL}")

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    config.ensure_dirs()
    config.set_seed(config.SEED)
    make_splits(verbose=True)
