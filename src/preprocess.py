"""
preprocess.py — IEEE-CIS data loading, merging, cleaning and feature engineering.

This file grows across Phase 1. Right now it implements **Task 1.1 (Load &
merge)**: read the raw transaction + identity CSVs and left-merge identity onto
transaction on `TransactionID`, then report basic diagnostics.

Inputs:
    data/raw/train_transaction.csv   (target + transaction fields)
    data/raw/train_identity.csv      (device / identity fields)

Outputs (Task 1.1):
    none on disk yet — returns the merged DataFrame and prints diagnostics.
    (clean.csv is written later in Task 1.3.)

Notes on memory: the transaction CSV is ~650 MB on disk and expands to well
over a gigabyte in RAM at default float64/int64 dtypes. We downcast numeric
columns to the smallest safe dtype on load to keep the merged frame manageable
on a CPU-only laptop. No information is lost relevant to fraud modelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable so `import config` works no matter where
# this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in-place to the smallest safe dtype.

    Floats -> float32, integers -> the smallest signed/unsigned int that holds
    the range. Object/categorical columns are left untouched. This roughly
    halves the in-memory footprint of the IEEE-CIS frames.
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], downcast="float")
    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_and_merge(verbose: bool = True) -> pd.DataFrame:
    """Load the raw transaction + identity CSVs and left-merge them.

    Steps:
        1. Verify both raw files exist (clear error if a download is missing).
        2. Read each CSV, downcasting numerics to shrink memory.
        3. Left-merge identity onto transaction on `TransactionID` — a left join
           keeps every transaction; the many transactions with no identity row
           simply get NaN in the identity columns.
        4. Print diagnostics: shape, column count, fraud ratio, top-20 missing.

    Returns:
        The merged pandas DataFrame.

    Raises:
        ValueError if a required raw file or the `isFraud` column is missing.
    """
    tx_path = config.RAW_TRANSACTION_CSV
    id_path = config.RAW_IDENTITY_CSV

    for p in (tx_path, id_path):
        if not p.exists():
            raise ValueError(
                f"Required raw file not found: {p}. Download the IEEE-CIS "
                f"dataset into {config.RAW_DIR} (see README.md)."
            )

    if verbose:
        print(f"[preprocess] Loading transactions from {tx_path.name} ...")
    tx = _downcast(pd.read_csv(tx_path))
    if verbose:
        print(f"[preprocess]   transactions: {tx.shape[0]:,} rows x {tx.shape[1]} cols")
        print(f"[preprocess] Loading identity from {id_path.name} ...")
    idn = _downcast(pd.read_csv(id_path))
    if verbose:
        print(f"[preprocess]   identity:     {idn.shape[0]:,} rows x {idn.shape[1]} cols")

    if "isFraud" not in tx.columns:
        raise ValueError(
            "Expected target column 'isFraud' not found in train_transaction.csv."
        )

    # Left join: keep ALL transactions; identity columns become NaN where a
    # transaction has no matching identity row (which is the common case).
    df = tx.merge(idn, how="left", on="TransactionID")

    if verbose:
        _report(df)

    return df


def _report(df: pd.DataFrame) -> None:
    """Print the Task 1.1 diagnostics: shape, fraud ratio, missing-value table."""
    n_rows, n_cols = df.shape
    fraud_rate = df["isFraud"].mean()
    n_with_identity = df["id_01"].notna().sum() if "id_01" in df.columns else float("nan")

    print("\n" + "=" * 62)
    print("Task 1.1 — Load & merge diagnostics")
    print("=" * 62)
    print(f"Merged shape          : {n_rows:,} rows x {n_cols} columns")
    print(f"isFraud ratio         : {fraud_rate:.4%}  "
          f"({int(df['isFraud'].sum()):,} fraud / {n_rows:,} total)")
    print(f"Rows w/ identity match: {int(n_with_identity):,} "
          f"({n_with_identity / n_rows:.2%})")

    # Top 20 columns by percent missing — sanity check for the cleaning step.
    miss = df.isna().mean().sort_values(ascending=False)
    miss = (miss[miss > 0] * 100).round(2)
    print("\nTop 20 columns by % missing:")
    if miss.empty:
        print("  (no missing values)")
    else:
        for col, pct in miss.head(20).items():
            print(f"  {col:<22} {pct:6.2f}%")
    print("=" * 62 + "\n")


# --------------------------------------------------------------------------- #
# Task 1.2 — Clean
# --------------------------------------------------------------------------- #

# Columns used to build the pseudo-user key. IEEE-CIS has no real user id, so
# we concatenate the most stable card-identity fields (community-standard
# approach). These are all low-missing and never dropped by the >90% rule.
CLIENT_ID_COLS = ["card1", "card2", "card3", "card5", "addr1"]

# Fraction-missing threshold above which a column is dropped entirely. Columns
# this sparse cannot be imputed responsibly and add little signal.
MISSING_DROP_THRESHOLD = 0.90


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean the merged IEEE-CIS frame (Task 1.2).

    Steps (each decision documented inline):
        1. Drop columns with > 90% missing — too sparse to impute usefully.
        2. Build the `client_id` pseudo-user key from raw card/addr fields,
           mapping any NaN component to the literal "NA" (done BEFORE imputation
           so the key reflects truly-missing entities, not median fill-ins).
        3. Impute remaining NaN: numeric -> column median, categorical/object
           -> the string "unknown".

    Row count is never changed (no rows dropped). Returns the cleaned frame.

    Raises:
        ValueError if any required client_id column is absent, or if any NaN
        survives in a kept column (no silent failures).
    """
    df = df.copy()
    n_rows_before = len(df)

    # --- Step 1: drop > 90% missing columns ------------------------------- #
    miss_frac = df.isna().mean()
    drop_cols = miss_frac[miss_frac > MISSING_DROP_THRESHOLD].index.tolist()
    # Never drop the target or the columns we need to build client_id, even in
    # the unlikely event they cross the threshold.
    keep_always = set(CLIENT_ID_COLS) | {"isFraud", "TransactionID", "TransactionDT"}
    drop_cols = [c for c in drop_cols if c not in keep_always]
    df = df.drop(columns=drop_cols)
    if verbose:
        print(f"[clean] Dropped {len(drop_cols)} columns with "
              f">{MISSING_DROP_THRESHOLD:.0%} missing:")
        for c in drop_cols:
            print(f"        - {c:<20} ({miss_frac[c]:.2%} missing)")

    # --- Step 2: build client_id BEFORE imputation ------------------------ #
    # We want the key to reflect the real entity. A NaN component becomes "NA"
    # (not a median fill-in) so genuinely-missing identities aren't collapsed
    # onto an imputed value. Build from string parts to avoid "1.0" style
    # float artefacts: cast to a nullable type, then to str with NaN -> "NA".
    for c in CLIENT_ID_COLS:
        if c not in df.columns:
            raise ValueError(
                f"client_id source column '{c}' missing — cannot build key."
            )
    parts = []
    for c in CLIENT_ID_COLS:
        s = df[c]
        # Integer-valued floats (e.g. card1=13926.0) -> "13926"; keep NaN as NA.
        if pd.api.types.is_float_dtype(s):
            s = s.map(lambda v: "NA" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))
        else:
            s = s.astype("object").map(lambda v: "NA" if pd.isna(v) else str(v))
        parts.append(s)
    df["client_id"] = parts[0].str.cat(parts[1:], sep="_")
    if verbose:
        n_unique = df["client_id"].nunique()
        print(f"[clean] Built client_id from {CLIENT_ID_COLS} "
              f"-> {n_unique:,} unique pseudo-users")

    # --- Step 3: impute remaining NaN ------------------------------------- #
    # Numeric NaN -> median (robust to the heavy-tailed amount/count columns).
    # Categorical/object NaN -> "unknown" (an explicit category the models can
    # learn from rather than a silent gap).
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    n_num_filled = int(df[num_cols].isna().sum().sum())
    n_cat_filled = int(df[cat_cols].isna().sum().sum())

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("unknown")

    if verbose:
        print(f"[clean] Imputed {n_num_filled:,} numeric cells (median) and "
              f"{n_cat_filled:,} categorical cells ('unknown')")

    # --- Acceptance guards ------------------------------------------------ #
    if len(df) != n_rows_before:
        raise ValueError(
            f"Row count changed during cleaning: {n_rows_before} -> {len(df)}."
        )
    remaining_na = df.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if not remaining_na.empty:
        raise ValueError(
            f"NaN still present after cleaning in columns: "
            f"{remaining_na.to_dict()}"
        )
    if "client_id" not in df.columns:
        raise ValueError("client_id column was not created.")

    if verbose:
        print(f"[clean] Done — {df.shape[0]:,} rows x {df.shape[1]} cols, "
              f"0 NaN remaining, client_id present.")
    return df


# --------------------------------------------------------------------------- #
# Task 1.3 — Feature engineering
# --------------------------------------------------------------------------- #

# Engineered columns added by feature_engineer(). Kept as a named list so the
# split/scaling step (Task 1.4) and the model views can reference them.
ENGINEERED_NUMERIC = [
    "hour",
    "day",
    "client_txn_count",
    "client_mean_amt",
    "client_std_amt",
    "amt_vs_client_mean",
    "log_amt",
]


def _add_causal_client_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal (leakage-free) per-client amount stats + amt_vs_client_mean.

    For each transaction, the client stats summarise ONLY that client's
    transactions that happened strictly BEFORE it (ordered by TransactionDT):

        client_txn_count   number of prior transactions  (0 for the first)
        client_mean_amt    mean amount of prior txns      (0 for the first)
        client_std_amt     sample std (ddof=1) of priors  (0 if <2 priors)
        amt_vs_client_mean TransactionAmt / (client_mean_amt + 1), but for a
                           client's FIRST transaction (no history) we fall back
                           to 1.0 — a neutral "matches expectation" value —
                           since dividing by the +1-only denominator would just
                           re-encode the raw amount and be misleading.

    Implemented with shifted cumulative sums (sum and sum-of-squares) so it is
    fully vectorised. Row alignment with the input frame is preserved.
    """
    # Stable sort by (client, time); mergesort keeps original order on DT ties.
    order = df.sort_values(["client_id", "TransactionDT"], kind="mergesort").index
    s = df.loc[order, "TransactionAmt"].to_numpy(dtype="float64")
    cid = df.loc[order, "client_id"].to_numpy()

    g = pd.Series(s).groupby(pd.Series(cid))
    prior_count = g.cumcount().to_numpy(dtype="float64")          # priors before row
    cum_sum_incl = g.cumsum().to_numpy(dtype="float64")           # includes current
    cum_sumsq_incl = (
        pd.Series(s * s).groupby(pd.Series(cid)).cumsum().to_numpy(dtype="float64")
    )
    prior_sum = cum_sum_incl - s
    prior_sumsq = cum_sumsq_incl - (s * s)

    has_prior = prior_count > 0
    safe_count = np.where(has_prior, prior_count, 1.0)            # avoid /0
    mean = np.where(has_prior, prior_sum / safe_count, 0.0)

    # Sample variance over priors: (Σx² - (Σx)²/n) / (n-1); needs >=2 priors.
    has2 = prior_count >= 2
    denom = np.where(has2, prior_count - 1.0, 1.0)
    var = np.where(has2, (prior_sumsq - prior_sum**2 / safe_count) / denom, 0.0)
    var = np.clip(var, 0.0, None)                                # kill fp negatives
    std = np.sqrt(var)

    # First transaction of a client -> neutral ratio 1.0 (documented above).
    ratio = np.where(has_prior, s / (mean + 1.0), 1.0)

    # Reassemble in the sorted order, then restore the original row order.
    stats = pd.DataFrame(
        {
            "client_txn_count": prior_count.astype("int32"),
            "client_mean_amt": mean.astype("float32"),
            "client_std_amt": std.astype("float32"),
            "amt_vs_client_mean": ratio.astype("float32"),
        },
        index=order,
    ).reindex(df.index)

    out = df.copy()
    for col in stats.columns:
        out[col] = stats[col]
    return out


def feature_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Derive time, per-client and amount features (Task 1.3).

    New columns:
        hour                 pseudo hour-of-day  = (TransactionDT // 3600) % 24
        day                  pseudo day-of-week  = (TransactionDT // 86400) % 7
        client_txn_count     # of this client's PRIOR transactions (causal)
        client_mean_amt      mean amount of this client's prior txns (0 if none)
        client_std_amt       std of prior txns (ddof=1; 0 if <2 priors)
        amt_vs_client_mean   TransactionAmt / (client_mean_amt + 1); 1.0 if no prior
        log_amt              log1p(TransactionAmt)

    The per-client stats are CAUSAL — they use only each client's earlier
    transactions, never future ones (corrected from an earlier full-history
    version that leaked future info into past rows).

    NOTE on scaling: scaling must be fit on the TRAIN split only,
    which isn't known yet. So this function computes the RAW engineered columns
    and leaves standardization to `scale_numeric()` (called in Task 1.4 after
    the split). Returns the augmented frame.
    """
    df = df.copy()
    for col in ("TransactionDT", "TransactionAmt", "client_id"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing for Task 1.3.")

    # --- Time features ---------------------------------------------------- #
    # TransactionDT is a seconds offset (not a real timestamp). Floor-divide so
    # hour/day are clean integer buckets in [0,24) / [0,7) rather than floats.
    df["hour"] = (df["TransactionDT"] // 3600 % 24).astype("int16")
    df["day"] = (df["TransactionDT"] // 86400 % 7).astype("int16")

    # --- Per-client aggregates: CAUSAL (no future leakage) ---------------- #
    # IMPORTANT (corrected for causality): each transaction's client stats use
    # ONLY that client's earlier transactions (strictly before it in time), not
    # the full client history. Using the full history would leak future rows
    # into past ones, inconsistent with the causal sequence design in Task 2.1.
    #
    # We sort by (client_id, TransactionDT) — stable sort so tied timestamps
    # keep their original order — then compute expanding count/mean/std over
    # PRIOR rows only via shifted cumulative sums (fast, vectorised).
    df = _add_causal_client_stats(df)
    # log1p compresses the heavy right tail of transaction amounts.
    df["log_amt"] = np.log1p(df["TransactionAmt"]).astype("float32")

    # Engineering must not introduce NaN.
    new_na = df[ENGINEERED_NUMERIC].isna().sum()
    new_na = new_na[new_na > 0]
    if not new_na.empty:
        raise ValueError(f"NaN in engineered columns: {new_na.to_dict()}")

    if verbose:
        print(f"[feateng] Added {len(ENGINEERED_NUMERIC)} columns: "
              f"{ENGINEERED_NUMERIC}")
        print("[feateng] .describe() of engineered columns:")
        with pd.option_context("display.width", 120,
                               "display.max_columns", None):
            print(df[ENGINEERED_NUMERIC].describe().round(3).to_string())

    return df


def scale_numeric(df, numeric_cols, train_mask, scaler=None):
    """Standardize numeric columns, fitting the scaler on TRAIN rows only.

    Deferred from Task 1.3 and called in Task 1.4 once the split is known. If
    `scaler` is None a fresh `sklearn.preprocessing.StandardScaler` is created,
    fit on `df.loc[train_mask, numeric_cols]`, then used to transform ALL rows
    (preventing test/val statistics from leaking into the fit). Returns the
    transformed frame and the fitted scaler.
    """
    from sklearn.preprocessing import StandardScaler  # local import: optional dep

    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df.loc[train_mask, numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


def build_clean_csv(verbose: bool = True):
    """Full Phase-1 pipeline up to Task 1.3: load -> clean -> engineer -> save.

    Writes `data/processed/clean.csv` (raw, unscaled engineered features) and
    returns the DataFrame.
    """
    merged = load_and_merge(verbose=verbose)
    cleaned = clean(merged, verbose=verbose)
    feats = feature_engineer(cleaned, verbose=verbose)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[feateng] Writing {feats.shape[0]:,} rows x {feats.shape[1]} "
              f"cols to {config.CLEAN_CSV} ...")
    feats.to_csv(config.CLEAN_CSV, index=False)
    if verbose:
        size_mb = config.CLEAN_CSV.stat().st_size / 1024 / 1024
        print(f"[feateng] Saved clean.csv ({size_mb:.1f} MB).")
    return feats


if __name__ == "__main__":
    config.ensure_dirs()
    config.set_seed(config.SEED)
    build_clean_csv(verbose=True)
