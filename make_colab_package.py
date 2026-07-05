"""
make_colab_package.py — bundle everything needed to train Models 1-4 on Colab.

Builds `colab_package.zip` in the project root with the SAME internal layout as
the project, so that after unzipping on Colab and running from the unzip root,
config.py's relative paths all resolve unchanged:

    config.py
    requirements.txt
    src/*.py
    data/processed/sequences.pt      (transformer branch — all models)
    data/processed/graph.pt          (GNN — all of m2/m3/m4)
    data/processed/splits.npz
    data/processed/clean.csv         (SLIM: only hour/day/TransactionDT/TransactionID
                                      — all that STHGNNBranch reads; ~1 GB -> a few MB)
    models/scaler.pkl
    results/comparison.csv           (only if it already exists locally; normally
                                      ABSENT now — all 4 models train on Colab, which
                                      writes the table fresh from m1 onward)

Why a slim clean.csv: m4's `temporal_features_from_clean` reads clean.csv with
usecols=[hour, day, TransactionDT, TransactionID] and hard-asserts its
TransactionID order matches the graph. A 4-column copy in the same row order
satisfies that exactly, at a tiny fraction of the 1 GB full file. The real local
clean.csv is never modified.

Large .pt files are stored UNCOMPRESSED (float32 tensors barely deflate, and it
keeps zipping fast); text/csv/code are deflated.

Run:  python make_colab_package.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

ZIP_PATH = ROOT / "colab_package.zip"
SLIM_CLEAN_COLS = ["hour", "day", "TransactionDT", "TransactionID"]

# (source path, arcname, store-uncompressed?) — sources are checked for existence.
STORE = zipfile.ZIP_STORED       # no compression (fast; for big binaries)
DEFLATE = zipfile.ZIP_DEFLATED   # compressed (for text/code)


def _build_slim_clean(tmp_path: Path) -> None:
    """Write a 4-column clean.csv (same row order) to tmp_path."""
    if not config.CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"{config.CLEAN_CSV} not found — needed to build the slim clean.csv "
            "for m4's temporal features."
        )
    print(f"[pkg] reading {len(SLIM_CLEAN_COLS)} cols from clean.csv (1 GB) ...")
    df = pd.read_csv(config.CLEAN_CSV, usecols=SLIM_CLEAN_COLS)
    # Preserve the exact column order the loader expects (usecols may reorder).
    df = df[SLIM_CLEAN_COLS]
    df.to_csv(tmp_path, index=False)
    print(f"[pkg] slim clean.csv: {len(df):,} rows -> {tmp_path.stat().st_size/1e6:.1f} MB")


def main() -> None:
    config.set_seed(config.SEED)

    # Required input data files (must exist).
    required = {
        config.SEQUENCES_PT: ("data/processed/sequences.pt", STORE),
        config.GRAPH_PT: ("data/processed/graph.pt", STORE),
        config.SPLITS_NPZ: ("data/processed/splits.npz", STORE),
        config.SCALER_PKL: ("models/scaler.pkl", DEFLATE),
    }
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required data files:\n  " + "\n  ".join(missing)
            + "\nRun the Phase 1-2 build steps first."
        )

    # Code: config.py + every src/*.py + requirements.txt (if present).
    code_files: list[tuple[Path, str]] = [(ROOT / "config.py", "config.py")]
    for py in sorted((ROOT / "src").glob("*.py")):
        code_files.append((py, f"src/{py.name}"))
    req = ROOT / "requirements.txt"
    if req.exists():
        code_files.append((req, "requirements.txt"))

    # Build the slim clean.csv in a temp file we add as data/processed/clean.csv.
    slim_clean = ROOT / "_slim_clean.csv.tmp"
    _build_slim_clean(slim_clean)

    try:
        print(f"[pkg] writing {ZIP_PATH.name} ...")
        with zipfile.ZipFile(ZIP_PATH, "w", allowZip64=True) as zf:
            for src, (arc, comp) in required.items():
                print(f"[pkg]   + {arc}  ({src.stat().st_size/1e6:.1f} MB, "
                      f"{'stored' if comp == STORE else 'deflated'})")
                zf.write(src, arcname=arc, compress_type=comp)

            zf.write(slim_clean, arcname="data/processed/clean.csv", compress_type=DEFLATE)
            print("[pkg]   + data/processed/clean.csv  (slim)")

            for src, arc in code_files:
                zf.write(src, arcname=arc, compress_type=DEFLATE)
            print(f"[pkg]   + {len(code_files)} code files (config.py, src/*.py, requirements.txt)")

            if config.COMPARISON_CSV.exists():
                zf.write(config.COMPARISON_CSV, arcname="results/comparison.csv",
                         compress_type=DEFLATE)
                print("[pkg]   + results/comparison.csv (existing local table)")
            else:
                print("[pkg]   . results/comparison.csv not present - expected; "
                      "Colab writes the full m1-m4 table fresh.")
    finally:
        slim_clean.unlink(missing_ok=True)

    size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"\n[pkg] DONE -> {ZIP_PATH}")
    print(f"[pkg] total zip size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")


if __name__ == "__main__":
    main()
