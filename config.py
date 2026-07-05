"""
config.py — central configuration for the fraud-detection project.

What it does:
    Defines every filesystem path (as pathlib.Path), the global hyperparameters
    reused across all scripts, a reproducibility helper `set_seed`, and a
    device helper `get_device`. Import this module everywhere instead of
    hard-coding paths or magic numbers.

Inputs:  none.
Outputs: none (module of constants + two helper functions). Running it as a
         script creates the project folders, prints the device, and seeds RNGs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# torch is heavy; import lazily-safe at module load so config can still be
# imported in environments where torch isn't installed yet (paths/seeds for
# random/numpy still work). The device/seed helpers require torch.
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch should be installed per requirements
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Paths (all relative to this file so the project is location-independent)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
APP_DIR = ROOT_DIR / "app"

# Common artifact paths
RAW_TRANSACTION_CSV = RAW_DIR / "train_transaction.csv"
RAW_IDENTITY_CSV = RAW_DIR / "train_identity.csv"
CLEAN_CSV = PROCESSED_DIR / "clean.csv"
SPLITS_NPZ = PROCESSED_DIR / "splits.npz"
SEQUENCES_PT = PROCESSED_DIR / "sequences.pt"
GRAPH_PT = PROCESSED_DIR / "graph.pt"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PT = MODELS_DIR / "best_model.pt"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

# All directories that must exist for the pipeline to run.
ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SRC_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    APP_DIR,
]


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Global hyperparameters (reused everywhere — single source of truth)
# --------------------------------------------------------------------------- #
SEED = 42
EMBED_DIM = 128          # output dim of BOTH transformer and GNN branches
MAX_SEQ_LEN = 20         # transactions per client sequence (pad/truncate)
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_FF = 256
GNN_HIDDEN = 128
GNN_LAYERS = 2
# GAT attention heads. The design spec called for 4, but the GAT (Model 3) and
# ST-HGNN (Model 4) checkpoints were trained on the free-tier Colab T4 GPU, whose
# memory could not fit 4-head attention over the full transaction graph (OOM).
# Reduced to 1 head as a documented HARDWARE CONSTRAINT — not an oversight. The
# saved m3_gat.pt / m4_sthgnn.pt checkpoints have a 1-head architecture, so this
# value MUST stay 1 for those checkpoints to load.
GAT_HEADS = 1
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 7  # stop if val PR-AUC doesn't improve
EARLY_STOP_MIN_DELTA = 1e-4  # min val PR-AUC gain over the running best that
                             # counts as a real improvement and RESETS the
                             # early-stop patience counter (the best checkpoint is
                             # still saved on any strict gain). Keeps noise-level
                             # creep from preventing early stopping indefinitely.
# Focal-loss params (Task 4.1 imbalance loss; ~3.5% fraud). gamma down-weights
# easy/well-classified examples; alpha up-weights the positive (fraud) class.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, stratified, time-aware


# --------------------------------------------------------------------------- #
# Reproducibility & device helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs (random, numpy, torch, torch.cuda) for reproducibility.

    Also forces cuDNN into deterministic mode so repeated runs on the same
    hardware produce identical results. Call at the top of every script.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Return the torch device (cuda if available, else cpu) and print it.

    Raises a clear error if torch is not installed, since every model script
    needs it.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install it via requirements.txt before "
            "calling get_device()."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[config] Using device: {device} ({name})")
    return device


if __name__ == "__main__":
    ensure_dirs()
    print(f"[config] Project root: {ROOT_DIR}")
    print(f"[config] Seed: {SEED}")
    set_seed(SEED)
    if _TORCH_AVAILABLE:
        get_device()
        print(f"[config] torch version: {torch.__version__}")
    else:
        print("[config] torch not installed yet (paths/seeds OK; install per requirements.txt).")
    print("[config] All project folders verified/created.")
