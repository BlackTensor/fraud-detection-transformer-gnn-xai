"""
compute_confusion.py — confusion matrix for the best model on the test split.

What it does:
    Loads the winning checkpoint (default m3_gat = Transformer + GAT), runs it on
    the SAME held-out TEST split used in Phase 5, applies that model's best-F1
    decision threshold (read from results/comparison.csv), and saves the four
    confusion-matrix counts (TN/FP/FN/TP) plus precision/recall to a small JSON.

    This is a one-off, offline precompute so the Streamlit Analysis & Findings page
    can render a clean confusion-matrix heatmap WITHOUT loading the 1.5 GB sequence
    tensor at view time.

Inputs:  sequences.pt, splits.npz, graph.pt, models/<ckpt>.pt, results/comparison.csv
Output:  results/confusion_<ckpt>.json
Run with:  python src/compute_confusion.py            (defaults to m3_gat)
           python src/compute_confusion.py --model m2 (any MODEL_SPECS key)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from train import (  # noqa: E402
    MODEL_SPECS,
    SequenceDataset,
    _build_gnn_branch,
    load_sequence_view,
    load_split_indices,
)
from evaluate import _timed_predict  # noqa: E402


def _best_f1_threshold_for(ckpt_name: str, fallback: float = 0.5) -> tuple[float, str]:
    """Read the best-F1 threshold for `ckpt_name` from comparison.csv (else fallback)."""
    import pandas as pd

    if config.COMPARISON_CSV.exists():
        cmp = pd.read_csv(config.COMPARISON_CSV)
        row = cmp.loc[cmp["model"] == ckpt_name]
        if len(row) and not pd.isna(row.iloc[0].get("best_f1_threshold")):
            return float(row.iloc[0]["best_f1_threshold"]), "best-F1 threshold (comparison.csv)"
    return fallback, "default 0.5"


def main(model_key: str = "m3") -> dict:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"model_key must be one of {list(MODEL_SPECS)}, got {model_key!r}.")
    spec = MODEL_SPECS[model_key]

    config.set_seed(config.SEED)
    device = config.get_device()

    from transformer_model import SequenceTransformer
    from fusion_model import FusionClassifier
    from gnn_models import _load_graph

    print(f"[confusion] {model_key}: {spec['label']} — loading views ...", flush=True)
    X_seq, mask, y, _txn_id, f = load_sequence_view()
    _train_idx, _val_idx, test_idx = load_split_indices()
    graph = _load_graph()
    print(f"[confusion] test split = {len(test_idx):,} rows", flush=True)

    full_ds = SequenceDataset(X_seq, mask, y, node_idx=torch.arange(X_seq.shape[0]))
    test_loader = DataLoader(Subset(full_ds, test_idx.tolist()),
                             batch_size=config.BATCH_SIZE, shuffle=False)

    ckpt_path = config.MODELS_DIR / f"{spec['ckpt']}.pt"
    if not ckpt_path.exists():
        raise ValueError(f"{ckpt_path} not found — train {model_key} first.")
    gnn = _build_gnn_branch(spec["gnn"], graph)
    model = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=gnn).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])

    # repeats=1: we only need predictions here, not a timing estimate.
    y_true, y_prob, _ = _timed_predict(model, test_loader, device, graph=graph, repeats=1)

    threshold, thr_source = _best_f1_threshold_for(spec["ckpt"])
    y_pred = (y_prob >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

    out = {
        "model": spec["ckpt"],
        "label": spec["label"],
        "threshold": float(threshold),
        "threshold_source": thr_source,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
        "n_fraud": int((y_true == 1).sum()),
        "n_legit": int((y_true == 0).sum()),
        "precision": precision,
        "recall": recall,
    }

    out_path = config.RESULTS_DIR / f"confusion_{spec['ckpt']}.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[confusion] wrote {out_path}", flush=True)
    print(f"[confusion] TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,} "
          f"| precision {precision:.3f} | recall {recall:.3f} "
          f"| threshold {threshold:.4f} ({thr_source})", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_SPECS), default="m3",
                        help="which model to compute the confusion matrix for (default m3).")
    args = parser.parse_args()
    main(args.model)
