"""
evaluate.py — Phase 5 evaluation of all four comparison models (Tasks 5.1-5.3).

What it does:
    Loads each trained checkpoint (m1_transformer, m2_sage, m3_gat, m4_sthgnn),
    runs it on the SAME held-out TEST split, and reports the full metric panel:
    Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC (the headline metric for
    this ~3.5%-fraud problem), at both the fixed 0.5 threshold and the best-F1
    threshold. It also times inference and reports mean ms per 1,000 transactions.

    Outputs:
      * results/comparison.csv  — one row per model x metrics (Task 5.2)
      * results/plots/roc.png   — ROC curves, all 4 models overlaid (Task 5.3)
      * results/plots/pr.png    — Precision-Recall curves, all 4 overlaid
      * results/plots/metrics_bar.png — grouped bars of the key metrics
      * a printed 4 x metrics table (Task 5.1)

Inputs:  sequences.pt, splits.npz, graph.pt, models/m{1..4}_*.pt, scaler implicit.
Outputs: the files listed above.

IMPORTANT — GAT_HEADS: the m3_gat / m4_sthgnn checkpoints were trained on a free
Colab T4 with GAT_HEADS=1 (a documented memory constraint). config.GAT_HEADS is
therefore pinned to 1, so the branches rebuilt here match the saved weights.
Run with:  python src/evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils import compute_metrics  # noqa: E402
from train import (  # noqa: E402
    COMPARISON_COLUMNS,
    MODEL_SPECS,
    SequenceDataset,
    _build_gnn_branch,
    load_sequence_view,
    load_split_indices,
)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep the PR curve for the threshold maximising F1. Returns (threshold, f1)."""
    from sklearn.metrics import precision_recall_curve, f1_score

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5, float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0))
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_i = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_thr = float(thr[best_i])
    best_f1 = float(f1_score(y_true, (y_prob >= best_thr).astype(int), zero_division=0))
    return best_thr, best_f1


def _timed_predict(model, loader, device, *, graph=None, repeats: int = 3):
    """Run inference over `loader`, returning (y_true, y_prob, ms_per_1k).

    KEY OPTIMISATION (CPU-friendly): GNN branches are TRANSDUCTIVE — they embed
    EVERY transaction node in one full-graph pass. So we compute that full-graph
    embedding ONCE per model here, then per batch we only run the (cheap) sequence
    transformer + fusion head and GATHER the matching graph rows by node_idx. This
    avoids re-running the whole-graph GNN forward on every mini-batch (the per-batch
    recompute is what made the earlier full eval crawl on CPU).

    Times the forward work only (excludes metric computation), averaged over
    `repeats` passes. The amortised one-off GNN embedding cost is included in the
    reported time. ms_per_1k = mean wall-time to score 1,000 transactions.
    """
    needs_graph = getattr(model, "gnn", None) is not None
    x_dict = edge_index_dict = None
    if needs_graph:
        graph = graph.to(device)
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict

    model.eval()
    n_total = len(loader.dataset)

    # Time `repeats` passes; collect predictions on the LAST pass for metrics.
    y_true_chunks, y_prob_chunks = [], []
    times = []
    for r in range(repeats):
        collect = (r == repeats - 1)
        if collect:
            y_true_chunks, y_prob_chunks = [], []
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # ONE full-graph GNN pass for the whole model (not per batch).
            graph_emb_all = None
            if needs_graph:
                graph_emb_all = model.gnn(x_dict, edge_index_dict)  # [num_txn, 128]

            for X_seq, mask, y, node_idx in loader:
                X_seq = X_seq.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                node_idx = node_idx.to(device, non_blocking=True).long()

                seq_emb = model.transformer(X_seq, mask)  # [B, 128]
                if needs_graph:
                    graph_emb = graph_emb_all[node_idx]  # gather this batch's rows
                    fused = model.fuse(torch.cat([seq_emb, graph_emb], dim=1))
                    logits = model.classifier(fused)
                else:
                    logits = model.classifier(seq_emb)
                if collect:
                    y_true_chunks.append(y.view(-1))
                    y_prob_chunks.append(torch.sigmoid(logits.float()).cpu().view(-1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    y_true = torch.cat(y_true_chunks).numpy()
    y_prob = torch.cat(y_prob_chunks).numpy()
    ms_per_1k = (float(np.mean(times)) / n_total) * 1000.0 * 1000.0  # sec/txn -> ms/1k
    return y_true, y_prob, ms_per_1k


def evaluate_all(batch_size: int = config.BATCH_SIZE) -> "pd.DataFrame":
    """Evaluate all four checkpoints on the test split and write comparison.csv."""
    import pandas as pd

    config.set_seed(config.SEED)
    device = config.get_device()

    from transformer_model import SequenceTransformer
    from fusion_model import FusionClassifier
    from gnn_models import _load_graph

    print("[eval] loading views ...")
    X_seq, mask, y, _txn_id, f = load_sequence_view()
    _train_idx, _val_idx, test_idx = load_split_indices()
    print(f"[eval] sequences X_seq {tuple(X_seq.shape)} (F={f}); test split = {len(test_idx):,} rows")

    graph = _load_graph()

    full_ds = SequenceDataset(X_seq, mask, y, node_idx=torch.arange(X_seq.shape[0]))
    test_loader = DataLoader(Subset(full_ds, test_idx.tolist()), batch_size=batch_size, shuffle=False)

    rows = []
    curves = {}  # model -> (y_true, y_prob) for plotting
    for key, spec in MODEL_SPECS.items():
        ckpt_path = config.MODELS_DIR / f"{spec['ckpt']}.pt"
        if not ckpt_path.exists():
            raise ValueError(f"{ckpt_path} not found — train {key} first (Tasks 4.2-4.5).")

        gnn = _build_gnn_branch(spec["gnn"], graph)
        model = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=gnn).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])

        y_true, y_prob, ms_per_1k = _timed_predict(model, test_loader, device, graph=graph)
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        best_thr, best_f1 = _best_f1_threshold(y_true, y_prob)

        curves[spec["ckpt"]] = (y_true, y_prob)
        rows.append({
            "model": spec["ckpt"],
            "mode": getattr(model, "mode", spec["label"]),
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "accuracy": m["accuracy"],
            "threshold": 0.5,
            "best_f1": best_f1,
            "best_f1_threshold": best_thr,
            "ms_per_1k": ms_per_1k,
            "val_pr_auc": float(ck.get("val_pr_auc", float("nan"))),
            "best_epoch": int(ck.get("epoch", -1)),
        })
        print(f"[eval] {spec['ckpt']:<14} | PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} "
              f"| F1 {m['f1']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f} "
              f"| Acc {m['accuracy']:.4f} | bestF1 {best_f1:.4f}@{best_thr:.3f} "
              f"| {ms_per_1k:.2f} ms/1k", flush=True)

    # Enforce the shared schema/order so the file matches train.py's writer.
    df = pd.DataFrame(rows).reindex(columns=COMPARISON_COLUMNS)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.COMPARISON_CSV, index=False)
    print(f"[eval] wrote {config.COMPARISON_CSV}")

    _plot_curves(curves)
    _plot_metric_bars(df)

    print("\n[eval] ===== TEST-SET COMPARISON (Task 5.1) =====")
    show_cols = ["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy",
                 "best_f1", "best_f1_threshold", "ms_per_1k"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df[show_cols].to_string(index=False))
    return df


def _plot_curves(curves: dict) -> None:
    """Overlay ROC and PR curves for all models -> results/plots/{roc,pr}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(6, 5))
    for name, (yt, yp) in curves.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — test set"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "roc.png", dpi=150); plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    base = None
    for name, (yt, yp) in curves.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        base = float(np.mean(yt))
    if base is not None:
        plt.axhline(base, color="k", ls="--", alpha=0.4, label=f"base rate={base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — test set"); plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "pr.png", dpi=150); plt.close()
    print(f"[eval] wrote {config.PLOTS_DIR / 'roc.png'} and {config.PLOTS_DIR / 'pr.png'}")


def _plot_metric_bars(df) -> None:
    """Grouped bar chart of key metrics per model -> results/plots/metrics_bar.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    plt.figure(figsize=(10, 5))
    for i, mdl in enumerate(models):
        vals = df.loc[df["model"] == mdl, metrics].values.ravel()
        plt.bar(x + i * width, vals, width, label=mdl)
    plt.xticks(x + width * (len(models) - 1) / 2, metrics, rotation=0)
    plt.ylim(0, 1.0); plt.ylabel("score")
    plt.title("Model comparison — test metrics"); plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "metrics_bar.png", dpi=150); plt.close()
    print(f"[eval] wrote {config.PLOTS_DIR / 'metrics_bar.png'}")


if __name__ == "__main__":
    evaluate_all()
