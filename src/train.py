"""
train.py — generic, model-agnostic trainer (Task 4.1).

What it does:
    ONE training loop that drives every model in the comparison study (Model 1
    transformer-only, and Models 2-4 transformer + a GNN), because they all share
    the FusionClassifier forward contract. Given a model, a train loader and a val
    loader it:

      * optimises with Adam (LR / weight_decay from config.py),
      * minimises a class-imbalance loss — FOCAL loss (gamma=2) by default, or
        weighted BCE — both computed on raw LOGITS for stability,
      * tracks validation PR-AUC every epoch (the project's headline metric since
        fraud is ~3.5% of rows),
      * EARLY-STOPS when val PR-AUC stops improving for `patience` epochs,
      * saves the best-so-far checkpoint to models/<name>.pt,
      * uses CUDA mixed precision (autocast + GradScaler) when a GPU is present,
        and plain fp32 on CPU.

How models are fed (the model-agnostic part):
    Batches are (X_seq, mask, y, node_idx). For Model 1 the GNN args are omitted.
    For Models 2-4 the SAME full graph (x_dict, edge_index_dict) is passed every
    step and `node_idx` selects this batch's transaction rows from the GNN's
    full-graph output. node_idx is the GLOBAL transaction-node index (= clean.csv
    row index), which the sequence and graph views share by construction (Task 2.3).

Inputs (for the trainer): an nn.Module (FusionClassifier), two DataLoaders over a
    SequenceDataset, a device, and — for GNN models — the loaded HeteroData graph.
Outputs: a dict of history + best metric, and a written checkpoint at
    models/<name>.pt containing the best model_state and metadata.

Running this file as a script executes a 2-epoch smoke run on a small synthetic
subset (Model 1) and asserts a checkpoint is written — the Task 4.1 acceptance
check. It also runs a tiny real-graph Model-2 step if graph.pt is present.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils import compute_metrics, focal_loss_with_logits, weighted_bce_with_logits  # noqa: E402


class SequenceDataset(Dataset):
    """Wrap the sequence view tensors as a Dataset yielding (X, mask, y, node_idx).

    `node_idx` is the GLOBAL transaction-node index for each row. When a subset
    (e.g. a split) is passed, give the original global indices so GNN models can
    gather the right rows from the full-graph embedding; if omitted it defaults to
    arange(N) (i.e. the rows already ARE the full set in natural order).
    """

    def __init__(
        self,
        X_seq: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        node_idx: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        n = X_seq.shape[0]
        if mask.shape[0] != n or y.shape[0] != n:
            raise ValueError(
                f"X_seq/mask/y must share dim 0: {X_seq.shape[0]}/{mask.shape[0]}/{y.shape[0]}."
            )
        self.X_seq = X_seq
        self.mask = mask
        self.y = y.float().view(-1)
        if node_idx is None:
            node_idx = torch.arange(n, dtype=torch.long)
        else:
            node_idx = torch.as_tensor(np.asarray(node_idx)).long().view(-1)
            if node_idx.shape[0] != n:
                raise ValueError(
                    f"node_idx length {node_idx.shape[0]} != number of rows {n}."
                )
        self.node_idx = node_idx

    def __len__(self) -> int:
        return self.X_seq.shape[0]

    def __getitem__(self, i: int):
        return self.X_seq[i], self.mask[i], self.y[i], self.node_idx[i]


def _run_epoch(
    model,
    loader,
    device,
    *,
    loss_fn,
    optimizer=None,
    scaler=None,
    x_dict=None,
    edge_index_dict=None,
    use_amp: bool = False,
):
    """Run one pass over `loader`. Train if `optimizer` given, else evaluate.

    Returns (mean_loss, y_true [N], y_prob [N]). y_prob is collected for metrics.
    """
    is_train = optimizer is not None
    model.train(is_train)

    needs_graph = getattr(model, "gnn", None) is not None
    total_loss, n_seen = 0.0, 0
    all_true, all_prob = [], []

    for X_seq, mask, y, node_idx in loader:
        X_seq = X_seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)
        node_idx = node_idx.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                if needs_graph:
                    logits = model(
                        X_seq, mask, x_dict, edge_index_dict, node_idx,
                        return_logits=True,
                    )
                else:
                    logits = model(X_seq, mask, return_logits=True)
                loss = loss_fn(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.detach()) * bs
        n_seen += bs
        all_true.append(y.detach().cpu().view(-1))
        all_prob.append(torch.sigmoid(logits.detach().float()).cpu().view(-1))

    mean_loss = total_loss / max(n_seen, 1)
    y_true = torch.cat(all_true).numpy()
    y_prob = torch.cat(all_prob).numpy()
    return mean_loss, y_true, y_prob


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    *,
    name: str,
    graph=None,
    loss: str = "focal",
    focal_gamma: float = config.FOCAL_GAMMA,
    focal_alpha: float | None = config.FOCAL_ALPHA,
    pos_weight: float | None = None,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.EARLY_STOP_PATIENCE,
    min_delta: float = config.EARLY_STOP_MIN_DELTA,
    ckpt_dir: Path = config.MODELS_DIR,
    use_amp: bool | None = None,
    verbose: bool = True,
) -> dict:
    """Train `model`, early-stopping on val PR-AUC; save best to models/<name>.pt.

    Args:
        model:        a FusionClassifier (any of the 4 configs).
        train_loader/val_loader: DataLoaders over SequenceDataset.
        device:       torch device.
        name:         checkpoint base name -> ckpt_dir/<name>.pt.
        graph:        loaded HeteroData (REQUIRED for GNN models; ignored for
                      Model 1). x_dict/edge_index_dict are taken from it.
        loss:         'focal' (gamma) or 'weighted_bce' (pos_weight).
        focal_gamma/focal_alpha: focal loss params (used iff loss='focal').
        pos_weight:   weighted-BCE positive weight; if None and loss='weighted_bce'
                      it is derived from the train labels (#neg/#pos).
        lr/weight_decay/max_epochs/patience: optimisation + early-stop controls.
        min_delta:    minimum val PR-AUC gain over the running best that counts as
                      a real improvement for EARLY STOPPING. The best checkpoint is
                      still saved on any strict improvement; min_delta only governs
                      whether the patience counter resets, so trivial noise-level
                      gains (e.g. 1e-9) can't keep training alive indefinitely.
        ckpt_dir:     where to write <name>.pt.
        verbose:      print per-epoch logs.

    Returns:
        dict with keys: best_val_pr_auc, best_epoch, ckpt_path, history (list of
        per-epoch dicts), best_val_metrics.
    """
    model = model.to(device)
    needs_graph = getattr(model, "gnn", None) is not None

    x_dict = edge_index_dict = None
    if needs_graph:
        if graph is None:
            raise ValueError(
                f"model mode {getattr(model, 'mode', '?')!r} uses a GNN but no graph "
                "was passed; provide the loaded HeteroData via graph=..."
            )
        graph = graph.to(device)
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict

    # Loss selection (both on logits).
    if loss == "focal":
        def loss_fn(logits, targets):
            return focal_loss_with_logits(logits, targets, focal_gamma, focal_alpha)
    elif loss == "weighted_bce":
        if pos_weight is None:
            y_train = torch.as_tensor(train_loader.dataset.y)  # type: ignore[attr-defined]
            from utils import pos_weight_from_labels
            pos_weight = pos_weight_from_labels(y_train)
        pw = torch.tensor(float(pos_weight), device=device)

        def loss_fn(logits, targets):
            return weighted_bce_with_logits(logits, targets, pw)
    else:
        raise ValueError(f"loss must be 'focal' or 'weighted_bce', got {loss!r}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Mixed precision: auto-on for CUDA, off for CPU; `use_amp` can force either
    # (e.g. --no-amp on Colab if a GNN's fp16 attention misbehaves).
    if use_amp is None:
        use_amp = device.type == "cuda"
    elif use_amp and device.type != "cuda":
        use_amp = False  # AMP only meaningful on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"

    if verbose:
        print(f"[train] {name}: mode={getattr(model, 'mode', '?')}, loss={loss}, "
              f"device={device}, amp={use_amp}, lr={lr}, max_epochs={max_epochs}, "
              f"patience={patience}")

    best_pr_auc = -float("inf")
    best_epoch = -1
    best_val_metrics: dict = {}
    history: list[dict] = []
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, _, _ = _run_epoch(
            model, train_loader, device,
            loss_fn=loss_fn, optimizer=optimizer, scaler=scaler,
            x_dict=x_dict, edge_index_dict=edge_index_dict, use_amp=use_amp,
        )
        val_loss, y_true, y_prob = _run_epoch(
            model, val_loader, device,
            loss_fn=loss_fn, optimizer=None, scaler=None,
            x_dict=x_dict, edge_index_dict=edge_index_dict, use_amp=use_amp,
        )
        val_metrics = compute_metrics(y_true, y_prob)
        val_pr_auc = val_metrics["pr_auc"]
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "seconds": dt,
        })

        # Compute both flags against the OLD best before mutating it: we save the
        # checkpoint on any strict improvement (keep the genuinely-best weights),
        # but only a gain of at least `min_delta` resets the patience counter so
        # noise-level creep can't prevent early stopping (see Model-1 5h post-mortem).
        improved = val_pr_auc > best_pr_auc
        improved_meaningfully = val_pr_auc > best_pr_auc + min_delta
        if verbose:
            flag = "  *best*" if improved else ""
            print(f"[train] {name} epoch {epoch:>3}/{max_epochs} | "
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"val PR-AUC {val_pr_auc:.4f} | ROC-AUC {val_metrics['roc_auc']:.4f} | "
                  f"F1 {val_metrics['f1']:.4f} | {dt:.1f}s{flag}", flush=True)

        if improved:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": getattr(model, "mode", None),
                    "name": name,
                    "epoch": epoch,
                    "val_pr_auc": val_pr_auc,
                    "val_metrics": val_metrics,
                    "config": {
                        "embed_dim": config.EMBED_DIM,
                        "max_seq_len": config.MAX_SEQ_LEN,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "loss": loss,
                        "focal_gamma": focal_gamma,
                        "focal_alpha": focal_alpha,
                    },
                },
                ckpt_path,
            )

        if improved_meaningfully:
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"[train] {name}: early stop at epoch {epoch} (no val "
                          f"PR-AUC gain >= {min_delta:g} for {patience} epochs).",
                          flush=True)
                break

    if verbose:
        print(f"[train] {name}: best val PR-AUC {best_pr_auc:.4f} at epoch "
              f"{best_epoch}; checkpoint -> {ckpt_path}")

    return {
        "best_val_pr_auc": best_pr_auc,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "ckpt_path": str(ckpt_path),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Real-data experiment runner (drives Tasks 4.2 - 4.5 from the actual views)
# --------------------------------------------------------------------------- #
# Registry of the four comparison models. `gnn` selects the branch (None = the
# transformer-only Model 1); `ckpt` is the checkpoint base name written to models/.
MODEL_SPECS: dict[str, dict] = {
    "m1": {"ckpt": "m1_transformer", "gnn": None, "label": "Transformer only"},
    "m2": {"ckpt": "m2_sage", "gnn": "sage", "label": "Transformer + GraphSAGE"},
    "m3": {"ckpt": "m3_gat", "gnn": "gat", "label": "Transformer + GAT"},
    "m4": {"ckpt": "m4_sthgnn", "gnn": "sthgnn", "label": "Transformer + ST-HGNN"},
}

# Canonical column schema for results/comparison.csv. BOTH writers (train.py's
# append_comparison_row after a single training run, and evaluate.py's full
# Phase-5 sweep) emit exactly these columns in this order, so the file is
# unambiguous no matter which script wrote it. `val_pr_auc` is the saved
# checkpoint's best validation PR-AUC; `ms_per_1k` (inference time) is only
# measured by evaluate.py, so train.py leaves it blank (NaN).
COMPARISON_COLUMNS = [
    "model", "mode", "pr_auc", "roc_auc", "f1", "precision", "recall",
    "accuracy", "threshold", "best_f1", "best_f1_threshold", "ms_per_1k",
    "val_pr_auc", "best_epoch",
]


def load_sequence_view():
    """Load sequences.pt -> (X_seq, mask, y, transaction_id, F)."""
    if not config.SEQUENCES_PT.exists():
        raise ValueError(
            f"{config.SEQUENCES_PT} not found — run build_sequences.py (Task 2.1) first."
        )
    try:
        blob = torch.load(config.SEQUENCES_PT, weights_only=False)
    except TypeError:  # older torch without weights_only
        blob = torch.load(config.SEQUENCES_PT)
    X_seq = blob["X_seq"].float()
    mask = blob["mask"].bool()
    y = blob["y"].float().view(-1)
    transaction_id = blob["transaction_id"]
    f = X_seq.shape[2]
    return X_seq, mask, y, transaction_id, f


def load_split_indices():
    """Load splits.npz -> (train_idx, val_idx, test_idx) as long tensors."""
    if not config.SPLITS_NPZ.exists():
        raise ValueError(
            f"{config.SPLITS_NPZ} not found — run split.py (Task 1.4) first."
        )
    npz = np.load(config.SPLITS_NPZ)
    return (
        torch.as_tensor(npz["train_idx"]).long(),
        torch.as_tensor(npz["val_idx"]).long(),
        torch.as_tensor(npz["test_idx"]).long(),
    )


def _build_gnn_branch(kind: str | None, graph):
    """Construct the GNN branch named by `kind`, or None for Model 1."""
    if kind is None:
        return None
    from gnn_models import (
        GATBranch, GraphSAGEBranch, STHGNNBranch,
        _in_channels_dict, temporal_features_from_clean,
    )
    inc = _in_channels_dict(graph)
    meta = graph.metadata()
    if kind == "sage":
        return GraphSAGEBranch(meta, inc)
    if kind == "gat":
        return GATBranch(meta, inc)
    if kind == "sthgnn":
        return STHGNNBranch(meta, inc, temporal_features_from_clean(graph))
    raise ValueError(f"unknown gnn kind {kind!r}.")


def evaluate_on_loader(model, loader, device, *, graph=None):
    """Run the model over `loader` and return (metrics_dict, y_true, y_prob).

    Metrics are reported at the default 0.5 threshold AND, additionally, the
    best-F1 threshold and its F1 (handy for Phase 5); PR-AUC/ROC-AUC are
    threshold-free. The model is reloaded by the caller from its best checkpoint
    before this is called.
    """
    needs_graph = getattr(model, "gnn", None) is not None
    x_dict = edge_index_dict = None
    if needs_graph:
        graph = graph.to(device)
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict

    # No optimizer -> evaluation pass. Loss fn is unused for metrics; pass a dummy.
    _, y_true, y_prob = _run_epoch(
        model, loader, device,
        loss_fn=lambda lo, ta: torch.tensor(0.0),
        optimizer=None, scaler=None,
        x_dict=x_dict, edge_index_dict=edge_index_dict, use_amp=False,
    )
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    # Best-F1 threshold via the PR curve (threshold-swept F1).
    from sklearn.metrics import precision_recall_curve, f1_score
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    if len(thr) > 0:
        best_i = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
        best_thr = float(thr[best_i])
        best_f1 = float(f1_score(y_true, (y_prob >= best_thr).astype(int), zero_division=0))
    else:
        best_thr, best_f1 = 0.5, metrics["f1"]
    metrics["best_f1_threshold"] = best_thr
    metrics["best_f1"] = best_f1
    return metrics, y_true, y_prob


def append_comparison_row(row: dict) -> None:
    """Insert/replace this model's row in results/comparison.csv (idempotent).

    Uses the shared COMPARISON_COLUMNS schema so the file matches what
    evaluate.py writes; columns this writer doesn't measure (e.g. ms_per_1k)
    are left blank (NaN) rather than dropped.
    """
    import pandas as pd

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if config.COMPARISON_CSV.exists():
        df = pd.read_csv(config.COMPARISON_CSV)
        df = df[df["model"] != row["model"]]  # replace any prior run of this model
    else:
        df = pd.DataFrame(columns=COMPARISON_COLUMNS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=COMPARISON_COLUMNS)
    df.to_csv(config.COMPARISON_CSV, index=False)


def run_experiment(
    model_key: str,
    *,
    loss: str = "focal",
    batch_size: int = config.BATCH_SIZE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.EARLY_STOP_PATIENCE,
    min_delta: float = config.EARLY_STOP_MIN_DELTA,
    use_amp: bool | None = None,
) -> dict:
    """Train one of the four comparison models on the REAL views, then record test
    metrics to results/comparison.csv.

    Steps: load sequences + splits (+ graph for GNN models) -> build the
    FusionClassifier for `model_key` -> train (early-stop on val PR-AUC, best
    checkpoint to models/<ckpt>.pt) -> reload best -> evaluate on the held-out
    TEST split -> append a row to comparison.csv.
    """
    if model_key not in MODEL_SPECS:
        raise ValueError(f"model_key must be one of {list(MODEL_SPECS)}, got {model_key!r}.")
    spec = MODEL_SPECS[model_key]

    config.set_seed(config.SEED)
    device = config.get_device()

    from torch.utils.data import Subset
    from transformer_model import SequenceTransformer
    from fusion_model import FusionClassifier

    print(f"[run] {model_key}: {spec['label']} — loading views ...")
    X_seq, mask, y, _txn_id, f = load_sequence_view()
    train_idx, val_idx, test_idx = load_split_indices()
    print(f"[run] sequences: X_seq {tuple(X_seq.shape)} (F={f}); "
          f"splits train/val/test = {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")

    graph = None
    if spec["gnn"] is not None:
        from gnn_models import _load_graph
        graph = _load_graph()

    # Full-data dataset with GLOBAL node indices; Subset selects each split without
    # copying the (large) sequence tensor.
    full_ds = SequenceDataset(X_seq, mask, y, node_idx=torch.arange(X_seq.shape[0]))
    train_loader = DataLoader(Subset(full_ds, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx.tolist()), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_ds, test_idx.tolist()), batch_size=batch_size, shuffle=False)

    gnn = _build_gnn_branch(spec["gnn"], graph)
    model = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=gnn)

    out = train_model(
        model, train_loader, val_loader, device,
        name=spec["ckpt"], graph=graph, loss=loss,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        use_amp=use_amp,
    )

    # Reload the BEST checkpoint before scoring the test set.
    ckpt = torch.load(out["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"[run] {model_key}: scoring TEST split with best checkpoint "
          f"(val PR-AUC {ckpt['val_pr_auc']:.4f} @ epoch {ckpt['epoch']}) ...")
    test_metrics, _, _ = evaluate_on_loader(model, test_loader, device, graph=graph)

    print(f"[run] {model_key} TEST: PR-AUC {test_metrics['pr_auc']:.4f} | "
          f"ROC-AUC {test_metrics['roc_auc']:.4f} | F1 {test_metrics['f1']:.4f} | "
          f"P {test_metrics['precision']:.4f} | R {test_metrics['recall']:.4f} | "
          f"Acc {test_metrics['accuracy']:.4f} | "
          f"best-F1 {test_metrics['best_f1']:.4f} @ thr {test_metrics['best_f1_threshold']:.3f}")

    append_comparison_row({
        "model": spec["ckpt"],
        "mode": getattr(model, "mode", spec["label"]),
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "accuracy": test_metrics["accuracy"],
        "threshold": 0.5,
        "best_f1": test_metrics["best_f1"],
        "best_f1_threshold": test_metrics["best_f1_threshold"],
        # `val_pr_auc` = the saved checkpoint's best validation PR-AUC (same
        # quantity evaluate.py records under this name). ms_per_1k is left to
        # evaluate.py, so it stays NaN for a single train-time row.
        "val_pr_auc": out["best_val_pr_auc"],
        "best_epoch": out["best_epoch"],
    })
    print(f"[run] {model_key}: appended test metrics -> {config.COMPARISON_CSV}")
    return {"train": out, "test_metrics": test_metrics}


def _smoke_test() -> None:
    """2-epoch smoke run on a small synthetic subset -> assert a checkpoint exists.

    This is the Task 4.1 acceptance check. It exercises the FULL trainer path
    (focal loss, Adam, per-epoch val PR-AUC, early-stop bookkeeping, checkpoint
    write) on Model 1 (transformer only) with cheap synthetic data so it runs in
    seconds on CPU. If graph.pt is available it additionally runs ONE tiny
    Model-2 (transformer+SAGE) epoch to prove the model-agnostic GNN path works.
    """
    config.set_seed(config.SEED)
    device = config.get_device()

    from transformer_model import SequenceTransformer
    from fusion_model import FusionClassifier

    # --- Synthetic sequence subset (N small) -------------------------------- #
    n, f = 400, 32
    X = torch.randn(n, config.MAX_SEQ_LEN, f)
    lengths = torch.randint(1, config.MAX_SEQ_LEN + 1, (n,))
    mask = torch.zeros(n, config.MAX_SEQ_LEN, dtype=torch.bool)
    for i, k in enumerate(lengths):
        mask[i, config.MAX_SEQ_LEN - int(k):] = True
    X[~mask] = 0.0
    # Labels weakly tied to the last-step mean so PR-AUC can rise above base rate.
    signal = X[torch.arange(n), -1].mean(dim=1)
    prob = torch.sigmoid(3.0 * signal)
    y = (torch.rand(n) < (0.035 + 0.3 * prob)).float()

    n_tr = 300
    train_ds = SequenceDataset(X[:n_tr], mask[:n_tr], y[:n_tr])
    val_ds = SequenceDataset(X[n_tr:], mask[n_tr:], y[n_tr:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=None)
    out = train_model(
        model, train_loader, val_loader, device,
        name="smoke_m1", loss="focal", max_epochs=2, patience=7,
    )

    ckpt = Path(out["ckpt_path"])
    assert ckpt.exists(), f"checkpoint not written: {ckpt}"
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state" in loaded and loaded["mode"] == "transformer"
    print(f"[train] smoke (Model 1): checkpoint OK -> {ckpt} "
          f"(best PR-AUC {out['best_val_pr_auc']:.4f} @ epoch {out['best_epoch']})")

    # --- Optional tiny GNN path check (only if the real graph exists) -------- #
    if config.GRAPH_PT.exists():
        try:
            from gnn_models import (
                GraphSAGEBranch, _in_channels_dict, _load_graph,
            )
            graph = _load_graph()
            inc = _in_channels_dict(graph)
            gnn = GraphSAGEBranch(graph.metadata(), inc)
            # Use the first few transaction nodes as a micro-batch, with node_idx
            # = their GLOBAL indices so the GNN gathers the matching rows.
            m = 64
            node_idx = torch.arange(m)
            ds = SequenceDataset(X[:m], mask[:m], y[:m], node_idx=node_idx)
            ld = DataLoader(ds, batch_size=32, shuffle=False)
            model2 = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=gnn)
            out2 = train_model(
                model2, ld, ld, device,
                name="smoke_m2", graph=graph, loss="focal", max_epochs=1, patience=7,
            )
            assert Path(out2["ckpt_path"]).exists()
            print(f"[train] smoke (Model 2, GNN path): checkpoint OK -> "
                  f"{out2['ckpt_path']}")
        except Exception as e:  # don't fail the core acceptance check on the optional path
            print(f"[train] optional GNN-path smoke skipped ({type(e).__name__}: {e})")
    else:
        print("[train] graph.pt not found — skipped optional GNN-path smoke.")

    print("[train] smoke test passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generic trainer. No --model -> Task 4.1 smoke test; "
                    "--model m1|m2|m3|m4 -> real training run on the data views."
    )
    parser.add_argument(
        "--model", choices=list(MODEL_SPECS), default=None,
        help="which comparison model to train on the real data (else smoke test).",
    )
    parser.add_argument("--loss", choices=["focal", "weighted_bce"], default="focal")
    parser.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument(
        "--min-delta", type=float, default=config.EARLY_STOP_MIN_DELTA,
        help="min val PR-AUC gain that resets early-stop patience "
             f"(default {config.EARLY_STOP_MIN_DELTA:g}).",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--no-amp", action="store_true",
        help="disable CUDA mixed precision (use if a GNN's fp16 attention misbehaves).",
    )
    args = parser.parse_args()

    if args.model is None:
        _smoke_test()
    else:
        run_experiment(
            args.model,
            loss=args.loss,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            use_amp=False if args.no_amp else None,
        )
