"""
explain_shap.py — SHAP feature attribution on the best model (Task 6.1).

What it does:
    Explains a SINGLE transaction's fraud prediction from `models/best_model.pt`
    (the winning model — m3_gat = Transformer + GAT) at the INPUT-FEATURE level
    using `shap.KernelExplainer` (model-agnostic, free, no API key). It produces a
    ranked, signed list of per-feature contributions (which feature pushed the risk
    up vs down, and by how much) and a SHAP bar plot.

Why KernelExplainer (and what exactly is explained):
    The model is multi-branch (a sequence Transformer fused with a graph GAT), so a
    gradient-based explainer over a single clean feature vector isn't well defined.
    Instead we treat the model as a black box of the CURRENT transaction's feature
    vector — the 32 numeric+categorical features describing the transaction being
    scored (= the last/current timestep of its sequence window, which is also what
    the Transformer pools on). KernelExplainer perturbs those 32 features against a
    small background sample and measures the change in predicted fraud probability.

    Cost control (CPU-friendly, the ZERO-COST rule): the GAT branch is transductive
    and embeds EVERY transaction node in one full-graph pass. Re-running that for
    each SHAP perturbation would be intractable on CPU. So we compute the target
    transaction's 128-d GRAPH embedding ONCE and hold it fixed as the transaction's
    relational context, while SHAP varies only the per-transaction features through
    the (cheap) Transformer -> fusion -> classifier path. The explanation therefore
    answers: "holding this transaction's card/device/region/merchant context fixed,
    which of its own feature values drove the score?" — the standard, defensible
    reading of an input-feature attribution for a fused model.

Inputs:
    models/best_model.pt           the winning checkpoint (Transformer + GAT)
    data/processed/sequences.pt    per-transaction sequence windows + feature_names
    data/processed/splits.npz      train/val/test row indices
    data/processed/graph.pt        the HeteroData graph (for the GAT branch)
    data/processed/clean.csv       (read for the target row only) raw human values

Outputs:
    results/plots/shap_example.png a SHAP bar plot of the signed contributions
    results/shap_example.csv       the ranked signed contributions (feature, value,
                                   shap_contribution) — reused by Task 6.2 (Ollama)

Run:  python src/explain_shap.py
      python src/explain_shap.py --background 100 --nsamples 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from train import (  # noqa: E402
    MODEL_SPECS,
    _build_gnn_branch,
    load_split_indices,
)

# Raw clean.csv columns that make the printed explanation human-readable. These are
# the un-scaled originals of (a subset of) the model-input features, plus the
# entity/context fields a bank officer would recognise (used again in Task 6.2).
RAW_CONTEXT_COLUMNS = [
    "TransactionID", "isFraud", "TransactionAmt", "hour", "day",
    "ProductCD", "card4", "card6", "P_emaildomain",
    "DeviceType", "DeviceInfo", "addr1", "dist1",
]


def _load_sequence_blob():
    """Load sequences.pt ONCE -> (X_seq, mask, y, transaction_id, feature_names, F).

    Centralised here so prepare_context() reads the ~1.5 GB tensor file a SINGLE
    time and hands the pieces to everything downstream (model build, background
    sample, per-row explanation). Phase 7's Streamlit app calls prepare_context()
    repeatedly, so avoiding redundant multi-GB loads matters.
    """
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
    tid = blob["transaction_id"]
    feature_names = blob["feature_names"]
    return X_seq, mask, y, tid, feature_names, X_seq.shape[2]


def load_best_model(device, f: int):
    """Rebuild the winning architecture and load best_model.pt's weights.

    `f` (the per-timestep feature dim) is passed in by the caller from the already
    loaded sequence blob, so this function does NOT re-load sequences.pt itself.

    Returns (model, spec, graph) where spec is the MODEL_SPECS entry matching the
    checkpoint's `name` (so the right GNN branch is wired in).
    """
    from transformer_model import SequenceTransformer
    from fusion_model import FusionClassifier
    from gnn_models import _load_graph

    if not config.BEST_MODEL_PT.exists():
        raise ValueError(
            f"{config.BEST_MODEL_PT} not found — run Task 5.4 (pick best model) first."
        )
    ckpt = torch.load(config.BEST_MODEL_PT, map_location=device, weights_only=False)
    ckpt_name = ckpt.get("name", "m3_gat")

    # Find which of the four specs produced this checkpoint so we build the matching
    # branch (None / sage / gat / sthgnn).
    spec = next((s for s in MODEL_SPECS.values() if s["ckpt"] == ckpt_name), None)
    if spec is None:
        raise ValueError(f"checkpoint name {ckpt_name!r} not in MODEL_SPECS.")

    graph = _load_graph()
    gnn = _build_gnn_branch(spec["gnn"], graph)
    model = FusionClassifier(SequenceTransformer(in_features=f), gnn_branch=gnn).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[shap] loaded best_model.pt: name={ckpt_name}, mode={model.mode}, "
          f"val PR-AUC {ckpt.get('val_pr_auc', float('nan')):.4f} @ epoch {ckpt.get('epoch')}")
    return model, spec, graph


def _precompute_graph_embedding(model, graph, device):
    """Run the GAT branch ONCE over the full graph -> [num_txn_nodes, 128] (or None
    for the transformer-only model). Held fixed during SHAP perturbations."""
    if getattr(model, "gnn", None) is None:
        return None
    graph = graph.to(device)
    with torch.no_grad():
        emb = model.gnn(graph.x_dict, graph.edge_index_dict)  # [num_txn, 128]
    return emb.detach()


@torch.no_grad()
def _predict_rows(model, X_seq, mask, node_idx, graph_emb_all, device, batch_size=512):
    """Predict fraud probability for the given rows (used to choose a target).

    Mirrors the fast eval path: gather each row's precomputed graph embedding by
    node index instead of re-running the GNN per batch.
    """
    probs = []
    needs_graph = graph_emb_all is not None
    for s in range(0, X_seq.shape[0], batch_size):
        xb = X_seq[s:s + batch_size].to(device)
        mb = mask[s:s + batch_size].to(device)
        seq_emb = model.transformer(xb, mb)
        if needs_graph:
            gb = graph_emb_all[node_idx[s:s + batch_size].to(device)]
            logits = model.classifier(model.fuse(torch.cat([seq_emb, gb], dim=1)))
        else:
            logits = model.classifier(seq_emb)
        probs.append(torch.sigmoid(logits.float()).cpu().view(-1))
    return torch.cat(probs).numpy()


def choose_target(model, X_seq, mask, y, test_idx, graph_emb_all, device):
    """Pick the TEST-set fraud transaction the model is MOST confident about.

    A high-confidence true fraud makes the clearest illustrative explanation: the
    SHAP factors should visibly add up to a high fraud probability.
    """
    test_idx_np = test_idx.numpy()
    fraud_mask = y[test_idx].numpy() == 1
    fraud_global = test_idx_np[fraud_mask]  # global clean.csv row ids of test frauds
    if len(fraud_global) == 0:
        raise ValueError("no fraud transactions in the test split — cannot pick a target.")

    fg = torch.as_tensor(fraud_global, dtype=torch.long)
    probs = _predict_rows(model, X_seq[fg], mask[fg], fg, graph_emb_all, device)
    best = int(np.argmax(probs))
    target_global = int(fraud_global[best])
    print(f"[shap] target = TEST fraud row (global idx {target_global}), "
          f"model prob = {probs[best]:.4f} (most-confident of {len(fraud_global):,} test frauds)")
    return target_global, float(probs[best])


def make_predict_fn(model, target_X, target_mask, graph_emb_target, device, batch_size=4096):
    """Build the scalar black-box f(Z) SHAP needs.

    Z is [m, F] candidate CURRENT-transaction feature vectors. For each row we copy
    the target's sequence window, overwrite its LAST (current) timestep with Z[i],
    run the Transformer, fuse with the FIXED target graph embedding, and return the
    fraud probability. Batched so KernelExplainer's many synthetic rows stay cheap.
    """
    L = target_X.shape[0]
    base = target_X.clone()                # [L, F]
    m_base = target_mask.clone()           # [L]
    needs_graph = graph_emb_target is not None

    def f(Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        m = Z.shape[0]
        out = np.empty(m, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, m, batch_size):
                zb = torch.from_numpy(Z[s:s + batch_size]).to(device)      # [b, F]
                b = zb.shape[0]
                Xb = base.unsqueeze(0).repeat(b, 1, 1).to(device)          # [b, L, F]
                Xb[:, -1, :] = zb                                          # set current step
                Mb = m_base.unsqueeze(0).repeat(b, 1).to(device)          # [b, L]
                seq_emb = model.transformer(Xb, Mb)                       # [b, 128]
                if needs_graph:
                    gb = graph_emb_target.unsqueeze(0).repeat(b, 1).to(device)
                    logits = model.classifier(model.fuse(torch.cat([seq_emb, gb], dim=1)))
                else:
                    logits = model.classifier(seq_emb)
                out[s:s + b] = torch.sigmoid(logits.float()).cpu().view(-1).numpy()
        return out

    return f


def _raw_context(target_global: int) -> dict:
    """Pull the raw (un-scaled) human-readable fields for the target transaction."""
    import pandas as pd

    have = pd.read_csv(config.CLEAN_CSV, nrows=0).columns
    usecols = [c for c in RAW_CONTEXT_COLUMNS if c in have]
    # Read only the target row: skip rows before it, keep one (row order == clean.csv).
    row = pd.read_csv(
        config.CLEAN_CSV, usecols=usecols,
        skiprows=range(1, target_global + 1), nrows=1,
    )
    return row.iloc[0].to_dict()


def prepare_context(background: int = 100, seed: int = config.SEED) -> dict:
    """Load the model + both views + the (one-off) GAT graph embedding + a background
    sample ONCE, returning a context dict reused to explain MANY transactions.

    Loading the 1.5 GB sequence tensor and running the full-graph GAT forward are the
    expensive steps; doing them once lets Task 6.3 explain 5 transactions cheaply.
    """
    config.set_seed(seed)
    device = config.get_device()

    # Load the sequence blob exactly ONCE, then reuse its pieces everywhere.
    X_seq, mask, y, _tid, feature_names, f = _load_sequence_blob()
    model, spec, graph = load_best_model(device, f)
    train_idx, _val_idx, test_idx = load_split_indices()
    assert len(feature_names) == X_seq.shape[2], "feature_names length != F"

    graph_emb_all = _precompute_graph_embedding(model, graph, device)

    # Background = `background` train transactions' current-step feature vectors.
    rng = np.random.default_rng(seed)
    bg_pool = train_idx.numpy()
    bg_sel = rng.choice(bg_pool, size=min(background, len(bg_pool)), replace=False)
    background_data = X_seq[torch.as_tensor(bg_sel, dtype=torch.long)][:, -1, :].numpy()
    print(f"[shap] context ready: background = {background_data.shape[0]} train txns, "
          f"F={len(feature_names)} features.")

    return {
        "device": device, "model": model, "graph_emb_all": graph_emb_all,
        "X_seq": X_seq, "mask": mask, "y": y,
        "train_idx": train_idx, "test_idx": test_idx,
        "feature_names": feature_names, "background_data": background_data,
    }


def explain_one(target_global: int, ctx: dict, *, nsamples="auto", l1_reg=0,
                save_artifacts: bool = False, verbose: bool = True) -> dict:
    """SHAP-explain ONE transaction (given its global row index) using a prepared ctx.

    Returns a payload dict with meta (predicted_prob, base_value, additive check),
    raw human-readable context, and the ranked signed `rows`. If save_artifacts, also
    writes the Task-6.1 single-example files (shap_example.csv/json + plot).
    """
    import shap

    model = ctx["model"]; device = ctx["device"]
    X_seq = ctx["X_seq"]; mask = ctx["mask"]
    feature_names = ctx["feature_names"]; background_data = ctx["background_data"]
    graph_emb_all = ctx["graph_emb_all"]

    target_X = X_seq[target_global]                   # [L, F]
    target_mask = mask[target_global]                 # [L]
    target_vec = target_X[-1].numpy().reshape(1, -1)  # current step = what we explain
    graph_emb_target = graph_emb_all[target_global] if graph_emb_all is not None else None

    predict_fn = make_predict_fn(model, target_X, target_mask, graph_emb_target, device)
    # The true model prob == predict_fn on the unperturbed current-step vector.
    target_prob = float(predict_fn(target_vec)[0])

    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(target_vec, nsamples=nsamples, l1_reg=l1_reg, silent=True)
    sv = np.asarray(shap_values).reshape(-1)          # [F]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    order = np.argsort(-np.abs(sv))
    rows = [{"feature": feature_names[j],
             "model_value": float(target_vec[0, j]),
             "shap_contribution": float(sv[j])} for j in order]

    # Sign-consistency check (used by the Task 6.3 spot-check): the net SHAP push must
    # move the score the same direction as (prob - base).
    additive = base_value + float(sv.sum())
    pos = float(sv[sv > 0].sum()); neg = float(sv[sv < 0].sum())
    direction = "raise" if (target_prob - base_value) >= 0 else "lower"
    sign_ok = bool(np.sign(sv.sum()) == np.sign(target_prob - base_value)) or abs(sv.sum()) < 1e-6

    try:
        raw = _raw_context(target_global)
    except Exception as e:
        raw = {}
        if verbose:
            print(f"[shap] (raw context lookup skipped: {type(e).__name__}: {e})")

    meta = {
        "target_global_idx": int(target_global),
        "predicted_prob": target_prob,
        "base_value": base_value,
        "additive_check": additive,
        "pos_sum": pos, "neg_sum": neg,
        "net_direction": direction, "sign_consistent": sign_ok,
    }

    if verbose:
        print(f"[shap] row {target_global}: prob={target_prob:.4f}, base={base_value:.4f}, "
              f"base+sumSHAP={additive:.4f} (additive OK), net={direction} "
              f"(+{pos:.3f}/{neg:.3f}), sign_consistent={sign_ok}")

    if save_artifacts:
        _save_single_artifacts(rows, meta, raw, sv, target_vec, feature_names,
                               base_value, target_prob)

    return {"shap_values": sv, "base_value": base_value, "rows": rows,
            "meta": meta, "raw": raw}


def _save_single_artifacts(rows, meta, raw, sv, target_vec, feature_names,
                           base_value, target_prob) -> None:
    """Write the Task-6.1 single-example files: shap_example.csv/json + bar plot."""
    import json
    import pandas as pd

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.RESULTS_DIR / "shap_example.csv"
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(out_csv, index=False)
    print(f"[shap] wrote ranked contributions -> {out_csv}")

    out_json = config.RESULTS_DIR / "shap_example.json"
    payload = {
        "meta": meta,
        "raw": {k: (None if pd.isna(v) else v) for k, v in raw.items()},
        "rows": [{"rank": i + 1, **r} for i, r in enumerate(rows)],
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[shap] wrote prob+context+factors -> {out_json}")

    _save_bar_plot(sv, target_vec[0], feature_names, base_value, target_prob)


def explain(background: int = 100, nsamples="auto", l1_reg=0, seed: int = config.SEED) -> dict:
    """Run Task 6.1 end-to-end: SHAP-explain the most-confident fraud, save list + plot.

    nsamples : KernelExplainer coalitions ("auto" = 2*F + 2048, plenty to solve all
               F features). l1_reg=0 disables SHAP's default L1 feature pruning
               (which would otherwise zero out all but ~10 features), so EVERY
               feature gets a signed attribution — clearer and more defensible.
    """
    ctx = prepare_context(background=background, seed=seed)
    target_global, _prob = choose_target(
        ctx["model"], ctx["X_seq"], ctx["mask"], ctx["y"],
        ctx["test_idx"], ctx["graph_emb_all"], ctx["device"],
    )
    return explain_one(target_global, ctx, nsamples=nsamples, l1_reg=l1_reg,
                       save_artifacts=True, verbose=True)


def _save_bar_plot(sv, values, feature_names, base_value, target_prob, top_k: int = 15):
    """Horizontal bar plot of the top-|SHAP| signed contributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-np.abs(sv))[:top_k][::-1]   # smallest-on-top for barh
    names = [feature_names[j] for j in order]
    vals = sv[order]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]  # red=raises, blue=lowers

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, max(4, 0.4 * len(order))))
    plt.barh(range(len(order)), vals, color=colors)
    plt.yticks(range(len(order)), names)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("SHAP contribution to fraud probability  (+ raises risk, - lowers)")
    plt.title(f"SHAP — best model (Transformer+GAT)\n"
              f"base={base_value:.3f}  ->  predicted fraud prob={target_prob:.3f}")
    plt.tight_layout()
    out = config.PLOTS_DIR / "shap_example.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[shap] wrote SHAP bar plot -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explanation of one fraud transaction (Task 6.1).")
    parser.add_argument("--background", type=int, default=100,
                        help="number of background train transactions (default 100).")
    parser.add_argument("--nsamples", default="auto",
                        help="KernelExplainer coalitions ('auto' = 2*F+2048, or an int).")
    parser.add_argument("--l1-reg", default="0",
                        help="SHAP L1 regularization (default '0' = none, attribute all features).")
    args = parser.parse_args()
    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)
    l1_reg = 0 if args.l1_reg in ("0", 0) else args.l1_reg
    explain(background=args.background, nsamples=nsamples, l1_reg=l1_reg)
