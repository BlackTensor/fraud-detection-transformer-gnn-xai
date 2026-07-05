"""
streamlit_app.py — FraudShield demo dashboard (Phase 7).

What it does:
    A professional, local, free (ZERO-COST) Streamlit dashboard for the project. It
    presents the comparative study and a live fraud-assessment console driven by the
    winning model (models/best_model.pt = m3_gat, Transformer + GAT):

        Section 1  Hero / landing + headline stats (from results/comparison.csv)
        Section 2  How it works (sequence / graph / fusion+explainability)
        Section 3  Comparative study — 4-model metric table + ROC/PR curves
        Section 4  Live assessment — sample test transactions OR custom input,
                   returning a fraud probability + verdict + SHAP top-5 factors +
                   a local-LLM (Ollama) natural-language explanation
        Section 5  Research context

    The live engine reuses the tested Phase-6 machinery:
      * features:        build_sequences feature lists + the shared train-fit scaler
      * graph context:   a synthetic GAT embedding = average of the historical
                         transactions sharing this transaction's merchant/device/
                         region (its would-be neighbours; progressive fallback)
      * SHAP:            explain_shap (KernelExplainer over the 32 input features)
      * NL explanation:  explain_ollama (local model; templated fallback if Ollama
                         is unavailable so the demo never breaks)

ZERO-COST: everything runs locally with free, open-source tools. No API key, no
cloud. Run with:  streamlit run app/streamlit_app.py

Inputs (artifacts):
    models/best_model.pt, models/scaler.pkl,
    data/processed/{graph.pt, sequences.pt, splits.npz, clean.csv},
    results/comparison.csv, results/plots/{roc.png, pr.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------- #
# Make the project importable: src/ holds the model code (bare imports like
# `from train import ...`), and the repo root holds config.py.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

import config  # noqa: E402
from build_sequences import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

# Order of the 32 model-input features (must match build_sequences feature_names).
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_NEIGHBOURS = 25          # neighbour count below which we relax the graph match
LIVE_BACKGROUND = 80         # background sample size for the live SHAP explainer
LIVE_NSAMPLES = 400          # KernelExplainer coalitions per live explanation
BEST_CKPT = "m3_gat"         # winning checkpoint name (for highlighting + threshold)

# Honest disclaimer shown on the Custom Input and Batch Input tabs (NOT the Sample
# tab). A form cannot supply the engineered behavioural features the model leans on
# most, so those are held at dataset averages and custom scores sit in a compressed
# low band — illustrative of the pipeline, not calibrated risk.
FORM_DISCLAIMER = (
    "The engineered behavioural features — a card's transaction-count and spending "
    "history, plus the anonymized C/D features that carry most of the model's fraud "
    "signal — cannot be entered from a form, so they are held at dataset averages. "
    "Custom predictions therefore demonstrate the full pipeline end to end but sit in "
    "a compressed, low-probability band and should not be read as calibrated risk "
    "scores. The Sample Transactions tab runs on complete records and reflects the "
    "model's true performance."
)

# Palette (light, premium; applied via CSS + .streamlit/config.toml).
NAVY = "#1B2A4A"        # headings + stat numbers
BLUE = "#2563EB"        # section titles + card border accents
BODY = "#374151"        # body text
SUBTEXT = "#6B7280"     # labels / subtext
FRAUD_COLOR = "#DC2626"
LEGIT_COLOR = "#16A34A"

CUSTOM_CSS = """
<style>
/* ----- Global light, warm-grey canvas ----- */
.stApp { background-color: #F5F5F7; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1100px; }

/* Generous breathing room around the horizontal dividers between sections. */
hr { margin: 2.4rem 0 !important; border-color: #E5E7EB !important; }

/* ----- Hero ----- */
.fs-hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
    color: #1B2A4A; margin-bottom: 0.35rem; line-height: 1.12;
}
.fs-hero-sub { font-size: 1.08rem; color: #6B7280; margin-bottom: 1.6rem;
               max-width: 760px; line-height: 1.55; }

/* ----- Hero stat cards: white, navy number, grey label, blue top border ----- */
.fs-statbar { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.4rem 0 0.2rem 0; }
.fs-stat {
    flex: 1 1 0; min-width: 180px; background: #FFFFFF;
    border-top: 3px solid #2563EB; border-radius: 12px;
    padding: 1.15rem 1.25rem; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.fs-stat .v { font-size: 1.85rem; font-weight: 800; color: #1B2A4A; line-height: 1.1; }
.fs-stat .l { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
              color: #6B7280; margin-top: 0.35rem; font-weight: 600; }
.fs-stat .s { font-size: 0.78rem; color: #9CA3AF; margin-top: 0.15rem; }

/* ----- Generic white cards ----- */
.fs-card {
    background: #FFFFFF; border-radius: 12px; padding: 1.35rem 1.45rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); height: 100%;
}
/* How-It-Works step cards: thin blue left accent. */
.fs-card.step-card { border-left: 3px solid #2563EB; }
.fs-card h4 { color: #1B2A4A; margin: 0 0 0.55rem 0; font-size: 1.1rem; font-weight: 700; }
.fs-card p { color: #374151; font-size: 0.93rem; margin: 0; line-height: 1.55; }
.fs-card .step { font-size: 0.72rem; color: #2563EB; letter-spacing: 0.1em;
                 font-weight: 700; margin-bottom: 0.3rem; }

/* ----- Section headings ----- */
.fs-section { font-size: 1.55rem; font-weight: 600; color: #2563EB;
              margin: 0.2rem 0 0.15rem 0; }
.fs-section-sub { color: #6B7280; font-size: 0.95rem; margin-bottom: 1rem;
                  max-width: 820px; line-height: 1.5; }

/* ----- Gauge ----- */
.fs-gauge-wrap { margin: 0.5rem 0 0.2rem 0; }
.fs-gauge-track {
    position: relative; height: 26px; border-radius: 13px;
    background: linear-gradient(90deg, #16A34A 0%, #EAB308 55%, #DC2626 100%);
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.06);
}
.fs-gauge-marker {
    position: absolute; top: -6px; width: 4px; height: 38px;
    background: #1B2A4A; border-radius: 2px; box-shadow: 0 0 4px rgba(0,0,0,0.35);
}
.fs-gauge-thr {
    position: absolute; top: -3px; width: 2px; height: 32px;
    background: #1B2A4A; opacity: 0.45;
}

/* ----- Verdict badge (the only dark-on-color element) ----- */
.fs-verdict {
    border-radius: 12px; padding: 1.15rem 1.3rem; text-align: center;
    font-weight: 800; font-size: 1.4rem; color: #FFFFFF; letter-spacing: 0.02em;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
}
.fs-verdict .sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; margin-top: 0.2rem; }

/* ----- NL explanation card ----- */
.fs-nl {
    background: #FFFFFF; border-left: 3px solid #2563EB; border-radius: 12px;
    padding: 1.1rem 1.25rem; color: #374151; font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.fs-nl .src { display:block; margin-top: 0.8rem; font-size: 0.74rem; color: #9CA3AF; }

/* ----- Misc: heading colors for st.markdown('#### ...') in results ----- */
h4 { color: #1B2A4A !important; }

/* ----- Left sidebar navigation (matches the light card theme) ----- */
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] .fs-nav-brand {
    font-size: 1.4rem; font-weight: 800; color: #1B2A4A; letter-spacing: -0.01em;
    margin: 0.2rem 0 0.1rem 0;
}
[data-testid="stSidebar"] .fs-nav-tag {
    font-size: 0.78rem; color: #6B7280; margin-bottom: 0.4rem; line-height: 1.4;
}

/* ----- Artifact download cards ----- */
.fs-artifact {
    background: #FFFFFF; border-radius: 12px; border-left: 3px solid #2563EB;
    padding: 1.1rem 1.25rem; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 0.55rem;
}
.fs-artifact.primary { border-left: 3px solid #16A34A; }
.fs-artifact .fname { font-size: 1.02rem; font-weight: 700; color: #1B2A4A; }
.fs-artifact .what { font-size: 0.9rem; color: #374151; margin-top: 0.25rem; line-height: 1.5; }
.fs-artifact .size { font-size: 0.78rem; color: #9CA3AF; margin-top: 0.3rem; }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    """Load results/comparison.csv (tiny — needed for the landing + study sections)."""
    if not config.COMPARISON_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(config.COMPARISON_CSV)


def _load_resources_impl() -> dict:
    """Load model + sequences + graph embeddings + scaler + category frequencies ONCE.

    Built on explain_shap.prepare_context (model, X_seq, mask, y, graph_emb_all,
    background_data, train/test indices, feature_names) and extended with the extra
    pieces the live form needs (scaler, frequency maps, entity keys, UI options,
    decision threshold, Ollama model). Plain function so it can be tested headlessly.
    """
    required = {
        "best model": config.BEST_MODEL_PT,
        "scaler": config.SCALER_PKL,
        "graph": config.GRAPH_PT,
        "sequences": config.SEQUENCES_PT,
        "splits": config.SPLITS_NPZ,
        "clean.csv": config.CLEAN_CSV,
    }
    missing = [n for n, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s): "
            + ", ".join(f"{n} ({required[n]})" for n in missing)
            + ". Run the earlier phases before launching the app."
        )

    from explain_shap import prepare_context

    ctx = prepare_context(background=LIVE_BACKGROUND, seed=config.SEED)
    device = ctx["device"]
    model = ctx["model"]
    # graph embeddings on CPU for cheap neighbour averaging.
    if ctx["graph_emb_all"] is not None:
        ctx["graph_emb_all"] = ctx["graph_emb_all"].cpu()
        global_graph_mean = ctx["graph_emb_all"].mean(dim=0, keepdim=True)
    else:
        global_graph_mean = None

    # --- Scaler payload (per-column mean_/scale_) --------------------------- #
    import joblib

    payload = joblib.load(config.SCALER_PKL)
    scaler, scaler_cols = payload["scaler"], payload["columns"]
    col_to_idx = {c: i for i, c in enumerate(scaler_cols)}

    # --- Category frequencies (TRAIN rows only) + entity keys --------------- #
    train_idx = ctx["train_idx"].numpy()
    ent_cols = ["ProductCD", "DeviceType", "addr1"]
    usecols = list(dict.fromkeys(CATEGORICAL_FEATURES + ent_cols))
    cat_df = pd.read_csv(config.CLEAN_CSV, usecols=usecols)
    train_cat = cat_df.iloc[train_idx]

    freq_maps, cat_neutral = {}, {}
    for col in CATEGORICAL_FEATURES:
        freq = train_cat[col].astype(str).value_counts(normalize=True)
        freq_maps[col] = freq
        cat_neutral[col] = float(train_cat[col].astype(str).map(freq).fillna(0.0).mean())

    entities = pd.DataFrame({
        "ProductCD": cat_df["ProductCD"].astype(str),
        "DeviceType": cat_df["DeviceType"].astype(str),
        "addr1": pd.to_numeric(cat_df["addr1"], errors="coerce"),
    })
    options = {
        "ProductCD": sorted(cat_df["ProductCD"].astype(str).unique()),
        "DeviceType": sorted(cat_df["DeviceType"].astype(str).unique()),
        "card4": sorted(cat_df["card4"].astype(str).unique()),
        "card6": sorted(cat_df["card6"].astype(str).unique()),
    }

    threshold, thr_source = _load_threshold(BEST_CKPT)

    # --- Resolve a local Ollama model (fast list call; None if unavailable) - #
    try:
        from explain_ollama import pick_model, DEFAULT_MODEL
        ollama_model = pick_model(DEFAULT_MODEL)
    except Exception:
        ollama_model = None

    R = dict(ctx)
    R.update({
        "model": model, "device": device, "mode": model.mode, "ckpt": BEST_CKPT,
        "global_graph_mean": global_graph_mean,
        "scaler": scaler, "col_to_idx": col_to_idx,
        "freq_maps": freq_maps, "cat_neutral": cat_neutral,
        "entities": entities, "options": options,
        "threshold": threshold, "thr_source": thr_source,
        "ollama_model": ollama_model,
    })
    return R


def _load_threshold(ckpt_name: str) -> tuple[float, str]:
    """Best-F1 decision threshold for `ckpt_name` from comparison.csv (else 0.5)."""
    cmp = load_comparison()
    if len(cmp):
        row = cmp.loc[cmp["model"] == ckpt_name]
        if len(row) and not pd.isna(row.iloc[0].get("best_f1_threshold")):
            return float(row.iloc[0]["best_f1_threshold"]), "best-F1 threshold (Task 5.1)"
    return 0.5, "default 0.5"


@st.cache_resource(show_spinner="Loading model, sequences and graph embeddings (one-time)…")
def load_resources() -> dict:
    """Cached wrapper around _load_resources_impl (heavy load runs once per session)."""
    return _load_resources_impl()


@st.cache_resource(show_spinner="Selecting sample test transactions…")
def get_sample_targets(_R: dict) -> list[dict]:
    """Pick 3 confident frauds + 2 confident legits from the TEST split for the demo.

    `_R` is prefixed with an underscore so Streamlit does not try to hash it.
    """
    from explain_ollama import _select_spotcheck_targets
    from explain_shap import _predict_rows, _raw_context

    targets = _select_spotcheck_targets(_R, n_fraud=3, n_legit=2)
    out = []
    for t in targets:
        g = t["global"]
        gt = torch.as_tensor([g], dtype=torch.long)
        prob = float(_predict_rows(
            _R["model"], _R["X_seq"][gt], _R["mask"][gt], gt,
            _R["graph_emb_all"], _R["device"],
        )[0])
        try:
            raw = _raw_context(g)
        except Exception:
            raw = {}
        out.append({"global": g, "true": t["true"], "prob": prob, "raw": raw})
    return out


# --------------------------------------------------------------------------- #
# Feature-vector construction + custom prediction
# --------------------------------------------------------------------------- #
def _scale_num(R: dict, col: str, raw: float) -> float:
    idx = R["col_to_idx"][col]
    return float((raw - R["scaler"].mean_[idx]) / R["scaler"].scale_[idx])


def build_feature_vector(inputs: dict, R: dict) -> np.ndarray:
    """Build the 32-d current-transaction vector (mirrors the training pipeline).

    Numeric features default to the dataset average (scaled 0); categoricals to
    their TRAIN-mean frequency encoding. Only user-supplied fields are overridden.
    """
    vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    fidx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for col in CATEGORICAL_FEATURES:
        vec[fidx[col]] = R["cat_neutral"][col]

    amt = float(inputs["amount"])
    vec[fidx["TransactionAmt"]] = _scale_num(R, "TransactionAmt", amt)
    vec[fidx["log_amt"]] = _scale_num(R, "log_amt", float(np.log1p(amt)))
    vec[fidx["hour"]] = _scale_num(R, "hour", float(inputs["hour"]))
    vec[fidx["amt_vs_client_mean"]] = _scale_num(R, "amt_vs_client_mean", 1.0)

    def _enc(col, value):
        return float(R["freq_maps"][col].get(str(value), 0.0))

    vec[fidx["ProductCD"]] = _enc("ProductCD", inputs["merchant"])
    vec[fidx["DeviceType"]] = _enc("DeviceType", inputs["device"])
    if inputs.get("card4"):
        vec[fidx["card4"]] = _enc("card4", inputs["card4"])
    if inputs.get("card6"):
        vec[fidx["card6"]] = _enc("card6", inputs["card6"])
    return vec


def compute_graph_context(inputs: dict, R: dict) -> tuple[torch.Tensor | None, str]:
    """Synthetic GAT context: average of historical neighbours' graph embeddings."""
    emb = R["graph_emb_all"]
    if emb is None:
        return None, "no graph branch (transformer-only model)"

    ent = R["entities"]
    m_merch = (ent["ProductCD"] == str(inputs["merchant"])).to_numpy()
    m_dev = (ent["DeviceType"] == str(inputs["device"])).to_numpy()
    region = inputs.get("region")
    have_region = region is not None and not (isinstance(region, float) and np.isnan(region))
    m_reg = (ent["addr1"] == float(region)).to_numpy() if have_region else np.zeros(len(ent), bool)

    levels = []
    if have_region:
        levels.append(("merchant + device + region", m_merch & m_dev & m_reg))
    levels += [("merchant + device", m_merch & m_dev), ("merchant", m_merch)]
    if have_region:
        levels.append(("region", m_reg))

    for desc, mask in levels:
        idx = np.where(mask)[0]
        if len(idx) >= MIN_NEIGHBOURS:
            ctx = emb[torch.as_tensor(idx, dtype=torch.long)].mean(dim=0, keepdim=True)
            return ctx, f"average of {len(idx):,} historical transactions matching {desc}"
    return R["global_graph_mean"], "global average (no close neighbours found)"


def _custom_tensors(inputs: dict, R: dict):
    """Build (vec, target_X[L,F], target_mask[L], graph_ctx[1,128] or None, ctx_desc)."""
    vec = build_feature_vector(inputs, R)
    L, Fd = config.MAX_SEQ_LEN, len(FEATURE_NAMES)
    target_X = torch.zeros(L, Fd, dtype=torch.float32)
    target_X[-1] = torch.from_numpy(vec)
    target_mask = torch.zeros(L, dtype=torch.bool)
    target_mask[-1] = True
    ctx, ctx_desc = compute_graph_context(inputs, R)
    return vec, target_X, target_mask, ctx, ctx_desc


@torch.no_grad()
def _forward_prob(R: dict, target_X: torch.Tensor, target_mask: torch.Tensor,
                  graph_ctx: torch.Tensor | None) -> float:
    """Run the fused model on a single transaction window -> fraud probability."""
    model, device = R["model"], R["device"]
    X = target_X.unsqueeze(0).to(device)
    m = target_mask.unsqueeze(0).to(device)
    seq_emb = model.transformer(X, m)
    if model.gnn is None:
        logits = model.classifier(seq_emb)
    else:
        logits = model.classifier(model.fuse(torch.cat([seq_emb, graph_ctx.to(device)], dim=1)))
    return float(torch.sigmoid(logits.float()).view(-1)[0].cpu())


# --------------------------------------------------------------------------- #
# SHAP + NL explanation
# --------------------------------------------------------------------------- #
def _rank_rows(sv: np.ndarray, target_vec: np.ndarray, feature_names) -> list[dict]:
    order = np.argsort(-np.abs(sv))
    return [{"feature": feature_names[j], "model_value": float(target_vec[0, j]),
             "shap_contribution": float(sv[j])} for j in order]


def shap_custom(target_X, target_mask, graph_ctx, R, nsamples=LIVE_NSAMPLES):
    """SHAP for a synthetic (custom) transaction. Returns (rows, base_value, prob)."""
    import shap
    from explain_shap import make_predict_fn

    graph_emb_target = graph_ctx.squeeze(0) if graph_ctx is not None else None
    predict_fn = make_predict_fn(R["model"], target_X, target_mask, graph_emb_target, R["device"])
    tv = target_X[-1].numpy().reshape(1, -1).astype(np.float32)
    prob = float(predict_fn(tv)[0])
    explainer = shap.KernelExplainer(predict_fn, R["background_data"])
    sv = np.asarray(explainer.shap_values(tv, nsamples=nsamples, l1_reg=0, silent=True)).reshape(-1)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return _rank_rows(sv, tv, R["feature_names"]), base, prob


def shap_sample(global_idx: int, R, nsamples=LIVE_NSAMPLES):
    """SHAP for a real test transaction. Returns (rows, base_value, prob, raw)."""
    from explain_shap import explain_one
    res = explain_one(global_idx, R, nsamples=nsamples, l1_reg=0,
                       save_artifacts=False, verbose=False)
    return res["rows"], res["base_value"], res["meta"]["predicted_prob"], res["raw"]


def nl_explanation(payload: dict, R: dict) -> tuple[str, str]:
    """Generate the NL explanation via local Ollama, falling back to a template."""
    from explain_ollama import build_prompt, run_ollama, templated_explanation

    prompt = build_prompt(payload, top_k=5)
    model_name = R.get("ollama_model")
    if model_name:
        try:
            return run_ollama(prompt, model_name), f"ollama:{model_name}"
        except Exception:
            pass
    return templated_explanation(payload, top_k=5), "templated fallback (Ollama not running)"


def shap_bar_figure(rows: list[dict], top_k: int = 5):
    """Horizontal bar chart of the top-k signed SHAP factors (dark theme)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[:top_k][::-1]
    names = [r["feature"] for r in top]
    vals = [r["shap_contribution"] for r in top]
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(range(len(top)), vals, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, color="#374151", fontsize=9)
    ax.axvline(0, color="#9CA3AF", lw=0.8)
    ax.tick_params(axis="x", colors="#6B7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#E5E7EB")
    ax.set_xlabel("SHAP contribution  (positive raises risk, negative lowers)",
                  color="#6B7280", fontsize=8)
    fig.tight_layout()
    return fig


# Sequential blue->navy ramp for the four models (matches the navy/blue theme).
MODEL_COLORS = {
    "m1_transformer": "#BFDBFE",
    "m2_sage": "#60A5FA",
    "m3_gat": "#2563EB",
    "m4_sthgnn": "#1B2A4A",
}


def metrics_bar_figure(cmp: pd.DataFrame):
    """Themed grouped bar chart: PR-AUC, ROC-AUC and F1 across the four models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1")]
    models = cmp["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for i, mdl in enumerate(models):
        vals = [float(cmp.loc[cmp["model"] == mdl, key].iloc[0]) for key, _ in metrics]
        offs = x + (i - (len(models) - 1) / 2) * width
        bars = ax.bar(offs, vals, width, label=mdl,
                      color=MODEL_COLORS.get(mdl, BLUE), edgecolor="#FFFFFF", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.3, color=BODY)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], color=NAVY, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score", color=SUBTEXT, fontsize=9)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=8)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#E5E7EB")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=8, frameon=False, labelcolor=BODY)
    fig.tight_layout()
    return fig


def confusion_figure(cm: dict):
    """Confusion-matrix heatmap for the best model (rows=actual, cols=predicted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)        # actual x predicted
    row_tot = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(counts, np.clip(row_tot, 1, None))     # within-actual-class share
    cell_lbl = np.array([["TN", "FP"], ["FN", "TP"]])

    cmap = LinearSegmentedColormap.from_list("fs_blue", ["#FFFFFF", "#2563EB"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.imshow(row_frac, cmap=cmap, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            txt_color = "#FFFFFF" if row_frac[i, j] > 0.55 else NAVY
            ax.text(j, i, f"{cell_lbl[i, j]}\n{int(counts[i, j]):,}\n{row_frac[i, j]:.1%}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=txt_color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nLegit", "Predicted\nFraud"], fontsize=9, color=NAVY)
    ax.set_yticklabels(["Actual\nLegit", "Actual\nFraud"], fontsize=9, color=NAVY)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _arch_box(ax, cx, cy, text, *, w=17, h=15, fill="#FFFFFF", border=BLUE, fg=NAVY):
    """Draw one rounded box (centred at cx,cy on a 0-100 canvas) for the diagram."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.6,rounding_size=2.2",
                         linewidth=1.6, edgecolor=border, facecolor=fill)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=fg, linespacing=1.25)


def architecture_figure():
    """Two-branch architecture diagram (boxes + arrows), styled to the theme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    cx = [10, 31, 52, 72, 91]   # column centres: input, branch, fusion, prob, explain
    y_top, y_bot, y_mid = 72, 28, 50

    # Inputs (light grey) -> branches (white, blue border).
    _arch_box(ax, cx[0], y_top, "Transaction\nSequence", fill="#EEF2F7", border="#CBD5E1")
    _arch_box(ax, cx[0], y_bot, "Transaction\nGraph", fill="#EEF2F7", border="#CBD5E1")
    _arch_box(ax, cx[1], y_top, "Transformer\nEncoder")
    _arch_box(ax, cx[1], y_bot, "GNN Branch\n(GAT)")
    # Merge -> fusion (blue) -> probability (navy) -> explanation (white).
    _arch_box(ax, cx[2], y_mid, "Fusion\nLayer", fill=BLUE, border=BLUE, fg="#FFFFFF")
    _arch_box(ax, cx[3], y_mid, "Fraud\nProbability", fill=NAVY, border=NAVY, fg="#FFFFFF")
    _arch_box(ax, cx[4], y_mid, "SHAP + LLM\nExplanation", w=19)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.6,
                                     color="#94A3B8", shrinkA=2, shrinkB=2))

    arrow(cx[0] + 8.5, y_top, cx[1] - 8.5, y_top)     # seq input  -> transformer
    arrow(cx[0] + 8.5, y_bot, cx[1] - 8.5, y_bot)     # graph input -> gnn
    arrow(cx[1] + 8.5, y_top - 2, cx[2] - 8.5, y_mid + 4)   # transformer -> fusion
    arrow(cx[1] + 8.5, y_bot + 2, cx[2] - 8.5, y_mid - 4)   # gnn         -> fusion
    arrow(cx[2] + 8.5, y_mid, cx[3] - 8.5, y_mid)     # fusion -> probability
    arrow(cx[3] + 8.5, y_mid, cx[4] - 9.5, y_mid)     # probability -> explanation

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def _stat(value: str, label: str, sub: str = "") -> str:
    return (f'<div class="fs-stat"><div class="v">{value}</div>'
            f'<div class="l">{label}</div><div class="s">{sub}</div></div>')


def render_hero(cmp: pd.DataFrame) -> None:
    st.markdown(
        '<div class="fs-hero-title">Fraud<span class="accent">Shield</span> — '
        'Sequence-Aware Fraud Detection with Graph Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fs-hero-sub">A transaction-fraud detector that fuses a Transformer '
        'over a card\'s spending history with a Graph Neural Network over shared '
        'cards, devices, merchants and regions — then explains every decision.</div>',
        unsafe_allow_html=True,
    )

    best_model, best_pr, best_roc = "—", "—", "—"
    if len(cmp):
        best_row = cmp.loc[cmp["pr_auc"].idxmax()]
        best_model = str(best_row["model"])
        best_pr = f"{best_row['pr_auc']:.4f}"
        best_roc = f"{cmp['roc_auc'].max():.4f}"

    st.markdown(
        '<div class="fs-statbar">'
        + _stat(best_model, "Best Model", "by test PR-AUC")
        + _stat(best_pr, "Best PR-AUC", "primary metric (imbalanced)")
        + _stat(best_roc, "Best ROC-AUC", "ranking quality")
        + _stat("590,540", "Transactions", "IEEE-CIS, ~3.5% fraud")
        + "</div>",
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    st.markdown('<div class="fs-section">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="fs-section-sub">Two complementary views of every '
                'transaction, fused into one explainable score.</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="fs-card step-card"><div class="step">STEP 01</div>'
            '<h4>Sequence Modeling</h4><p>A Transformer encoder reads a card\'s recent '
            'transaction history and learns temporal spending patterns — order, '
            'recency and rhythm that a single snapshot cannot capture.</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="fs-card step-card"><div class="step">STEP 02</div>'
            '<h4>Graph Intelligence</h4><p>A Graph Neural Network learns relationship '
            'patterns across the network — transactions sharing cards, devices, '
            'merchants and regions — surfacing organized fraud rings.</p></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="fs-card step-card"><div class="step">STEP 03</div>'
            '<h4>Fusion + Explainability</h4><p>The two embeddings are fused into a '
            'fraud probability, then SHAP attributes the score to individual features '
            'and a local LLM turns that into a plain-English rationale.</p></div>',
            unsafe_allow_html=True)


def render_comparison(cmp: pd.DataFrame) -> None:
    st.markdown('<div class="fs-section">Comparative Study — Four Models</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fs-section-sub">All models evaluated on the identical '
                'time-aware test split. The winning row (Transformer + GAT) is '
                'highlighted.</div>', unsafe_allow_html=True)

    if not len(cmp):
        st.warning("results/comparison.csv not found — run Phase 5 to populate it.")
        return

    cols = ["model", "mode", "pr_auc", "roc_auc", "f1", "precision", "recall"]
    disp = cmp[cols].rename(columns={
        "model": "Model", "mode": "Architecture", "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC", "f1": "F1", "precision": "Precision", "recall": "Recall",
    })
    num_cols = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"]

    def _highlight(row):
        if row["Model"] == BEST_CKPT:
            return ["background-color: #EFF6FF; color: #1B2A4A; font-weight: 700"] * len(row)
        return [""] * len(row)

    styler = (disp.style
              .format({c: "{:.4f}" for c in num_cols})
              .apply(_highlight, axis=1))
    st.dataframe(styler, width="stretch", hide_index=True)

    # Themed grouped bar chart of the headline metrics, right beside the table.
    st.markdown("##### Headline metrics at a glance")
    st.pyplot(metrics_bar_figure(cmp))

    p1, p2 = st.columns(2)
    roc_png, pr_png = config.PLOTS_DIR / "roc.png", config.PLOTS_DIR / "pr.png"
    with p1:
        if roc_png.exists():
            st.image(str(roc_png), caption="ROC curves — all four models",
                     width="stretch")
        else:
            st.info("results/plots/roc.png not found.")
    with p2:
        if pr_png.exists():
            st.image(str(pr_png), caption="Precision-Recall curves — all four models",
                     width="stretch")
        else:
            st.info("results/plots/pr.png not found.")


def render_gauge(prob: float, threshold: float) -> None:
    pct = max(0.0, min(prob, 1.0)) * 100
    thr = max(0.0, min(threshold, 1.0)) * 100
    st.markdown(
        '<div class="fs-gauge-wrap"><div class="fs-gauge-track">'
        f'<div class="fs-gauge-thr" style="left:{thr:.1f}%;"></div>'
        f'<div class="fs-gauge-marker" style="left:calc({pct:.1f}% - 2px);"></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"White marker = fraud probability ({prob:.1%}). "
               f"Grey line = decision threshold ({threshold:.1%}).")


def render_result(prob, threshold, rows, base_value, raw, ctx_desc, R) -> None:
    is_fraud = prob >= threshold
    color = FRAUD_COLOR if is_fraud else LEGIT_COLOR
    verdict = "FRAUD" if is_fraud else "LEGIT"

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown(f"#### Fraud probability: {prob:.1%}")
        render_gauge(prob, threshold)
    with g2:
        st.markdown(
            f'<div class="fs-verdict" style="background:{color};">{verdict}'
            f'<div class="sub">probability {prob:.1%} '
            f'{"≥" if is_fraud else "<"} threshold {threshold:.1%}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown("##### Top contributing factors (SHAP)")
        st.pyplot(shap_bar_figure(rows, top_k=5))
    with r2:
        st.markdown("##### Natural-language explanation")
        payload = {
            "meta": {"predicted_prob": prob, "base_value": base_value},
            "raw": raw or {}, "rows": rows,
        }
        with st.spinner("Generating explanation with the local model…"):
            text, source = nl_explanation(payload, R)
        st.markdown(f'<div class="fs-nl">{text}<span class="src">Source: {source}</span></div>',
                    unsafe_allow_html=True)

    st.caption(f"Graph context: {ctx_desc}.")


# --------------------------------------------------------------------------- #
# Batch Input — paste raw CSV, visualize the full pipeline step by step
# --------------------------------------------------------------------------- #
# Minimum columns a pasted batch must contain (TransactionID + isFraud optional).
REQUIRED_BATCH_COLS = ["TransactionAmt", "hour", "ProductCD", "DeviceType", "addr1"]
OPTIONAL_BATCH_COLS = ["TransactionID", "isFraud", "card4", "card6"]

# Distinct colours per entity-node type in the mini graph.
ENTITY_STYLE = {
    "merchant": ("#2563EB", "Merchant (ProductCD)"),
    "device":   ("#7C3AED", "Device (DeviceType)"),
    "region":   ("#EA580C", "Region (addr1)"),
    "card":     ("#0D9488", "Card"),
}
UNKNOWN_NODE = "#9CA3AF"


def batch_example_csv(R: dict) -> str:
    """Build example CSV text from 5 REAL test transactions (3 fraud + 2 legit).

    Reuses get_sample_targets (the same confident-fraud / confident-legit picks the
    Sample tab uses), so the example is genuine labelled test data, not invented.
    """
    samples = get_sample_targets(R)

    def _num(v, as_int=False):
        try:
            f = float(v)
            return str(int(round(f))) if as_int else (f"{f:.3f}".rstrip("0").rstrip("."))
        except (TypeError, ValueError):
            return ""

    header = "TransactionID,TransactionAmt,hour,ProductCD,DeviceType,addr1,isFraud"
    lines = [header]
    for s in samples:
        raw = s["raw"]
        tid = raw.get("TransactionID", s["global"])
        lines.append(",".join([
            _num(tid, as_int=True),
            _num(raw.get("TransactionAmt")),
            _num(raw.get("hour"), as_int=True),
            str(raw.get("ProductCD", "")),
            str(raw.get("DeviceType", "")),
            _num(raw.get("addr1"), as_int=True),
            str(int(s["true"])),
        ]))
    return "\n".join(lines)


def parse_batch_csv(text: str) -> pd.DataFrame:
    """Parse pasted CSV text into a validated DataFrame (clear errors on problems)."""
    import io

    text = (text or "").strip()
    if not text:
        raise ValueError("No data pasted. Paste CSV rows or click “Load example data”.")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:  # malformed CSV
        raise ValueError(f"Could not parse the pasted text as CSV: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_BATCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
            + ".  Required columns are: " + ", ".join(REQUIRED_BATCH_COLS)
            + "  (TransactionID and isFraud are optional)."
        )
    if df.empty:
        raise ValueError("The CSV has a header but no transaction rows.")
    return df.reset_index(drop=True)


def _row_to_inputs(row: pd.Series) -> dict:
    """Map one parsed CSV row to the inputs dict the model pipeline expects."""
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    region = _f(row.get("addr1"))
    return {
        "amount": _f(row.get("TransactionAmt")) or 0.0,
        "hour": _f(row.get("hour")) or 0.0,
        "merchant": str(row.get("ProductCD", "")),
        "device": str(row.get("DeviceType", "")),
        "region": region,
        "card4": (str(row["card4"]) if "card4" in row and pd.notna(row["card4"]) else None),
        "card6": (str(row["card6"]) if "card6" in row and pd.notna(row["card6"]) else None),
    }


def batch_predict(df: pd.DataFrame, R: dict) -> list[dict]:
    """Score every parsed transaction. Returns one record per row with prob + verdict."""
    threshold = R["threshold"]
    records = []
    for i, row in df.iterrows():
        inputs = _row_to_inputs(row)
        _, tX, tM, gctx, _ = _custom_tensors(inputs, R)
        prob = _forward_prob(R, tX, tM, gctx)
        tid = row.get("TransactionID", i)
        actual = int(row["isFraud"]) if ("isFraud" in df.columns and pd.notna(row.get("isFraud"))) else None
        records.append({
            "row": i, "tid": tid, "inputs": inputs,
            "amount": inputs["amount"], "prob": prob,
            "verdict": "FRAUD" if prob >= threshold else "LEGIT",
            "actual": actual,
        })
    return records


def batch_graph_figure(df: pd.DataFrame):
    """Mini transaction–entity graph: txn circles (colour=ground truth) + entity squares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    has_card = any(c in df.columns for c in ("card", "card1"))
    card_col = "card" if "card" in df.columns else ("card1" if "card1" in df.columns else None)

    G = nx.Graph()
    txn_nodes, txn_colors = [], []
    entity_nodes = {k: [] for k in ENTITY_STYLE}      # type -> [node_id, ...]
    labels = {}

    for i, row in df.iterrows():
        tid = row.get("TransactionID", i)
        tnode = f"txn:{tid}"
        G.add_node(tnode)
        txn_nodes.append(tnode)
        if "isFraud" in df.columns and pd.notna(row.get("isFraud")):
            txn_colors.append(FRAUD_COLOR if int(row["isFraud"]) == 1 else LEGIT_COLOR)
        else:
            txn_colors.append(UNKNOWN_NODE)

        # Build this transaction's entity nodes and connect them.
        ent_specs = [
            ("merchant", f"M:{row.get('ProductCD')}", str(row.get("ProductCD", ""))),
            ("device",   f"D:{row.get('DeviceType')}", str(row.get("DeviceType", ""))),
        ]
        if pd.notna(row.get("addr1")):
            ent_specs.append(("region", f"R:{row.get('addr1')}", f"addr {row.get('addr1')}"))
        if has_card and pd.notna(row.get(card_col)):
            ent_specs.append(("card", f"C:{row.get(card_col)}", f"card {row.get(card_col)}"))

        for etype, enode, elabel in ent_specs:
            if enode not in G:
                G.add_node(enode)
                entity_nodes[etype].append(enode)
                labels[enode] = elabel
            G.add_edge(tnode, enode)

    pos = nx.spring_layout(G, seed=config.SEED, k=0.9)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#D1D5DB", width=1.0)
    # Transaction nodes (circles).
    nx.draw_networkx_nodes(G, pos, nodelist=txn_nodes, node_color=txn_colors,
                           node_shape="o", node_size=320, edgecolors="#FFFFFF",
                           linewidths=1.2, ax=ax)
    # Entity nodes (squares), one colour per type.
    legend_handles = []
    for etype, (color, legend_label) in ENTITY_STYLE.items():
        nodes = entity_nodes[etype]
        if not nodes:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                               node_shape="s", node_size=620, edgecolors="#FFFFFF",
                               linewidths=1.2, ax=ax)
        legend_handles.append(Line2D([0], [0], marker="s", color="w", label=legend_label,
                                     markerfacecolor=color, markersize=10))
    # Label entity nodes only (transaction nodes are too many / not informative).
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8,
                            font_color="#1B2A4A", ax=ax)

    legend_handles += [
        Line2D([0], [0], marker="o", color="w", label="Txn — fraud",
               markerfacecolor=FRAUD_COLOR, markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Txn — legit",
               markerfacecolor=LEGIT_COLOR, markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Txn — unknown",
               markerfacecolor=UNKNOWN_NODE, markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              frameon=True, facecolor="#FFFFFF", edgecolor="#E5E7EB",
              bbox_to_anchor=(1.0, 1.0))
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_batch(R: dict) -> None:
    """Batch Input tab: paste CSV -> parsed table -> graph -> predictions -> explanations."""
    st.markdown("Paste raw transaction rows in **CSV format** and watch the full "
                "pipeline run step by step — parsing, the relationship graph, model "
                "predictions, and SHAP + LLM explanations for every flagged fraud.")

    st.info(FORM_DISCLAIMER)

    st.markdown("**Required columns:** `TransactionAmt, hour, ProductCD, DeviceType, "
                "addr1`  ·  **Optional:** `TransactionID, isFraud` (for ground-truth "
                "comparison).")

    if st.button("Load example data", key="batch_load_example",
                 help="Pre-fill with 5 real test transactions (3 fraud, 2 legit)."):
        st.session_state["batch_text"] = batch_example_csv(R)

    text = st.text_area(
        "Transaction data (CSV)", key="batch_text", height=200,
        placeholder=("TransactionID,TransactionAmt,hour,ProductCD,DeviceType,addr1,isFraud\n"
                     "3529001,54.225,13,C,mobile,299,1\n..."),
    )
    run = st.button("Run pipeline", type="primary", key="batch_run")

    if not run:
        return

    # ---- STEP 1/2: parse + validate --------------------------------------- #
    try:
        df = parse_batch_csv(text)
    except ValueError as e:
        st.error(str(e))
        return

    st.divider()
    st.markdown("#### Step 1 — Parsed transactions")
    has_truth = "isFraud" in df.columns
    if has_truth:
        n = len(df)
        n_fraud = int(pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).sum())
        st.caption(f"{n} transaction(s) · {n_fraud} labelled fraud · "
                   f"fraud rate {n_fraud / n:.1%} (from the isFraud column)")
    else:
        st.caption(f"{len(df)} transaction(s) · no isFraud column (ground truth unknown)")
    st.dataframe(df, width="stretch", hide_index=True)

    # ---- STEP 3: graph ---------------------------------------------------- #
    st.markdown("#### Step 2 — Relationship graph")
    st.caption("Circles are transactions (red = fraud, green = legit, grey = unknown). "
               "Squares are the entities they share — merchant, device, region and card. "
               "Transactions linked to the same square share that entity.")
    try:
        st.pyplot(batch_graph_figure(df))
    except Exception as e:
        st.warning(f"Could not render the graph: {type(e).__name__}: {e}")

    # ---- STEP 4: predictions ---------------------------------------------- #
    st.markdown("#### Step 3 — Model predictions")
    with st.spinner("Scoring all transactions with the best model…"):
        records = batch_predict(df, R)
    threshold = R["threshold"]

    table = []
    for rec in records:
        out = {
            "TransactionID": rec["tid"],
            "Amount": f"{rec['amount']:,.2f}",
            "Predicted Probability": f"{rec['prob']:.1%}",
            "Verdict": rec["verdict"],
        }
        if has_truth and rec["actual"] is not None:
            out["Actual"] = "FRAUD" if rec["actual"] == 1 else "LEGIT"
            correct = (rec["actual"] == 1) == (rec["verdict"] == "FRAUD")
            out["Result"] = "Correct" if correct else "Wrong"
        table.append(out)
    res_df = pd.DataFrame(table)

    def _highlight_wrong(row):
        if row.get("Result") == "Wrong":
            return ["background-color: #FEE2E2; color: #7F1D1D"] * len(row)
        return [""] * len(row)

    styler = res_df.style.apply(_highlight_wrong, axis=1) if "Result" in res_df.columns else res_df.style
    st.dataframe(styler, width="stretch", hide_index=True)
    st.caption(f"Verdict uses the model's {R['thr_source']} ({threshold:.1%}).")

    if "Result" in res_df.columns:
        n_correct = int((res_df["Result"] == "Correct").sum())
        st.caption(f"Accuracy on these rows: {n_correct}/{len(res_df)} correct "
                   "(wrong predictions highlighted in red).")

    # ---- STEP 5: explanations for flagged frauds -------------------------- #
    flagged = [rec for rec in records if rec["verdict"] == "FRAUD"]
    st.markdown("#### Step 4 — Explanations for flagged frauds")
    if not flagged:
        st.info("No transactions were flagged as FRAUD by the model — nothing to explain.")
        return

    st.caption(f"{len(flagged)} transaction(s) flagged. Expand each for its SHAP top-5 "
               "factors and a generated explanation.")
    for rec in flagged:
        with st.expander(f"TransactionID {rec['tid']}  ·  probability {rec['prob']:.1%}  "
                         f"·  amount {rec['amount']:,.2f}"):
            with st.spinner("Explaining (SHAP over 32 features)…"):
                _, tX, tM, gctx, _ = _custom_tensors(rec["inputs"], R)
                rows, base, prob = shap_custom(tX, tM, gctx, R)
            e1, e2 = st.columns([1, 1])
            with e1:
                st.markdown("**Top contributing factors (SHAP)**")
                st.pyplot(shap_bar_figure(rows, top_k=5))
            with e2:
                st.markdown("**Natural-language explanation**")
                inp = rec["inputs"]
                raw = {
                    "TransactionAmt": inp["amount"], "hour": inp["hour"],
                    "ProductCD": inp["merchant"], "DeviceType": inp["device"],
                    "addr1": inp["region"], "card4": inp["card4"], "card6": inp["card6"],
                }
                payload = {
                    "meta": {"predicted_prob": prob, "base_value": base},
                    "raw": raw, "rows": rows,
                }
                with st.spinner("Generating explanation with the local model…"):
                    nl_text, source = nl_explanation(payload, R)
                st.markdown(
                    f'<div class="fs-nl">{nl_text}<span class="src">Source: {source}</span></div>',
                    unsafe_allow_html=True,
                )


def render_live(R: dict) -> None:
    st.markdown('<div class="fs-section">Live Assessment — Transformer + GAT '
                '(Best Model)</div>', unsafe_allow_html=True)
    st.markdown('<div class="fs-section-sub">Score a real test transaction or enter '
                'your own. Each assessment returns a probability, a verdict, the top '
                'SHAP factors and a generated explanation.</div>',
                unsafe_allow_html=True)

    tab_sample, tab_custom, tab_batch = st.tabs(
        ["Sample Transactions", "Custom Input", "Batch Input"])

    # --- Tab 1: sample test transactions ---------------------------------- #
    with tab_sample:
        samples = get_sample_targets(R)
        labels = []
        for i, s in enumerate(samples):
            raw = s["raw"]
            kind = "FRAUD" if s["true"] == 1 else "LEGIT"
            tid = raw.get("TransactionID", s["global"])
            amt = raw.get("TransactionAmt", "?")
            pcd = raw.get("ProductCD", "?")
            labels.append(f"[{kind}]  TxnID {tid}  ·  amount {amt}  ·  product {pcd}")
        choice = st.radio("Pick a labelled test transaction:", list(range(len(samples))),
                          format_func=lambda i: labels[i], key="sample_choice")

        sel = samples[choice]
        raw = sel["raw"]
        st.caption(
            f"True label: {'FRAUD' if sel['true'] == 1 else 'LEGIT'}  ·  "
            f"hour {raw.get('hour', '?')}  ·  device {raw.get('DeviceType', '?')}  ·  "
            f"region(addr1) {raw.get('addr1', '?')}  ·  email {raw.get('P_emaildomain', '?')}"
        )

        if st.button("Assess sample transaction", type="primary", key="assess_sample"):
            with st.spinner("Scoring and explaining (SHAP over 32 features)…"):
                rows, base, prob, raw_full = shap_sample(sel["global"], R)
            ctx_desc = (f"real transaction node embedded in the trained graph "
                        f"(TransactionID {raw_full.get('TransactionID', sel['global'])})")
            st.divider()
            render_result(prob, R["threshold"], rows, base, raw_full, ctx_desc, R)

    # --- Tab 2: custom input ---------------------------------------------- #
    with tab_custom:
        st.info(FORM_DISCLAIMER)
        opts = R["options"]
        with st.form("custom_form"):
            c1, c2 = st.columns(2)
            with c1:
                amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
                hour = st.slider("Hour of day", 0, 23, 13)
                merchant = st.selectbox("Merchant (ProductCD)", opts["ProductCD"],
                                        index=_default_index(opts["ProductCD"], "W"))
            with c2:
                device = st.selectbox("Device type", opts["DeviceType"],
                                      index=_default_index(opts["DeviceType"], "desktop"))
                region = st.number_input("Region (addr1)", min_value=0, value=299, step=1)
            with st.expander("Optional: card details"):
                card4 = st.selectbox("Card network (card4)", ["(unspecified)"] + opts["card4"])
                card6 = st.selectbox("Card type (card6)", ["(unspecified)"] + opts["card6"])
            submitted = st.form_submit_button("Assess transaction", type="primary")

        if submitted:
            inputs = {
                "amount": amount, "hour": hour, "merchant": merchant, "device": device,
                "region": float(region),
                "card4": None if card4 == "(unspecified)" else card4,
                "card6": None if card6 == "(unspecified)" else card6,
            }
            with st.spinner("Scoring and explaining (SHAP over 32 features)…"):
                vec, tX, tM, gctx, ctx_desc = _custom_tensors(inputs, R)
                rows, base, prob = shap_custom(tX, tM, gctx, R)
            raw = {
                "TransactionAmt": amount, "hour": hour, "ProductCD": merchant,
                "DeviceType": device, "addr1": region,
                "card4": inputs["card4"], "card6": inputs["card6"],
            }
            st.divider()
            render_result(prob, R["threshold"], rows, base, raw, ctx_desc, R)
            st.caption("Features you did not specify are held at the dataset average, so "
                       "this is the model's estimate given your inputs, not a full case file.")

    # --- Tab 3: batch input ----------------------------------------------- #
    with tab_batch:
        render_batch(R)


def render_research(cmp: pd.DataFrame) -> None:
    st.markdown('<div class="fs-section">Research Context</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="fs-card"><h4>Dataset</h4><p>IEEE-CIS Fraud Detection — '
            '590,540 transactions, ~3.5% fraud. Time-aware 70/15/15 split with a '
            'shared train-fit scaler to prevent leakage.</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="fs-card"><h4>Architecture</h4><p>A 2-layer Transformer '
            'sequence encoder fused with one of four graph branches '
            '(GraphSAGE, GAT, ST-HGNN) — each projecting to a 128-d embedding before '
            'a shared fusion classifier.</p></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="fs-card"><h4>Explainability</h4><p>SHAP KernelExplainer '
            'attributes each score to the 32 input features; a local Ollama LLM '
            'converts the signed factors into a grounded plain-English explanation '
            '(no external API).</p></div>',
            unsafe_allow_html=True)

    st.info(
        "Documented limitation: the GAT and ST-HGNN checkpoints were trained with "
        "GAT_HEADS = 1 (reduced from the design's 4) because the free-tier Colab T4 "
        "GPU ran out of memory with 4-head attention over the full transaction graph. "
        "This is an intentional, recorded hardware constraint — reported metrics for "
        "those models are a likely lower bound, and the qualitative finding that the "
        "graph branch improves over the sequence-only baseline is robust to it."
    )


def render_architecture() -> None:
    """Two-branch system-architecture diagram (sequence + graph -> fusion -> explain)."""
    st.markdown('<div class="fs-section">System Architecture</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fs-section-sub">Two complementary branches — a Transformer '
                'over the card\'s transaction sequence and a GNN over the transaction '
                'graph — fused into a single fraud probability, then explained with '
                'SHAP and a local LLM.</div>', unsafe_allow_html=True)
    st.pyplot(architecture_figure())


def render_confusion() -> None:
    """Best-model (m3_gat) confusion matrix on the test set at its best-F1 threshold."""
    import json

    st.markdown('<div class="fs-section">Best Model — Error Breakdown</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fs-section-sub">How the winning Transformer + GAT model '
                'splits the held-out test set at its best-F1 decision threshold.</div>',
                unsafe_allow_html=True)

    cm_path = config.RESULTS_DIR / "confusion_m3_gat.json"
    if not cm_path.exists():
        st.info("results/confusion_m3_gat.json not found — run evaluation to generate it.")
        return
    cm = json.loads(cm_path.read_text())
    thr = cm.get("threshold", 0.5)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(confusion_figure(cm))
    with c2:
        st.markdown(
            '<div class="fs-card">'
            f'<h4>{cm.get("label", "Best model")}</h4>'
            f'<p>At a decision threshold of <b>{thr:.3f}</b> '
            f'({cm.get("threshold_source", "best-F1")}), the model catches '
            f'<b>{cm["tp"]:,}</b> of {cm["n_fraud"]:,} frauds '
            f'(recall {cm["recall"]:.1%}) while raising <b>{cm["fp"]:,}</b> false '
            f'alarms on {cm["n_legit"]:,} legitimate transactions — a precision of '
            f'{cm["precision"]:.1%}.</p></div>',
            unsafe_allow_html=True)
    st.caption("Lowering the threshold catches more fraud (higher recall) at the cost of "
               "more false alarms (lower precision); this operating point maximizes F1.")


def _default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


# --------------------------------------------------------------------------- #
# Reproducibility & Artifacts page
# --------------------------------------------------------------------------- #
def _human_size(num_bytes: int) -> str:
    """Human-readable file size (e.g. '3.9 MB')."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@st.cache_data(show_spinner=False)
def _read_bytes(path_str: str) -> bytes:
    """Read a file from disk as bytes (cached so reruns don't re-read)."""
    return Path(path_str).read_bytes()


# (path, download filename, what it is, MIME, primary?) for each downloadable artifact.
ARTIFACTS = [
    (config.MODELS_DIR / "m1_transformer.pt", "m1_transformer.pt",
     "Model 1 — Transformer only (sequence-only baseline).", "application/octet-stream"),
    (config.MODELS_DIR / "m2_sage.pt", "m2_sage.pt",
     "Model 2 — Transformer + GraphSAGE.", "application/octet-stream"),
    (config.MODELS_DIR / "m3_gat.pt", "m3_gat.pt",
     "Model 3 — Transformer + GAT.", "application/octet-stream"),
    (config.MODELS_DIR / "m4_sthgnn.pt", "m4_sthgnn.pt",
     "Model 4 — Transformer + ST-HGNN.", "application/octet-stream"),
]


def _artifact_card(path, filename: str, what: str, mime: str,
                   primary: bool = False) -> None:
    """Render one artifact as a card + a download button (or a warning if missing)."""
    if not path.exists():
        st.warning(f"{filename} not found at {path} — run the earlier phases to "
                   "generate it before it can be downloaded.")
        return
    size = _human_size(path.stat().st_size)
    st.markdown(
        f'<div class="fs-artifact{" primary" if primary else ""}">'
        f'<div class="fname">{filename}</div>'
        f'<div class="what">{what}</div>'
        f'<div class="size">{size}</div></div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label=f"Download {filename}",
        data=_read_bytes(str(path)),
        file_name=filename,
        mime=mime,
        key=f"dl_{filename}",
        width="stretch",
    )


def render_artifacts() -> None:
    st.markdown('<div class="fs-section">Reproducibility &amp; Artifacts</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="fs-section-sub">The trained models, the preprocessing scaler and '
        'the full results table are openly available for download. Together with the '
        'pipeline code they let anyone reproduce, inspect or re-use this study\'s '
        'results end to end.</div>',
        unsafe_allow_html=True,
    )

    # --- Winning model (highlighted) -------------------------------------- #
    st.markdown("#### Best model")
    _artifact_card(config.BEST_MODEL_PT, "best_model.pt",
                   "The winning model — Transformer + GAT (selected by test PR-AUC). "
                   "This is the checkpoint the live demo runs on.",
                   "application/octet-stream", primary=True)

    # --- All four trained checkpoints ------------------------------------- #
    st.markdown("#### All trained checkpoints")
    cols = st.columns(2)
    for i, (path, fname, what, mime) in enumerate(ARTIFACTS):
        with cols[i % 2]:
            _artifact_card(path, fname, what, mime)

    # --- Preprocessing + results ------------------------------------------ #
    st.markdown("#### Preprocessing and results")
    c1, c2 = st.columns(2)
    with c1:
        _artifact_card(config.SCALER_PKL, "scaler.pkl",
                       "StandardScaler fit on the training split only (shared by every "
                       "model so feature scaling is identical and leak-free).",
                       "application/octet-stream")
    with c2:
        _artifact_card(config.COMPARISON_CSV, "comparison.csv",
                       "The full four-model results table — PR-AUC, ROC-AUC, F1, "
                       "precision, recall and inference time on the shared test split.",
                       "text/csv")

    # --- How to load note ------------------------------------------------- #
    st.info(
        "Note: the .pt files are model weights (state dictionaries), not standalone "
        "programs. To load them you need this project's model architecture code — the "
        "same `SequenceTransformer` + GNN branch + `FusionClassifier` definitions used "
        "for training — plus `scaler.pkl` for feature scaling. The complete, "
        "reproducible pipeline (data prep, training, evaluation and this app) is "
        "available at: [your repo URL]"
    )

    st.divider()
    render_run_instructions()


def render_run_instructions() -> None:
    """Task 7.3: self-documented setup/run steps, shown inside the app itself."""
    st.markdown("#### Run this app locally")
    st.markdown(
        "1. **Install dependencies** — `pip install -r requirements.txt` "
        "(everything is free & open-source).\n"
        "2. **Install Ollama** (free, local LLM runtime) — download from "
        "[ollama.com](https://ollama.com), then pull a small free model: "
        "`ollama pull llama3.2:3b` (the default this app uses) or `ollama pull phi3`.\n"
        "3. **Start Ollama** if it isn't already running as a background service "
        "(`ollama serve`).\n"
        "4. **Launch the app** from the project root: "
        "`streamlit run app/streamlit_app.py`."
    )
    st.info(
        "**Free-hosting caveat:** Streamlit Community Cloud cannot run Ollama, so a "
        "cloud-hosted copy of this app cannot serve the local LLM. If Ollama isn't "
        "reachable — locally or in the cloud — the app automatically falls back to a "
        "deterministic, template-based explanation (still grounded in the same SHAP "
        "factors) instead of failing; the natural-language card always states its "
        "source (`ollama:<model>` vs `templated fallback`) so this is never hidden."
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_fraud_detection(cmp: pd.DataFrame) -> None:
    """Main page: landing + how-it-works + the live fraud-assessment console."""
    render_hero(cmp)
    st.divider()
    render_how_it_works()
    st.divider()

    try:
        R = load_resources()
    except Exception as e:
        st.markdown('<div class="fs-section">Live Assessment</div>', unsafe_allow_html=True)
        st.error(f"Could not load the model/artifacts:\n\n{type(e).__name__}: {e}")
        st.stop()

    render_live(R)


def page_analysis(cmp: pd.DataFrame) -> None:
    """Analysis & Findings page: architecture, comparative study, error breakdown, context."""
    render_architecture()
    st.divider()
    render_comparison(cmp)
    st.divider()
    render_confusion()
    st.divider()
    render_research(cmp)


def page_artifacts(cmp: pd.DataFrame) -> None:
    """Reproducibility & Artifacts page: download the trained models + results."""
    render_artifacts()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PAGES = {
    "Fraud Detection": page_fraud_detection,
    "Analysis & Findings": page_analysis,
    "Reproducibility & Artifacts": page_artifacts,
}


def main() -> None:
    st.set_page_config(page_title="FraudShield", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    cmp = load_comparison()

    # Left-sidebar navigation across the app's pages.
    st.sidebar.markdown('<div class="fs-nav-brand">FraudShield</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown('<div class="fs-nav-tag">Sequence-aware fraud detection '
                        'with graph intelligence</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", list(PAGES.keys()),
                            label_visibility="collapsed")

    PAGES[page](cmp)


if __name__ == "__main__":
    main()
