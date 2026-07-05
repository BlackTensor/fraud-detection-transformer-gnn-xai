"""
make_ch5_figures.py — generates the Chapter 5 (Results) figure(s) that are NOT
already-existing project plots.

The ROC / PR / metrics-bar charts are real project assets copied straight from
results/plots/ into assets/ (fig_5_1..fig_5_3). This script builds the one
remaining figure:

  fig_5_4_confusion.png : a clean 2x2 confusion-matrix heatmap for the winning
                          model (Transformer + GAT) on the held-out test split,
                          at its best-F1 decision threshold. Read straight from
                          results/confusion_m3_gat.json so the counts are the
                          real ones, not hand-typed.

Free/open-source only (matplotlib + numpy). Deterministic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
import json
import numpy as np

try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\times.ttf")
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\timesbd.ttf")
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = 12

ROOT = Path(__file__).resolve().parent.parent.parent          # project root
ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#1B2631"


def confusion():
    data = json.loads((ROOT / "results" / "confusion_m3_gat.json").read_text())
    tn, fp, fn, tp = data["tn"], data["fp"], data["fn"], data["tp"]
    thr = data["threshold"]
    # rows = actual (Legit, Fraud); cols = predicted (Legit, Fraud)
    mat = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    # colour each cell on its own row-scale so the rare-fraud row is readable
    # (a global scale would wash the fraud row out against the huge TN count).
    norm = np.zeros_like(mat, dtype=float)
    for r in range(2):
        row = mat[r].astype(float)
        norm[r] = row / row.max()

    greens = plt.cm.Greens
    reds = plt.cm.Reds
    # correct cells (TN, TP) in green; error cells (FP, FN) in red
    colours = np.empty((2, 2), dtype=object)
    colours[0, 0] = greens(0.25 + 0.6 * norm[0, 0])   # TN
    colours[0, 1] = reds(0.20 + 0.6 * norm[0, 1])     # FP
    colours[1, 0] = reds(0.20 + 0.6 * norm[1, 0])     # FN
    colours[1, 1] = greens(0.25 + 0.6 * norm[1, 1])   # TP

    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=colours[r, c],
                                       edgecolor="white", lw=3, zorder=1))
            label = [["True negatives", "False positives"],
                     ["False negatives", "True positives"]][r][c]
            txtcol = "white" if norm[r, c] > 0.6 else INK
            ax.text(c + 0.5, 1 - r + 0.60, f"{mat[r, c]:,}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=txtcol, zorder=2)
            ax.text(c + 0.5, 1 - r + 0.30, label, ha="center", va="center",
                    fontsize=11, color=txtcol, zorder=2)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([1.5, 0.5])
    ax.set_xticklabels(["Predicted legit", "Predicted fraud"], fontsize=12)
    ax.set_yticklabels(["Actual legit", "Actual fraud"], fontsize=12, rotation=90, va="center")
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title(f"Confusion matrix — Transformer + GAT (test set)\n"
                 f"decision threshold = {thr:.3f} (best-F1)  ·  "
                 f"precision {data['precision']:.2f}, recall {data['recall']:.2f}",
                 fontsize=12.5, color=INK, pad=28)

    fig.tight_layout()
    out = ASSETS / "fig_5_4_confusion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out, "| counts TN/FP/FN/TP =", tn, fp, fn, tp)


if __name__ == "__main__":
    confusion()
