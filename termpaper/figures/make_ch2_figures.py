"""
make_ch2_figures.py — generates the two Chapter 2 figures for the term paper.

Outputs (saved to ../assets/):
  fig_2_1_attention.png   : self-attention intuition — one transaction in a
                            sequence attending, with different weights, to the
                            earlier transactions of the same card.
  fig_2_2_message_passing.png : message passing in a GNN — a transaction node
                            gathering messages from its neighbouring entities.

Clean, presentation-quality, serif-styled to sit next to Times New Roman body text.
Free/open-source only (matplotlib + numpy). Deterministic layout (no randomness).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager
from pathlib import Path
import numpy as np

# --- try to match the document's Times New Roman; fall back to a serif ---
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\times.ttf")
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\timesbd.ttf")
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = 12

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# --- shared palette (kept consistent with the Chapter 1 figures) ---
FRAUD   = "#C0392B"   # deep red
LEGIT   = "#2E86C1"   # blue
QUERY   = "#B9770E"   # amber (the transaction currently being scored)
ENTITY  = "#566573"   # slate
CARDBG  = "#EAF2F8"
GREY    = "#7F8C8D"
INK     = "#1B2631"


# =====================================================================
# Figure 2.1 — self-attention over a card's transaction sequence
# =====================================================================
def attention():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.1, 4.6, "Reading a card's sequence with attention", fontsize=13,
            fontweight="bold", color=INK, ha="left")

    # a row of six transactions of one card, left (oldest) -> right (newest)
    n = 6
    xs = np.linspace(1.0, 9.0, n)
    y = 2.4
    # the newest transaction (rightmost) is the one being scored = the "query"
    query_ix = n - 1
    # made-up but sensible attention weights: the query leans on two past txns
    weights = [0.05, 0.35, 0.10, 0.08, 0.30, 0.0]

    qx = xs[query_ix]
    # draw curved attention edges from the query back to each earlier txn
    for i in range(query_ix):
        w = weights[i]
        col = FRAUD if w >= 0.25 else GREY
        arc = FancyArrowPatch(
            (qx, y + 0.28), (xs[i], y + 0.28),
            connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
            mutation_scale=13, lw=0.8 + 6.0 * w, color=col, alpha=0.75, zorder=1,
        )
        ax.add_patch(arc)

    # transaction nodes
    for i, x in enumerate(xs):
        if i == query_ix:
            fc, ec, tc = "#FDEBD0", QUERY, INK
        else:
            fc, ec, tc = CARDBG, LEGIT, INK
        ax.add_patch(Circle((x, y), 0.40, facecolor=fc, edgecolor=ec, lw=2.0, zorder=3))
        ax.text(x, y, f"t{i+1}", ha="center", va="center", fontsize=11,
                color=tc, fontweight="bold", zorder=4)

    # time arrow underneath
    ax.annotate("", xy=(9.4, 1.5), xytext=(0.7, 1.5),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1))
    ax.text(0.7, 1.15, "older", fontsize=10, color=GREY, ha="left")
    ax.text(9.4, 1.15, "most recent (being scored)", fontsize=10, color=GREY, ha="right")

    # labels
    ax.text(qx, y + 0.75, "query", fontsize=10.5, color=QUERY, ha="center",
            fontweight="bold")
    ax.text(5.0, 0.35, "thicker, red links = the past transactions this one "
            "leans on most\nwhen deciding whether it looks normal",
            fontsize=10.5, color=INK, ha="center", va="bottom")

    fig.tight_layout()
    out = ASSETS / "fig_2_1_attention.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# Figure 2.2 — message passing in a GNN
# =====================================================================
def message_passing():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(-4.1, 3.0, "How a graph network updates one transaction",
            fontsize=13, fontweight="bold", color=INK, ha="left")

    centre = (0.0, 0.0)

    # neighbouring entity nodes around the transaction
    neigh = [
        (( 2.6,  1.4), "card",     CARDBG),
        (( 2.6, -1.4), "device",   CARDBG),
        ((-2.6,  1.4), "merchant", CARDBG),
        ((-2.6, -1.4), "region",   CARDBG),
    ]

    # edges + message arrows flowing inward
    for (nx_, ny_), lab, fc in neigh:
        ax.plot([centre[0], nx_], [centre[1], ny_], color=GREY, lw=1.1, zorder=1)
        arr = FancyArrowPatch((nx_, ny_), (centre[0] * 0.0 + 0.55 * np.sign(nx_),
                                           0.55 * np.sign(ny_)),
                              arrowstyle="-|>", mutation_scale=14, color=LEGIT,
                              lw=2.0, alpha=0.8, zorder=2,
                              connectionstyle="arc3,rad=0.0", shrinkA=14, shrinkB=10)
        ax.add_patch(arr)
        ax.add_patch(FancyBboxPatch((nx_ - 0.62, ny_ - 0.30), 1.24, 0.60,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor=fc, edgecolor=ENTITY, lw=1.3, zorder=3))
        ax.text(nx_, ny_, lab, ha="center", va="center", fontsize=10.5,
                color=INK, zorder=4)

    # central transaction node
    ax.add_patch(Circle(centre, 0.62, facecolor="#FDEBD0", edgecolor=QUERY,
                        lw=2.2, zorder=5))
    ax.text(centre[0], centre[1], "txn", ha="center", va="center", fontsize=11.5,
            color=INK, fontweight="bold", zorder=6)

    ax.text(0.0, -2.55,
            "The transaction's new description = its own features combined with\n"
            "messages gathered from every entity it is connected to.",
            ha="center", fontsize=10.5, color=GREY, style="italic")

    fig.tight_layout()
    out = ASSETS / "fig_2_2_message_passing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    attention()
    message_passing()
