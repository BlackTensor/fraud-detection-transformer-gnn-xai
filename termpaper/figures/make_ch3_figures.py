"""
make_ch3_figures.py — generates the three Chapter 3 (Methodology) figures.

Outputs (saved to ../assets/):
  fig_3_1_two_views.png   : how one flat transaction table becomes two views —
                            per-card ordered sequences and a relationship graph.
  fig_3_2_hetero_graph.png: the heterogeneous graph we built — transaction nodes
                            wired to card / device / merchant / region entities,
                            with the real node counts and typed edges.
  fig_3_3_fusion.png      : the fusion classifier — two 128-d embeddings concat
                            into an MLP that outputs a fraud probability.

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

# --- shared palette (consistent with Chapters 1 and 2) ---
FRAUD   = "#C0392B"
LEGIT   = "#2E86C1"
SEQCOL  = "#2874A6"
GRAPHC  = "#1E8449"
FUSE    = "#7D3C98"
QUERY   = "#B9770E"
ENTITY  = "#566573"
CARDBG  = "#EAF2F8"
GREY    = "#7F8C8D"
INK     = "#1B2631"


def box(ax, x, y, w, h, text, fc, ec, fs=11, tc=INK, bold=False, rounding=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                 facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, color=INK, lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                 color=color, lw=lw, shrinkA=2, shrinkB=2, zorder=1))


# =====================================================================
# Figure 3.1 — from one flat table to two views
# =====================================================================
def two_views():
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ---- left: the flat transaction table ----
    ax.text(1.9, 6.4, "IEEE-CIS transactions", fontsize=12, fontweight="bold",
            color=INK, ha="center")
    tx, ty, tw, rh = 0.4, 2.4, 3.0, 0.52
    header = ["card", "time", "amount", "..."]
    rows = [
        ["A", "t1", "12", "…"],
        ["A", "t2", "15", "…"],
        ["B", "t1", "40", "…"],
        ["A", "t3", "220", "…"],
        ["B", "t2", "38", "…"],
    ]
    ncol = len(header)
    cw = tw / ncol
    # header
    for j, hcell in enumerate(header):
        box(ax, tx + j * cw, ty + len(rows) * rh, cw, rh, hcell, "#D6DBDF",
            ENTITY, fs=9.5, bold=True, rounding=0.02)
    for i, row in enumerate(rows):
        yy = ty + (len(rows) - 1 - i) * rh
        fraud = (i == 3)  # the big jump on card A
        for j, cell in enumerate(row):
            fc = "#FADBD8" if fraud else "white"
            box(ax, tx + j * cw, yy, cw, rh, cell, fc, GREY, fs=9.5, rounding=0.02)

    ax.text(1.9, 2.0, "one row per transaction\n(no order, no links shown)",
            fontsize=9.5, color=GREY, ha="center", style="italic")

    # ---- arrows to the two views ----
    arrow(ax, (3.7, 4.6), (5.2, 5.5), color=SEQCOL, lw=1.8)
    arrow(ax, (3.7, 3.2), (5.2, 2.0), color=GRAPHC, lw=1.8)

    # ---- top-right: sequence view ----
    ax.text(9.1, 6.5, "View 1 — card sequences", fontsize=12, fontweight="bold",
            color=SEQCOL, ha="center")
    # card A row of ordered transactions
    seqx = 5.4
    for k, (lab, col) in enumerate([("t1", LEGIT), ("t2", LEGIT), ("t3", FRAUD)]):
        cx = seqx + k * 1.15
        ax.add_patch(Circle((cx, 5.55), 0.34, facecolor=CARDBG, edgecolor=col,
                            lw=2.0, zorder=3))
        ax.text(cx, 5.55, lab, ha="center", va="center", fontsize=9.5,
                color=INK, zorder=4)
        if k > 0:
            arrow(ax, (cx - 1.15 + 0.34, 5.55), (cx - 0.34, 5.55), color=GREY, lw=1.2)
    ax.text(seqx + 1.15, 4.85, "card A, in time order", fontsize=9.5, color=GREY,
            ha="center", style="italic")
    ax.text(9.1, 4.35, "→ a Transformer reads each card's recent run",
            fontsize=10, color=SEQCOL, ha="center")

    # ---- bottom-right: graph view ----
    ax.text(9.1, 3.5, "View 2 — relationship graph", fontsize=12, fontweight="bold",
            color=GRAPHC, ha="center")
    # tiny graph: two txns sharing a card
    ta = (6.6, 1.9); tb = (8.1, 1.15); shared = (7.35, 2.75)
    for p, col in [(ta, FRAUD), (tb, LEGIT)]:
        ax.plot([p[0], shared[0]], [p[1], shared[1]], color=GREY, lw=1.2, zorder=1)
        ax.add_patch(Circle(p, 0.26, facecolor=col, edgecolor="white", lw=1.2, zorder=3))
        ax.text(p[0], p[1], "txn", ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold", zorder=4)
    box(ax, shared[0] - 0.55, shared[1] - 0.22, 1.1, 0.44, "card A", CARDBG,
        ENTITY, fs=9.5, rounding=0.06)
    ax.text(10.6, 1.5, "→ a graph network\nlets shared entities\nlink transactions",
            fontsize=10, color=GRAPHC, ha="center", va="center")

    fig.tight_layout()
    out = ASSETS / "fig_3_1_two_views.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# Figure 3.2 — the heterogeneous graph we built
# =====================================================================
def hetero_graph():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.6, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0, 3.5, "The heterogeneous transaction graph we built",
            fontsize=13, fontweight="bold", color=INK, ha="center")

    # three transaction nodes down the middle
    txn_pos = [(0, 1.4), (0, 0.0), (0, -1.4)]
    txn_fraud = [False, True, False]

    # entity nodes: (pos, label, count, edge-name)
    entities = [
        ((-3.6, 1.9),  "card\n(42,946)",     "made_by"),
        ((-3.6, -1.9), "device\n(1,943)",    "on_device"),
        ((3.6, 1.9),   "merchant\n(5)",      "at_merchant"),
        ((3.6, -1.9),  "region\n(332)",      "in_region"),
    ]

    # wire each transaction to one of each entity type (illustrative)
    # card: t0,t1 share card L; t2 -> card R (drawn as same 'card' box for simplicity)
    def link(p, q, col=GREY, ls="-"):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=1.1, ls=ls, zorder=1)

    # edges from every txn to card & device (left) and merchant & region (right)
    for tp in txn_pos:
        link(tp, entities[0][0])   # card
        link(tp, entities[1][0])   # device
        link(tp, entities[2][0])   # merchant
        link(tp, entities[3][0])   # region

    # entity boxes
    for pos, lab, _ in entities:
        ax.add_patch(FancyBboxPatch((pos[0] - 0.95, pos[1] - 0.45), 1.9, 0.9,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor=CARDBG, edgecolor=ENTITY, lw=1.5, zorder=3))
        ax.text(pos[0], pos[1], lab, ha="center", va="center", fontsize=10,
                color=INK, zorder=4)

    # transaction nodes
    for tp, fr in zip(txn_pos, txn_fraud):
        col = FRAUD if fr else LEGIT
        ax.add_patch(Circle(tp, 0.42, facecolor=col, edgecolor="white", lw=1.5, zorder=5))
        ax.text(tp[0], tp[1], "txn", ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=6)

    # edge-type labels along one representative edge each
    ax.text(-2.05, 1.05, "made_by", fontsize=8.5, color=ENTITY, ha="center",
            rotation=18, style="italic")
    ax.text(-2.05, -1.05, "on_device", fontsize=8.5, color=ENTITY, ha="center",
            rotation=-18, style="italic")
    ax.text(2.05, 1.05, "at_merchant", fontsize=8.5, color=ENTITY, ha="center",
            rotation=-18, style="italic")
    ax.text(2.05, -1.05, "in_region", fontsize=8.5, color=ENTITY, ha="center",
            rotation=18, style="italic")

    ax.text(0, -3.15,
            "590,540 transaction nodes wired to four entity types by typed edges "
            "(shown both ways in the model).\nEvidence flows between transactions that "
            "share an entity.",
            ha="center", fontsize=10, color=GREY, style="italic")

    # legend
    ax.scatter([], [], s=110, color=FRAUD, label="fraudulent transaction")
    ax.scatter([], [], s=110, color=LEGIT, label="legitimate transaction")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(-0.02, 1.0))

    fig.tight_layout()
    out = ASSETS / "fig_3_2_hetero_graph.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# Figure 3.3 — the fusion classifier
# =====================================================================
def fusion():
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # two branch outputs
    box(ax, 0.3, 3.9, 2.7, 1.1, "Transformer\nembedding", "#D6EAF8", SEQCOL, bold=True)
    box(ax, 0.3, 1.0, 2.7, 1.1, "Graph\nembedding", "#D6F5E3", GRAPHC, bold=True)
    ax.text(1.65, 3.65, "128-d", fontsize=9.5, color=GREY, ha="center", va="top")
    ax.text(1.65, 0.75, "128-d", fontsize=9.5, color=GREY, ha="center", va="top")

    # concat
    box(ax, 3.9, 2.45, 2.0, 1.1, "concat\n→ 256-d", "#FDEBD0", QUERY, bold=True)
    arrow(ax, (3.0, 4.45), (3.9, 3.3))
    arrow(ax, (3.0, 1.55), (3.9, 2.7))

    # MLP block
    box(ax, 6.7, 2.45, 3.0, 1.1, "Linear 256→128\nReLU · Dropout 0.3", "#E8DAEF", FUSE)
    arrow(ax, (5.9, 3.0), (6.7, 3.0))

    box(ax, 10.4, 2.45, 2.3, 1.1, "Linear 128→1\nSigmoid", "#E8DAEF", FUSE)
    arrow(ax, (9.7, 3.0), (10.4, 3.0))

    # output
    ax.annotate("fraud\nprobability", xy=(12.7, 3.0), xytext=(11.9, 4.9),
                fontsize=11, color=FRAUD, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=FRAUD, lw=1.4))

    ax.text(6.5, 0.5,
            "Model 1 (Transformer only) skips the concat and classifies the 128-d "
            "sequence embedding directly (Linear 128→1).",
            fontsize=9.5, color=GREY, ha="center", style="italic")

    fig.tight_layout()
    out = ASSETS / "fig_3_3_fusion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    two_views()
    hetero_graph()
    fusion()
