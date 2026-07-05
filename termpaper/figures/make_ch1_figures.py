"""
make_ch1_figures.py — generates the two Chapter 1 figures for the term paper.

Outputs (saved to ../assets/):
  fig_1_1_scenario.png      : the motivating fraud scenario (behaviour-over-time
                              timeline + a shared-device fraud ring)
  fig_1_2_architecture.png  : the high-level two-branch architecture

Clean, presentation-quality, serif-styled to sit next to Times New Roman body text.
Free/open-source only (matplotlib + networkx). Deterministic layout (no randomness).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager
from pathlib import Path

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

# --- shared palette ---
FRAUD   = "#C0392B"   # deep red
LEGIT   = "#2E86C1"   # blue
DEVICE  = "#566573"   # slate
CARDBG  = "#EAF2F8"
GREY    = "#7F8C8D"
INK     = "#1B2631"


# =====================================================================
# Figure 1.1 — the scenario
# =====================================================================
def scenario():
    fig, (ax_t, ax_g) = plt.subplots(
        2, 1, figsize=(8, 6.6), gridspec_kw={"height_ratios": [1, 1.35]}
    )

    # ---------- top: behaviour over time ----------
    ax_t.set_title("(a)  One card's behaviour over time", fontsize=13,
                   fontweight="bold", loc="left", color=INK, pad=8)
    ax_t.axhline(0, color=GREY, lw=1.2, zorder=1)
    # normal small local purchases, then a sudden jump
    xs      = [0.6, 1.3, 2.0, 2.7, 3.4, 4.1, 5.3, 6.1]
    amt     = [0.30, 0.45, 0.28, 0.50, 0.35, 0.42, 1.55, 1.70]
    fraudix = {6, 7}
    for i, (x, a) in enumerate(zip(xs, amt)):
        col = FRAUD if i in fraudix else LEGIT
        ax_t.plot([x, x], [0, a], color=col, lw=2.4, zorder=2)
        ax_t.scatter([x], [a], s=90, color=col, zorder=3, edgecolor="white", linewidth=1)
    ax_t.annotate("small, local, familiar purchases",
                  xy=(2.0, 0.55), xytext=(1.4, 1.35), fontsize=11, color=GREY,
                  ha="center",
                  arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax_t.annotate("sudden high-value spend\nin a new city  →  flagged",
                  xy=(5.3, 1.60), xytext=(5.7, 2.05), fontsize=11, color=FRAUD,
                  ha="center", va="bottom", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=FRAUD, lw=1.3))
    ax_t.annotate("", xy=(6.6, 0), xytext=(0.2, 0),
                  arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax_t.text(6.6, -0.30, "time", fontsize=11, color=GREY, ha="right")
    ax_t.set_xlim(0, 6.9)
    ax_t.set_ylim(-0.5, 2.5)
    ax_t.axis("off")

    # ---------- bottom: shared-device fraud ring ----------
    ax_g.set_title("(b)  The same payments seen as relationships", fontsize=13,
                   fontweight="bold", loc="left", color=INK, pad=8)
    import numpy as np

    # central shared device
    dev = (0.0, 0.0)
    ax_g.add_patch(Circle(dev, 0.48, facecolor=DEVICE, edgecolor="black", lw=1.2, zorder=3))
    ax_g.text(*dev, "shared\ndevice", ha="center", va="center", color="white",
              fontsize=9.5, fontweight="bold", zorder=4)

    # five transactions around it, each on a different card; three are fraud
    n = 5
    labels = ["card A", "card B", "card C", "card D", "card E"]
    is_fraud = [True, True, True, False, False]
    angles = np.linspace(90, 90 + 360, n, endpoint=False)
    for ang, lab, fr in zip(angles, labels, is_fraud):
        rad = np.deg2rad(ang)
        tx = (1.9 * np.cos(rad), 1.35 * np.sin(rad))
        cx = (2.95 * np.cos(rad), 2.05 * np.sin(rad))
        col = FRAUD if fr else LEGIT
        # edge device -> transaction
        ax_g.plot([dev[0], tx[0]], [dev[1], tx[1]], color=GREY, lw=1.3, zorder=1)
        # edge transaction -> card
        ax_g.plot([tx[0], cx[0]], [tx[1], cx[1]], color=GREY, lw=1.0, ls=":", zorder=1)
        # transaction node
        ax_g.add_patch(Circle(tx, 0.24, facecolor=col, edgecolor="white", lw=1.2, zorder=3))
        ax_g.text(tx[0], tx[1], "txn", ha="center", va="center", color="white",
                  fontsize=9, fontweight="bold", zorder=4)
        # card node
        ax_g.add_patch(FancyBboxPatch((cx[0] - 0.34, cx[1] - 0.16), 0.68, 0.32,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       facecolor=CARDBG, edgecolor=DEVICE, lw=1.0, zorder=2))
        ax_g.text(cx[0], cx[1], lab, ha="center", va="center", fontsize=9.5, color=INK, zorder=3)

    # legend (upper-left, clear of the caption at the bottom)
    ax_g.scatter([], [], s=110, color=FRAUD, label="fraudulent transaction")
    ax_g.scatter([], [], s=110, color=LEGIT, label="legitimate transaction")
    ax_g.legend(loc="upper left", frameon=False, fontsize=10, bbox_to_anchor=(-0.02, 1.02))

    ax_g.text(0.0, -2.75, "One device ties together many cards — a pattern a single "
              "transaction row cannot show.", ha="center", fontsize=10.5,
              color=GREY, style="italic")
    ax_g.set_xlim(-3.4, 3.4)
    ax_g.set_ylim(-3.0, 2.5)
    ax_g.set_aspect("equal")
    ax_g.axis("off")

    fig.tight_layout(h_pad=1.5)
    out = ASSETS / "fig_1_1_scenario.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# Figure 1.2 — high-level architecture
# =====================================================================
def box(ax, x, y, w, h, text, fc, ec, fs=11, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, color=INK):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                 color=color, lw=1.6, shrinkA=2, shrinkB=2, zorder=1))


def architecture():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # input
    box(ax, 0.2, 2.9, 2.0, 1.2, "A transaction\nto score", "#FDEBD0", "#CA6F1E", bold=True)

    # two views
    box(ax, 3.0, 4.6, 2.9, 1.5, "Card's recent\ntransactions\n(sequence view)", "#EAF2F8", LEGIT)
    box(ax, 3.0, 0.9, 2.9, 1.5, "Transaction graph\n(shared card, device,\nmerchant, region)", "#EAF2F8", LEGIT)

    # branch encoders
    box(ax, 6.4, 4.7, 2.5, 1.3, "Transformer\nencoder", "#D6EAF8", "#2874A6", bold=True)
    box(ax, 6.4, 1.0, 2.5, 1.3, "Graph neural\nnetwork", "#D6EAF8", "#2874A6", bold=True)

    # 128-d tags
    ax.text(9.05, 5.35, "128-d", fontsize=9.5, color=GREY, va="center")
    ax.text(9.05, 1.65, "128-d", fontsize=9.5, color=GREY, va="center")

    # fusion
    box(ax, 9.6, 2.9, 2.2, 1.2, "Fusion\n(concat → MLP)", "#E8DAEF", "#7D3C98", bold=True)

    # outputs (stacked to the right/below fusion)
    box(ax, 9.6, 5.3, 2.2, 1.1, "Fraud\nprobability", "#FADBD8", FRAUD, bold=True)
    box(ax, 9.6, 0.3, 2.2, 1.5, "SHAP + local LLM\n→ plain-English\nexplanation", "#D5F5E3", "#1E8449", bold=True)

    # arrows
    arrow(ax, (2.2, 3.5), (3.0, 5.35))            # input -> seq view
    arrow(ax, (2.2, 3.5), (3.0, 1.65))            # input -> graph view
    arrow(ax, (5.9, 5.35), (6.4, 5.35))           # seq view -> transformer
    arrow(ax, (5.9, 1.65), (6.4, 1.65))           # graph view -> gnn
    arrow(ax, (8.9, 5.35), (10.7, 4.1))           # transformer -> fusion
    arrow(ax, (8.9, 1.65), (10.7, 2.9))           # gnn -> fusion
    arrow(ax, (10.7, 4.1), (10.7, 5.3))           # fusion -> prob
    arrow(ax, (10.7, 2.9), (10.7, 1.8))           # fusion -> explanation

    fig.tight_layout()
    out = ASSETS / "fig_1_2_architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    scenario()
    architecture()
