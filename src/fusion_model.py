"""
fusion_model.py — FusionClassifier (Task 3.5).

What it does:
    Ties the two views together into the final fraud classifier used by all four
    models in the comparison study. It takes the SequenceTransformer branch and,
    optionally, ONE GNN branch (GraphSAGE / GAT / ST-HGNN), and produces a fraud
    probability per transaction:

        Model 1 (transformer only, gnn_branch=None):
            seq_emb [B,128] -> Linear(128 -> 1) -> sigmoid

        Models 2-4 (transformer + a GNN):
            seq_emb  [B,128]                 (from the sequence window)
            graph_emb[B,128]                 (the SAME transactions, gathered from
                                              the full-graph GNN output by node id)
            concat -> Linear(256 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
            -> sigmoid

    Keeping every branch at EMBED_DIM (128) is what makes this fusion head uniform
    across all four configs. A `mode` label records which config is wired in.

Inputs (forward):
    X_seq           : float32 [B, L, F]   sequence windows (from sequences.pt)
    mask            : bool    [B, L]       valid-mask (True = real timestep)
    x_dict          : dict    node features        (graph.x_dict; needed iff GNN)
    edge_index_dict : dict    edge indices         (graph.edge_index_dict; iff GNN)
    node_idx        : long    [B]          transaction-node indices for this batch,
                                           used to gather the matching graph rows
    return_logits   : bool                 if True return raw logits (for a stable
                                           BCEWithLogits/focal loss); else sigmoid prob

Outputs (forward):
    [B, 1]  fraud probability in (0,1)  — or logits if return_logits=True.

Note on the GNN call: GNN branches are transductive — they embed ALL transaction
nodes in one full-graph pass, then we index `node_idx` to pull out this batch's
rows. The trainer (Task 4.x) may cache that full-graph forward once per step
instead of recomputing per mini-batch; this module supports either by simply
running the branch whenever a graph is provided.

Running this file as a script executes a CPU smoke test that builds all four
configs (none / SAGE / GAT / ST-HGNN) and asserts each returns a [B, 1] prob.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TARGET_NODE = "transaction"

# Human-readable mode label inferred from the GNN branch class name.
_MODE_BY_CLASS = {
    "GraphSAGEBranch": "transformer+sage",
    "GATBranch": "transformer+gat",
    "STHGNNBranch": "transformer+sthgnn",
}


class FusionClassifier(nn.Module):
    """Fuse a SequenceTransformer with an optional GNN branch -> fraud probability."""

    def __init__(
        self,
        transformer_branch: nn.Module,
        gnn_branch: nn.Module | None = None,
        embed_dim: int = config.EMBED_DIM,
        dropout: float = config.DROPOUT,
        target_node: str = TARGET_NODE,
    ) -> None:
        """
        Args:
            transformer_branch: a SequenceTransformer (-> [B, embed_dim]). Required.
            gnn_branch:         one of GraphSAGEBranch / GATBranch / STHGNNBranch,
                                or None for the transformer-only Model 1.
            embed_dim:          per-branch embedding width (128).
            dropout:            dropout in the fusion head.
            target_node:        node type the GNN embeds ('transaction').
        """
        super().__init__()
        self.transformer = transformer_branch
        self.gnn = gnn_branch
        self.embed_dim = embed_dim
        self.target_node = target_node

        if gnn_branch is None:
            # Model 1: classify on the transformer embedding alone.
            self.mode = "transformer"
            self.fuse = None
            self.classifier = nn.Linear(embed_dim, 1)
        else:
            # Models 2-4: concat the two 128-d views, then an MLP head.
            self.mode = _MODE_BY_CLASS.get(
                type(gnn_branch).__name__, f"transformer+{type(gnn_branch).__name__}"
            )
            self.fuse = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(
        self,
        X_seq: torch.Tensor,
        mask: torch.Tensor,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
        node_idx: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Return [B, 1] fraud probability (or logits if return_logits=True)."""
        seq_emb = self.transformer(X_seq, mask)  # [B, embed_dim]

        if self.gnn is None:
            logits = self.classifier(seq_emb)  # [B, 1]
        else:
            if x_dict is None or edge_index_dict is None or node_idx is None:
                raise ValueError(
                    f"mode {self.mode!r} needs x_dict, edge_index_dict and node_idx "
                    "to gather the matching graph embeddings."
                )
            graph_emb_all = self.gnn(x_dict, edge_index_dict)  # [num_txn_nodes, embed_dim]
            if node_idx.dtype not in (torch.long, torch.int64):
                node_idx = node_idx.long()
            graph_emb = graph_emb_all[node_idx]  # [B, embed_dim]
            if graph_emb.shape[0] != seq_emb.shape[0]:
                raise ValueError(
                    f"batch mismatch: seq {seq_emb.shape[0]} vs graph {graph_emb.shape[0]} "
                    "— node_idx must align with the sequence batch."
                )
            fused = self.fuse(torch.cat([seq_emb, graph_emb], dim=1))  # [B, embed_dim]
            logits = self.classifier(fused)  # [B, 1]

        if return_logits:
            return logits
        return torch.sigmoid(logits)


def _smoke_test() -> None:
    """CPU smoke test: all four configs return a [B, 1] probability on a dummy batch."""
    config.set_seed(config.SEED)
    device = torch.device("cpu")

    from transformer_model import SequenceTransformer
    from gnn_models import (
        GATBranch,
        GraphSAGEBranch,
        STHGNNBranch,
        _in_channels_dict,
        _load_graph,
        temporal_features_from_clean,
    )

    f = 32  # sequence feature dim (matches sequences.pt)
    b = 8

    # Real graph so the GNN branches have something structurally valid to run on.
    data = _load_graph().to(device)
    n_txn = data[TARGET_NODE].num_nodes
    inc = _in_channels_dict(data)
    node_idx = torch.arange(b, device=device)  # first b transaction nodes as the batch

    # Dummy sequence batch aligned (positionally) with node_idx.
    X = torch.randn(b, config.MAX_SEQ_LEN, f, device=device)
    lengths = torch.randint(1, config.MAX_SEQ_LEN + 1, (b,))
    mask = torch.zeros(b, config.MAX_SEQ_LEN, dtype=torch.bool)
    for i, n_real in enumerate(lengths):
        mask[i, config.MAX_SEQ_LEN - int(n_real):] = True
    X[~mask] = 0.0

    def make_transformer():
        return SequenceTransformer(in_features=f).to(device)

    temporal = temporal_features_from_clean(data).to(device)
    configs = {
        "transformer (Model 1)": None,
        "transformer+sage (Model 2)": GraphSAGEBranch(data.metadata(), inc).to(device),
        "transformer+gat (Model 3)": GATBranch(data.metadata(), inc).to(device),
        "transformer+sthgnn (Model 4)": STHGNNBranch(data.metadata(), inc, temporal).to(device),
    }

    for label, gnn in configs.items():
        model = FusionClassifier(make_transformer(), gnn).to(device).eval()
        with torch.no_grad():
            if gnn is None:
                prob = model(X, mask)
                logits = model(X, mask, return_logits=True)
            else:
                prob = model(X, mask, data.x_dict, data.edge_index_dict, node_idx)
                logits = model(
                    X, mask, data.x_dict, data.edge_index_dict, node_idx,
                    return_logits=True,
                )
        assert prob.shape == (b, 1), f"{label}: bad prob shape {tuple(prob.shape)}"
        assert torch.isfinite(prob).all(), f"{label}: non-finite prob"
        assert ((prob >= 0) & (prob <= 1)).all(), f"{label}: prob out of [0,1]"
        assert logits.shape == (b, 1), f"{label}: bad logits shape"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[fusion] {label:<32} mode={model.mode:<20} "
              f"params={n_params:>10,} -> prob {tuple(prob.shape)} OK")

    print("[fusion] smoke test passed (all four configs produce [B,1] probabilities).")


if __name__ == "__main__":
    _smoke_test()
