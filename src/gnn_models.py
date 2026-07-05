"""
gnn_models.py — graph-view GNN branches, each producing a 128-d embedding.

This file holds the three GNN branches used in the comparison study. Each one
consumes the PyG HeteroData graph built in build_graph.py and returns ONE
EMBED_DIM (128-d) embedding per TRANSACTION node, so the fusion classifier
(Task 3.5) can wire any of them in next to the SequenceTransformer.

    GraphSAGEBranch  (Task 3.2)  — implemented here.
    GATBranch        (Task 3.3)  — implemented here.
    STHGNNBranch     (Task 3.4)  — implemented here (real ST-HGNN, not the fallback).

Output contract (shared by every branch):
    forward(x_dict, edge_index_dict) -> [num_transaction_nodes, EMBED_DIM]

Mask note: the branch embeds ALL transaction nodes in one pass (full-graph,
transductive message passing). Train/val/test selection is done downstream by
indexing these embeddings with the transaction masks stored on the graph.

Running this file as a script executes a CPU smoke test that loads
data/processed/graph.pt and asserts the GraphSAGE output is
[num_transaction_nodes, EMBED_DIM].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TARGET_NODE = "transaction"


class GraphSAGEBranch(nn.Module):
    """Heterogeneous GraphSAGE encoder -> [num_transaction_nodes, EMBED_DIM].

    Design:
      1. Per-node-type input projection (Linear) maps each type's raw features
         (transaction = F-dim scaled numerics, entities = 1-d log-degree) into a
         common GNN_HIDDEN space, so the relational layers see uniform widths.
      2. GNN_LAYERS HeteroConv layers, each wrapping one SAGEConv per edge type
         (forward AND the rev_* edges from ToUndirected, so messages flow both
         ways). Per layer: SAGE message passing -> ReLU -> Dropout.
      3. A final Linear maps the target node's GNN_HIDDEN vector to EMBED_DIM.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels_dict: dict[str, int],
        hidden: int = config.GNN_HIDDEN,
        out_dim: int = config.EMBED_DIM,
        num_layers: int = config.GNN_LAYERS,
        dropout: float = config.DROPOUT,
        target_node: str = TARGET_NODE,
    ) -> None:
        """
        Args:
            metadata:         (node_types, edge_types) — graph.metadata().
            in_channels_dict: raw input feature dim per node type.
            hidden:           hidden width of the SAGE layers.
            out_dim:          output embedding dim (EMBED_DIM, for fusion).
            num_layers:       number of HeteroConv message-passing layers.
            dropout:          dropout after each conv layer.
            target_node:      node type whose embeddings are returned ('transaction').
        """
        super().__init__()
        node_types, edge_types = metadata
        if target_node not in node_types:
            raise ValueError(
                f"target_node {target_node!r} not in graph node types {node_types}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.target_node = target_node
        self.num_layers = num_layers
        self.dropout = dropout

        # 1) Project every node type's raw features into the shared hidden space.
        self.input_proj = nn.ModuleDict(
            {
                ntype: nn.Linear(in_channels_dict[ntype], hidden)
                for ntype in node_types
            }
        )

        # 2) Stacked heterogeneous SAGE layers (one SAGEConv per edge type).
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    etype: SAGEConv(hidden, hidden)
                    for etype in edge_types
                },
                aggr="sum",  # sum messages across the different edge types
            )
            self.convs.append(conv)

        # 3) Read-out head: hidden -> EMBED_DIM for the target node only.
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Map node features + edges -> [num_transaction_nodes, EMBED_DIM]."""
        if self.target_node not in x_dict:
            raise ValueError(
                f"x_dict missing target node {self.target_node!r}; "
                f"got {list(x_dict.keys())}."
            )

        # Input projection per node type (+ a nonlinearity so it's an encoder).
        h_dict = {
            ntype: F.relu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        # Relational message passing.
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                ntype: F.dropout(F.relu(h), p=self.dropout, training=self.training)
                for ntype, h in h_dict.items()
            }

        return self.out_proj(h_dict[self.target_node])  # [num_txn_nodes, EMBED_DIM]


class GATBranch(nn.Module):
    """Heterogeneous Graph Attention encoder -> [num_transaction_nodes, EMBED_DIM].

    Same output contract as GraphSAGEBranch, but uses attention-weighted message
    passing so a transaction can learn WHICH neighbouring entities matter most.

    Design:
      1. Per-node-type input projection (Linear + ReLU) -> shared GNN_HIDDEN space.
      2. GNN_LAYERS HeteroConv layers, each wrapping one multi-head GATConv per
         edge type (incl. the rev_* edges). Heads are CONCATenated, so a layer
         outputs GNN_HIDDEN * GAT_HEADS per node; a per-type Linear then projects
         that back to GNN_HIDDEN, followed by ReLU + Dropout.
         add_self_loops=False — the graph is bipartite per edge type (a node type
         is only ever src OR dst of a given relation), where GAT self-loops are
         ill-defined; bidirectional flow already comes from the rev_* edges.
      3. A final Linear maps the target node's GNN_HIDDEN vector to EMBED_DIM.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels_dict: dict[str, int],
        hidden: int = config.GNN_HIDDEN,
        out_dim: int = config.EMBED_DIM,
        num_layers: int = config.GNN_LAYERS,
        heads: int = config.GAT_HEADS,
        dropout: float = config.DROPOUT,
        target_node: str = TARGET_NODE,
    ) -> None:
        """
        Args:
            metadata:         (node_types, edge_types) — graph.metadata().
            in_channels_dict: raw input feature dim per node type.
            hidden:           hidden width per node (after head concat + projection).
            out_dim:          output embedding dim (EMBED_DIM, for fusion).
            num_layers:       number of HeteroConv (GAT) message-passing layers.
            heads:            attention heads per GATConv (concatenated).
            dropout:          dropout on attention and after each layer.
            target_node:      node type whose embeddings are returned ('transaction').
        """
        super().__init__()
        node_types, edge_types = metadata
        if target_node not in node_types:
            raise ValueError(
                f"target_node {target_node!r} not in graph node types {node_types}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}.")

        self.target_node = target_node
        self.num_layers = num_layers
        self.dropout = dropout

        # 1) Project every node type's raw features into the shared hidden space.
        self.input_proj = nn.ModuleDict(
            {
                ntype: nn.Linear(in_channels_dict[ntype], hidden)
                for ntype in node_types
            }
        )

        # 2) Stacked heterogeneous GAT layers, each followed by a per-node-type
        #    projection from (hidden * heads) concatenated heads back to hidden.
        self.convs = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    etype: GATConv(
                        (-1, -1),       # lazy init -> handles bipartite edge types
                        hidden,
                        heads=heads,
                        concat=True,    # concat heads -> hidden * heads
                        dropout=dropout,
                        add_self_loops=False,
                    )
                    for etype in edge_types
                },
                aggr="sum",  # sum attention outputs across the different edge types
            )
            self.convs.append(conv)
            self.post.append(
                nn.ModuleDict(
                    {ntype: nn.Linear(hidden * heads, hidden) for ntype in node_types}
                )
            )

        # 3) Read-out head: hidden -> EMBED_DIM for the target node only.
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Map node features + edges -> [num_transaction_nodes, EMBED_DIM]."""
        if self.target_node not in x_dict:
            raise ValueError(
                f"x_dict missing target node {self.target_node!r}; "
                f"got {list(x_dict.keys())}."
            )

        h_dict = {
            ntype: F.relu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        for conv, post in zip(self.convs, self.post):
            h_dict = conv(h_dict, edge_index_dict)  # [n, hidden * heads] per type
            # Project concatenated heads back to hidden, then ReLU + dropout.
            h_dict = {
                ntype: F.dropout(
                    F.relu(post[ntype](h)), p=self.dropout, training=self.training
                )
                for ntype, h in h_dict.items()
            }

        return self.out_proj(h_dict[self.target_node])  # [num_txn_nodes, EMBED_DIM]


class STHGNNBranch(nn.Module):
    """Spatio-Temporal Heterogeneous GNN -> [num_transaction_nodes, EMBED_DIM].

    The "advanced" branch of the comparison study. Two ideas distinguish it from
    the plain SAGE / GAT branches:

      * Heterogeneous message passing that MIXES conv types per relation: edges
        flowing INTO a transaction (the rev_* edges, dst == 'transaction') use a
        multi-head GATConv so a transaction can attend over its card/device/
        merchant/region context; edges flowing into ENTITY nodes use SAGEConv.
        Both convs emit GNN_HIDDEN (GAT with concat=False averages heads), so the
        HeteroConv 'sum' aggregation across edge types stays dimension-safe for
        any graph topology.

      * A light TEMPORAL signal injected as extra transaction node features:
        hour, day (pseudo time-of-day / day-of-week) and a global TransactionDT
        rank in [0, 1]. Built once with `temporal_features_from_clean` and held
        as a buffer, so the shared forward(x_dict, edge_index_dict) contract is
        unchanged — the branch augments x_dict['transaction'] internally.

    (GAT was the designated fallback if this proved too heavy; it trains
    fine here, so the real ST-HGNN is used and the 4-model comparison stands.)
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels_dict: dict[str, int],
        temporal_features: torch.Tensor,
        hidden: int = config.GNN_HIDDEN,
        out_dim: int = config.EMBED_DIM,
        num_layers: int = config.GNN_LAYERS,
        heads: int = config.GAT_HEADS,
        dropout: float = config.DROPOUT,
        target_node: str = TARGET_NODE,
    ) -> None:
        """
        Args:
            metadata:           (node_types, edge_types) — graph.metadata().
            in_channels_dict:   raw input feature dim per node type.
            temporal_features:  [num_transaction_nodes, T] temporal block appended
                                to the transaction features (row-aligned with the
                                transaction nodes). See temporal_features_from_clean.
            hidden:             hidden width of the conv layers.
            out_dim:            output embedding dim (EMBED_DIM, for fusion).
            num_layers:         number of HeteroConv message-passing layers.
            heads:              attention heads for the GAT (transaction-incoming) edges.
            dropout:            dropout on attention and after each layer.
            target_node:        node type whose embeddings are returned ('transaction').
        """
        super().__init__()
        node_types, edge_types = metadata
        if target_node not in node_types:
            raise ValueError(
                f"target_node {target_node!r} not in graph node types {node_types}."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if temporal_features.dim() != 2:
            raise ValueError(
                f"temporal_features must be 2-D [N, T], got {tuple(temporal_features.shape)}."
            )

        self.target_node = target_node
        self.num_layers = num_layers
        self.dropout = dropout
        # Held as a (non-trained) buffer so it moves with .to(device) and is saved
        # with the checkpoint, keeping the forward signature identical to the others.
        self.register_buffer("temporal_features", temporal_features.float())

        # The transaction encoder sees its raw features PLUS the temporal block.
        in_dims = dict(in_channels_dict)
        in_dims[target_node] = in_channels_dict[target_node] + temporal_features.shape[1]

        # 1) Per-node-type input projection into the shared hidden space.
        self.input_proj = nn.ModuleDict(
            {ntype: nn.Linear(in_dims[ntype], hidden) for ntype in node_types}
        )

        # 2) Heterogeneous layers: GAT on edges into the target node, SAGE elsewhere.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs_by_edge = {}
            for etype in edge_types:
                _src, _rel, dst = etype
                if dst == target_node:
                    convs_by_edge[etype] = GATConv(
                        (-1, -1),
                        hidden,
                        heads=heads,
                        concat=False,         # average heads -> stays GNN_HIDDEN-wide
                        dropout=dropout,
                        add_self_loops=False,  # bipartite per-relation edges
                    )
                else:
                    convs_by_edge[etype] = SAGEConv(hidden, hidden)
            self.convs.append(HeteroConv(convs_by_edge, aggr="sum"))

        # 3) Read-out head: hidden -> EMBED_DIM for the target node only.
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Map node features + edges -> [num_transaction_nodes, EMBED_DIM]."""
        if self.target_node not in x_dict:
            raise ValueError(
                f"x_dict missing target node {self.target_node!r}; "
                f"got {list(x_dict.keys())}."
            )
        x_txn = x_dict[self.target_node]
        if x_txn.shape[0] != self.temporal_features.shape[0]:
            raise ValueError(
                f"transaction node count {x_txn.shape[0]} != temporal_features rows "
                f"{self.temporal_features.shape[0]} — graph/temporal misalignment."
            )

        # Inject the temporal signal into the transaction features.
        x_dict = dict(x_dict)
        x_dict[self.target_node] = torch.cat([x_txn, self.temporal_features], dim=1)

        h_dict = {
            ntype: F.relu(self.input_proj[ntype](x)) for ntype, x in x_dict.items()
        }

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                ntype: F.dropout(F.relu(h), p=self.dropout, training=self.training)
                for ntype, h in h_dict.items()
            }

        return self.out_proj(h_dict[self.target_node])  # [num_txn_nodes, EMBED_DIM]


def temporal_features_from_clean(data) -> torch.Tensor:
    """Build the ST-HGNN temporal block, row-aligned with the transaction nodes.

    Columns: [hour/24, day/7, TransactionDT-rank in [0,1]]. Read from clean.csv in
    natural row order (build_graph indexed transaction nodes in that same order),
    and verified against the graph's stored transaction_id so they can never drift.

    Returns: float32 tensor [num_transaction_nodes, 3].
    """
    import numpy as np
    import pandas as pd

    if not config.CLEAN_CSV.exists():
        raise ValueError(
            f"{config.CLEAN_CSV} not found — run preprocess (Tasks 1.1-1.3) first."
        )
    df = pd.read_csv(config.CLEAN_CSV, usecols=["hour", "day", "TransactionDT", "TransactionID"])

    n_txn = data[TARGET_NODE].num_nodes
    if len(df) != n_txn:
        raise ValueError(
            f"clean.csv rows {len(df)} != transaction nodes {n_txn} — view misalignment."
        )
    # Hard alignment check against the graph (same order, no silent drift).
    graph_ids = data[TARGET_NODE].transaction_id.cpu().numpy()
    if not np.array_equal(df["TransactionID"].to_numpy(np.int64), graph_ids):
        raise ValueError("clean.csv TransactionID order != graph transaction_id order.")

    hour = (df["hour"].to_numpy(np.float32)) / 24.0
    day = (df["day"].to_numpy(np.float32)) / 7.0
    # Global temporal position: rank of TransactionDT scaled to [0, 1].
    dt = df["TransactionDT"].to_numpy()
    order = np.argsort(dt, kind="stable")
    rank = np.empty(n_txn, dtype=np.float32)
    rank[order] = np.arange(n_txn, dtype=np.float32)
    dt_rank = rank / max(n_txn - 1, 1)

    return torch.from_numpy(np.stack([hour, day, dt_rank], axis=1)).float()  # [N, 3]


def _in_channels_dict(data) -> dict[str, int]:
    """Per-node-type input feature width from a HeteroData object."""
    return {ntype: data[ntype].x.shape[1] for ntype in data.node_types}


def _load_graph():
    """Load the saved HeteroData graph (weights_only=False — it's a full object)."""
    if not config.GRAPH_PT.exists():
        raise ValueError(
            f"{config.GRAPH_PT} not found — run build_graph.py (Task 2.2) first."
        )
    try:
        return torch.load(config.GRAPH_PT, weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(config.GRAPH_PT)


def _smoke_one(tag: str, model: nn.Module, data, n_txn: int) -> None:
    """Forward one branch on the full graph and assert the output contract."""
    model = model.eval()
    with torch.no_grad():
        emb = model(data.x_dict, data.edge_index_dict)

    assert emb.shape == (n_txn, config.EMBED_DIM), f"unexpected shape {tuple(emb.shape)}"
    assert torch.isfinite(emb).all(), "non-finite values in embedding"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params: {n_params:,}")
    print(f"[{tag}] graph ({n_txn:,} txn nodes) -> embedding {tuple(emb.shape)} OK")


def _smoke_test() -> None:
    """CPU smoke test: each branch forward on graph.pt -> [num_txn_nodes, EMBED_DIM]."""
    config.set_seed(config.SEED)
    device = torch.device("cpu")

    data = _load_graph().to(device)
    n_txn = data[TARGET_NODE].num_nodes
    in_channels = _in_channels_dict(data)
    print(f"[gnn] node types: {data.node_types}")
    print(f"[gnn] in_channels: {in_channels}")
    print(f"[gnn] hidden={config.GNN_HIDDEN}, layers={config.GNN_LAYERS}, "
          f"heads={config.GAT_HEADS}, out={config.EMBED_DIM}, dropout={config.DROPOUT}")

    sage = GraphSAGEBranch(
        metadata=data.metadata(), in_channels_dict=in_channels
    ).to(device)
    _smoke_one("graphsage", sage, data, n_txn)

    gat = GATBranch(
        metadata=data.metadata(), in_channels_dict=in_channels
    ).to(device)
    _smoke_one("gat", gat, data, n_txn)

    temporal = temporal_features_from_clean(data).to(device)
    print(f"[sthgnn] temporal block: {tuple(temporal.shape)} (hour, day, DT-rank)")
    sthgnn = STHGNNBranch(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        temporal_features=temporal,
    ).to(device)
    _smoke_one("sthgnn", sthgnn, data, n_txn)

    print("[gnn] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
