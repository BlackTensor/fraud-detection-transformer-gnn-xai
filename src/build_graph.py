"""
build_graph.py — clean.csv -> PyG HeteroData graph (Task 2.2).

What it does (the "graph view" of the same transactions):
    Builds a heterogeneous transaction graph where each transaction is linked to
    the real-world entities it touches. This lets the GNN branches (Phase 3) pick
    up relational fraud patterns the sequence model can't see — e.g. many
    transactions fanning out from one card/device, or sharing an address.

    Node types:
        transaction  one node per row of clean.csv
        card         = client_id  (the pseudo-user key: card1_card2_card3_card5_addr1)
        device       = DeviceInfo + DeviceType  (combined)
        merchant     = ProductCD
        addr         = addr1  (region)

    Edge types (forward; reverse edges added by ToUndirected so messages flow
    BOTH ways):
        (transaction, made_by,     card)
        (transaction, on_device,   device)
        (transaction, at_merchant, merchant)
        (transaction, in_region,   addr)

    Node features:
        transaction : the 25 standardized NUMERIC features only
                      (build_sequences.NUMERIC_FEATURES), scaled with the shared
                      train-fit scaler.pkl. These are the same NUMERIC columns and
                      identical numbers the sequence view uses — no leakage (scaler
                      fit on TRAIN only). NOTE: the graph deliberately does NOT
                      include the 7 frequency-encoded CATEGORICAL features that the
                      sequence view appends, so the transaction node vector is 25-d,
                      whereas the sequence per-timestep vector is 32-d (25 numeric +
                      7 categorical). The categorical signal reaches the model only
                      through the sequence branch; the graph branch relies on the
                      entity NODES (card/device/merchant/addr) for that information.
        entity nodes: a 1-d degree/count feature = log1p(#transactions touching
                      the node). Models that prefer learned embeddings can key an
                      nn.Embedding by node index (num_nodes is available per type).

    Labels & masks are attached to the TRANSACTION nodes only:
        y, train_mask, val_mask, test_mask  — derived from splits.npz so the graph
        view trains/evaluates on exactly the same rows as the sequence view.
        transaction_id is stored for the Task 2.3 alignment check.

Inputs:
    data/processed/clean.csv
    data/processed/splits.npz
    models/scaler.pkl

Outputs:
    data/processed/graph.pt  (a PyG HeteroData object)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
# Reuse the exact numeric feature list + scaling helper from the sequence view so
# the two views are perfectly consistent.
from build_sequences import NUMERIC_FEATURES, _scaled_numeric  # noqa: E402

# Columns we need beyond the numeric features: the entity keys + label/id/time.
# HIDDEN COUPLING: like build_sequences' curated lists, every column named here
# (and in the reused NUMERIC_FEATURES) must survive preprocess.clean()'s
# >90%-missing drop (preprocess.MISSING_DROP_THRESHOLD = 0.90). DeviceInfo/
# DeviceType are ~76% missing on IEEE-CIS — kept under the threshold, but lower it
# (or change datasets) and _load_frame() will raise on the missing column. Keep
# this list and that threshold in sync.
ENTITY_SOURCE_COLUMNS = ["client_id", "DeviceInfo", "DeviceType", "ProductCD", "addr1"]
META_COLUMNS = ["isFraud", "TransactionID", "TransactionDT"]

# (forward relation name, node type) for each transaction->entity edge.
RELATIONS = [
    ("made_by", "card", "client_id"),
    ("on_device", "device", None),       # device key built from two columns below
    ("at_merchant", "merchant", "ProductCD"),
    ("in_region", "addr", "addr1"),
]


def _load_frame() -> pd.DataFrame:
    """Load only the columns the graph needs (keeps memory modest)."""
    if not config.CLEAN_CSV.exists():
        raise ValueError(
            f"{config.CLEAN_CSV} not found — run preprocess (Tasks 1.1-1.3) first."
        )
    usecols = list(dict.fromkeys(NUMERIC_FEATURES + ENTITY_SOURCE_COLUMNS + META_COLUMNS))
    df = pd.read_csv(config.CLEAN_CSV, usecols=usecols)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"clean.csv is missing expected columns: {missing}")
    return df


def _entity_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the string key Series that defines each entity node type."""
    # device collapses DeviceInfo + DeviceType into one entity (both imputed to
    # "unknown" for the many no-identity rows -> a single 'unknown__unknown' node).
    device_key = df["DeviceInfo"].astype(str) + "__" + df["DeviceType"].astype(str)
    return {
        "card": df["client_id"].astype(str),
        "device": device_key,
        # addr1 is numeric (median-imputed for missing) -> str so equal regions
        # share a node; imputed-missing rows collapse onto the median region node.
        "merchant": df["ProductCD"].astype(str),
        "addr": df["addr1"].astype(str),
    }


def build_graph(verbose: bool = True) -> HeteroData:
    """Run Task 2.2: build and save the HeteroData transaction graph."""
    splits = np.load(config.SPLITS_NPZ)
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]

    if verbose:
        print(f"[graph] Loading {config.CLEAN_CSV.name} (selected columns only) ...")
    df = _load_frame()
    n = len(df)

    # --- Transaction node features (shared, scaled numeric vector) --------- #
    x_txn = torch.from_numpy(_scaled_numeric(df))            # float32 [N, F]
    f = x_txn.shape[1]

    data = HeteroData()
    data["transaction"].x = x_txn
    data["transaction"].y = torch.from_numpy(df["isFraud"].to_numpy(np.int64))
    data["transaction"].transaction_id = torch.from_numpy(
        df["TransactionID"].to_numpy(np.int64)
    )

    # Boolean split masks on transaction nodes (align exactly with splits.npz).
    for name, idx in (("train_mask", train_idx), ("val_mask", val_idx),
                      ("test_mask", test_idx)):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        data["transaction"][name] = m

    # --- Entity nodes + transaction->entity edges ------------------------- #
    keys = _entity_keys(df)
    txn_ids = np.arange(n, dtype=np.int64)
    for rel, ntype, _src in RELATIONS:
        codes, uniques = pd.factorize(keys[ntype], sort=False)   # codes in [0, n_type)
        n_type = len(uniques)
        # Entity node feature = log1p(degree) where degree = #transactions on it.
        degree = np.bincount(codes, minlength=n_type).astype(np.float32)
        data[ntype].x = torch.from_numpy(np.log1p(degree)).unsqueeze(1)  # [n_type, 1]
        data[ntype].num_nodes = n_type
        # Forward edge transaction -> entity (reverse added by ToUndirected).
        edge_index = torch.from_numpy(np.vstack([txn_ids, codes.astype(np.int64)]))
        data["transaction", rel, ntype].edge_index = edge_index

    # Add reverse edges so messages can flow entity -> transaction as well.
    data = T.ToUndirected()(data)

    # --- Acceptance asserts (no silent failures) -------------------------- #
    assert data["transaction"].num_nodes == n, "transaction node count != #rows"
    tr = data["transaction"]
    assert int(tr.train_mask.sum()) == len(train_idx), "train_mask size mismatch"
    assert int(tr.val_mask.sum()) == len(val_idx), "val_mask size mismatch"
    assert int(tr.test_mask.sum()) == len(test_idx), "test_mask size mismatch"
    # Masks must be disjoint and cover every transaction (same partition as splits).
    assert torch.equal(tr.train_mask & tr.val_mask, torch.zeros(n, dtype=torch.bool))
    assert torch.equal(tr.train_mask & tr.test_mask, torch.zeros(n, dtype=torch.bool))
    assert torch.equal(tr.val_mask & tr.test_mask, torch.zeros(n, dtype=torch.bool))
    assert int((tr.train_mask | tr.val_mask | tr.test_mask).sum()) == n, "masks miss rows"
    assert np.array_equal(np.where(tr.train_mask.numpy())[0], np.sort(train_idx)), \
        "train_mask positions != splits.npz train_idx"

    # --- Save + report ---------------------------------------------------- #
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, config.GRAPH_PT)

    if verbose:
        size_mb = config.GRAPH_PT.stat().st_size / 1024 / 1024
        print(f"[graph] transaction features F = {f} (same as sequence view)")
        print("[graph] Node counts:")
        for ntype in data.node_types:
            print(f"          {ntype:<12} {data[ntype].num_nodes:>10,}")
        print("[graph] Edge counts (forward + reverse):")
        for et in data.edge_types:
            src, rel, dst = et
            ne = data[et].edge_index.shape[1]
            print(f"          ({src:>11}, {rel:<14}, {dst:<11}) {ne:>10,}")
        print(f"[graph] Labels/masks on transaction nodes | "
              f"train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_idx):,}")
        print(f"[graph] Fraud rate (all): {tr.y.float().mean():.4%}")
        print(f"[graph] Saved -> {config.GRAPH_PT} ({size_mb:.1f} MB)")

    return data


if __name__ == "__main__":
    config.ensure_dirs()
    config.set_seed(config.SEED)
    build_graph(verbose=True)
