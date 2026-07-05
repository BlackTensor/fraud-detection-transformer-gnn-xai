"""
check_alignment.py — sequence/graph view alignment sanity check (Task 2.3).

What it does:
    Both views are built in the natural clean.csv row order (0..N-1), so row i of
    sequences.pt MUST describe the same transaction as transaction-node i of
    graph.pt. This script proves it by asserting, element-for-element, that:
      * the two views have the same number of transactions,
      * their labels (isFraud) are identical,
      * their TransactionIDs are identical (so both map to the same real row).
    It then prints a 5-row sample with TransactionID and isFraud from each view
    side by side.

Inputs:
    data/processed/sequences.pt   (X_seq, mask, y, transaction_id, ...)
    data/processed/graph.pt       (HeteroData; transaction.y, transaction.transaction_id)

Outputs:
    none on disk — asserts + a printed sample table. Raises ValueError on any
    mismatch (no silent failures).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def check_alignment(verbose: bool = True) -> None:
    """Run Task 2.3: assert the sequence and graph views are row-aligned."""
    for p in (config.SEQUENCES_PT, config.GRAPH_PT):
        if not p.exists():
            raise ValueError(f"{p} not found — build it first (Task 2.1 / 2.2).")

    seq = torch.load(config.SEQUENCES_PT, weights_only=False)
    graph = torch.load(config.GRAPH_PT, weights_only=False)
    txn = graph["transaction"]

    # Pull the comparable vectors from each view.
    seq_y = seq["y"].to(torch.int64)
    graph_y = txn.y.to(torch.int64)
    seq_id = seq["transaction_id"].to(torch.int64)
    graph_id = txn.transaction_id.to(torch.int64)

    # --- Count must match -------------------------------------------------- #
    n_seq, n_graph = seq_y.numel(), graph_y.numel()
    if n_seq != n_graph:
        raise ValueError(
            f"Row-count mismatch: sequences.pt has {n_seq:,} rows but "
            f"graph.pt has {n_graph:,} transaction nodes."
        )

    # --- TransactionIDs identical in the same order ------------------------ #
    if not torch.equal(seq_id, graph_id):
        n_bad = int((seq_id != graph_id).sum())
        first = int((seq_id != graph_id).nonzero()[0, 0])
        raise ValueError(
            f"TransactionID mismatch between views: {n_bad:,} rows differ; "
            f"first at row {first} "
            f"(seq={int(seq_id[first])} vs graph={int(graph_id[first])})."
        )

    # --- Labels identical in the same order -------------------------------- #
    if not torch.equal(seq_y, graph_y):
        n_bad = int((seq_y != graph_y).sum())
        first = int((seq_y != graph_y).nonzero()[0, 0])
        raise ValueError(
            f"isFraud label mismatch between views: {n_bad:,} rows differ; "
            f"first at row {first} "
            f"(seq={int(seq_y[first])} vs graph={int(graph_y[first])})."
        )

    if verbose:
        print(f"[align] OK - {n_seq:,} transactions align across both views.")
        print(f"[align] TransactionID: identical in order  |  isFraud: identical in order")
        print("\n[align] 5-row sample (same row order in both views):")
        print(f"  {'row':>4}  {'TransactionID':>14}  {'isFraud(seq)':>12}  {'isFraud(graph)':>14}")
        for i in range(5):
            print(f"  {i:>4}  {int(seq_id[i]):>14}  {int(seq_y[i]):>12}  {int(graph_y[i]):>14}")


if __name__ == "__main__":
    config.set_seed(config.SEED)
    check_alignment(verbose=True)
