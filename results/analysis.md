# Phase 5 — Model Comparison Analysis (Task 5.4)

## Selection criterion
Best model selected by **PR-AUC** on the held-out test set (primary metric for this
~3.5%-fraud, heavily imbalanced problem), with **F1** as the tie-break. All four models
were evaluated on the *identical* time-aware test split (last 15% of transactions).

## Ranking (by test PR-AUC)

| Rank | Model | Mode | PR-AUC | ROC-AUC | F1 (thr=0.5) | Best-F1 | ms/1k |
|------|-------|------|--------|---------|--------------|---------|-------|
| 1 | **m3_gat** | transformer+GAT | **0.4211** | 0.8348 | 0.3726 | 0.4496 | 320.8 |
| 2 | m2_sage | transformer+GraphSAGE | 0.4198 | 0.8341 | 0.3538 | 0.4409 | 293.4 |
| 3 | m1_transformer | transformer-only | 0.4142 | 0.8270 | 0.3205 | 0.4553 | 216.4 |
| 4 | m4_sthgnn | transformer+ST-HGNN | 0.4093 | 0.8210 | 0.3341 | 0.4547 | 303.8 |

**Winner: `m3_gat` (Transformer + GAT).** Checkpoint copied to `models/best_model.pt`.

## Analysis

The Transformer+GAT fusion model won on PR-AUC (0.4211), but the margin is slim — just
+0.0013 over GraphSAGE (0.4198) and +0.0069 over the transformer-only baseline (0.4142).
Both graph-augmented models (GAT and SAGE) beat the sequence-only Model 1, which is the
result the comparison study was designed to surface: relational structure — sharing a
card, device, region, or merchant across transactions — carries fraud signal that a
per-client temporal sequence alone cannot see, since organized fraud rings reuse the same
entities across many "clients." GAT edged out SAGE most clearly on F1 at the default 0.5
threshold (0.373 vs 0.354) and on recall (0.242 vs 0.228), suggesting its attention-weighted
neighbor aggregation picks up a few more true fraud cases than SAGE's mean aggregation.
Notably, the heaviest model — ST-HGNN — finished *last* on PR-AUC (0.4093), below even the
transformer baseline; its extra heterogeneous/temporal machinery did not pay off here and
may have been mildly under-trained or over-regularized for this data. The transformer-only
model is the fastest (216 ms/1k vs ~320 for GAT), so if inference latency mattered more than
a ~0.7 PR-AUC point, Model 1 would be a defensible choice.

## ⚠️ Caveat — GAT_HEADS reduced from 4 → 1 (hardware constraint)

The GAT (Model 3) and ST-HGNN (Model 4) checkpoints were trained on the **free Colab T4 GPU**,
which ran out of memory with 4-head attention over the full ~590k-node transaction graph.
`GAT_HEADS` was therefore reduced from the project default of 4 down to **1** for these two
models (see the comment in `config.py`). This means the winning GAT result and the ST-HGNN
result were obtained with single-head attention rather than the intended 4-head configuration.
The very narrow PR-AUC gaps among the top three models should be read with this in mind: the
GAT/ST-HGNN numbers are likely a *lower bound* on what these architectures could achieve with
full 4-head attention on a larger GPU. The qualitative conclusion — graph relations help over
sequence-only modeling — is robust to this caveat, but the precise ordering of GAT vs SAGE,
given a 0.0013 margin, should not be over-interpreted.
