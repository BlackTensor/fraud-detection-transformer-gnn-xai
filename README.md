# Sequence-Aware Transformer-Based Fraud Detection with LLM-Driven Explainability

A ₹0/$0 fraud-detection project that compares a **sequence view** (per-client
transaction history via a Transformer encoder) against a **graph view**
(shared card/device/merchant/region relations via three GNN variants), fuses
the two, and explains individual predictions with **SHAP + a local LLM**
(Ollama) — no paid APIs, no paid compute.

---

## What this project does

1. **Sequence branch** — a `TransformerEncoder` reads each transaction plus
   its client's previous transactions (causal, no future leakage) and
   produces a 128-d embedding.
2. **Graph branch** — a heterogeneous transaction graph (nodes: transaction,
   card, device, merchant, addr) is encoded with one of three GNNs —
   **GraphSAGE**, **GAT**, or **ST-HGNN** (heterogeneous + temporal) — also
   producing a 128-d embedding.
3. **Fusion** — the two embeddings are concatenated and passed through a
   small MLP classifier to output a fraud probability. Four models are
   trained: Transformer-only, Transformer+SAGE, Transformer+GAT,
   Transformer+ST-HGNN — this is the project's **comparison study**.
4. **Explainability** — SHAP explains the winning model's prediction at the
   input-feature level; a local Ollama LLM (`llama3.2:3b`) turns the SHAP
   output into a plain-English paragraph, grounded only in the given
   evidence.
5. **Deployment** — a Streamlit app takes a user-described transaction,
   scores it with the best model, and shows the SHAP factors + LLM
   explanation.

## Architecture

```
                 ┌─────────────────────┐        ┌──────────────────────────┐
                 │   Sequence view     │        │       Graph view         │
                 │  (per-client, up    │        │  HeteroData: transaction │
                 │  to 20 past txns)   │        │  <-> card/device/        │
                 │                     │        │  merchant/addr           │
                 └─────────┬───────────┘        └────────────┬─────────────┘
                           │                                  │
                 TransformerEncoder                 GraphSAGE / GAT / ST-HGNN
                  (2 layers, 4 heads)                   (2 layers, 128 hidden)
                           │                                  │
                       [B, 128]                            [B, 128]
                           └───────────────┬──────────────────┘
                                           concat [B, 256]
                                Linear(256→128) → ReLU → Dropout
                                       Linear(128→1) → Sigmoid
                                                │
                                        fraud probability
                                                │
                              ┌─────────────────┴─────────────────┐
                              │   SHAP (KernelExplainer, 32 feats)  │
                              │   → top-5 signed factors            │
                              │   → local Ollama (llama3.2:3b)      │
                              │   → grounded NL explanation          │
                              └──────────────────────────────────────┘
```

## Results — the comparison study

Evaluated on the identical, time-aware held-out test split (88,581
transactions, last 15% by `TransactionDT`). Primary metric is **PR-AUC**
(the dataset is ~3.5% fraud, so ROC-AUC alone is misleading).

| Rank | Model | Mode | PR-AUC | ROC-AUC | F1 (thr=0.5) | Best-F1 | Inference (ms/1k) |
|---|---|---|---|---|---|---|---|
| 1 | **m3_gat** ⭐ | Transformer + GAT | **0.4211** | 0.8348 | 0.3726 | 0.4496 | 303.0 |
| 2 | m2_sage | Transformer + GraphSAGE | 0.4198 | 0.8341 | 0.3538 | 0.4409 | 295.2 |
| 3 | m1_transformer | Transformer only | 0.4142 | 0.8270 | 0.3205 | 0.4553 | 226.9 |
| 4 | m4_sthgnn | Transformer + ST-HGNN | 0.4093 | 0.8210 | 0.3341 | 0.4547 | 283.4 |

*(PR-AUC/ROC-AUC/F1 are exact re-verifications from a full pipeline reproduce — see
"Reproducibility" below; inference ms/1k is a live wall-clock benchmark on this CPU and
varies a little run to run, unlike the other columns.)*

**Winner: Transformer + GAT** (`models/best_model.pt`). Both graph-augmented
models beat the sequence-only baseline — relational structure shared across
card/device/merchant/region catches organized fraud that a single client's
transaction sequence can't see on its own. The margin between GAT and
GraphSAGE is narrow (+0.0013 PR-AUC); ST-HGNN, the heaviest model, finished
last, likely under-trained relative to its complexity on free-tier compute.

**Known constraint:** `GAT_HEADS` was reduced from the design's 4 to **1**
for the GAT/ST-HGNN checkpoints — the free Colab T4 GPU ran out of memory
with 4-head attention over the full ~590k-node graph. This is a documented
hardware constraint (`config.py`), not an oversight; see
[`results/analysis.md`](results/analysis.md) for the full write-up and its
implications for the GAT-vs-SAGE ranking.

Full metrics: [`results/comparison.csv`](results/comparison.csv). Plots:
[`results/plots/roc.png`](results/plots/roc.png),
[`pr.png`](results/plots/pr.png),
[`metrics_bar.png`](results/plots/metrics_bar.png).

## Explainability example

For a true-fraud test transaction, SHAP attributes the model's 75.4%
probability to factors like `C1` (+0.222), `C11` (+0.121), `C8` (+0.084),
`DeviceType` (+0.033) against a base rate of 0.150 — exactly additive. The
local LLM turns this into, e.g.:

> "This transaction was flagged with a 75% probability of fraud. The
> strongest signal was factor C1, which raised the risk score by 0.222,
> followed by C11 (+0.121) and C8 (+0.084)... these factors outweighed the
> smaller risk-lowering effects of C14 and C7."

See [`results/explanations_samples.md`](results/explanations_samples.md) for
5 worked examples (3 fraud, 2 legit) and
[`results/plots/shap_example.png`](results/plots/shap_example.png) for the
SHAP bar chart.

## Tech stack (all free & open-source)

| Stage | Library |
|---|---|
| Data / preprocessing | `pandas`, `numpy`, `scikit-learn` |
| Sequence model | `PyTorch` (`nn.TransformerEncoder`) |
| Graph construction & models | `torch_geometric` (`SAGEConv`, `GATConv`, `HeteroConv`), `networkx` |
| Fusion classifier | `PyTorch` (`nn.Linear`/`ReLU`/`Dropout`) |
| Metrics | `scikit-learn.metrics` |
| Feature explainability | `shap` |
| NL explainability | `ollama` (local `llama3.2:3b`, no API key) |
| Deployment | `streamlit` |
| Plotting | `matplotlib`, `seaborn` |

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
(Kaggle, free with a free account). No paid APIs, cloud credits, or
subscriptions are used anywhere in this project.

## Project structure

```
├── config.py                 # paths, hyperparameters, set_seed(), get_device()
├── requirements.txt
├── src/
│   ├── preprocess.py          # merge + clean → clean.csv
│   ├── split.py                # time-aware stratified split + scaler
│   ├── build_sequences.py      # clean.csv → per-client padded sequences
│   ├── build_graph.py          # clean.csv → PyG HeteroData graph
│   ├── check_alignment.py      # verifies sequence/graph views are row-aligned
│   ├── transformer_model.py    # SequenceTransformer branch
│   ├── gnn_models.py           # GraphSAGEBranch, GATBranch, STHGNNBranch
│   ├── fusion_model.py         # FusionClassifier (4 modes)
│   ├── train.py                # generic trainer, focal loss, early stopping
│   ├── evaluate.py             # metrics, thresholds, inference time, plots
│   ├── explain_shap.py         # SHAP on best_model.pt
│   ├── explain_ollama.py       # SHAP → local LLM explanation
│   └── utils.py                # focal loss, metrics, seed re-exports
├── app/streamlit_app.py        # interactive demo
├── models/                     # *.pt checkpoints + best_model.pt + scaler.pkl
├── results/                     # comparison.csv, analysis.md, plots/, SHAP/LLM examples
└── data/{raw,processed}/        # Kaggle CSVs (gitignored) + processed tensors
```

## How to run

### 1. Setup (free)
```bash
pip install -r requirements.txt
```
Get the free dataset (requires a free Kaggle account):
```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
# unzip train_transaction.csv, train_identity.csv into data/raw/
```
For the local LLM explanations, install [Ollama](https://ollama.com) (free)
and pull the model used by this project:
```bash
ollama pull llama3.2:3b
```

### 2. Reproduce the pipeline, in order
```bash
python config.py                    # sanity check: device, seed, folders
python src/preprocess.py            # → data/processed/clean.csv
python src/split.py                 # → data/processed/splits.npz, models/scaler.pkl
python src/build_sequences.py       # → data/processed/sequences.pt
python src/build_graph.py           # → data/processed/graph.pt
python src/check_alignment.py       # sanity check: both views agree
python src/train.py --model m1      # repeat for m1/m2/m3/m4 (GPU recommended — see note below)
python src/evaluate.py              # → results/comparison.csv, results/plots/*.png
python src/explain_shap.py          # → results/shap_example.png/.csv/.json
python src/explain_ollama.py        # → results/explanation_example.md
```

**Note on training compute:** all four models in this repo's `models/`
directory were trained on **Google Colab's free-tier T4 GPU** (not locally)
— see `colab_train_models_1to4.ipynb` and `COLAB_INSTRUCTIONS.md`. Training
locally on CPU works but is much slower (a single-model smoke test is fine;
a full 50-epoch run can take hours per model on CPU). Both paths are $0.

### 3. Run the demo
```bash
ollama serve                              # start Ollama (separate terminal)
streamlit run app/streamlit_app.py
```
Open the local URL Streamlit prints, fill in a transaction, and view the
fraud probability, SHAP factors, and LLM explanation. If Ollama isn't
running, the app automatically falls back to a templated (non-LLM)
explanation instead of crashing — this is also what happens on free hosts
like Streamlit Community Cloud, which can't run Ollama.

## Reproducibility

The full pipeline — split → build sequence/graph views → train → evaluate → explain →
app — has been verified end-to-end from `clean.csv` in a single reproduce pass (~15
minutes on this CPU-only machine, dominated by the sequence/graph rebuild and the
4-model evaluation). Every regenerated artifact matched the historical values exactly:
same split sizes and fraud ratios, same sequence/graph shapes and node/edge counts, same
PR-AUC ranking and SHAP target/attributions, and a clean Streamlit launch with no errors.

## Notes & constraints

- **Zero-cost throughout:** every library, dataset, and compute resource
  used is free (see the table above).
- **Ollama model:** the project's actual default is `llama3.2:3b`. `phi3`
  was also tested and found to hallucinate more on this task's prompts.
- **`client_id`** is a constructed pseudo-user key (`card1_card2_card3_card5_addr1`)
  since IEEE-CIS has no explicit user ID — the standard community approach
  for this dataset.
- **PR-AUC**, not accuracy, is the headline metric throughout because of the
  ~3.5% fraud class imbalance.
