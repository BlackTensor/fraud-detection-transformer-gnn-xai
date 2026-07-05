# Training Models 1–4 on Free Colab GPU

This trains **all four** comparison models (Tasks 4.2–4.5) on a free T4 GPU.

Why all four (not just the GNNs): local CPU training is far too slow. Benchmarked
on the project machine, even the Transformer-only Model 1 ran at **~10 min/epoch**
(≈8.5 h for a full 50-epoch run), and the GNN models (which re-run a 590k-node
graph every batch) are worse. The free T4 GPU does each model in a few minutes, so
the entire comparison study runs on Colab.

**Cost: ₹0 / $0** — Colab free tier + free Drive storage, no paid APIs.

---

## Files involved (all in your project root)
- `colab_package.zip` — everything Colab needs (code + processed data). **Built by
  `make_colab_package.py`.**
- `colab_train_models_1to4.ipynb` — the notebook you run on Colab.

---

## Step 1 — Build the package (local, one command)
```
python make_colab_package.py
```
This prints the final zip size. It bundles `config.py`, all of `src/`, the
processed views (`sequences.pt`, `graph.pt`, `splits.npz`), `scaler.pkl`, and a
**slim 4-column `clean.csv`** (all that Model 4's temporal features need). It does
**not** need a local `comparison.csv` — Colab writes that table fresh, starting
with Model 1.

Expected size: **~1.6 GB** (the float32 `sequences.pt` dominates and doesn't
compress).

## Step 2 — Upload to Google Drive
1. Go to <https://drive.google.com> → **My Drive**.
2. Drag `colab_package.zip` into the root of My Drive (not a subfolder, or you'll
   need to edit the `ZIP` path in the notebook).
3. Wait for the upload to finish (a ~1.6 GB upload can take a while on home
   internet — that's normal).

## Step 3 — Open the notebook in Colab
1. Go to <https://colab.research.google.com>.
2. **File → Upload notebook** → upload `colab_train_models_1to4.ipynb` from your
   project root. (Alternatively put the .ipynb in Drive and open it from there.)

## Step 4 — Enable the free GPU
1. **Runtime → Change runtime type**.
2. **Hardware accelerator → T4 GPU** (this is the free tier).
3. **Save**.

## Step 5 — Run the cells, top to bottom
| Cell | What it does |
|------|--------------|
| 1 | `nvidia-smi` — confirms a GPU is attached |
| 2 | `pip install torch_geometric==2.8.0` (matches `graph.pt`) + asserts CUDA |
| 3 | Mounts Google Drive (click through the auth popup) |
| 4 | Unzips `colab_package.zip` → `/content/ntcc` |
| 5 | Prints `device = cuda` (sanity check) |
| 6 | **Trains Model 1 (Transformer only)** → `models/m1_transformer.pt` |
| 7 | **Trains Model 2 (GraphSAGE)** → `models/m2_sage.pt` |
| 8 | **Trains Model 3 (GAT)** → `models/m3_gat.pt` |
| 9 | **Trains Model 4 (ST-HGNN)** → `models/m4_sthgnn.pt` |
| 10 | Shows the `comparison.csv` table |
| 11 | Copies the 4 checkpoints + `comparison.csv` → `My Drive/colab_outputs/` |

Run cells 6–9 one at a time and watch each finish (each prints per-epoch val
PR-AUC and a final TEST line). On a T4 each model should take only a few minutes.
Early stopping uses a `min_delta` of 1e-4 on val PR-AUC, so trivial noise-level
gains no longer keep training running for the full 50 epochs.

> **If cell 9 (ST-HGNN) throws a CUDA fp16 / autocast error**, rerun it with the
> AMP fallback (commented in that cell):
> `!cd /content/ntcc && python src/train.py --model m4 --no-amp`

## Step 6 — Pull results back to your local project
From `My Drive/colab_outputs/`, download:
- `models/m1_transformer.pt`, `m2_sage.pt`, `m3_gat.pt`, `m4_sthgnn.pt` → into your
  local `models/`
- `comparison.csv` → into your local `results/comparison.csv` (it now has all
  **four** model rows).

That completes Tasks 4.2–4.5. Phase 5 (evaluate + plots) then runs fine locally.

---

## Notes / troubleshooting
- **Don't close the Colab tab** while training; the free runtime disconnects if
  idle. Running cells keep it alive.
- **`graph.pt` fails to unpickle** → the PyG version differs. Try
  `!pip install -U torch_geometric` and rerun from the unzip cell.
- **Out of memory** (unlikely on T4 for this size): lower the batch size, e.g.
  `python src/train.py --model m2 --batch-size 128`.
- The `scaler.pkl` and full feature engineering are already baked into the
  prebuilt `sequences.pt` / `graph.pt`, so Colab does **no** preprocessing — it
  only trains.
- **Why not train Model 1 locally?** It works, but at ~10 min/epoch on CPU a full
  run is ~8.5 h. Keeping all four on the GPU is faster and keeps the comparison
  consistent (same device/precision for every model).
