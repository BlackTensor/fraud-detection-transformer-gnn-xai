"""
transformer_model.py — SequenceTransformer branch (Task 3.1).

What it does:
    The "sequence view" of a transaction is a left-padded causal window of shape
    [MAX_SEQ_LEN, F] (built in build_sequences.py): the current transaction sits
    in the LAST slot, with up to MAX_SEQ_LEN-1 of the same client's prior
    transactions before it, and zero-padding at the FRONT for short histories.

    SequenceTransformer turns each such window into a single EMBED_DIM (128-d)
    embedding that the fusion classifier (Task 3.5) consumes. Pipeline:
      1. Linear input projection  F -> EMBED_DIM.
      2. Add a LEARNED positional encoding over the MAX_SEQ_LEN slots.
      3. nn.TransformerEncoder (TRANSFORMER_LAYERS layers, TRANSFORMER_HEADS heads,
         dim_feedforward=TRANSFORMER_FF, batch_first=True), with the padding mask
         passed as src_key_padding_mask so padded slots never attend / are attended.
      4. Pool to one vector by taking the LAST REAL timestep (the current
         transaction). A masked-mean pooling option is also provided.

    Output: a [B, EMBED_DIM] tensor — NOT a probability. Classification happens in
    the fusion head so this branch can be reused unchanged across all four models.

Inputs (forward):
    X_seq : float32 [B, L, F]   left-padded sequences
    mask  : bool    [B, L]      True = REAL timestep, False = padding
                                (the VALID-mask convention from build_sequences.py)

Outputs (forward):
    emb   : float32 [B, EMBED_DIM]

Mask convention note: build_sequences.py stores a VALID-mask (True = real). The
TransformerEncoder wants src_key_padding_mask where True = "ignore", so we pass
`~mask` internally. The current transaction is always the last slot, so no row is
ever fully padded (which would otherwise produce NaNs from full-row masking).

Running this file as a script executes a CPU smoke test that pushes a dummy batch
through and asserts the output shape is [B, EMBED_DIM].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


class SequenceTransformer(nn.Module):
    """Encode a [B, L, F] causal sequence window into a [B, EMBED_DIM] vector."""

    def __init__(
        self,
        in_features: int,
        embed_dim: int = config.EMBED_DIM,
        max_seq_len: int = config.MAX_SEQ_LEN,
        num_layers: int = config.TRANSFORMER_LAYERS,
        num_heads: int = config.TRANSFORMER_HEADS,
        dim_feedforward: int = config.TRANSFORMER_FF,
        dropout: float = config.DROPOUT,
        pooling: str = "last",
    ) -> None:
        """
        Args:
            in_features:     F, the per-timestep feature dimension (32 for our data).
            embed_dim:       output / model dimension (must be divisible by num_heads).
            max_seq_len:     L, length of every (padded) sequence window.
            num_layers:      number of TransformerEncoder layers.
            num_heads:       attention heads per layer.
            dim_feedforward: hidden size of the position-wise feed-forward block.
            dropout:         dropout used inside the encoder.
            pooling:         "last" (last real timestep) or "mean" (masked mean).
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}.")

        self.in_features = in_features
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pooling = pooling

        # 1) Project each timestep's raw feature vector into the model space.
        self.input_proj = nn.Linear(in_features, embed_dim)

        # 2) Learned positional encoding, one vector per slot (0..max_seq_len-1).
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.input_dropout = nn.Dropout(dropout)

        # 3) Stacked self-attention encoder (batch_first so shapes are [B, L, E]).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable to train than the default.
        )
        # enable_nested_tensor=False: the nested-tensor fast path is unused with
        # pre-norm (norm_first=True) and would otherwise emit a benign warning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, X_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map [B, L, F] + valid-mask [B, L] -> [B, EMBED_DIM] embedding."""
        if X_seq.dim() != 3:
            raise ValueError(f"X_seq must be [B, L, F], got shape {tuple(X_seq.shape)}.")
        b, length, f = X_seq.shape
        if f != self.in_features:
            raise ValueError(
                f"X_seq feature dim {f} != model in_features {self.in_features}."
            )
        if length != self.max_seq_len:
            raise ValueError(
                f"X_seq length {length} != model max_seq_len {self.max_seq_len}."
            )
        if mask.shape != (b, length):
            raise ValueError(
                f"mask must be [B, L]={ (b, length) }, got {tuple(mask.shape)}."
            )

        mask = mask.bool()

        # Project + add learned positions, then dropout.
        h = self.input_proj(X_seq) + self.pos_embedding[:, :length, :]
        h = self.input_dropout(h)

        # TransformerEncoder expects True = "ignore"; our mask is True = "real".
        src_key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        if self.pooling == "mean":
            # Masked mean over real timesteps only.
            valid = mask.unsqueeze(-1).type_as(h)            # [B, L, 1]
            summed = (h * valid).sum(dim=1)                  # [B, E]
            counts = valid.sum(dim=1).clamp(min=1.0)         # [B, 1]
            return summed / counts

        # "last": index of the last REAL timestep per row. Padding is at the front
        # and the current transaction is always the last slot, so this is L-1 in
        # practice; we still derive it from the mask to stay robust.
        last_idx = (mask.cumsum(dim=1).argmax(dim=1))        # [B]
        gather_idx = last_idx.view(b, 1, 1).expand(b, 1, self.embed_dim)
        return h.gather(dim=1, index=gather_idx).squeeze(1)  # [B, E]


def _smoke_test() -> None:
    """CPU smoke test: dummy batch -> assert output is [B, EMBED_DIM]."""
    config.set_seed(config.SEED)
    device = torch.device("cpu")

    b, length, f = 8, config.MAX_SEQ_LEN, 32
    model = SequenceTransformer(in_features=f).to(device).eval()

    X = torch.randn(b, length, f, device=device)
    # Random left-padded valid-masks: each row has 1..L real timesteps, with the
    # last slot always real (matches build_sequences.py).
    lengths = torch.randint(1, length + 1, (b,))
    mask = torch.zeros(b, length, dtype=torch.bool)
    for i, n_real in enumerate(lengths):
        mask[i, length - int(n_real):] = True
    X[~mask] = 0.0  # zero the padded slots, as in the real data

    with torch.no_grad():
        emb = model(X, mask)

    assert emb.shape == (b, config.EMBED_DIM), f"unexpected shape {tuple(emb.shape)}"
    assert torch.isfinite(emb).all(), "non-finite values in embedding"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[transformer] in_features={f}, embed_dim={config.EMBED_DIM}, "
          f"layers={config.TRANSFORMER_LAYERS}, heads={config.TRANSFORMER_HEADS}, "
          f"ff={config.TRANSFORMER_FF}")
    print(f"[transformer] params: {n_params:,}")
    print(f"[transformer] dummy batch {tuple(X.shape)} -> embedding {tuple(emb.shape)} OK")
    print("[transformer] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
