"""
check_env.py — environment sanity check for the fraud-detection project.

What it does:
    Prints the installed versions of the core libraries (torch, torch_geometric,
    scikit-learn, shap, streamlit, plus numpy/pandas), reports CUDA availability,
    and runs a tiny PyTorch Geometric smoke test: build a 3-node graph and run a
    single SAGEConv forward pass, printing the output tensor shape.

Inputs:  none.
Outputs: none (prints a report to stdout). Exits non-zero if a core import or
         the PyG smoke test fails, so it doubles as a setup gate.

Run:     python check_env.py
"""

from __future__ import annotations

import importlib
import sys

# Modules to report versions for: (import name, friendly label).
_CORE_MODULES = [
    ("torch", "torch"),
    ("torch_geometric", "torch_geometric"),
    ("sklearn", "scikit-learn"),
    ("shap", "shap"),
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
]


def print_versions() -> None:
    """Import each core module and print its version (or a clear failure)."""
    print("=== Library versions ===")
    failed = []
    for import_name, label in _CORE_MODULES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {label:18s} {version}")
        except ImportError as exc:
            print(f"  {label:18s} NOT INSTALLED ({exc})")
            failed.append(label)
    if failed:
        raise ImportError(f"Missing core packages: {', '.join(failed)}")


def print_cuda() -> None:
    """Report CUDA availability and device name (falls back to CPU)."""
    import torch

    print("\n=== Compute ===")
    available = torch.cuda.is_available()
    print(f"  CUDA available: {available}")
    if available:
        print(f"  CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version:   {torch.version.cuda}")
    else:
        print("  Using CPU (no GPU detected) - fine for this project.")


def pyg_smoke_test() -> None:
    """Build a 3-node graph and run one SAGEConv forward; print output shape.

    The graph: 3 nodes, each with a 4-dim feature vector, connected in a small
    chain 0->1->2 (plus reverse edges). A single SAGEConv maps 4 -> 8 features.
    """
    import torch
    from torch_geometric.nn import SAGEConv

    print("\n=== PyG smoke test ===")
    # 3 nodes, 4 input features each.
    x = torch.randn(3, 4)
    # edge_index: shape [2, num_edges]; undirected chain 0-1-2.
    edge_index = torch.tensor(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]],
        dtype=torch.long,
    )

    conv = SAGEConv(in_channels=4, out_channels=8)
    out = conv(x, edge_index)

    print(f"  input  shape: {tuple(x.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    assert out.shape == (3, 8), f"unexpected SAGEConv output shape: {tuple(out.shape)}"
    print("  SAGEConv forward OK.")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}\n")
    print_versions()
    print_cuda()
    pyg_smoke_test()
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
