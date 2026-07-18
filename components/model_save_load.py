"""Demo: save / load a small model in several formats (PyTorch, safetensors, GGUF).

GGUF design note
----------------
Like ``torch.save(state_dict)`` / safetensors, we only persist **weights**.
Architecture (layer sizes, depth, etc.) already lives in the Python
``nn.Module`` definition — there is no need to hand-write every
``in_features`` / ``out_features`` into GGUF metadata.

Load pattern (same as PyTorch / HuggingFace)::

    model = MyBigModel(config)          # build empty shell from code/config
    load_gguf("model.gguf", model)      # fill weights from file

Production LLM GGUFs *do* store arch KV (n_layer, n_head, …) because
llama.cpp has no Python class — a C++ runtime must rebuild the graph from
metadata alone. For in-Python round-trips, that is unnecessary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from gguf import GGUFReader, GGUFWriter
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

M = TypeVar("M", bound=nn.Module)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def save_gguf(self, path: str | Path) -> Path:
        """Convenience wrapper: dump this module's state_dict to GGUF."""
        return save_gguf(self, path)


# ---------------------------------------------------------------------------
# Generic GGUF helpers — work for any nn.Module
# ---------------------------------------------------------------------------


def save_gguf(
    model: nn.Module,
    path: str | Path,
    *,
    arch: str = "pytorch",
    name: str | None = None,
) -> Path:
    """Save an ``nn.Module``'s ``state_dict`` into a GGUF file.

    Only tensors are written. No per-layer hyper-parameters needed — those
    already exist on ``model`` (and/or your config object).

    Args:
        model: Any ``torch.nn.Module``.
        path: Output ``.gguf`` path.
        arch: GGUF ``general.architecture`` tag (free-form for this demo).
        name: Optional ``general.name``; defaults to the class name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(path), arch=arch)
    writer.add_name(name or type(model).__name__)

    # One loop over state_dict — scales to transformers of any size.
    for tensor_name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().float().contiguous().numpy()
        writer.add_tensor(tensor_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


def load_state_dict_gguf(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a GGUF file into a plain ``state_dict`` (name → tensor)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    reader = GGUFReader(str(path))
    state_dict: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        # Copy out of memmap so the file handle can be released.
        data = np.array(tensor.data, copy=True)
        state_dict[tensor.name] = torch.from_numpy(data)
    return state_dict


def load_gguf(path: str | Path, model: M | None = None) -> M:
    """Load GGUF weights into a model.

    Args:
        path: Path to ``.gguf``.
        model: Pre-built module whose parameter names/shapes match the file.
            If ``None``, constructs a default :class:`SimpleModel` (demo only).

    Returns:
        The same ``model`` instance with weights loaded (``eval`` mode).
    """
    if model is None:
        model = SimpleModel()  # type: ignore[assignment]

    state_dict = load_state_dict_gguf(path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Format demos
# ---------------------------------------------------------------------------


def torch_save_load():
    model = SimpleModel()
    Path("tmp").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "tmp/model.pth")
    state_dict = torch.load("tmp/model.pth", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def huggingface_save_load():
    model = SimpleModel()
    Path("tmp").mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), "tmp/model.safetensors")
    state_dict = load_file("tmp/model.safetensors")
    model.load_state_dict(state_dict)
    return model


def tokenizer_save_load():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("tmp/tokenizer")
    path = Path("tmp/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer


def model_save_gguf():
    """Save SimpleModel as GGUF and load it back (round-trip demo)."""
    Path("tmp").mkdir(parents=True, exist_ok=True)
    model = SimpleModel()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)

    # Same mental model as torch / safetensors:
    #   1) dump weights
    #   2) build empty architecture in code
    #   3) load weights into it
    save_gguf(model, "tmp/model.gguf")
    loaded = load_gguf("tmp/model.gguf", SimpleModel())

    x = torch.randn(4, 10)
    with torch.no_grad():
        if not torch.allclose(model(x), loaded(x), atol=1e-6):
            raise AssertionError("GGUF round-trip produced different outputs")
    return loaded


if __name__ == "__main__":
    torch_save_load()
    print("torch_save_load: ok")
    huggingface_save_load()
    print("huggingface_save_load: ok")
    model_save_gguf()
    print("model_save_gguf: ok -> tmp/model.gguf")
    try:
        tokenizer_save_load()
        print("tokenizer_save_load: ok")
    except Exception as e:
        print(f"tokenizer_save_load skipped: {e}")
