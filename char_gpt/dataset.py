"""Dataset and batch sampler for character-level language modeling."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


def load_bundled_corpus() -> str:
    """Return the bundled Shakespeare corpus as a single string."""
    path = resources.files("char_gpt.data").joinpath("tiny_shakespeare.txt")
    return path.read_text(encoding="utf-8")


def load_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


class CharDataset(Dataset):
    """A sliding-window dataset over a flat sequence of token ids.

    Each item is a ``(x, y)`` pair where ``y`` is ``x`` shifted by one
    position, ready for a standard next-token-prediction loss.
    """

    def __init__(self, ids: torch.Tensor, block_size: int) -> None:
        if ids.dim() != 1:
            raise ValueError("ids must be a 1-D tensor")
        if len(ids) <= block_size:
            raise ValueError(f"corpus too short: {len(ids)} <= block_size={block_size}")
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.block_size + 1]
        return chunk[:-1].long(), chunk[1:].long()


def get_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random mini-batch directly from a flat token tensor."""
    if len(ids) <= block_size:
        raise ValueError("ids too short for the requested block_size")
    starts = torch.randint(
        0, len(ids) - block_size, (batch_size,), generator=generator
    )
    x = torch.stack([ids[i : i + block_size] for i in starts])
    y = torch.stack([ids[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device).long(), y.to(device).long()


def prepare_data(
    text: str, *, block_size: int, val_fraction: float = 0.1
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Tokenize ``text`` and split into train/val tensors."""
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * val_fraction)
    if n_val < block_size + 1:
        n_val = block_size + 1
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]
    return tok, train_ids, val_ids
