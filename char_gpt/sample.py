"""Load a trained checkpoint and generate text."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, CharTokenizer]:
    """Load a model + its tokenizer from a checkpoint saved by :func:`train`."""
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    model.to(device)
    stoi = {str(k): int(v) for k, v in payload["tokenizer_stoi"].items()}
    itos = {v: k for k, v in stoi.items()}
    tokenizer = CharTokenizer(stoi=stoi, itos=itos)
    return model, tokenizer


def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | torch.device = "cpu",
) -> str:
    """Sample a continuation of ``prompt`` of length ``max_new_tokens``."""
    device = torch.device(device)
    if not prompt:
        # seed with the first character of the vocabulary if nothing was given
        prompt = next(iter(tokenizer.stoi))
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
