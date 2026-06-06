"""Training loop for char-gpt."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .dataset import get_batch
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    max_iters: int = 2000
    batch_size: int = 32
    eval_interval: int = 200
    eval_iters: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    warmup_iters: int = 100
    seed: int = 0
    device: str = "auto"
    checkpoint: str = "checkpoints/char_gpt.pt"
    log_every: int = 50

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class TrainResult:
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    best_val_loss: float = float("inf")
    losses: list[tuple[int, float, float]] = field(default_factory=list)
    checkpoint_path: str | None = None


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    out = {}
    for name, data in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device=device)
            _, loss = model(x, y)
            losses[k] = float(loss)
        out[name] = float(losses.mean())
    model.train()
    return out["train"], out["val"]


def _lr_for(it: int, config: TrainConfig) -> float:
    if it < config.warmup_iters:
        return config.lr * (it + 1) / max(1, config.warmup_iters)
    progress = (it - config.warmup_iters) / max(1, config.max_iters - config.warmup_iters)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model: GPT,
    tokenizer: CharTokenizer,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    config: TrainConfig,
) -> TrainResult:
    """Train ``model`` for ``config.max_iters`` steps.

    Saves the best-val-loss checkpoint to ``config.checkpoint`` so the
    sampler can pick it up later.
    """
    torch.manual_seed(config.seed)
    device = config.resolve_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )

    ckpt_path = Path(config.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    result = TrainResult()
    print(f"device: {device}")
    print(f"model:  GPT  ({model.num_parameters():,} params)")
    print(f"data:   train={len(train_ids):,} val={len(val_ids):,} tokens")

    model.train()
    block_size = model.config.block_size
    t0 = time.time()
    for it in range(config.max_iters):
        lr = _lr_for(it, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_ids, block_size, config.batch_size, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if it % config.log_every == 0:
            print(
                f"iter {it:5d}  loss={float(loss.detach()):.4f}  lr={lr:.2e}  "
                f"({time.time() - t0:.1f}s)"
            )

        if it > 0 and (it % config.eval_interval == 0 or it == config.max_iters - 1):
            train_loss, val_loss = estimate_loss(
                model, train_ids, val_ids, block_size, config.batch_size, config.eval_iters, device
            )
            result.losses.append((it, train_loss, val_loss))
            print(f"  -- eval  train={train_loss:.4f}  val={val_loss:.4f}")
            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model.config.__dict__,
                        "tokenizer_stoi": tokenizer.stoi,
                        "iter": it,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )
                result.checkpoint_path = str(ckpt_path)
                print(f"  -> new best val, saved {ckpt_path}")

    result.final_train_loss = float(loss.detach())
    if result.losses:
        result.final_val_loss = result.losses[-1][2]
    return result
