"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import load_bundled_corpus, load_corpus, prepare_data
from .model import GPT, GPTConfig
from .sample import generate, load_checkpoint
from .train import TrainConfig, train


def _cmd_train(args: argparse.Namespace) -> int:
    text = load_corpus(args.corpus) if args.corpus else load_bundled_corpus()
    if args.max_chars:
        text = text[: args.max_chars]
    print(f"corpus: {len(text):,} characters")

    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    tokenizer, train_ids, val_ids = prepare_data(text, block_size=config.block_size)
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config)

    train_config = TrainConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        log_every=args.log_every,
    )
    result = train(model, tokenizer, train_ids, val_ids, train_config)
    print(
        f"\ntraining done. best val loss={result.best_val_loss:.4f}  "
        f"checkpoint={result.checkpoint_path}"
    )
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    model, tokenizer = load_checkpoint(ckpt_path, device=args.device)
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(text)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    model, tokenizer = load_checkpoint(ckpt_path, device=args.device)
    print(f"checkpoint: {ckpt_path}")
    print(f"params:     {model.num_parameters():,}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"config:     {model.config}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="char-gpt",
        description="Train and sample from a tiny character-level GPT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("train", help="train a model from scratch")
    p_tr.add_argument("--corpus", help="path to a text file (defaults to bundled tinyshakespeare)")
    p_tr.add_argument("--max-chars", type=int, default=None, help="truncate corpus for fast smoke tests")
    p_tr.add_argument("--block-size", type=int, default=64)
    p_tr.add_argument("--n-layer", type=int, default=4)
    p_tr.add_argument("--n-head", type=int, default=4)
    p_tr.add_argument("--n-embd", type=int, default=128)
    p_tr.add_argument("--dropout", type=float, default=0.1)
    p_tr.add_argument("--max-iters", type=int, default=2000)
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--eval-interval", type=int, default=200)
    p_tr.add_argument("--eval-iters", type=int, default=50)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--device", default="auto")
    p_tr.add_argument("--checkpoint", default="checkpoints/char_gpt.pt")
    p_tr.add_argument("--log-every", type=int, default=50)
    p_tr.set_defaults(func=_cmd_train)

    p_sm = sub.add_parser("sample", help="generate text from a trained checkpoint")
    p_sm.add_argument("--checkpoint", default="checkpoints/char_gpt.pt")
    p_sm.add_argument("--prompt", default="\n")
    p_sm.add_argument("--max-new-tokens", type=int, default=300)
    p_sm.add_argument("--temperature", type=float, default=0.9)
    p_sm.add_argument("--top-k", type=int, default=None)
    p_sm.add_argument("--device", default="cpu")
    p_sm.set_defaults(func=_cmd_sample)

    p_in = sub.add_parser("info", help="describe a checkpoint")
    p_in.add_argument("--checkpoint", default="checkpoints/char_gpt.pt")
    p_in.add_argument("--device", default="cpu")
    p_in.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
