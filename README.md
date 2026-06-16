---
title: char-gpt
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
---

# char-gpt

> **A tiny GPT, built from scratch in pure PyTorch, trained on a Jane Austen novel.**

**Live demo:** <https://huggingface.co/spaces/jainsarthak0205/char-gpt>

No `transformers` library, no pretrained weights — just one short file
that implements multi-head causal self-attention, a position-wise MLP,
pre-norm residual blocks, weight-tied token embeddings, and a top-k
sampler. Train it on a laptop CPU in a few minutes and generate text
that reads like a (very small) Austen.

## Why this exists

You can use a transformer without understanding it. Implementing one
from scratch removes that option. Every piece is right here:

```
char_gpt/
├── model.py        # GPTConfig, CausalSelfAttention, MLP, Block, GPT
├── tokenizer.py    # character-level codec with save/load
├── data.py         # bundled corpus loader, batch sampler, train/val split
├── train.py        # AdamW + cosine LR + warmup + grad clipping + eval
├── sample.py       # checkpoint loader + autoregressive sampler
├── cli.py          # python -m char_gpt {train,sample,info}
└── data/
    ├── pride_and_prejudice.txt  # 694k chars, Jane Austen (default)
    └── tiny_shakespeare.txt     # 1.1 MB, Karpathy's classic corpus
```

## Install

```bash
pip install -r requirements.txt        # core (CPU torch is fine)
pip install -r requirements-dev.txt    # + pytest
```

## Quick start

Train a tiny model on the bundled Pride and Prejudice corpus:

```bash
python -m char_gpt train --block-size 128 --max-iters 3000
```

Then sample from it:

```bash
python -m char_gpt sample --prompt "Mr. Darcy" --max-new-tokens 400 --temperature 0.8
```

After ~20 minutes of CPU training the output looks like (very-small-model)
Austen: real English words, somewhat sensible sentence structure, no real
plot coherence — a character-level model with under a million parameters
can only do so much.

(Want Shakespeare instead? `--corpus char_gpt/data/tiny_shakespeare.txt`.)

Inspect a checkpoint:

```bash
python -m char_gpt info
# checkpoint: checkpoints/char_gpt.pt
# params:     813,825
# vocab_size: 65
# config:     GPTConfig(vocab_size=65, block_size=64, n_layer=4, ...)
```

## Useful flags

| Flag             | Default  | Notes                                    |
| ---------------- | -------- | ---------------------------------------- |
| `--corpus`       | bundled  | path to a UTF-8 text file                |
| `--max-chars`    | none     | truncate the corpus (smoke testing)      |
| `--block-size`   | 64       | context length                           |
| `--n-layer`      | 4        | transformer blocks                       |
| `--n-head`       | 4        | attention heads (must divide n_embd)     |
| `--n-embd`       | 128      | embedding dimension                      |
| `--max-iters`    | 2000     | training steps                           |
| `--lr`           | 3e-4     | AdamW peak learning rate                 |
| `--temperature`  | 0.9      | sampling temperature                     |
| `--top-k`        | none     | restrict sampling to top-k logits        |

## Python API

```python
from char_gpt import GPT, GPTConfig, prepare_data, train, TrainConfig, generate

text = open("my_corpus.txt", encoding="utf-8").read()
config = GPTConfig(block_size=128, n_layer=4, n_head=4, n_embd=128)
tokenizer, train_ids, val_ids = prepare_data(text, block_size=config.block_size)
config.vocab_size = tokenizer.vocab_size

model = GPT(config)
result = train(model, tokenizer, train_ids, val_ids, TrainConfig(max_iters=2000))
print(generate(model, tokenizer, prompt="Once upon a time", max_new_tokens=300))
```

## What's inside the model

- **Token embedding** (vocab_size × n_embd) and a learned **position
  embedding** (block_size × n_embd), added together.
- N transformer **blocks**, each one `x = x + attn(ln(x))` then
  `x = x + mlp(ln(x))` — the classic pre-norm residual design.
- The attention layer fuses Q/K/V into a single `Linear(n_embd, 3*n_embd)`
  projection, splits, reshapes to `(B, n_head, T, head_dim)`, computes
  scaled dot-product attention with a registered lower-triangular
  causal mask, then re-projects.
- The MLP expands by 4× with `GELU` then projects back, the standard
  GPT-2 ratio.
- The output head shares its weight matrix with the token embedding
  (weight tying), which saves parameters and tends to improve quality.

It's about 80 lines of meaningful PyTorch in [model.py](char_gpt/model.py).

## Tests

```bash
pytest
```

The suite verifies:

- Tokenizer round-tripping and save/load
- Forward pass shapes and a positive cross-entropy loss
- That attention is genuinely causal (modifying token *t* must not
  change the logits at positions before *t*)
- The autoregressive sampler preserves the prompt and extends it by
  exactly `max_new_tokens`
- A 80-step training run on a toy corpus actually lowers the loss
- Checkpoint save → load → generate round-trips

## What you can talk about in interviews

- **Pre-norm vs post-norm.** This implementation uses pre-norm
  (`x + sublayer(ln(x))`) because it trains more stably at depth
  without learning-rate warmup tuning. The original GPT-2 paper uses the
  same pattern.
- **Weight tying.** The input embedding and output projection share a
  weight matrix. Halves the parameter count of those two layers and
  acts as a soft regularizer.
- **Causal mask as a buffer.** Registered with
  `register_buffer("causal_mask", ...)`, so it moves with the module
  to GPU/CPU but isn't a learnable parameter.
- **Cosine LR with linear warmup.** `train.py` warms up for 100 steps
  to `--lr`, then cosine-decays to zero — a standard and effective
  schedule for small transformers.

## License

MIT — see [LICENSE](LICENSE).
