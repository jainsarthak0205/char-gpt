"""Tests for tokenizer, model, data, and a short training smoke run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from char_gpt import (
    CharDataset,
    CharTokenizer,
    GPT,
    GPTConfig,
    TrainConfig,
    generate,
    get_batch,
    train,
)
from char_gpt.dataset import load_bundled_corpus, prepare_data
from char_gpt.sample import load_checkpoint


def test_tokenizer_roundtrip():
    text = "hello, World!\n123."
    tok = CharTokenizer.from_text(text)
    assert tok.vocab_size == len(set(text))
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_tokenizer_save_load(tmp_path: Path):
    tok = CharTokenizer.from_text("abcd")
    path = tok.save(tmp_path / "tok.json")
    reloaded = CharTokenizer.load(path)
    assert reloaded.stoi == tok.stoi


def test_bundled_corpus_loads():
    text = load_bundled_corpus()
    assert len(text) > 100_000
    assert "ROMEO" in text


def test_gpt_forward_shapes():
    config = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(0, 20, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 20)
    assert loss is None

    targets = torch.randint(0, 20, (4, 8))
    _, loss = model(idx, targets)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_gpt_respects_block_size():
    config = GPTConfig(vocab_size=10, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0)
    model = GPT(config)
    too_long = torch.zeros((1, 5), dtype=torch.long)
    with pytest.raises(ValueError):
        model(too_long)


def test_attention_is_causal():
    """Modifying a later token must not change the logits at earlier positions."""
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    model.eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    b = a.clone()
    b[0, -1] = 9  # change the final token only
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    # logits at positions 0..6 (everything but the last) must be identical
    torch.testing.assert_close(la[:, :-1, :], lb[:, :-1, :])


def test_generate_extends_prompt():
    config = GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
    # the prompt portion must be preserved
    assert torch.equal(out[:, :3], idx)


def test_get_batch_shapes_and_shift():
    ids = torch.arange(100)
    x, y = get_batch(ids, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted right by one position
    torch.testing.assert_close(y[:, :-1], x[:, 1:])


def test_char_dataset_yields_shifted_pairs():
    ids = torch.arange(20)
    ds = CharDataset(ids, block_size=4)
    assert len(ds) == 16
    x, y = ds[0]
    torch.testing.assert_close(x, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(y, torch.tensor([1, 2, 3, 4]))


def test_smoke_training_lowers_loss(tmp_path: Path):
    """Train for a handful of steps on a tiny corpus and verify loss goes down."""
    text = "hello world. " * 200  # ~2.6k chars, learnable trivially
    config = GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    tokenizer, train_ids, val_ids = prepare_data(text, block_size=config.block_size)
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config)

    initial_loss = None
    with torch.no_grad():
        x, y = get_batch(train_ids, config.block_size, batch_size=8)
        _, l = model(x, y)
        initial_loss = float(l)

    train_config = TrainConfig(
        max_iters=80,
        batch_size=8,
        eval_interval=40,
        eval_iters=5,
        lr=3e-3,
        warmup_iters=10,
        device="cpu",
        checkpoint=str(tmp_path / "ckpt.pt"),
        log_every=200,  # suppress per-step logs in tests
    )
    result = train(model, tokenizer, train_ids, val_ids, train_config)
    assert result.final_train_loss < initial_loss
    assert (tmp_path / "ckpt.pt").exists()


def test_checkpoint_save_load_and_generate(tmp_path: Path):
    text = "abcabcabcabc " * 100
    config = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    tokenizer, train_ids, val_ids = prepare_data(text, block_size=config.block_size)
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config)
    train_config = TrainConfig(
        max_iters=20,
        batch_size=4,
        eval_interval=10,
        eval_iters=3,
        lr=3e-3,
        warmup_iters=5,
        device="cpu",
        checkpoint=str(tmp_path / "ckpt.pt"),
        log_every=200,
    )
    train(model, tokenizer, train_ids, val_ids, train_config)

    loaded, tok = load_checkpoint(tmp_path / "ckpt.pt")
    out = generate(loaded, tok, prompt="abc", max_new_tokens=10, temperature=1.0)
    assert out.startswith("abc")
    assert len(out) == 13
