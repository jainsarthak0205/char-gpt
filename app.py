"""Gradio app for char-gpt — Hugging Face Spaces entrypoint.

Loads the bundled checkpoint and exposes a small UI for generating
Shakespeare-flavored text from a prompt.

The checkpoint file (`checkpoints/char_gpt.pt`) is uploaded to the HF
Space repo via xet storage out-of-band — it is NOT committed to git
(see .gitignore). Local runs can re-train via `python -m char_gpt train`.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch

from char_gpt.sample import generate, load_checkpoint

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "char_gpt.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model, _tokenizer = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)


def sample(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    return generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k and top_k > 0 else None,
        device=DEVICE,
    )


with gr.Blocks(title="char-gpt") as demo:
    gr.Markdown(
        """
# char-gpt

A tiny decoder-only transformer built from scratch in PyTorch and trained
on **Pride and Prejudice** (Jane Austen, public domain, ~694k characters).
No `transformers` library — just causal multi-head self-attention,
pre-norm residual blocks, weight-tied embeddings, and AdamW + cosine LR.

Sub-million-parameter character-level models can't follow instructions
or answer questions. They learn the *statistical shape* of the training
text — spelling, common words, sentence cadence. Expect plausible-looking
English with no plot coherence.

Source: <https://github.com/jainsarthak0205/char-gpt>
        """
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Mr. Darcy", lines=2)
            max_new_tokens = gr.Slider(20, 600, value=300, step=10, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k (0 = off)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Sample", lines=18)
    btn.click(sample, [prompt, max_new_tokens, temperature, top_k], output)
    gr.Examples(
        examples=[
            ["Mr. Darcy", 300, 0.8, 40],
            ["Elizabeth was", 300, 0.8, 40],
            ["It is a truth universally acknowledged,", 300, 0.7, 40],
            ["My dear Mr. Bennet,", 300, 0.85, 40],
        ],
        inputs=[prompt, max_new_tokens, temperature, top_k],
    )


if __name__ == "__main__":
    demo.launch()
