"""char_gpt — a from-scratch character-level GPT language model in pure PyTorch."""

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .dataset import CharDataset, get_batch
from .train import TrainConfig, train
from .sample import generate

__all__ = [
    "GPT",
    "GPTConfig",
    "CharTokenizer",
    "CharDataset",
    "get_batch",
    "TrainConfig",
    "train",
    "generate",
]
__version__ = "0.1.0"
