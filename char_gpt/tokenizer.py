"""Character-level tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharTokenizer:
    """Reversible character-to-integer codec.

    Use :meth:`from_text` to learn the vocabulary from a corpus, then
    :meth:`encode` / :meth:`decode` to convert between strings and lists
    of integer token ids.
    """

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {v: k for k, v in stoi.items()}
        return cls(stoi=stoi, itos=itos)
