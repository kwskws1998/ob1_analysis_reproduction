"""Load the frozen ET1 BiLSTM in its native T5-token coordinate system."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .et_checkpoint import ensure_et1_checkpoint
from .et1_tokenizer import ET1_TOKENIZER_SIGNATURE, load_et1_tokenizer
from .tokenizer_fingerprint import tokenizer_fingerprint


class BiLSTMRegression(nn.Module):
    """Predict one TRT value per native T5 token."""

    def __init__(self, embedding: nn.Embedding, hidden_dim: int, dropout: float):
        super().__init__()
        self.emb = embedding
        self.emb.requires_grad_(False)
        self.lstm = nn.LSTM(
            input_size=self.emb.weight.size(1),
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token-level TRT predictions."""
        hidden_states = self.dropout(self.emb(input_ids))
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.head(hidden_states).squeeze(-1)


class ET1NativeModel:
    """Expose the original ET1 checkpoint without reward-tokenizer remapping."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        tokenizer_path: Path | None = None,
        tokenizer_cache_dir: Path | None = None,
    ) -> None:
        self.model = BiLSTMRegression(nn.Embedding(32128, 512), 128, 0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fixTokenizer = load_et1_tokenizer(
            tokenizer_path=tokenizer_path,
            cache_dir=tokenizer_cache_dir,
        )
        fingerprint = tokenizer_fingerprint(self.fixTokenizer)
        self.cache_signature = (
            "huangxt39/SelectiveCacheForLM@"
            "eccc93f969745b04ce1e4911d6513d85565cc919:"
            "T5-tokenizer-BiLSTM-TRT-12-concat-3:"
            f"tokenizer_{ET1_TOKENIZER_SIGNATURE}:fingerprint_{fingerprint}"
        )
        resolved_checkpoint = ensure_et1_checkpoint(checkpoint_path)
        state_dict = torch.load(
            resolved_checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict native T5-token TRT and return its attention mask."""
        encoded = self.fixTokenizer(
            sentences,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            values = self.model(encoded["input_ids"].to(self.device))
        return values, encoded["attention_mask"].to(self.device)
