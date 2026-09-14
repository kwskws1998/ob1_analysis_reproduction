"""Tests for the stable tokenizer behavior fingerprint."""

from __future__ import annotations

import json
import unittest

from ob1_repro.tokenizer_fingerprint import tokenizer_fingerprint


class _BackendTokenizer:
    """Expose a mutable serialized backend state for fingerprint tests."""

    def __init__(self) -> None:
        self.payload = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "model": {"type": "Unigram", "vocab": [["a", 0.0]]},
        }

    def to_str(self) -> str:
        """Serialize the current backend state like tokenizers.Tokenizer."""
        return json.dumps(self.payload, separators=(",", ":"))


class _Tokenizer:
    """Supply the tokenizer attributes consumed by tokenizer_fingerprint."""

    def __init__(self) -> None:
        self.backend_tokenizer = _BackendTokenizer()
        self.model_max_length = 2048
        self.padding_side = "right"
        self.special_tokens_map = {"pad_token": "<pad>"}
        self.truncation_side = "right"


class TokenizerFingerprintTests(unittest.TestCase):
    """Require inference-time padding state not to change the fingerprint."""

    def test_padding_state_does_not_change_fingerprint(self) -> None:
        """Ignore mutable batch-padding and truncation backend settings."""
        tokenizer = _Tokenizer()
        before = tokenizer_fingerprint(tokenizer)
        tokenizer.backend_tokenizer.payload["padding"] = {
            "strategy": "BatchLongest",
            "direction": "Right",
            "pad_id": 0,
            "pad_token": "<pad>",
        }
        tokenizer.backend_tokenizer.payload["truncation"] = {
            "max_length": 2048,
            "stride": 0,
            "strategy": "LongestFirst",
            "direction": "Right",
        }
        self.assertEqual(before, tokenizer_fingerprint(tokenizer))

    def test_model_state_changes_fingerprint(self) -> None:
        """Continue detecting behaviorally relevant backend changes."""
        tokenizer = _Tokenizer()
        before = tokenizer_fingerprint(tokenizer)
        tokenizer.backend_tokenizer.payload["model"]["vocab"].append(["b", -1.0])
        self.assertNotEqual(before, tokenizer_fingerprint(tokenizer))


if __name__ == "__main__":
    unittest.main()
