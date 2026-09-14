from __future__ import annotations

import hashlib
import json


def tokenizer_fingerprint(tokenizer) -> str:
    """Return a stable digest of tokenizer behavior relevant to gaze caching."""
    digest = hashlib.sha256()
    tokenizer_type = (
        f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
    )
    digest.update(tokenizer_type.encode("utf-8"))

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        backend_state = backend_tokenizer.to_str()
    else:
        vocabulary = tokenizer.get_vocab()
        backend_state = json.dumps(
            sorted(vocabulary.items(), key=lambda item: (item[1], item[0])),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    digest.update(backend_state.encode("utf-8"))

    metadata = {
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    digest.update(
        json.dumps(
            metadata,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()
