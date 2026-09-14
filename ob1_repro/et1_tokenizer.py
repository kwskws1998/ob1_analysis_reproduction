from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from transformers import AutoTokenizer


ET1_TOKENIZER_REPO_ID = "t5-small"
ET1_TOKENIZER_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
ET1_TOKENIZER_ENV = "GAZE_REWARD_ET1_TOKENIZER"
ET1_TOKENIZER_CACHE_ENV = "GAZE_REWARD_ET1_TOKENIZER_CACHE"
ET1_TOKENIZER_MODEL_MAX_LENGTH = 2048
ET1_TOKENIZER_SIGNATURE = f"{ET1_TOKENIZER_REPO_ID}@{ET1_TOKENIZER_REVISION}"


def default_et1_tokenizer_cache_dir() -> Path:
    """Return the configured cache directory for the pinned T5 tokenizer."""
    configured_path = os.environ.get(ET1_TOKENIZER_CACHE_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "cache" / "models"


def load_et1_tokenizer(
    tokenizer_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """Load an offline tokenizer override or the pinned T5 tokenizer."""
    configured_path = tokenizer_path or os.environ.get(ET1_TOKENIZER_ENV)
    resolved_cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_et1_tokenizer_cache_dir()
    )
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    if configured_path:
        local_path = Path(configured_path).expanduser().resolve()
        if not local_path.is_dir():
            raise FileNotFoundError(
                "GAZE_REWARD_ET1_TOKENIZER must point to a local "
                f"tokenizer directory: {local_path}"
            )
        return AutoTokenizer.from_pretrained(
            str(local_path),
            local_files_only=True,
            model_max_length=ET1_TOKENIZER_MODEL_MAX_LENGTH,
        )

    return AutoTokenizer.from_pretrained(
        ET1_TOKENIZER_REPO_ID,
        revision=ET1_TOKENIZER_REVISION,
        cache_dir=str(resolved_cache_dir),
        model_max_length=ET1_TOKENIZER_MODEL_MAX_LENGTH,
    )
