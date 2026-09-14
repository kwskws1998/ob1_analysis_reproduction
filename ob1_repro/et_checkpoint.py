from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen

from filelock import FileLock


ET1_CHECKPOINT_FILENAME = "T5-tokenizer-BiLSTM-TRT-12-concat-3"
ET1_SOURCE_REPOSITORY = "huangxt39/SelectiveCacheForLM"
ET1_SOURCE_COMMIT = "eccc93f969745b04ce1e4911d6513d85565cc919"
ET1_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/"
    f"{ET1_SOURCE_REPOSITORY}/{ET1_SOURCE_COMMIT}/"
    f"FPmodels/{ET1_CHECKPOINT_FILENAME}"
)
ET1_CHECKPOINT_SIZE = 70_144_951
ET1_CHECKPOINT_SHA256 = (
    "f872e426de7a2a5473f096f1576df9e912efef71e33c5e4a664cabe84c6baa03"
)
ET1_CHECKPOINT_ENV = "GAZE_REWARD_ET1_CHECKPOINT"


class CheckpointValidationError(RuntimeError):
    """Raised when a downloaded or cached checkpoint fails validation."""


def default_et1_checkpoint_path() -> Path:
    """Return the configured ET1 checkpoint path."""
    configured_path = os.environ.get(ET1_CHECKPOINT_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models" / ET1_CHECKPOINT_FILENAME


def checkpoint_sha256(path: Union[str, Path]) -> str:
    """Compute the SHA-256 digest of a checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_et1_checkpoint(
    path: Union[str, Path],
    expected_size: int = ET1_CHECKPOINT_SIZE,
    expected_sha256: str = ET1_CHECKPOINT_SHA256,
) -> Path:
    """Validate the ET1 checkpoint size and SHA-256 digest."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointValidationError(
            f"ET1 checkpoint does not exist: {checkpoint_path}"
        )

    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected_size:
        raise CheckpointValidationError(
            f"ET1 checkpoint size mismatch: expected {expected_size}, got {actual_size}"
        )

    actual_sha256 = checkpoint_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise CheckpointValidationError(
            "ET1 checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return checkpoint_path


def _checkpoint_lock_path(checkpoint_path: Path) -> Path:
    """Return a process-safe lock path outside the project tree."""
    lock_id = hashlib.sha256(str(checkpoint_path.resolve()).encode()).hexdigest()
    return Path(tempfile.gettempdir()) / f"gaze_reward_et1_{lock_id}.lock"


def _download_et1_checkpoint(
    checkpoint_path: Path,
    request: Request,
    expected_size: int,
    expected_sha256: str,
    timeout_seconds: int,
) -> None:
    """Download, validate, and atomically install one ET1 checkpoint."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=checkpoint_path.parent,
            prefix=f"{checkpoint_path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(request, timeout=timeout_seconds) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    temporary_file.write(chunk)

        validate_et1_checkpoint(
            temporary_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ensure_et1_checkpoint(
    path: Optional[Union[str, Path]] = None,
    url: str = ET1_CHECKPOINT_URL,
    expected_size: int = ET1_CHECKPOINT_SIZE,
    expected_sha256: str = ET1_CHECKPOINT_SHA256,
    timeout_seconds: int = 120,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> Path:
    """Return a validated checkpoint, downloading it atomically when needed."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")

    checkpoint_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else default_et1_checkpoint_path()
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(_checkpoint_lock_path(checkpoint_path)), timeout=600):
        try:
            return validate_et1_checkpoint(
                checkpoint_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except CheckpointValidationError:
            pass

        request = Request(
            url,
            headers={"User-Agent": "gaze-reward-et1-checkpoint-downloader/1.0"},
        )
        for attempt in range(1, max_attempts + 1):
            print(
                "Downloading ET1 checkpoint "
                f"from {url} (attempt {attempt}/{max_attempts})"
            )
            try:
                _download_et1_checkpoint(
                    checkpoint_path=checkpoint_path,
                    request=request,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    timeout_seconds=timeout_seconds,
                )
                break
            except Exception as error:
                if attempt == max_attempts:
                    if isinstance(error, CheckpointValidationError):
                        raise
                    raise RuntimeError(
                        "Failed to download the ET1 checkpoint after "
                        f"{max_attempts} attempts from {url}"
                    ) from error
                delay = retry_delay_seconds * (2 ** (attempt - 1))
                print(f"ET1 checkpoint download failed; retrying in {delay:g} seconds")
                time.sleep(delay)

        print(f"ET1 checkpoint saved to {checkpoint_path}")
        return checkpoint_path
