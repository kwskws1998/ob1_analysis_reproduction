"""Extract learned widths and define fixed and RMS symmetric controls."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch


STATE_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
LEFT_SUFFIX = "log_sigma_left"
RIGHT_SUFFIX = "log_sigma_right"
FIXED_SYMMETRIC_SIGMA = 1.0


def rms_scale_symmetric_sigma(
    sigma_left: float,
    sigma_right: float,
) -> float:
    """Return the RMS of two positive learned side-scale parameters."""
    values = (float(sigma_left), float(sigma_right))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Learned sigma values must be finite and positive")
    return math.sqrt((values[0] ** 2 + values[1] ** 2) / 2.0)


def sigmas_from_ratio_and_rms_scale(
    right_left_ratio: float,
    rms_side_scale: float,
) -> tuple[float, float]:
    """Recover side scales with a fixed ratio and fixed side-scale RMS."""
    ratio = float(right_left_ratio)
    rms_scale = float(rms_side_scale)
    if (
        not math.isfinite(ratio)
        or ratio <= 0
        or not math.isfinite(rms_scale)
        or rms_scale <= 0
    ):
        raise ValueError("Ratio and RMS side scale must be finite and positive")
    sigma_left = rms_scale * math.sqrt(2.0 / (1.0 + ratio**2))
    sigma_right = ratio * sigma_left
    return sigma_left, sigma_right


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_state_path(checkpoint: Path) -> Path:
    """Resolve a checkpoint file or one unambiguous state file in a directory."""
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    direct = [checkpoint / name for name in STATE_FILENAMES]
    direct = [path for path in direct if path.is_file()]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise ValueError(
            f"Multiple state files in {checkpoint}: {[path.name for path in direct]}"
        )
    recursive = sorted(
        {path for name in STATE_FILENAMES for path in checkpoint.rglob(name)}
    )
    if len(recursive) != 1:
        raise ValueError(
            f"Expected one checkpoint state file under {checkpoint}, "
            f"found {[str(path) for path in recursive]}"
        )
    return recursive[0]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load safetensors or a PyTorch state dictionary on CPU."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    return loaded


def scalar_value(state: dict[str, torch.Tensor], key: str) -> float:
    """Read exactly one finite scalar tensor from a state dictionary."""
    tensor = state[key]
    if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
        raise ValueError(f"Sigma key is not a scalar tensor: {key}")
    value = float(tensor.detach().cpu().item())
    if not math.isfinite(value):
        raise ValueError(f"Sigma key is not finite: {key}={value}")
    return value


def sigma_prefixes(state: dict[str, torch.Tensor]) -> list[str]:
    """Return prefixes containing both left and right learned log-sigmas."""
    left = {key[: -len(LEFT_SUFFIX)] for key in state if key.endswith(LEFT_SUFFIX)}
    right = {key[: -len(RIGHT_SUFFIX)] for key in state if key.endswith(RIGHT_SUFFIX)}
    return sorted(left & right)


def select_sigma_prefix(
    state: dict[str, torch.Tensor],
    requested_prefix: str | None = None,
) -> str:
    """Select one explicit sigma pair without silently merging model copies."""
    prefixes = sigma_prefixes(state)
    if requested_prefix is not None:
        if requested_prefix not in prefixes:
            raise ValueError(
                f"Requested sigma prefix {requested_prefix!r} not found; "
                f"available prefixes: {prefixes}"
            )
        return requested_prefix
    if len(prefixes) == 1:
        return prefixes[0]
    module_copies = [
        prefix for prefix in prefixes if ".modules_to_save.default." in prefix
    ]
    if len(module_copies) == 1:
        return module_copies[0]
    raise ValueError(
        "Checkpoint must contain one unambiguous left/right sigma pair; "
        f"available prefixes: {prefixes}. Pass an explicit prefix."
    )


def extract_sigma_record(
    checkpoint: Path,
    requested_prefix: str | None = None,
    min_sigma: float = 1e-6,
    allow_initial_sigmas: bool = False,
) -> dict:
    """Extract learned log-sigmas and attach the independent fixed control."""
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")
    state_path = resolve_state_path(checkpoint)
    state = load_state_dict(state_path)
    prefix = select_sigma_prefix(state, requested_prefix)
    left_key = f"{prefix}{LEFT_SUFFIX}"
    right_key = f"{prefix}{RIGHT_SUFFIX}"
    log_sigma_left = scalar_value(state, left_key)
    log_sigma_right = scalar_value(state, right_key)
    sigma_left = math.exp(log_sigma_left) + min_sigma
    sigma_right = math.exp(log_sigma_right) + min_sigma
    initial_like = math.isclose(
        sigma_left,
        1.0 + min_sigma,
        rel_tol=0.0,
        abs_tol=1e-7,
    ) and math.isclose(
        sigma_right,
        1.0 + min_sigma,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    if initial_like and not allow_initial_sigmas:
        raise ValueError(
            "Checkpoint contains initial-like 1.0/1.0 widths. "
            "Pass allow_initial_sigmas=True only after confirming this is the "
            "intended trained checkpoint."
        )
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "state_bytes": state_path.stat().st_size,
        "available_sigma_prefixes": sigma_prefixes(state),
        "selected_sigma_prefix": prefix,
        "log_sigma_left_key": left_key,
        "log_sigma_right_key": right_key,
        "log_sigma_left": log_sigma_left,
        "log_sigma_right": log_sigma_right,
        "min_sigma": min_sigma,
        "sigma_left": sigma_left,
        "sigma_right": sigma_right,
        "right_left_ratio": sigma_right / sigma_left,
        "sigma_symmetric": FIXED_SYMMETRIC_SIGMA,
        "sigma_symmetric_fixed": FIXED_SYMMETRIC_SIGMA,
        "sigma_symmetric_rms_scale": rms_scale_symmetric_sigma(
            sigma_left,
            sigma_right,
        ),
        "symmetric_sigma_source": "fixed_independent_control",
        "rms_symmetric_sigma_source": "learned_side_scale_rms",
        "initial_like": initial_like,
    }


def direct_sigma_record(
    checkpoint_id: str,
    sigma_left: float,
    sigma_right: float,
    value_type: str = "effective",
    min_sigma: float = 1e-6,
    allow_initial_sigmas: bool = False,
    source_accuracy: float | None = None,
) -> dict:
    """Build a complete fixed-sigma record without a reward-model checkpoint."""
    if not checkpoint_id:
        raise ValueError("Direct sigma input requires a nonempty checkpoint ID")
    if value_type not in {"effective", "log"}:
        raise ValueError("sigma value type must be 'effective' or 'log'")
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")
    if not math.isfinite(sigma_left) or not math.isfinite(sigma_right):
        raise ValueError("Direct sigma values must be finite")
    if source_accuracy is not None and not math.isfinite(source_accuracy):
        raise ValueError("Sigma source accuracy must be finite")

    if value_type == "effective":
        if sigma_left <= min_sigma or sigma_right <= min_sigma:
            raise ValueError(
                f"Effective sigma values must be greater than min_sigma={min_sigma}"
            )
        effective_left = float(sigma_left)
        effective_right = float(sigma_right)
        log_sigma_left = math.log(effective_left - min_sigma)
        log_sigma_right = math.log(effective_right - min_sigma)
    else:
        log_sigma_left = float(sigma_left)
        log_sigma_right = float(sigma_right)
        effective_left = math.exp(log_sigma_left) + min_sigma
        effective_right = math.exp(log_sigma_right) + min_sigma

    initial_like = math.isclose(
        effective_left,
        1.0 + min_sigma,
        rel_tol=0.0,
        abs_tol=1e-7,
    ) and math.isclose(
        effective_right,
        1.0 + min_sigma,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    if initial_like and not allow_initial_sigmas:
        raise ValueError(
            "Direct input contains initial-like 1.0/1.0 widths. "
            "Use --allow-initial-sigmas only after confirming the values."
        )

    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint": f"direct-sigma:{checkpoint_id}",
        "state_path": None,
        "state_sha256": None,
        "state_bytes": None,
        "available_sigma_prefixes": [],
        "selected_sigma_prefix": None,
        "log_sigma_left_key": None,
        "log_sigma_right_key": None,
        "log_sigma_left": log_sigma_left,
        "log_sigma_right": log_sigma_right,
        "min_sigma": min_sigma,
        "sigma_left": effective_left,
        "sigma_right": effective_right,
        "right_left_ratio": effective_right / effective_left,
        "sigma_symmetric": FIXED_SYMMETRIC_SIGMA,
        "sigma_symmetric_fixed": FIXED_SYMMETRIC_SIGMA,
        "sigma_symmetric_rms_scale": rms_scale_symmetric_sigma(
            effective_left,
            effective_right,
        ),
        "symmetric_sigma_source": "fixed_independent_control",
        "rms_symmetric_sigma_source": "learned_side_scale_rms",
        "initial_like": initial_like,
        "sigma_source": "direct_cli",
        "sigma_input_value_type": value_type,
        "source_accuracy": source_accuracy,
    }
