"""Bounded symmetric metrics for normalized allocation distributions."""

from __future__ import annotations

import math

import numpy as np


def _normalized_probability_vector(
    values: np.ndarray,
    label: str,
) -> np.ndarray:
    """Validate and normalize one finite nonnegative probability vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} distribution must be a nonempty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} distribution contains non-finite values")
    if bool((vector < 0).any()):
        raise ValueError(f"{label} distribution contains negative values")
    total = float(vector.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label} distribution has zero mass")
    return vector / total


def distribution_similarity_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Compute Hellinger, total variation, and overlap on one distribution pair."""
    reference_distribution = _normalized_probability_vector(
        reference,
        "Reference",
    )
    candidate_distribution = _normalized_probability_vector(
        candidate,
        "Candidate",
    )
    if reference_distribution.shape != candidate_distribution.shape:
        raise ValueError(
            "Reference and candidate distributions must have the same shape"
        )

    root_difference = np.sqrt(reference_distribution) - np.sqrt(candidate_distribution)
    hellinger = float(np.linalg.norm(root_difference) / math.sqrt(2.0))
    total_variation = float(
        0.5 * np.abs(reference_distribution - candidate_distribution).sum()
    )
    hellinger = float(np.clip(hellinger, 0.0, 1.0))
    total_variation = float(np.clip(total_variation, 0.0, 1.0))
    overlap = float(np.clip(1.0 - total_variation, 0.0, 1.0))
    return {
        "hellinger_distance": hellinger,
        "total_variation_distance": total_variation,
        "overlap_coefficient": overlap,
    }
