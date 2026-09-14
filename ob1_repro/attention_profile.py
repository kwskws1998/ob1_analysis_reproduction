"""Compare redistribution kernels with OB1's internal attention profile."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance

from .attention_profile_diagnostics import (
    build_sigma_landscape,
    plot_metric_comparison,
    plot_parameter_diagnostics,
    plot_profile_regions,
    plot_sigma_landscape,
    safe_identifier,
    summarize_profile_regions,
)
from .distribution_metrics import (
    distribution_similarity_metrics,
)
from .evaluate import (
    bootstrap_mean,
    paired_sign_flip_pvalue,
)
from .sigmas import (
    FIXED_SYMMETRIC_SIGMA,
    rms_scale_symmetric_sigma,
    sigmas_from_ratio_and_rms_scale,
)


FIXED_PSYCHOPHYSICAL_RATIO = 4.0
PROFILE_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "rms_side_scale_symmetric",
    "fixed_ratio4_same_rms",
    "support_rms_displacement_symmetric",
    "support_rms_displacement_ratio4",
    "support_centered_sd_symmetric",
    "support_centered_sd_ratio4",
    "mirrored_learned",
    "fixed_ob1_gaussian",
    "learned_asymmetric",
)
SUPPORT_RMS_DISPLACEMENT_METHODS = (
    "support_rms_displacement_symmetric",
    "support_rms_displacement_ratio4",
)
SUPPORT_CENTERED_SD_METHODS = (
    "support_centered_sd_symmetric",
    "support_centered_sd_ratio4",
)
SUPPORT_CENTERED_SD_METADATA_COLUMNS = (
    "learned_support_centered_token_sd",
    "support_centered_sd_symmetric_sigma",
    "support_centered_sd_ratio4_sigma_left",
    "support_centered_sd_ratio4_sigma_right",
)
INFERENTIAL_PROFILE_METHODS = tuple(
    method for method in PROFILE_METHODS if method != "fixed_ob1_gaussian"
)
REVIEWER_PLOT_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "support_rms_displacement_symmetric",
    "support_rms_displacement_ratio4",
    "support_centered_sd_symmetric",
    "support_centered_sd_ratio4",
    "mirrored_learned",
    "learned_asymmetric",
)
PROFILE_DISPLAY_NAMES = {
    "raw_delta": (
        "No redistribution (all allocation weight remains at the source token)"
    ),
    "fixed_symmetric_sigma1": (
        "Fixed SymGaussian redistribution (sigma_left=sigma_right=1.0)"
    ),
    "rms_side_scale_symmetric": (
        "Symmetric redistribution at the learned kernel's quadratic side-scale RMS"
    ),
    "fixed_ratio4_same_rms": (
        "Fixed 4:1 Gaussian at the learned kernel's quadratic side-scale RMS"
    ),
    "support_rms_displacement_symmetric": (
        "Symmetric Gaussian matched to the learned kernel's realized RMS "
        "token displacement on the pooled fixation support"
    ),
    "support_rms_displacement_ratio4": ("4:1"),
    "support_centered_sd_symmetric": (
        "Symmetric Gaussian matched to the learned kernel's centered "
        "token-offset SD on the pooled fixation support"
    ),
    "support_centered_sd_ratio4": (
        "4:1 Gaussian matched to the learned kernel's centered token-offset "
        "SD on the pooled fixation support"
    ),
    "mirrored_learned": (
        "Learned side scales exchanged as a parameter-level direction "
        "reversal under the same fixation-conditioned support"
    ),
    "fixed_ob1_gaussian": ("Descriptive Gaussian fitted to the same OB1 profile"),
    "learned_asymmetric": "Learned asymmetric redistribution kernel",
}
PROFILE_SHORT_DISPLAY_NAMES = {
    "ob1_attention_profile": "OB1 attention",
    "raw_delta": "No redistribution",
    "fixed_symmetric_sigma1": "Symmetric σ=1",
    "rms_side_scale_symmetric": "Symmetric, parameter RMS",
    "fixed_ratio4_same_rms": "4:1, parameter RMS",
    "support_rms_displacement_symmetric": "Symmetric, matched spread",
    "support_rms_displacement_ratio4": "4:1",
    "support_centered_sd_symmetric": "Symmetric, matched centered SD",
    "support_centered_sd_ratio4": "4:1, matched centered SD",
    "mirrored_learned": "Mirrored learned",
    "fixed_ob1_gaussian": "OB1 fit (in-sample)",
    "learned_asymmetric": "Learned asymmetric",
}
PROFILE_METRIC_DIRECTIONS = {
    "profile_spearman": "higher",
    "js_divergence": "lower",
    "hellinger_distance": "lower",
    "total_variation_distance": "lower",
    "overlap_coefficient": "higher",
    "token_offset_wasserstein": "lower",
}
PROFILE_INFERENTIAL_METRIC_DIRECTIONS = {
    metric: direction
    for metric, direction in PROFILE_METRIC_DIRECTIONS.items()
    if metric != "overlap_coefficient"
}
PROFILE_COMPARISONS = (
    ("learned_asymmetric", "raw_delta"),
    ("learned_asymmetric", "fixed_symmetric_sigma1"),
    ("learned_asymmetric", "rms_side_scale_symmetric"),
    ("learned_asymmetric", "fixed_ratio4_same_rms"),
    ("learned_asymmetric", "support_rms_displacement_symmetric"),
    ("learned_asymmetric", "support_rms_displacement_ratio4"),
    ("learned_asymmetric", "support_centered_sd_symmetric"),
    ("learned_asymmetric", "support_centered_sd_ratio4"),
    ("learned_asymmetric", "mirrored_learned"),
    ("fixed_ratio4_same_rms", "rms_side_scale_symmetric"),
    (
        "support_rms_displacement_ratio4",
        "support_rms_displacement_symmetric",
    ),
    (
        "support_centered_sd_ratio4",
        "support_centered_sd_symmetric",
    ),
)
OB1_MIN_ATTENTION_WIDTH = 3.0
OB1_MAX_ATTENTION_WIDTH = 5.0
OB1_RESIDUAL_ATTENTION = 0.25
OB1_WINDOW_WORDS_LEFT = 1
OB1_WINDOW_WORDS_RIGHT = 3
CANDIDATE_SUPPORT_POLICIES = (
    "fixation_matched",
    "global",
)
PROFILE_MEAN_CI_SCOPE = (
    "95% percentile passage-bootstrap CI conditional on pooled OB1 "
    "simulations and fixed sigma values"
)
PROFILE_CONTRAST_CI_SCOPE = (
    "95% percentile paired-passage-bootstrap CI conditional on pooled OB1 "
    "simulations and fixed sigma values"
)
FIXATION_MATCHED_ANALYSIS_ESTIMAND = (
    "fixation-context-conditioned correspondence: each candidate "
    "unit-impulse kernel and the OB1 fixation-onset attention component are "
    "normalized on the same exact visible relative-token offsets for every "
    "fixation, then pooled with identical fixation weights; actual "
    "ET1-predicted TRT magnitudes are not used"
)
GLOBAL_SUPPORT_ANALYSIS_ESTIMAND = (
    "legacy intrinsic-kernel sensitivity: each candidate unit-impulse kernel "
    "is normalized once on the global union of observed relative-token "
    "offsets, while OB1 remains fixation-window-conditioned; actual "
    "ET1-predicted TRT magnitudes are not used"
)
PROFILE_ANALYSIS_ESTIMAND = FIXATION_MATCHED_ANALYSIS_ESTIMAND
RIGHTWARD_SHARE_SCOPE = (
    "descriptive point estimate from the fixation-weighted pooled offset "
    "profile under the declared candidate-support policy; no bootstrap CI"
)
PROFILE_SPEARMAN_SCOPE = (
    "Spearman correlation over the passage-specific union of offsets with "
    "positive OB1 or candidate mass; offsets absent from both profiles are "
    "excluded rather than treated as shared tied zeros"
)
PROFILE_DISTRIBUTION_METRIC_SCOPE = (
    "Hellinger and total variation distances are computed on complete "
    "normalized passage-level offset profiles; overlap coefficient and its "
    "intervals are derived exactly as one minus total variation distance"
)


def candidate_support_policy_metadata(policy: str) -> tuple[str, str]:
    """Return the display label and estimand for one support policy."""
    if policy == "fixation_matched":
        return (
            "Primary: candidate normalized on each fixation's exact visible "
            "relative-token support",
            FIXATION_MATCHED_ANALYSIS_ESTIMAND,
        )
    if policy == "global":
        return (
            "Legacy sensitivity: candidate normalized once on global support",
            GLOBAL_SUPPORT_ANALYSIS_ESTIMAND,
        )
    raise ValueError("candidate support policy must be fixation_matched or global")


def ob1_reference_display_name(profile_component: str) -> str:
    """Return an explicit label for the reconstructed OB1 reference."""
    if profile_component == "focused":
        component_label = "focused Gaussian component"
    elif profile_component == "full":
        component_label = "focused-plus-constant-residual profile"
    else:
        raise ValueError("OB1 profile component must be focused or full")
    return (
        f"OB1 fixation-onset {component_label} "
        "projected to relative native T5-token offsets"
    )


def add_attention_skew_provenance(
    table: pd.DataFrame,
    trajectory_attention_skew: float | None,
) -> None:
    """Label trajectory-matched rows and formula-only sensitivities."""
    matches = []
    roles = []
    for requested_skew in table["ob1_attention_skew"]:
        if trajectory_attention_skew is None:
            matches.append(pd.NA)
            roles.append("unverified: trajectory attention_skew unavailable")
        elif math.isclose(
            float(requested_skew),
            trajectory_attention_skew,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            matches.append(True)
            roles.append("trajectory-matched attention-function evaluation")
        else:
            matches.append(False)
            roles.append(
                "formula sensitivity: attention function re-evaluated on "
                "saved trajectory geometry"
            )
    table["requested_skew_matches_trajectory"] = pd.Series(
        matches,
        index=table.index,
        dtype="boolean",
    )
    table["attention_skew_analysis_role"] = roles


def clean_ob1_word(value: str) -> str:
    """Apply the vendored OB1 word normalization."""
    return re.sub(r"[^\w\s]", "", str(value)).lower().strip()


def whitespace_word_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open source-text spans for whitespace-delimited words."""
    return [match.span() for match in re.finditer(r"\S+", str(text))]


def build_t5_letter_geometry(
    passage_text: str,
    passage_tokens: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Map native T5 text tokens into OB1's cleaned-letter coordinates."""
    source_text = str(passage_text)
    word_spans = whitespace_word_spans(source_text)
    clean_words = [clean_ob1_word(source_text[start:end]) for start, end in word_spans]
    if not clean_words or any(not word for word in clean_words):
        raise ValueError("OB1 projection requires nonempty cleaned words")
    word_starts = []
    cursor = 0
    for word in clean_words:
        word_starts.append(cursor)
        cursor += len(word) + 1

    rows = []
    for token in passage_tokens.sort_values("token_index").itertuples():
        word_id = int(token.word_id_zero_based)
        if word_id < 0 or word_id >= len(word_spans):
            raise ValueError("T5 token word ID is outside the passage words")
        word_start, word_end = word_spans[word_id]
        clean_positions = [
            source_index
            for source_index in range(word_start, word_end)
            if re.match(r"\w", source_text[source_index])
        ]
        rank_by_source = {
            source_index: rank for rank, source_index in enumerate(clean_positions)
        }
        token_clean_ranks = [
            rank_by_source[source_index]
            for source_index in range(
                max(int(token.character_start), word_start),
                min(int(token.character_end), word_end),
            )
            if source_index in rank_by_source
        ]
        if token_clean_ranks:
            center_in_word = float(np.mean(token_clean_ranks))
            punctuation_only = False
        else:
            raw_width = max(word_end - word_start, 1)
            raw_center = (
                float(token.character_start) + float(token.character_end)
            ) / 2.0
            relative_center = np.clip(
                (raw_center - word_start) / raw_width,
                0.0,
                1.0,
            )
            center_in_word = relative_center * max(
                len(clean_words[word_id]) - 1,
                0,
            )
            punctuation_only = True
        rows.append(
            {
                "token_index": int(token.token_index),
                "word_id_zero_based": word_id,
                "ob1_letter_center": (float(word_starts[word_id]) + center_in_word),
                "punctuation_only_token": punctuation_only,
            }
        )
    geometry = pd.DataFrame(rows)
    if geometry.empty or geometry["token_index"].duplicated().any():
        raise ValueError("T5 letter geometry must contain unique tokens")
    return geometry, clean_words, word_starts


def effective_ob1_attention_width(
    recorded_width: float,
    saccade_type: object,
) -> float:
    """Recover the width used after OB1's per-fixation width update."""
    width = float(recorded_width)
    if not math.isfinite(width):
        raise ValueError("OB1 attentional width must be finite")
    normalized_type = "" if pd.isna(saccade_type) else str(saccade_type).strip().lower()
    if normalized_type == "regression":
        return max(width - 1.0, OB1_MIN_ATTENTION_WIDTH)
    return min(width + 0.5, OB1_MAX_ATTENTION_WIDTH)


def ob1_attention_weight(
    eccentricity: np.ndarray,
    attention_width: float,
    attention_skew: float,
    profile_component: str = "focused",
) -> np.ndarray:
    """Evaluate the vendored OB1 asymmetric attention equation."""
    eccentricity = np.asarray(eccentricity, dtype=float)
    if attention_width <= 0 or attention_skew < 1:
        raise ValueError("OB1 width must be positive and skew must be at least one")
    sigma = np.where(
        eccentricity < 0,
        attention_width / attention_skew,
        attention_width,
    )
    focused = np.exp(-0.5 * (np.abs(eccentricity) / sigma) ** 2) / attention_width
    if profile_component == "focused":
        return focused
    if profile_component == "full":
        return focused + OB1_RESIDUAL_ATTENTION
    raise ValueError("OB1 profile component must be focused or full")


def unique_et1_token_grid(token_values: pd.DataFrame) -> pd.DataFrame:
    """Validate repeated checkpoint metadata and retain one ET1 token grid."""
    required = {
        "checkpoint_id",
        "passage_id_zero_based",
        "token_index",
        "character_start",
        "character_end",
        "is_special",
        "attention_mask",
        "word_id_zero_based",
    }
    missing = sorted(required - set(token_values.columns))
    if missing:
        raise ValueError(f"ET1 token table is missing columns: {missing}")
    metadata_columns = [
        "passage_id_zero_based",
        "token_index",
        "character_start",
        "character_end",
        "is_special",
        "attention_mask",
        "word_id_zero_based",
    ]
    distinct = token_values[metadata_columns].drop_duplicates()
    duplicate_keys = distinct.duplicated(
        ["passage_id_zero_based", "token_index"],
        keep=False,
    )
    if bool(duplicate_keys.any()):
        raise ValueError("ET1 checkpoint rows disagree on token metadata")
    grid = distinct.loc[
        distinct["attention_mask"].eq(1)
        & distinct["is_special"].eq(0)
        & distinct["word_id_zero_based"].notna()
    ].copy()
    if grid.empty:
        raise ValueError("ET1 token grid contains no evaluable text tokens")
    grid["passage_id_zero_based"] = grid["passage_id_zero_based"].astype(int)
    grid["token_index"] = grid["token_index"].astype(int)
    grid["word_id_zero_based"] = grid["word_id_zero_based"].astype(int)
    return grid.sort_values(["passage_id_zero_based", "token_index"]).reset_index(
        drop=True
    )


def gaussian_weights(
    relative_offsets: np.ndarray,
    sigma_left: float,
    sigma_right: float,
) -> np.ndarray:
    """Return one normalized production-shaped kernel on visible offsets."""
    offsets = np.asarray(relative_offsets, dtype=float)
    if sigma_left <= 0 or sigma_right <= 0:
        raise ValueError("Gaussian sigmas must be positive")
    sigmas = np.where(offsets < 0, sigma_left, sigma_right)
    weights = np.exp(-0.5 * (np.abs(offsets) / sigmas) ** 2)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Gaussian kernel has invalid mass")
    return weights / total


def delta_weights(relative_offsets: np.ndarray) -> np.ndarray:
    """Return a normalized no-redistribution delta on the anchor token."""
    offsets = np.asarray(relative_offsets, dtype=int)
    weights = offsets.eq(0) if isinstance(offsets, pd.Series) else offsets == 0
    weights = np.asarray(weights, dtype=float)
    if int(weights.sum()) != 1:
        raise ValueError("Every fixation geometry must contain one anchor token")
    return weights


def project_ob1_attention_to_t5(
    passages: pd.DataFrame,
    token_values: pd.DataFrame,
    ob1_fixations: pd.DataFrame,
    attention_skews: tuple[float, ...],
    fixation_weighting: str,
    profile_component: str,
) -> tuple[
    dict[float, dict[int, dict[int, float]]],
    np.ndarray,
    dict[int, dict[tuple[int, ...], float]],
    dict,
]:
    """Project OB1 attention and aggregate exact fixation support patterns."""
    if fixation_weighting not in {"duration", "equal"}:
        raise ValueError("fixation weighting must be duration or equal")
    if profile_component not in {"focused", "full"}:
        raise ValueError("OB1 profile component must be focused or full")
    required_passages = {
        "passage_id_zero_based",
        "passage_text",
    }
    required_fixations = {
        "simulation_id",
        "seed",
        "text_id",
        "fixation_counter",
        "word_id",
        "word",
        "fixation_duration",
        "saccade_type",
        "attentional_width",
        "eye_position",
    }
    missing_passages = sorted(required_passages - set(passages.columns))
    missing_fixations = sorted(required_fixations - set(ob1_fixations.columns))
    if missing_passages:
        raise ValueError(f"Passage table is missing columns: {missing_passages}")
    if missing_fixations:
        raise ValueError(f"OB1 fixation table is missing columns: {missing_fixations}")
    skews = tuple(float(value) for value in attention_skews)
    if not skews or len(skews) != len(set(skews)):
        raise ValueError("OB1 attention skews must be nonempty and unique")
    if any(value < 1 for value in skews):
        raise ValueError("OB1 attention skews must be at least one")

    grid = unique_et1_token_grid(token_values)
    passage_lookup = {
        int(row.passage_id_zero_based): str(row.passage_text)
        for row in passages.itertuples()
    }
    fixation_passages = set(ob1_fixations["text_id"].astype(int).unique())
    if fixation_passages != set(passage_lookup):
        raise ValueError(
            "OB1 fixation passages do not match the canonical passage grid"
        )
    if set(grid["passage_id_zero_based"].unique()) != set(passage_lookup):
        raise ValueError("ET1 token passages do not match the canonical passage grid")

    sparse_profiles = {
        skew: {passage_id: defaultdict(float) for passage_id in passage_lookup}
        for skew in skews
    }
    support_pattern_weights = {
        passage_id: defaultdict(float) for passage_id in passage_lookup
    }
    fixation_count = 0
    duration_total = 0.0
    widths_used = []
    characters_per_token_values = []
    visible_token_counts = []
    anchor_token_indices = set()
    punctuation_only_token_count = 0
    minimum_offset = 0
    maximum_offset = 0

    for passage_id in sorted(passage_lookup):
        passage_tokens = grid.loc[grid["passage_id_zero_based"].eq(passage_id)].copy()
        geometry, clean_words, word_starts = build_t5_letter_geometry(
            passage_lookup[passage_id],
            passage_tokens,
        )
        clean_character_count = sum(len(word) for word in clean_words)
        token_count = len(geometry)
        if clean_character_count < 1 or token_count < 1:
            raise ValueError("Passage has no clean characters or T5 tokens")
        characters_per_token = clean_character_count / token_count
        characters_per_token_values.append(characters_per_token)
        punctuation_only_token_count += int(geometry["punctuation_only_token"].sum())
        passage_fixations = ob1_fixations.loc[
            ob1_fixations["text_id"].astype(int).eq(passage_id)
        ].sort_values(["simulation_id", "fixation_counter"])
        if passage_fixations.empty:
            raise ValueError(f"OB1 has no fixations for passage {passage_id}")
        passage_total_weight = 0.0
        for fixation in passage_fixations.itertuples():
            word_id = int(fixation.word_id)
            if word_id < 0 or word_id >= len(clean_words):
                raise ValueError("OB1 fixation word ID is outside the passage")
            if clean_ob1_word(fixation.word) != clean_words[word_id]:
                raise ValueError("OB1 fixation word does not match the passage")
            window_start = max(0, word_id - OB1_WINDOW_WORDS_LEFT)
            window_end = min(
                len(clean_words) - 1,
                word_id + OB1_WINDOW_WORDS_RIGHT,
            )
            visible = geometry.loc[
                geometry["word_id_zero_based"].between(
                    window_start,
                    window_end,
                )
            ].copy()
            anchor_candidates = visible.loc[visible["word_id_zero_based"].eq(word_id)]
            if visible.empty or anchor_candidates.empty:
                raise ValueError("OB1 fixation has no visible or anchor token")
            local_eye_position = float(fixation.eye_position)
            if not math.isfinite(local_eye_position):
                raise ValueError("OB1 eye position must be finite")
            global_eye_position = float(word_starts[window_start]) + local_eye_position
            anchor_distance = (
                anchor_candidates["ob1_letter_center"] - global_eye_position
            ).abs()
            anchor_index = int(
                anchor_candidates.loc[
                    anchor_distance.idxmin(),
                    "token_index",
                ]
            )
            anchor_token_indices.add((passage_id, anchor_index))
            relative_offsets = visible["token_index"].to_numpy(dtype=int) - anchor_index
            eccentricity = (
                visible["ob1_letter_center"].to_numpy(dtype=float) - global_eye_position
            )
            minimum_offset = min(
                minimum_offset,
                int(relative_offsets.min()),
            )
            maximum_offset = max(
                maximum_offset,
                int(relative_offsets.max()),
            )
            visible_token_counts.append(len(visible))
            width = effective_ob1_attention_width(
                fixation.attentional_width,
                fixation.saccade_type,
            )
            fixation_duration = float(fixation.fixation_duration)
            if not math.isfinite(fixation_duration) or fixation_duration <= 0:
                raise ValueError("OB1 fixation duration must be positive")
            fixation_weight = (
                fixation_duration if fixation_weighting == "duration" else 1.0
            )
            support_pattern = tuple(int(offset) for offset in relative_offsets)
            if len(support_pattern) != len(set(support_pattern)):
                raise ValueError("Fixation support contains duplicate token offsets")
            if 0 not in support_pattern:
                raise ValueError("Fixation support does not contain its anchor token")
            support_pattern_weights[passage_id][support_pattern] += fixation_weight
            for skew in skews:
                weights = ob1_attention_weight(
                    eccentricity,
                    width,
                    skew,
                    profile_component,
                )
                weights = weights / weights.sum()
                for offset, value in zip(relative_offsets, weights):
                    sparse_profiles[skew][passage_id][int(offset)] += (
                        fixation_weight * float(value)
                    )
            passage_total_weight += fixation_weight
            fixation_count += 1
            duration_total += fixation_duration
            widths_used.append(width)

        for skew in skews:
            sparse_profiles[skew][passage_id] = {
                offset: value / passage_total_weight
                for offset, value in sparse_profiles[skew][passage_id].items()
            }
    support = np.arange(minimum_offset, maximum_offset + 1, dtype=int)
    normalized_support_patterns = {
        passage_id: {offsets: float(weight) for offsets, weight in patterns.items()}
        for passage_id, patterns in support_pattern_weights.items()
    }
    global_pattern_count = len(
        {
            offsets
            for patterns in normalized_support_patterns.values()
            for offsets in patterns
        }
    )
    reference_profiles = {skew: {} for skew in skews}
    for skew in skews:
        for passage_id, sparse in sparse_profiles[skew].items():
            profile = dictionary_profile(sparse, support)
            reference_profiles[skew][passage_id] = {
                int(offset): float(value) for offset, value in zip(support, profile)
            }
    audit = {
        "fixation_count": int(fixation_count),
        "fixation_duration_total_ms": float(duration_total),
        "fixation_weighting": fixation_weighting,
        "passage_count": int(len(passage_lookup)),
        "simulation_count": int(ob1_fixations["simulation_id"].nunique()),
        "seed_count": int(ob1_fixations["seed"].nunique()),
        "minimum_relative_token_offset": int(minimum_offset),
        "maximum_relative_token_offset": int(maximum_offset),
        "distinct_fixation_support_patterns": int(global_pattern_count),
        "passage_support_pattern_count_sum": int(
            sum(len(patterns) for patterns in normalized_support_patterns.values())
        ),
        "fixation_support_patterns_aggregated": True,
        "minimum_visible_t5_token_count": int(min(visible_token_counts)),
        "maximum_visible_t5_token_count": int(max(visible_token_counts)),
        "mean_visible_t5_token_count": float(np.mean(visible_token_counts)),
        "distinct_anchor_t5_tokens": int(len(anchor_token_indices)),
        "punctuation_only_t5_token_count": int(punctuation_only_token_count),
        "minimum_effective_attention_width": float(min(widths_used)),
        "maximum_effective_attention_width": float(max(widths_used)),
        "minimum_clean_characters_per_t5_token": float(
            min(characters_per_token_values)
        ),
        "maximum_clean_characters_per_t5_token": float(
            max(characters_per_token_values)
        ),
        "mean_clean_characters_per_t5_token": float(
            np.mean(characters_per_token_values)
        ),
        "visible_words_left": OB1_WINDOW_WORDS_LEFT,
        "visible_words_right": OB1_WINDOW_WORDS_RIGHT,
        "residual_attention": OB1_RESIDUAL_ATTENTION,
        "profile_component": profile_component,
        "residual_attention_included": profile_component == "full",
        "special_tokens_included": False,
        "projection_method": (
            "fixation-onset OB1 letter attention evaluated at native T5 "
            "token centers; relative offsets are centered on the nearest "
            "T5 token in the fixated word"
        ),
    }
    return (
        reference_profiles,
        support,
        normalized_support_patterns,
        audit,
    )


def dictionary_profile(
    values: dict[int, float],
    support: np.ndarray,
) -> np.ndarray:
    """Convert one sparse offset profile into a normalized dense array."""
    profile = np.asarray(
        [float(values.get(int(offset), 0.0)) for offset in support],
        dtype=float,
    )
    total = float(profile.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Offset profile has invalid mass")
    return profile / total


def candidate_profile(
    support: np.ndarray,
    method: str,
    sigma_left: float | None = None,
    sigma_right: float | None = None,
) -> np.ndarray:
    """Build one candidate kernel on the declared relative-token support."""
    if method == "raw_delta":
        return delta_weights(support)
    if sigma_left is None or sigma_right is None:
        raise ValueError("Gaussian candidate requires both sigmas")
    return gaussian_weights(support, sigma_left, sigma_right)


def merge_support_pattern_weights(
    passage_patterns: dict[int, dict[tuple[int, ...], float]],
) -> dict[tuple[int, ...], float]:
    """Pool already aggregated fixation-support patterns across passages."""
    merged = defaultdict(float)
    for patterns in passage_patterns.values():
        for offsets, weight in patterns.items():
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Fixation-support pattern weight must be positive")
            merged[offsets] += float(weight)
    if not merged:
        raise ValueError("No fixation-support patterns were collected")
    return dict(merged)


def candidate_profile_on_support_patterns(
    pattern_weights: dict[tuple[int, ...], float],
    support: np.ndarray,
    method: str,
    sigma_left: float | None = None,
    sigma_right: float | None = None,
) -> np.ndarray:
    """Normalize on each exact fixation support and pool identical patterns."""
    sparse = defaultdict(float)
    total_weight = 0.0
    for offsets, pattern_weight in pattern_weights.items():
        weight = float(pattern_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Fixation-support pattern weight must be positive")
        local_offsets = np.asarray(offsets, dtype=int)
        local_profile = candidate_profile(
            local_offsets,
            method,
            sigma_left,
            sigma_right,
        )
        for offset, value in zip(local_offsets, local_profile):
            sparse[int(offset)] += weight * float(value)
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("Candidate fixation-support mass is empty")
    return dictionary_profile(
        {offset: value / total_weight for offset, value in sparse.items()},
        support,
    )


def candidate_profiles_by_passage(
    passage_patterns: dict[int, dict[tuple[int, ...], float]],
    support: np.ndarray,
    method: str,
    sigma_left: float | None,
    sigma_right: float | None,
    candidate_support_policy: str,
) -> dict[int, np.ndarray]:
    """Build passage-specific candidates under the declared support policy."""
    candidate_support_policy_metadata(candidate_support_policy)
    if candidate_support_policy == "global":
        shared = candidate_profile(
            support,
            method,
            sigma_left,
            sigma_right,
        )
        return {passage_id: shared.copy() for passage_id in passage_patterns}
    return {
        passage_id: candidate_profile_on_support_patterns(
            patterns,
            support,
            method,
            sigma_left,
            sigma_right,
        )
        for passage_id, patterns in passage_patterns.items()
    }


def pool_passage_profiles(
    passage_profiles: dict[int, np.ndarray],
    passage_patterns: dict[int, dict[tuple[int, ...], float]],
) -> np.ndarray:
    """Pool passage profiles with their exact total fixation weights."""
    passage_ids = sorted(passage_profiles)
    if passage_ids != sorted(passage_patterns):
        raise ValueError("Passage profiles and support patterns use different passages")
    weights = np.asarray(
        [sum(passage_patterns[passage_id].values()) for passage_id in passage_ids],
        dtype=float,
    )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Passage fixation weights must be positive")
    pooled = np.average(
        np.stack([passage_profiles[passage_id] for passage_id in passage_ids]),
        axis=0,
        weights=weights,
    )
    return pooled / pooled.sum()


def rms_token_displacement(
    profile: np.ndarray,
    support: np.ndarray,
) -> float:
    """Return root mean squared offset from the source token."""
    distribution = np.asarray(profile, dtype=float)
    offsets = np.asarray(support, dtype=float)
    if (
        distribution.ndim != 1
        or offsets.ndim != 1
        or distribution.shape != offsets.shape
        or distribution.size == 0
        or not np.isfinite(distribution).all()
        or not np.isfinite(offsets).all()
        or np.any(distribution < 0)
        or float(distribution.sum()) <= 0
    ):
        raise ValueError(
            "RMS token displacement requires aligned finite nonnegative "
            "profile and support vectors"
        )
    normalized = distribution / distribution.sum()
    return float(np.sqrt(np.sum(normalized * offsets**2)))


def fit_scale_to_rms_token_displacement(
    candidate_builder,
    support: np.ndarray,
    right_left_ratio: float,
    target_rms_displacement: float,
) -> dict:
    """Match realized pooled RMS token displacement at a fixed side ratio."""
    ratio = float(right_left_ratio)
    target = float(target_rms_displacement)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("RMS-displacement match ratio must be positive")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Target RMS token displacement must be finite and positive")
    lower_left = 0.05
    upper_left = min(30.0, 30.0 / ratio)
    if upper_left <= lower_left:
        raise ValueError("RMS-displacement match ratio exceeds scale bounds")

    def objective(log_sigma_left: float) -> float:
        sigma_left = math.exp(float(log_sigma_left))
        sigma_right = ratio * sigma_left
        realized = rms_token_displacement(
            candidate_builder(sigma_left, sigma_right),
            support,
        )
        return float((realized - target) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(math.log(lower_left), math.log(upper_left)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"RMS-displacement scale match failed: {result.message}")
    sigma_left = math.exp(float(result.x))
    sigma_right = ratio * sigma_left
    achieved = rms_token_displacement(
        candidate_builder(sigma_left, sigma_right),
        support,
    )
    if not math.isclose(
        achieved,
        target,
        rel_tol=1e-7,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Requested RMS token displacement is not attainable within "
            "the declared sigma bounds"
        )
    return {
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "right_left_ratio": ratio,
        "target_rms_token_displacement": target,
        "achieved_rms_token_displacement": achieved,
        "absolute_match_error": abs(achieved - target),
        "sigma_bounds": [0.05, 30.0],
        "fit_scope": (
            "pooled fixation-conditioned candidate support; target derived "
            "from the frozen learned kernel, not from OB1 attention weights"
        ),
    }


def centered_token_offset_sd(
    profile: np.ndarray,
    support: np.ndarray,
) -> float:
    """Return the probability-weighted SD around the profile's own mean."""
    distribution = np.asarray(profile, dtype=float)
    offsets = np.asarray(support, dtype=float)
    if (
        distribution.ndim != 1
        or offsets.ndim != 1
        or distribution.shape != offsets.shape
        or distribution.size == 0
        or not np.isfinite(distribution).all()
        or not np.isfinite(offsets).all()
        or np.any(distribution < 0)
        or float(distribution.sum()) <= 0
    ):
        raise ValueError(
            "Centered token-offset SD requires aligned finite nonnegative "
            "profile and support vectors"
        )
    normalized = distribution / distribution.sum()
    mean_offset = float(np.sum(normalized * offsets))
    variance = float(np.sum(normalized * (offsets - mean_offset) ** 2))
    return float(np.sqrt(max(variance, 0.0)))


def fit_scale_to_centered_token_offset_sd(
    candidate_builder,
    support: np.ndarray,
    right_left_ratio: float,
    target_centered_sd: float,
) -> dict:
    """Match pooled centered token-offset SD at a fixed side ratio."""
    ratio = float(right_left_ratio)
    target = float(target_centered_sd)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Centered-SD match ratio must be positive")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Target centered token-offset SD must be finite and positive")
    lower_left = 0.05
    upper_left = min(30.0, 30.0 / ratio)
    if upper_left <= lower_left:
        raise ValueError("Centered-SD match ratio exceeds scale bounds")

    def objective(log_sigma_left: float) -> float:
        sigma_left = math.exp(float(log_sigma_left))
        sigma_right = ratio * sigma_left
        realized = centered_token_offset_sd(
            candidate_builder(sigma_left, sigma_right),
            support,
        )
        return float((realized - target) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(math.log(lower_left), math.log(upper_left)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Centered-SD scale match failed: {result.message}")
    sigma_left = math.exp(float(result.x))
    sigma_right = ratio * sigma_left
    achieved = centered_token_offset_sd(
        candidate_builder(sigma_left, sigma_right),
        support,
    )
    if not math.isclose(
        achieved,
        target,
        rel_tol=1e-7,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Requested centered token-offset SD is not attainable within "
            "the declared sigma bounds"
        )
    return {
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "right_left_ratio": ratio,
        "target_centered_token_offset_sd": target,
        "achieved_centered_token_offset_sd": achieved,
        "absolute_match_error": abs(achieved - target),
        "sigma_bounds": [0.05, 30.0],
        "fit_scope": (
            "pooled fixation-conditioned candidate support; target derived "
            "from the frozen learned kernel, not from OB1 attention weights"
        ),
    }


def fit_ob1_gaussian_prior(
    reference_profile: np.ndarray,
    support: np.ndarray,
    candidate_builder=None,
    candidate_support_policy: str = "global",
) -> dict:
    """Fit two independently bounded Gaussian side scales."""
    reference = np.asarray(reference_profile, dtype=float)
    reference = reference / reference.sum()
    candidate_support_policy_metadata(candidate_support_policy)

    def objective(parameters: np.ndarray) -> float:
        sigma_left = math.exp(float(parameters[0]))
        sigma_right = math.exp(float(parameters[1]))
        candidate = (
            candidate_profile(
                support,
                "fixed_ob1_gaussian",
                float(sigma_left),
                float(sigma_right),
            )
            if candidate_builder is None
            else candidate_builder(
                float(sigma_left),
                float(sigma_right),
            )
        )
        return float(jensenshannon(reference, candidate, base=2.0) ** 2)

    log_bounds = (math.log(0.05), math.log(30.0))
    results = [
        minimize(
            objective,
            x0=np.log(np.asarray(initial, dtype=float)),
            method="L-BFGS-B",
            bounds=[log_bounds, log_bounds],
            options={"maxiter": 500, "ftol": 1e-14},
        )
        for initial in ((1.0, 1.0), (1.0, 4.0), (4.0, 1.0))
    ]
    successful = [
        result for result in results if result.success and np.isfinite(result.fun)
    ]
    if not successful:
        messages = "; ".join(str(result.message) for result in results)
        raise RuntimeError(f"OB1 Gaussian fit failed: {messages}")
    result = min(successful, key=lambda candidate: float(candidate.fun))
    sigma_left = math.exp(float(result.x[0]))
    sigma_right = math.exp(float(result.x[1]))
    return {
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "right_left_ratio": float(sigma_right / sigma_left),
        "fit_js_divergence": float(result.fun),
        "optimizer": "L-BFGS-B",
        "optimizer_iterations": int(result.nit),
        "optimizer_initializations": [
            [1.0, 1.0],
            [1.0, 4.0],
            [4.0, 1.0],
        ],
        "sigma_bounds": [0.05, 30.0],
        "fit_objective": "Jensen-Shannon divergence in T5 token-offset space",
        "candidate_support_policy": candidate_support_policy,
    }


def fit_ob1_gaussian_at_fixed_ratio(
    reference_profile: np.ndarray,
    support: np.ndarray,
    right_left_ratio: float,
    candidate_builder=None,
    candidate_support_policy: str = "global",
) -> dict:
    """Fit one scale while holding the Gaussian right-to-left ratio fixed."""
    reference = np.asarray(reference_profile, dtype=float)
    reference = reference / reference.sum()
    ratio = float(right_left_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Fixed Gaussian ratio must be finite and positive")
    candidate_support_policy_metadata(candidate_support_policy)
    upper_left = min(30.0, 30.0 / ratio)
    if upper_left <= 0.05:
        raise ValueError("Fixed Gaussian ratio is outside supported bounds")

    def objective(parameters: np.ndarray) -> float:
        sigma_left = math.exp(float(parameters[0]))
        sigma_right = ratio * sigma_left
        candidate = (
            candidate_profile(
                support,
                "fixed_ratio_gaussian",
                sigma_left,
                sigma_right,
            )
            if candidate_builder is None
            else candidate_builder(sigma_left, sigma_right)
        )
        return float(jensenshannon(reference, candidate, base=2.0) ** 2)

    initial_left = min(max(1.0 / math.sqrt(ratio), 0.051), upper_left)
    result = minimize(
        objective,
        x0=np.asarray([math.log(initial_left)]),
        method="L-BFGS-B",
        bounds=[(math.log(0.05), math.log(upper_left))],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Fixed-ratio OB1 Gaussian fit failed: {result.message}")
    sigma_left = math.exp(float(result.x[0]))
    sigma_right = ratio * sigma_left
    return {
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "right_left_ratio": ratio,
        "quadratic_side_scale": rms_scale_symmetric_sigma(
            sigma_left,
            sigma_right,
        ),
        "fit_js_divergence": float(result.fun),
        "optimizer": "L-BFGS-B",
        "optimizer_iterations": int(result.nit),
        "fit_objective": (
            "Jensen-Shannon divergence in T5 token-offset space at a "
            "fixed right-to-left sigma ratio"
        ),
        "candidate_support_policy": candidate_support_policy,
    }


def profile_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    """Compute profile shape metrics against projected OB1 attention."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    support = np.asarray(support)
    if (
        reference.ndim != 1
        or candidate.ndim != 1
        or support.ndim != 1
        or reference.shape != candidate.shape
        or reference.shape != support.shape
    ):
        raise ValueError("Reference, candidate, and support must be aligned vectors")
    if not np.isfinite(support).all() or bool(
        (np.diff(support.astype(float)) <= 0).any()
    ):
        raise ValueError("Attention-profile support must be finite and increasing")
    distribution_metrics = distribution_similarity_metrics(
        reference,
        candidate,
    )
    observed = (reference > 0) | (candidate > 0)
    if int(observed.sum()) < 2:
        raise ValueError(
            "Attention-profile Spearman needs at least two observed offsets"
        )
    rho = float(spearmanr(reference[observed], candidate[observed]).statistic)
    if not math.isfinite(rho):
        raise ValueError("Attention-profile Spearman is not finite")
    return {
        "profile_spearman": rho,
        "js_divergence": float(jensenshannon(reference, candidate, base=2.0) ** 2),
        **distribution_metrics,
        "token_offset_wasserstein": float(
            wasserstein_distance(
                support,
                support,
                u_weights=reference,
                v_weights=candidate,
            )
        ),
    }


def gaussian_parameter_status(
    model_id: str,
    fitted_to_ob1: bool,
) -> str:
    """Describe how one Gaussian parameter row was selected."""
    if fitted_to_ob1:
        return "descriptive in-sample fit to the same pooled OB1 profile"
    if model_id == "fixed_ratio4_same_rms":
        return (
            "not fitted to Provo or the current OB1 profile; the 4:1 ratio "
            "is the paper-stated OB1 asymmetry and the quadratic side-scale "
            "RMS comes from the frozen learned kernel"
        )
    if model_id in {
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
    }:
        return (
            "scale selected to match the frozen learned kernel's pooled "
            "realized RMS token displacement on the fixation-conditioned "
            "support; no OB1 attention weights used"
        )
    if model_id in SUPPORT_CENTERED_SD_METHODS:
        return (
            "scale selected to match the frozen learned kernel's pooled "
            "centered token-offset SD on the fixation-conditioned support; "
            "no OB1 attention weights used"
        )
    if model_id == "mirrored_learned":
        return (
            "frozen learned side scales exchanged at the parameter level "
            "under the same fixation-conditioned support"
        )
    if model_id == "learned_fixed":
        return (
            "frozen learned parameters; not fitted to Provo or the current OB1 profile"
        )
    if model_id == "fixed_same_rms_symmetric":
        return (
            "quadratic side-scale RMS derived from the frozen learned kernel; "
            "not fitted to Provo or the current OB1 profile"
        )
    return "fixed a priori without fitting to Provo or the current OB1 profile"


def gaussian_parameter_diagnostics(
    reference_profile: np.ndarray,
    support: np.ndarray,
    candidate_builder,
    checkpoint_record: dict,
    ob1_attention_skew: float,
    unconstrained_fit: dict,
    candidate_support_policy: str,
    include_support_rms_displacement_controls: bool = True,
    include_support_centered_sd_controls: bool = False,
) -> pd.DataFrame:
    """Compare fixed controls with constrained and unconstrained OB1 fits."""
    learned_left = float(checkpoint_record["learned_sigma_left"])
    learned_right = float(checkpoint_record["learned_sigma_right"])
    learned_ratio = learned_right / learned_left
    rms_scale = float(checkpoint_record["rms_side_scale_symmetric_sigma"])
    ratio4_left, ratio4_right = sigmas_from_ratio_and_rms_scale(
        FIXED_PSYCHOPHYSICAL_RATIO,
        rms_scale,
    )
    fixed_specs = [
        (
            "fixed_symmetric_sigma1",
            "Symmetric σ=1",
            FIXED_SYMMETRIC_SIGMA,
            FIXED_SYMMETRIC_SIGMA,
            False,
            0,
            "s",
            55,
            "white",
            "black",
        ),
        (
            "fixed_same_rms_symmetric",
            "Symmetric, parameter RMS",
            rms_scale,
            rms_scale,
            False,
            0,
            "o",
            48,
            "#56B4E9",
            "black",
        ),
        (
            "fixed_ratio4_same_rms",
            "4:1, parameter RMS",
            ratio4_left,
            ratio4_right,
            False,
            0,
            "D",
            52,
            "#E69F00",
            "black",
        ),
        (
            "learned_fixed",
            "Learned",
            learned_left,
            learned_right,
            False,
            0,
            "*",
            90,
            "#D55E00",
            "black",
        ),
        (
            "mirrored_learned",
            "Mirrored learned",
            learned_right,
            learned_left,
            False,
            0,
            "<",
            62,
            "#0072B2",
            "black",
        ),
    ]
    if include_support_rms_displacement_controls:
        fixed_specs[3:3] = [
            (
                "support_rms_displacement_symmetric",
                "Symmetric, matched spread",
                checkpoint_record["support_rms_displacement_symmetric_sigma"],
                checkpoint_record["support_rms_displacement_symmetric_sigma"],
                False,
                1,
                "v",
                54,
                "#88CCEE",
                "black",
            ),
            (
                "support_rms_displacement_ratio4",
                "4:1",
                checkpoint_record["support_rms_displacement_ratio4_sigma_left"],
                checkpoint_record["support_rms_displacement_ratio4_sigma_right"],
                False,
                1,
                "^",
                54,
                "#AA4499",
                "black",
            ),
        ]
    if include_support_centered_sd_controls:
        insert_at = 5 if include_support_rms_displacement_controls else 3
        fixed_specs[insert_at:insert_at] = [
            (
                "support_centered_sd_symmetric",
                "Symmetric, matched centered SD",
                checkpoint_record["support_centered_sd_symmetric_sigma"],
                checkpoint_record["support_centered_sd_symmetric_sigma"],
                False,
                1,
                "h",
                58,
                "#44AA99",
                "black",
            ),
            (
                "support_centered_sd_ratio4",
                "4:1, matched centered SD",
                checkpoint_record["support_centered_sd_ratio4_sigma_left"],
                checkpoint_record["support_centered_sd_ratio4_sigma_right"],
                False,
                1,
                ">",
                58,
                "#CC6677",
                "black",
            ),
        ]
    fitted_specs = []
    for model_id, label, ratio in (
        ("ob1_fit_symmetric", "OB1 fit, ratio 1", 1.0),
        ("ob1_fit_ratio3", "OB1 fit, ratio 3", 3.0),
        ("ob1_fit_ratio4", "OB1 fit, ratio 4", 4.0),
        (
            "ob1_fit_learned_ratio",
            "OB1 fit, learned ratio",
            learned_ratio,
        ),
    ):
        fitted = fit_ob1_gaussian_at_fixed_ratio(
            reference_profile,
            support,
            ratio,
            candidate_builder=candidate_builder,
            candidate_support_policy=candidate_support_policy,
        )
        marker = "P" if model_id == "ob1_fit_ratio4" else np.nan
        fitted_specs.append(
            (
                model_id,
                label,
                fitted["sigma_left"],
                fitted["sigma_right"],
                True,
                1,
                marker,
                58,
                "#009E73",
                "black",
            )
        )
    fitted_specs.append(
        (
            "ob1_fit_unconstrained",
            "OB1 fit, free ratio (bounded)",
            float(unconstrained_fit["sigma_left"]),
            float(unconstrained_fit["sigma_right"]),
            True,
            2,
            "X",
            68,
            "#CC79A7",
            "black",
        )
    )
    offsets = np.asarray(support, dtype=int)
    records = []
    for (
        model_id,
        label,
        sigma_left,
        sigma_right,
        fitted_to_ob1,
        fit_parameters,
        marker,
        marker_size,
        facecolor,
        edgecolor,
    ) in (*fixed_specs, *fitted_specs):
        candidate = candidate_builder(sigma_left, sigma_right)
        left_mass = float(candidate[offsets < 0].sum())
        center_mass = float(candidate[offsets == 0].sum())
        right_mass = float(candidate[offsets > 0].sum())
        record = {
            "checkpoint_id": checkpoint_record["checkpoint_id"],
            "ob1_attention_skew": float(ob1_attention_skew),
            "model_id": model_id,
            "short_label": label,
            "sigma_left": float(sigma_left),
            "sigma_right": float(sigma_right),
            "right_left_ratio": float(sigma_right / sigma_left),
            "quadratic_side_scale": rms_scale_symmetric_sigma(
                sigma_left,
                sigma_right,
            ),
            "realized_rms_token_displacement": rms_token_displacement(
                candidate,
                offsets,
            ),
            "realized_mean_token_offset": float(np.sum(candidate * offsets)),
            "realized_centered_token_offset_sd": (
                centered_token_offset_sd(candidate, offsets)
            ),
            "fitted_to_same_ob1_profile": fitted_to_ob1,
            "fit_parameter_count": fit_parameters,
            "parameter_status": gaussian_parameter_status(
                model_id,
                fitted_to_ob1,
            ),
            **profile_metrics(
                reference_profile,
                candidate,
                offsets,
            ),
            "left_mass": left_mass,
            "center_mass": center_mass,
            "right_mass": right_mass,
            "distant_right_mass": float(candidate[offsets >= 2].sum()),
            "right_share_of_noncenter_mass": (
                right_mass / (left_mass + right_mass)
                if left_mass + right_mass > 0
                else np.nan
            ),
            "metric_scope": (
                "descriptive fixation-weighted pooled profile; no passage bootstrap"
            ),
            "landscape_marker": marker,
            "landscape_marker_size": marker_size,
            "landscape_facecolor": facecolor,
            "landscape_edgecolor": edgecolor,
        }
        records.append(record)
    return pd.DataFrame(records)


def summarize_profile_metrics(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap method means separately for every learned sigma record."""
    rng = np.random.default_rng(seed)
    records = []
    for (checkpoint_id, skew, method), group in passage_metrics.groupby(
        ["checkpoint_id", "ob1_attention_skew", "method"],
        sort=True,
    ):
        record = {
            "checkpoint_id": checkpoint_id,
            "ob1_attention_skew": float(skew),
            "method": method,
            "display_name": PROFILE_DISPLAY_NAMES[method],
            "passages": int(len(group)),
        }
        for column in (
            "source_accuracy",
            "learned_sigma_left",
            "learned_sigma_right",
            "learned_right_left_ratio",
            "fixed_symmetric_sigma",
            "rms_side_scale_symmetric_sigma",
            "fixed_ratio4_same_rms_sigma_left",
            "fixed_ratio4_same_rms_sigma_right",
            "learned_support_rms_token_displacement",
            "support_rms_displacement_symmetric_sigma",
            "support_rms_displacement_ratio4_sigma_left",
            "support_rms_displacement_ratio4_sigma_right",
            *SUPPORT_CENTERED_SD_METADATA_COLUMNS,
        ):
            if column in group:
                values = group[column].drop_duplicates()
                if len(values) != 1:
                    raise ValueError(f"{column} is not constant within {checkpoint_id}")
                record[column] = values.iloc[0]
        for metric in PROFILE_INFERENTIAL_METRIC_DIRECTIONS:
            mean, low, high = bootstrap_mean(
                group[metric].to_numpy(),
                bootstrap_samples,
                rng,
            )
            record[metric] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        record["overlap_coefficient"] = 1.0 - record["total_variation_distance"]
        record["overlap_coefficient_ci_low"] = (
            1.0 - record["total_variation_distance_ci_high"]
        )
        record["overlap_coefficient_ci_high"] = (
            1.0 - record["total_variation_distance_ci_low"]
        )
        records.append(record)
    return pd.DataFrame(records)


def profile_paired_contrasts(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    comparisons: tuple[tuple[str, str], ...] = PROFILE_COMPARISONS,
) -> pd.DataFrame:
    """Compute paired passage-level contrasts for every sigma record."""
    bootstrap_rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed)
    records = []
    for (checkpoint_id, skew), skew_metrics in passage_metrics.groupby(
        ["checkpoint_id", "ob1_attention_skew"],
        sort=True,
    ):
        metadata = {
            "checkpoint_id": checkpoint_id,
            "ob1_attention_skew": float(skew),
        }
        for column in (
            "source_accuracy",
            "learned_sigma_left",
            "learned_sigma_right",
            "learned_right_left_ratio",
            "fixed_symmetric_sigma",
            "rms_side_scale_symmetric_sigma",
            "fixed_ratio4_same_rms_sigma_left",
            "fixed_ratio4_same_rms_sigma_right",
            "learned_support_rms_token_displacement",
            "support_rms_displacement_symmetric_sigma",
            "support_rms_displacement_ratio4_sigma_left",
            "support_rms_displacement_ratio4_sigma_right",
            *SUPPORT_CENTERED_SD_METADATA_COLUMNS,
        ):
            if column in skew_metrics:
                values = skew_metrics[column].drop_duplicates()
                if len(values) != 1:
                    raise ValueError(f"{column} is not constant within {checkpoint_id}")
                metadata[column] = values.iloc[0]
        for candidate_name, baseline_name in comparisons:
            candidate = skew_metrics.loc[
                skew_metrics["method"].eq(candidate_name)
            ].set_index("passage_id_zero_based")
            baseline = skew_metrics.loc[
                skew_metrics["method"].eq(baseline_name)
            ].set_index("passage_id_zero_based")
            common = candidate.index.intersection(baseline.index)
            if len(common) < 2:
                raise ValueError("Profile contrast needs at least two passages")
            for metric, direction in PROFILE_INFERENTIAL_METRIC_DIRECTIONS.items():
                if direction == "higher":
                    differences = (
                        candidate.loc[common, metric] - baseline.loc[common, metric]
                    ).to_numpy()
                else:
                    differences = (
                        baseline.loc[common, metric] - candidate.loc[common, metric]
                    ).to_numpy()
                mean, low, high = bootstrap_mean(
                    differences,
                    bootstrap_samples,
                    bootstrap_rng,
                )
                p_value = paired_sign_flip_pvalue(
                    differences,
                    bootstrap_samples,
                    permutation_rng,
                )
                record = {
                    **metadata,
                    "candidate": candidate_name,
                    "baseline": baseline_name,
                    "metric": metric,
                    "positive_means_improvement": True,
                    "passages": int(len(common)),
                    "mean_paired_improvement": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "permutation_p_two_sided": p_value,
                }
                records.append(record)
                if metric == "total_variation_distance":
                    overlap_record = dict(record)
                    overlap_record["metric"] = "overlap_coefficient"
                    records.append(overlap_record)
    return pd.DataFrame(records)


def compare_attention_profiles(
    passages: pd.DataFrame,
    token_values: pd.DataFrame,
    ob1_fixations: pd.DataFrame,
    sigma_records: dict | list[dict],
    attention_skews: tuple[float, ...] = (3.0, 4.0),
    fixation_weighting: str = "duration",
    profile_component: str = "focused",
    bootstrap_samples: int = 10000,
    seed: int = 20260725,
    trajectory_attention_skew: float | None = None,
    candidate_support_policy: str = "fixation_matched",
    landscape_sigma_values: np.ndarray | None = None,
    skip_support_rms_displacement_controls: bool = False,
    include_support_centered_sd_controls: bool = False,
) -> dict:
    """Compare allocation kernels with a projected OB1 attention component."""
    support_policy_label, analysis_estimand = candidate_support_policy_metadata(
        candidate_support_policy
    )
    active_profile_methods = tuple(
        method
        for method in PROFILE_METHODS
        if not (
            skip_support_rms_displacement_controls
            and method in SUPPORT_RMS_DISPLACEMENT_METHODS
        )
        and not (
            not include_support_centered_sd_controls
            and method in SUPPORT_CENTERED_SD_METHODS
        )
    )
    active_inferential_methods = tuple(
        method for method in active_profile_methods if method != "fixed_ob1_gaussian"
    )
    active_comparisons = tuple(
        (candidate, baseline)
        for candidate, baseline in PROFILE_COMPARISONS
        if candidate in active_inferential_methods
        and baseline in active_inferential_methods
    )
    if isinstance(sigma_records, dict):
        sigma_records = [sigma_records]
    if not sigma_records:
        raise ValueError("At least one sigma record is required")
    required_sigma_keys = {
        "checkpoint_id",
        "sigma_left",
        "sigma_right",
    }
    normalized_records = []
    for sigma_record in sigma_records:
        missing_sigma = sorted(required_sigma_keys - set(sigma_record))
        if missing_sigma:
            raise ValueError(f"Sigma record is missing keys: {missing_sigma}")
        rms_symmetric_sigma = rms_scale_symmetric_sigma(
            float(sigma_record["sigma_left"]),
            float(sigma_record["sigma_right"]),
        )
        ratio4_left, ratio4_right = sigmas_from_ratio_and_rms_scale(
            FIXED_PSYCHOPHYSICAL_RATIO,
            rms_symmetric_sigma,
        )
        normalized_record = {
            "checkpoint_id": str(sigma_record["checkpoint_id"]),
            "source_accuracy": sigma_record.get("source_accuracy"),
            "learned_sigma_left": float(sigma_record["sigma_left"]),
            "learned_sigma_right": float(sigma_record["sigma_right"]),
            "fixed_symmetric_sigma": FIXED_SYMMETRIC_SIGMA,
            "rms_side_scale_symmetric_sigma": rms_symmetric_sigma,
            "fixed_ratio4_same_rms_sigma_left": ratio4_left,
            "fixed_ratio4_same_rms_sigma_right": ratio4_right,
        }
        if (
            normalized_record["learned_sigma_left"] <= 0
            or normalized_record["learned_sigma_right"] <= 0
            or normalized_record["fixed_symmetric_sigma"] <= 0
            or normalized_record["rms_side_scale_symmetric_sigma"] <= 0
            or normalized_record["fixed_ratio4_same_rms_sigma_left"] <= 0
            or normalized_record["fixed_ratio4_same_rms_sigma_right"] <= 0
        ):
            raise ValueError("All effective sigma values must be positive")
        normalized_record["learned_right_left_ratio"] = (
            normalized_record["learned_sigma_right"]
            / normalized_record["learned_sigma_left"]
        )
        normalized_records.append(normalized_record)
    checkpoint_ids = [record["checkpoint_id"] for record in normalized_records]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Sigma record checkpoint IDs must be unique")
    references, support, support_patterns, collection_audit = (
        project_ob1_attention_to_t5(
            passages,
            token_values,
            ob1_fixations,
            attention_skews,
            fixation_weighting,
            profile_component,
        )
    )
    profile_rows = []
    metric_rows = []
    fixed_prior_records = []
    reference_global_profiles = {}
    parameter_diagnostic_frames = []
    global_support_patterns = merge_support_pattern_weights(support_patterns)
    if candidate_support_policy == "fixation_matched":

        def pooled_candidate_builder(
            sigma_left: float,
            sigma_right: float,
        ) -> np.ndarray:
            """Build a Gaussian on every pooled fixation-visible support."""
            return candidate_profile_on_support_patterns(
                global_support_patterns,
                support,
                "diagnostic_gaussian",
                sigma_left,
                sigma_right,
            )

    else:

        def pooled_candidate_builder(
            sigma_left: float,
            sigma_right: float,
        ) -> np.ndarray:
            """Build a Gaussian once on the global offset support."""
            return candidate_profile(
                support,
                "diagnostic_gaussian",
                sigma_left,
                sigma_right,
            )

    support_spread_matches = {}
    for sigma_record in normalized_records:
        if skip_support_rms_displacement_controls:
            sigma_record.update(
                {
                    "learned_support_rms_token_displacement": None,
                    "support_rms_displacement_symmetric_sigma": None,
                    "support_rms_displacement_ratio4_sigma_left": None,
                    "support_rms_displacement_ratio4_sigma_right": None,
                }
            )
            continue
        checkpoint_id = sigma_record["checkpoint_id"]
        learned_profile = pooled_candidate_builder(
            sigma_record["learned_sigma_left"],
            sigma_record["learned_sigma_right"],
        )
        learned_rms_displacement = rms_token_displacement(
            learned_profile,
            support,
        )
        symmetric_match = fit_scale_to_rms_token_displacement(
            pooled_candidate_builder,
            support,
            1.0,
            learned_rms_displacement,
        )
        ratio4_match = fit_scale_to_rms_token_displacement(
            pooled_candidate_builder,
            support,
            FIXED_PSYCHOPHYSICAL_RATIO,
            learned_rms_displacement,
        )
        sigma_record.update(
            {
                "learned_support_rms_token_displacement": (learned_rms_displacement),
                "support_rms_displacement_symmetric_sigma": (
                    symmetric_match["sigma_left"]
                ),
                "support_rms_displacement_ratio4_sigma_left": (
                    ratio4_match["sigma_left"]
                ),
                "support_rms_displacement_ratio4_sigma_right": (
                    ratio4_match["sigma_right"]
                ),
            }
        )
        support_spread_matches[checkpoint_id] = {
            "target_source": "frozen learned redistribution kernel",
            "target_profile_uses_ob1_attention_weights": False,
            "target_profile_uses_fixation_visible_support": True,
            "symmetric_ratio1": symmetric_match,
            "fixed_ratio4": ratio4_match,
        }

    support_centered_sd_matches = {}
    if include_support_centered_sd_controls:
        for sigma_record in normalized_records:
            checkpoint_id = sigma_record["checkpoint_id"]
            learned_profile = pooled_candidate_builder(
                sigma_record["learned_sigma_left"],
                sigma_record["learned_sigma_right"],
            )
            learned_centered_sd = centered_token_offset_sd(
                learned_profile,
                support,
            )
            symmetric_match = fit_scale_to_centered_token_offset_sd(
                pooled_candidate_builder,
                support,
                1.0,
                learned_centered_sd,
            )
            ratio4_match = fit_scale_to_centered_token_offset_sd(
                pooled_candidate_builder,
                support,
                FIXED_PSYCHOPHYSICAL_RATIO,
                learned_centered_sd,
            )
            sigma_record.update(
                {
                    "learned_support_centered_token_sd": (learned_centered_sd),
                    "support_centered_sd_symmetric_sigma": (
                        symmetric_match["sigma_left"]
                    ),
                    "support_centered_sd_ratio4_sigma_left": (
                        ratio4_match["sigma_left"]
                    ),
                    "support_centered_sd_ratio4_sigma_right": (
                        ratio4_match["sigma_right"]
                    ),
                }
            )
            support_centered_sd_matches[checkpoint_id] = {
                "target_source": "frozen learned redistribution kernel",
                "target_profile_uses_ob1_attention_weights": False,
                "target_profile_uses_fixation_visible_support": True,
                "symmetric_ratio1": symmetric_match,
                "fixed_ratio4": ratio4_match,
            }

    candidate_cache = {}
    for sigma_record in normalized_records:
        checkpoint_id = sigma_record["checkpoint_id"]
        fixed_symmetric_sigma = sigma_record["fixed_symmetric_sigma"]
        rms_symmetric_sigma = sigma_record["rms_side_scale_symmetric_sigma"]
        method_sigmas = {
            "raw_delta": (None, None),
            "fixed_symmetric_sigma1": (
                fixed_symmetric_sigma,
                fixed_symmetric_sigma,
            ),
            "rms_side_scale_symmetric": (
                rms_symmetric_sigma,
                rms_symmetric_sigma,
            ),
            "fixed_ratio4_same_rms": (
                sigma_record["fixed_ratio4_same_rms_sigma_left"],
                sigma_record["fixed_ratio4_same_rms_sigma_right"],
            ),
            "support_rms_displacement_symmetric": (
                sigma_record["support_rms_displacement_symmetric_sigma"],
                sigma_record["support_rms_displacement_symmetric_sigma"],
            ),
            "support_rms_displacement_ratio4": (
                sigma_record["support_rms_displacement_ratio4_sigma_left"],
                sigma_record["support_rms_displacement_ratio4_sigma_right"],
            ),
            "mirrored_learned": (
                sigma_record["learned_sigma_right"],
                sigma_record["learned_sigma_left"],
            ),
            "learned_asymmetric": (
                sigma_record["learned_sigma_left"],
                sigma_record["learned_sigma_right"],
            ),
        }
        if skip_support_rms_displacement_controls:
            for method in SUPPORT_RMS_DISPLACEMENT_METHODS:
                method_sigmas.pop(method)
        if include_support_centered_sd_controls:
            method_sigmas.update(
                {
                    "support_centered_sd_symmetric": (
                        sigma_record["support_centered_sd_symmetric_sigma"],
                        sigma_record["support_centered_sd_symmetric_sigma"],
                    ),
                    "support_centered_sd_ratio4": (
                        sigma_record["support_centered_sd_ratio4_sigma_left"],
                        sigma_record["support_centered_sd_ratio4_sigma_right"],
                    ),
                }
            )
        candidate_cache[checkpoint_id] = {}
        for method, (left, right) in method_sigmas.items():
            passage_candidates = candidate_profiles_by_passage(
                support_patterns,
                support,
                method,
                left,
                right,
                candidate_support_policy,
            )
            candidate_cache[checkpoint_id][method] = {
                "passages": passage_candidates,
                "global": pool_passage_profiles(
                    passage_candidates,
                    support_patterns,
                ),
            }

    for skew in sorted(references):
        passage_references = {
            passage_id: dictionary_profile(values, support)
            for passage_id, values in references[skew].items()
        }
        reference_global = pool_passage_profiles(
            passage_references,
            support_patterns,
        )
        reference_global_profiles[float(skew)] = reference_global
        fixed_prior = fit_ob1_gaussian_prior(
            reference_global,
            support,
            candidate_builder=pooled_candidate_builder,
            candidate_support_policy=candidate_support_policy,
        )
        fixed_passage_candidates = candidate_profiles_by_passage(
            support_patterns,
            support,
            "fixed_ob1_gaussian",
            fixed_prior["sigma_left"],
            fixed_prior["sigma_right"],
            candidate_support_policy,
        )
        fixed_global_candidate = pool_passage_profiles(
            fixed_passage_candidates,
            support_patterns,
        )
        fixed_prior_records.append(
            {
                "ob1_attention_skew": float(skew),
                "profile_component": profile_component,
                "fixation_weighting": fixation_weighting,
                "projection_coordinate": "relative_native_t5_token_index",
                "candidate_support_policy_label": support_policy_label,
                **fixed_prior,
            }
        )
        for sigma_record in normalized_records:
            parameter_diagnostic_frames.append(
                gaussian_parameter_diagnostics(
                    reference_global,
                    support,
                    pooled_candidate_builder,
                    sigma_record,
                    float(skew),
                    fixed_prior,
                    candidate_support_policy,
                    include_support_rms_displacement_controls=(
                        not skip_support_rms_displacement_controls
                    ),
                    include_support_centered_sd_controls=(
                        include_support_centered_sd_controls
                    ),
                )
            )
        for sigma_record in normalized_records:
            checkpoint_candidates = candidate_cache[sigma_record["checkpoint_id"]]
            method_candidates = {
                **checkpoint_candidates,
                "fixed_ob1_gaussian": {
                    "passages": fixed_passage_candidates,
                    "global": fixed_global_candidate,
                },
            }
            for index, offset in enumerate(support):
                profile_rows.append(
                    {
                        **sigma_record,
                        "ob1_attention_skew": float(skew),
                        "relative_t5_token_offset": int(offset),
                        "ob1_attention_profile": float(reference_global[index]),
                        **{
                            method: float(candidates["global"][index])
                            for method, candidates in method_candidates.items()
                        },
                    }
                )
            for passage_id, reference in passage_references.items():
                for method, candidates in method_candidates.items():
                    if method not in active_inferential_methods:
                        continue
                    candidate = candidates["passages"][passage_id]
                    metric_rows.append(
                        {
                            **sigma_record,
                            "ob1_attention_skew": float(skew),
                            "passage_id_zero_based": int(passage_id),
                            "method": method,
                            **profile_metrics(
                                reference,
                                candidate,
                                support,
                            ),
                        }
                    )

    parameter_diagnostics = pd.concat(
        parameter_diagnostic_frames,
        ignore_index=True,
    )
    sigma_landscape = None
    if landscape_sigma_values is not None:
        sigma_landscape = build_sigma_landscape(
            reference_global_profiles,
            support,
            pooled_candidate_builder,
            np.asarray(landscape_sigma_values, dtype=float),
            profile_metrics,
        )
    passage_metrics = pd.DataFrame(metric_rows)
    reference_profile = ob1_reference_display_name(profile_component)
    passage_metrics["reference_profile"] = reference_profile
    passage_metrics["analysis_estimand"] = analysis_estimand
    passage_metrics["actual_et1_trt_magnitudes_used"] = False
    passage_metrics["candidate_support_policy"] = candidate_support_policy
    passage_metrics["candidate_support_policy_label"] = support_policy_label
    result_table = summarize_profile_metrics(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    result_table["reference_profile"] = reference_profile
    result_table["analysis_estimand"] = analysis_estimand
    result_table["actual_et1_trt_magnitudes_used"] = False
    result_table["candidate_support_policy"] = candidate_support_policy
    result_table["candidate_support_policy_label"] = support_policy_label
    result_table["ci_scope"] = PROFILE_MEAN_CI_SCOPE
    contrasts = profile_paired_contrasts(
        passage_metrics,
        bootstrap_samples,
        seed,
        active_comparisons,
    )
    contrasts["candidate_display_name"] = contrasts["candidate"].map(
        PROFILE_DISPLAY_NAMES
    )
    contrasts["baseline_display_name"] = contrasts["baseline"].map(
        PROFILE_DISPLAY_NAMES
    )
    contrasts["reference_profile"] = reference_profile
    contrasts["analysis_estimand"] = analysis_estimand
    contrasts["actual_et1_trt_magnitudes_used"] = False
    contrasts["candidate_support_policy"] = candidate_support_policy
    contrasts["candidate_support_policy_label"] = support_policy_label
    contrasts["ci_scope"] = PROFILE_CONTRAST_CI_SCOPE
    trajectory_skew = (
        None if trajectory_attention_skew is None else float(trajectory_attention_skew)
    )
    skew_policy = (
        "trajectory attention skew was not available from provenance; "
        "requested skews reweight the saved fixation geometries without "
        "rerunning OB1"
        if trajectory_skew is None
        else (
            "requested skews reweight saved fixation geometries without "
            f"rerunning OB1; the saved trajectory used attention_skew="
            f"{trajectory_skew:g}"
        )
    )
    for table in (passage_metrics, result_table, contrasts):
        table["profile_component"] = profile_component
        table["fixation_weighting"] = fixation_weighting
        table["trajectory_attention_skew"] = trajectory_skew
        table["profile_spearman_scope"] = PROFILE_SPEARMAN_SCOPE
        table["distribution_metric_scope"] = PROFILE_DISTRIBUTION_METRIC_SCOPE
        add_attention_skew_provenance(table, trajectory_skew)
    audit = {
        **collection_audit,
        "checkpoint_count": len(normalized_records),
        "learned_sigma_records": normalized_records,
        "attention_skews": [float(value) for value in sorted(references)],
        "trajectory_attention_skew": trajectory_skew,
        "attention_skew_trajectory_policy": skew_policy,
        "attention_skew_analysis_roles": {
            f"{float(value):g}": (
                "unverified: trajectory attention_skew unavailable"
                if trajectory_skew is None
                else (
                    "trajectory-matched attention-function evaluation"
                    if math.isclose(
                        float(value),
                        trajectory_skew,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    else (
                        "formula sensitivity: attention function "
                        "re-evaluated on saved trajectory geometry"
                    )
                )
            )
            for value in sorted(references)
        },
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "ci_level": 0.95,
        "method_mean_ci": ("percentile bootstrap over passage-level metrics"),
        "method_contrast_ci": (
            "percentile bootstrap over paired passage-level differences"
        ),
        "ci_resampling_unit": "passage",
        "ob1_simulations_pooled_before_bootstrap": True,
        "ob1_simulation_ids_resampled": False,
        "sigma_values_treated_as_fixed": True,
        "rightward_share_scope": RIGHTWARD_SHARE_SCOPE,
        "profile_spearman_scope": PROFILE_SPEARMAN_SCOPE,
        "distribution_metric_scope": PROFILE_DISTRIBUTION_METRIC_SCOPE,
        "candidate_support_policy": candidate_support_policy,
        "candidate_support_policy_label": support_policy_label,
        "candidate_support_policy_is_primary": (
            candidate_support_policy == "fixation_matched"
        ),
        "candidate_support_policy_legacy_sensitivity": (
            candidate_support_policy == "global"
        ),
        "reference_coordinate": "OB1 cleaned-letter coordinates",
        "comparison_coordinate": "relative native T5 token index",
        "reference_profile": reference_profile,
        "reference_profile_exclusions": (
            "acuity, within-fixation attention shifts, lexical activation, "
            "saccade control, and final TVT"
        ),
        "analysis_estimand": analysis_estimand,
        "actual_et1_trt_magnitudes_used": False,
        "et1_token_table_usage": (
            "native T5 token geometry and character alignment only"
        ),
        "no_redistribution_definition": (
            "unit allocation weight at the source token and zero weight at "
            "all neighboring token offsets"
        ),
        "attention_width_update": (
            "regression: max(recorded-1,3); otherwise min(recorded+0.5,5)"
        ),
        "fixed_prior_fit_scope": (
            "all projected OB1 fixations under the declared candidate-support "
            "policy; its alignment to the same OB1 profile is descriptive, "
            "not held-out validation, and is excluded from passage-bootstrap "
            "confidence intervals and paired significance tests"
        ),
        "gaussian_parameter_diagnostic_scope": (
            " ".join(
                (
                    "a priori and frozen-parameter controls do not fit the "
                    "current Provo OB1 profile;",
                    (
                        "support-RMS-displacement controls are disabled;"
                        if skip_support_rms_displacement_controls
                        else (
                            "support-RMS-displacement controls use the "
                            "evaluation fixation supports but not OB1 "
                            "attention weights;"
                        )
                    ),
                    (
                        "support-centered-SD controls use the evaluation "
                        "fixation supports but not OB1 attention weights;"
                        if include_support_centered_sd_controls
                        else "support-centered-SD controls are disabled;"
                    ),
                    (
                        "constrained and free-ratio OB1 fits use the same "
                        "pooled reference and are descriptive in-sample "
                        "diagnostics"
                    ),
                )
            )
        ),
        "fixed_ratio4_control_definition": (
            "right_left_ratio=4 specified a priori from the paper-stated "
            "OB1 asymmetry, with the same quadratic side-scale RMS as the "
            "frozen learned kernel; this does not guarantee equal realized "
            "variance after support truncation and normalization"
        ),
        "quadratic_side_scale_definition": ("sqrt((sigma_left^2 + sigma_right^2) / 2)"),
        "mirrored_control_definition": (
            "learned sigma_left and sigma_right exchanged; parameter count "
            "and quadratic side-scale RMS are unchanged, but right-heavy "
            "visible support means the pooled output is not an exact mirror"
        ),
        "support_rms_displacement_matches": support_spread_matches,
        "support_rms_displacement_controls_enabled": (
            not skip_support_rms_displacement_controls
        ),
        "active_profile_methods": list(active_profile_methods),
        "support_rms_displacement_definition": (
            None
            if skip_support_rms_displacement_controls
            else (
                "sqrt(sum_d p(d) * d^2) in relative native T5-token offsets, "
                "with p pooled over the declared fixation-conditioned support"
            )
        ),
        "support_rms_displacement_control_scope": (
            None
            if skip_support_rms_displacement_controls
            else (
                "post hoc support-conditioned ablation: the target comes from "
                "the frozen learned candidate and does not use OB1 attention "
                "weights; the visible supports come from the evaluation "
                "fixation contexts"
            )
        ),
        "support_centered_sd_matches": support_centered_sd_matches,
        "support_centered_sd_controls_enabled": (include_support_centered_sd_controls),
        "support_centered_sd_definition": (
            (
                "sqrt(sum_d p(d) * (d - sum_j p(j) * j)^2) in relative "
                "native T5-token offsets, with p pooled over the declared "
                "fixation-conditioned support"
            )
            if include_support_centered_sd_controls
            else None
        ),
        "support_centered_sd_control_scope": (
            (
                "post hoc support-conditioned width ablation: the target "
                "comes from the frozen learned candidate and does not use "
                "OB1 attention weights; centering removes the candidate's "
                "mean directional shift before measuring dispersion"
            )
            if include_support_centered_sd_controls
            else None
        ),
        "sigma_landscape_enabled": landscape_sigma_values is not None,
        "sigma_landscape_scope": (
            "descriptive fixation-weighted pooled profile without passage bootstrap"
            if landscape_sigma_values is not None
            else None
        ),
        "sigma_landscape_min": (
            float(np.min(landscape_sigma_values))
            if landscape_sigma_values is not None
            else None
        ),
        "sigma_landscape_max": (
            float(np.max(landscape_sigma_values))
            if landscape_sigma_values is not None
            else None
        ),
        "sigma_landscape_points": (
            int(len(landscape_sigma_values))
            if landscape_sigma_values is not None
            else 0
        ),
    }
    if len(normalized_records) == 1:
        audit.update(normalized_records[0])
    attention_profiles = pd.DataFrame(profile_rows)
    attention_profiles["profile_component"] = profile_component
    attention_profiles["reference_profile"] = reference_profile
    attention_profiles["analysis_estimand"] = analysis_estimand
    attention_profiles["actual_et1_trt_magnitudes_used"] = False
    attention_profiles["candidate_support_policy"] = candidate_support_policy
    attention_profiles["candidate_support_policy_label"] = support_policy_label
    attention_profiles["trajectory_attention_skew"] = trajectory_skew
    add_attention_skew_provenance(
        attention_profiles,
        trajectory_skew,
    )
    profile_regions = summarize_profile_regions(
        attention_profiles,
        ("ob1_attention_profile", *active_profile_methods),
        {
            "ob1_attention_profile": reference_profile,
            **PROFILE_DISPLAY_NAMES,
        },
    )
    directionality = summarize_directionality(attention_profiles)
    directionality["display_name"] = directionality["method"].map(
        {
            "ob1_attention_profile": reference_profile,
            **PROFILE_DISPLAY_NAMES,
        }
    )
    directionality["reference_profile"] = reference_profile
    directionality["analysis_estimand"] = analysis_estimand
    directionality["actual_et1_trt_magnitudes_used"] = False
    directionality["candidate_support_policy"] = candidate_support_policy
    directionality["candidate_support_policy_label"] = support_policy_label
    directionality["profile_component"] = profile_component
    directionality["fixation_weighting"] = fixation_weighting
    directionality["trajectory_attention_skew"] = trajectory_skew
    add_attention_skew_provenance(directionality, trajectory_skew)
    directionality["rightward_share_scope"] = RIGHTWARD_SHARE_SCOPE
    reference_mask = directionality["method"].eq("ob1_attention_profile")
    reference_candidate_metadata = [
        "source_accuracy",
        "learned_sigma_left",
        "learned_sigma_right",
        "learned_right_left_ratio",
        "fixed_symmetric_sigma",
        "rms_side_scale_symmetric_sigma",
        "fixed_ratio4_same_rms_sigma_left",
        "fixed_ratio4_same_rms_sigma_right",
        "learned_support_rms_token_displacement",
        "support_rms_displacement_symmetric_sigma",
        "support_rms_displacement_ratio4_sigma_left",
        "support_rms_displacement_ratio4_sigma_right",
        *(
            column
            for column in SUPPORT_CENTERED_SD_METADATA_COLUMNS
            if column in directionality
        ),
    ]
    directionality.loc[
        reference_mask,
        reference_candidate_metadata,
    ] = np.nan
    reviewer_summary = build_reviewer_profile_summary(
        result_table,
        directionality,
    )
    return {
        "attention_profiles": attention_profiles,
        "directionality": directionality,
        "reviewer_summary": reviewer_summary,
        "passage_metrics": passage_metrics,
        "result_table": result_table,
        "contrasts": contrasts,
        "fixed_priors": fixed_prior_records,
        "parameter_diagnostics": parameter_diagnostics,
        "profile_regions": profile_regions,
        "sigma_landscape": sigma_landscape,
        "audit": audit,
    }


def build_reviewer_profile_summary(
    result_table: pd.DataFrame,
    directionality: pd.DataFrame,
) -> pd.DataFrame:
    """Build one explicit reviewer-facing table from profile artifacts."""
    reported_methods = (
        "raw_delta",
        "fixed_symmetric_sigma1",
        "rms_side_scale_symmetric",
        "fixed_ratio4_same_rms",
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
        "mirrored_learned",
        "learned_asymmetric",
    )
    key_columns = [
        "checkpoint_id",
        "ob1_attention_skew",
        "method",
    ]
    direction_columns = [
        *key_columns,
        "left_mass",
        "center_mass",
        "right_mass",
        "right_share_of_noncenter_mass",
    ]
    candidates = result_table.loc[result_table["method"].isin(reported_methods)].merge(
        directionality[direction_columns],
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    candidates["row_role"] = "candidate allocation kernel"
    references = directionality.loc[
        directionality["method"].eq("ob1_attention_profile")
    ].copy()
    reference_rows = []
    for reference in references.itertuples():
        reference_row = {
            "checkpoint_id": reference.checkpoint_id,
            "ob1_attention_skew": reference.ob1_attention_skew,
            "method": reference.method,
            "display_name": reference.display_name,
            "passages": result_table.loc[
                result_table["checkpoint_id"].eq(reference.checkpoint_id)
                & result_table["ob1_attention_skew"].eq(reference.ob1_attention_skew),
                "passages",
            ].iloc[0],
            "source_accuracy": np.nan,
            "learned_sigma_left": np.nan,
            "learned_sigma_right": np.nan,
            "learned_right_left_ratio": np.nan,
            "fixed_symmetric_sigma": np.nan,
            "rms_side_scale_symmetric_sigma": np.nan,
            "fixed_ratio4_same_rms_sigma_left": np.nan,
            "fixed_ratio4_same_rms_sigma_right": np.nan,
            "learned_support_rms_token_displacement": np.nan,
            "support_rms_displacement_symmetric_sigma": np.nan,
            "support_rms_displacement_ratio4_sigma_left": np.nan,
            "support_rms_displacement_ratio4_sigma_right": np.nan,
            "profile_spearman": np.nan,
            "profile_spearman_ci_low": np.nan,
            "profile_spearman_ci_high": np.nan,
            "js_divergence": np.nan,
            "js_divergence_ci_low": np.nan,
            "js_divergence_ci_high": np.nan,
            "hellinger_distance": np.nan,
            "hellinger_distance_ci_low": np.nan,
            "hellinger_distance_ci_high": np.nan,
            "total_variation_distance": np.nan,
            "total_variation_distance_ci_low": np.nan,
            "total_variation_distance_ci_high": np.nan,
            "overlap_coefficient": np.nan,
            "overlap_coefficient_ci_low": np.nan,
            "overlap_coefficient_ci_high": np.nan,
            "token_offset_wasserstein": np.nan,
            "token_offset_wasserstein_ci_low": np.nan,
            "token_offset_wasserstein_ci_high": np.nan,
            "reference_profile": reference.reference_profile,
            "analysis_estimand": reference.analysis_estimand,
            "actual_et1_trt_magnitudes_used": False,
            "candidate_support_policy": (reference.candidate_support_policy),
            "candidate_support_policy_label": (
                reference.candidate_support_policy_label
            ),
            "ci_scope": "not applicable: this row is the reference",
            "profile_component": reference.profile_component,
            "fixation_weighting": reference.fixation_weighting,
            "profile_spearman_scope": PROFILE_SPEARMAN_SCOPE,
            "distribution_metric_scope": (PROFILE_DISTRIBUTION_METRIC_SCOPE),
            "trajectory_attention_skew": (reference.trajectory_attention_skew),
            "requested_skew_matches_trajectory": (
                reference.requested_skew_matches_trajectory
            ),
            "attention_skew_analysis_role": (reference.attention_skew_analysis_role),
            "left_mass": reference.left_mass,
            "center_mass": reference.center_mass,
            "right_mass": reference.right_mass,
            "right_share_of_noncenter_mass": (reference.right_share_of_noncenter_mass),
            "row_role": "OB1 reference profile",
        }
        for column in SUPPORT_CENTERED_SD_METADATA_COLUMNS:
            if column in result_table:
                reference_row[column] = np.nan
        reference_rows.append(reference_row)
    combined = pd.concat(
        [candidates, pd.DataFrame(reference_rows)],
        ignore_index=True,
        sort=False,
    )
    method_order = {
        "raw_delta": 0,
        "fixed_symmetric_sigma1": 1,
        "rms_side_scale_symmetric": 2,
        "fixed_ratio4_same_rms": 3,
        "support_rms_displacement_symmetric": 4,
        "support_rms_displacement_ratio4": 5,
        "support_centered_sd_symmetric": 6,
        "support_centered_sd_ratio4": 7,
        "mirrored_learned": 8,
        "learned_asymmetric": 9,
        "ob1_attention_profile": 10,
    }
    combined["_method_order"] = combined["method"].map(method_order)
    combined["rightward_share_scope"] = RIGHTWARD_SHARE_SCOPE
    return (
        combined.sort_values(
            [
                "checkpoint_id",
                "ob1_attention_skew",
                "_method_order",
            ]
        )
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def write_attention_profile_outputs(
    output_dir: Path,
    artifacts: dict,
) -> None:
    """Write attention-profile tables, fixed priors, and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts["attention_profiles"].to_csv(
        output_dir / "kernel_profiles.csv",
        index=False,
    )
    artifacts["directionality"].to_csv(
        output_dir / "kernel_directionality.csv",
        index=False,
    )
    artifacts["reviewer_summary"].to_csv(
        output_dir / "reviewer_kernel_summary.csv",
        index=False,
    )
    artifacts["passage_metrics"].to_csv(
        output_dir / "kernel_alignment_by_passage.csv",
        index=False,
    )
    artifacts["result_table"].to_csv(
        output_dir / "kernel_alignment_result_table.csv",
        index=False,
    )
    artifacts["contrasts"].to_csv(
        output_dir / "kernel_alignment_contrasts.csv",
        index=False,
    )
    artifacts["profile_regions"].to_csv(
        output_dir / "kernel_profile_regions.csv",
        index=False,
    )
    artifacts["parameter_diagnostics"].to_csv(
        output_dir / "gaussian_parameter_diagnostics.csv",
        index=False,
    )
    if artifacts["sigma_landscape"] is not None:
        artifacts["sigma_landscape"].to_csv(
            output_dir / "sigma_landscape.csv",
            index=False,
        )
    with (output_dir / "fixed_ob1_priors.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            artifacts["fixed_priors"],
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (output_dir / "attention_profile_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            artifacts["audit"],
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    profiles = artifacts["attention_profiles"]
    checkpoint_ids = profiles["checkpoint_id"].drop_duplicates().tolist()
    if len(checkpoint_ids) == 1:
        checkpoint_id = checkpoint_ids[0]
        plot_attention_profiles(
            profiles,
            output_dir / "kernel_profiles.png",
        )
        plot_metric_comparison(
            artifacts["result_table"],
            output_dir / "kernel_metric_comparison.png",
            REVIEWER_PLOT_METHODS,
            PROFILE_SHORT_DISPLAY_NAMES,
        )
        plot_profile_regions(
            artifacts["profile_regions"],
            output_dir / "kernel_profile_regions.png",
            ("ob1_attention_profile", *REVIEWER_PLOT_METHODS),
            PROFILE_SHORT_DISPLAY_NAMES,
        )
        plot_parameter_diagnostics(
            artifacts["parameter_diagnostics"],
            output_dir / "gaussian_parameter_diagnostics.png",
        )
        if artifacts["sigma_landscape"] is not None:
            plot_sigma_landscape(
                artifacts["sigma_landscape"],
                artifacts["parameter_diagnostics"],
                output_dir,
                checkpoint_id,
            )
    else:
        plot_dir = output_dir / "kernel_profile_plots"
        metric_dir = output_dir / "kernel_metric_plots"
        region_dir = output_dir / "kernel_region_plots"
        diagnostic_dir = output_dir / "gaussian_parameter_plots"
        landscape_dir = output_dir / "sigma_landscape_plots"
        for checkpoint_id in checkpoint_ids:
            safe_id = safe_identifier(checkpoint_id)
            plot_attention_profiles(
                profiles.loc[profiles["checkpoint_id"].eq(checkpoint_id)],
                plot_dir / f"{safe_id}.png",
            )
            plot_metric_comparison(
                artifacts["result_table"].loc[
                    artifacts["result_table"]["checkpoint_id"].eq(checkpoint_id)
                ],
                metric_dir / f"{safe_id}.png",
                REVIEWER_PLOT_METHODS,
                PROFILE_SHORT_DISPLAY_NAMES,
            )
            plot_profile_regions(
                artifacts["profile_regions"].loc[
                    artifacts["profile_regions"]["checkpoint_id"].eq(checkpoint_id)
                ],
                region_dir / f"{safe_id}.png",
                ("ob1_attention_profile", *REVIEWER_PLOT_METHODS),
                PROFILE_SHORT_DISPLAY_NAMES,
            )
            plot_parameter_diagnostics(
                artifacts["parameter_diagnostics"].loc[
                    artifacts["parameter_diagnostics"]["checkpoint_id"].eq(
                        checkpoint_id
                    )
                ],
                diagnostic_dir / f"{safe_id}.png",
            )
            if artifacts["sigma_landscape"] is not None:
                plot_sigma_landscape(
                    artifacts["sigma_landscape"],
                    artifacts["parameter_diagnostics"],
                    landscape_dir / safe_id,
                    checkpoint_id,
                )


def plot_attention_profiles(
    profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot projected OB1 attention and every token-space control kernel."""
    skews = sorted(profiles["ob1_attention_skew"].unique())
    components = profiles["profile_component"].drop_duplicates().tolist()
    if len(components) != 1:
        raise ValueError("Profile plot requires one OB1 profile component")
    support_policies = profiles["candidate_support_policy"].drop_duplicates().tolist()
    if len(support_policies) != 1:
        raise ValueError("Profile plot requires one candidate-support policy")
    figure, axes = plt.subplots(
        len(skews),
        1,
        figsize=(9, 4.2 * len(skews)),
        sharex=True,
        squeeze=False,
    )
    series = [
        ("ob1_attention_profile", "OB1 attention", 2.5),
        ("raw_delta", "No redistribution", 1.5),
        (
            "fixed_symmetric_sigma1",
            "Symmetric Gaussian (sigma=1)",
            1.5,
        ),
        (
            "support_rms_displacement_symmetric",
            "Symmetric Gaussian (matched spread)",
            1.5,
        ),
        (
            "support_rms_displacement_ratio4",
            "4:1",
            1.5,
        ),
        (
            "support_centered_sd_symmetric",
            "Symmetric Gaussian (matched centered SD)",
            1.5,
        ),
        (
            "support_centered_sd_ratio4",
            "4:1 Gaussian (matched centered SD)",
            1.5,
        ),
        (
            "mirrored_learned",
            "Mirrored learned Gaussian",
            1.5,
        ),
        (
            "fixed_ob1_gaussian",
            "OB1-fitted Gaussian (descriptive)",
            1.5,
        ),
        (
            "learned_asymmetric",
            "Learned asymmetric Gaussian",
            1.8,
        ),
    ]
    series = [item for item in series if item[0] in profiles.columns]
    for axis, skew in zip(axes[:, 0], skews):
        skew_profiles = profiles.loc[
            profiles["ob1_attention_skew"].eq(skew)
        ].sort_values("relative_t5_token_offset")
        for column, label, width in series:
            axis.plot(
                skew_profiles["relative_t5_token_offset"],
                skew_profiles[column],
                marker="o",
                markersize=3,
                linewidth=width,
                label=label,
            )
        axis.axvline(0, color="black", linewidth=0.8, alpha=0.45)
        axis.set_title(
            f"OB1 attention skew = {skew:g}; candidate support = {support_policies[0]}"
        )
        axis.set_ylabel("Normalized mass")
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Relative native T5 token offset")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def summarize_directionality(profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize left, center, and right mass for every compared profile."""
    profile_columns = [
        "ob1_attention_profile",
        *(method for method in PROFILE_METHODS if method in profiles.columns),
    ]
    records = []
    for (checkpoint_id, skew), skew_profiles in profiles.groupby(
        ["checkpoint_id", "ob1_attention_skew"],
        sort=True,
    ):
        offsets = skew_profiles["relative_t5_token_offset"]
        for method in profile_columns:
            left = float(skew_profiles.loc[offsets.lt(0), method].sum())
            center = float(skew_profiles.loc[offsets.eq(0), method].sum())
            right = float(skew_profiles.loc[offsets.gt(0), method].sum())
            noncenter = left + right
            record = {
                "checkpoint_id": checkpoint_id,
                "ob1_attention_skew": float(skew),
                "method": method,
                "source_accuracy": skew_profiles["source_accuracy"].iloc[0],
                "learned_sigma_left": skew_profiles["learned_sigma_left"].iloc[0],
                "learned_sigma_right": skew_profiles["learned_sigma_right"].iloc[0],
                "learned_right_left_ratio": skew_profiles[
                    "learned_right_left_ratio"
                ].iloc[0],
                "fixed_symmetric_sigma": skew_profiles["fixed_symmetric_sigma"].iloc[0],
                "rms_side_scale_symmetric_sigma": skew_profiles[
                    "rms_side_scale_symmetric_sigma"
                ].iloc[0],
                "fixed_ratio4_same_rms_sigma_left": skew_profiles[
                    "fixed_ratio4_same_rms_sigma_left"
                ].iloc[0],
                "fixed_ratio4_same_rms_sigma_right": skew_profiles[
                    "fixed_ratio4_same_rms_sigma_right"
                ].iloc[0],
                "learned_support_rms_token_displacement": skew_profiles[
                    "learned_support_rms_token_displacement"
                ].iloc[0],
                "support_rms_displacement_symmetric_sigma": (
                    skew_profiles["support_rms_displacement_symmetric_sigma"].iloc[0]
                ),
                "support_rms_displacement_ratio4_sigma_left": (
                    skew_profiles["support_rms_displacement_ratio4_sigma_left"].iloc[0]
                ),
                "support_rms_displacement_ratio4_sigma_right": (
                    skew_profiles["support_rms_displacement_ratio4_sigma_right"].iloc[0]
                ),
                "left_mass": left,
                "center_mass": center,
                "right_mass": right,
                "right_minus_left_mass": right - left,
                "right_share_of_noncenter_mass": (
                    right / noncenter if noncenter > 0 else np.nan
                ),
            }
            for column in SUPPORT_CENTERED_SD_METADATA_COLUMNS:
                if column in skew_profiles:
                    record[column] = skew_profiles[column].iloc[0]
            records.append(record)
    return pd.DataFrame(records)
