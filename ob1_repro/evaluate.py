"""Evaluate Human, ET1 redistribution, and OB1 word-level allocation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance

from .distribution_metrics import (
    distribution_similarity_metrics,
)


METHOD_COLUMNS = {
    "et1_raw": "et1_raw_word_trt",
    "et1_symmetric": "et1_symmetric_word_trt",
    "et1_rms_side_scale_symmetric": ("et1_rms_side_scale_symmetric_word_trt"),
    "et1_asymmetric": "et1_asymmetric_word_trt",
    "ob1": "ob1_tvt",
}
DISPLAY_NAMES = {
    "et1_raw": "ET1 raw",
    "et1_symmetric": "ET1 + fixed SymGaussian (sigma=1.0)",
    "et1_rms_side_scale_symmetric": ("ET1 + RMS-of-side-scales symmetric"),
    "et1_asymmetric": "ET1 + learned asymmetric",
    "ob1": "OB1 baseline",
}
OB1_CLEAN_PASSAGE_POLICY = "exclude_passages_with_any_ob1_incompatible_word"
HUMAN_METRIC_DIRECTIONS = {
    "human_spearman": "higher",
    "js_divergence": "lower",
    "hellinger_distance": "lower",
    "total_variation_distance": "lower",
    "overlap_coefficient": "higher",
    "word_order_wasserstein": "lower",
}
COGNITIVE_METRIC_DIRECTIONS = {
    "ob1_spearman": "higher",
    "ob1_js_divergence": "lower",
    "ob1_hellinger_distance": "lower",
    "ob1_total_variation_distance": "lower",
    "ob1_overlap_coefficient": "higher",
    "ob1_word_order_wasserstein": "lower",
}
METRIC_DIRECTIONS = {
    **HUMAN_METRIC_DIRECTIONS,
    **COGNITIVE_METRIC_DIRECTIONS,
}
OVERLAP_COMPLEMENTS = {
    "overlap_coefficient": "total_variation_distance",
    "ob1_overlap_coefficient": "ob1_total_variation_distance",
}
INFERENTIAL_METRIC_DIRECTIONS = {
    metric: direction
    for metric, direction in METRIC_DIRECTIONS.items()
    if metric not in OVERLAP_COMPLEMENTS
}
BEHAVIOR_DISTRIBUTION_METRIC_SCOPE = (
    "Jensen-Shannon divergence, Hellinger distance, and total variation "
    "distance are computed on complete normalized passage-level word "
    "allocations; overlap coefficient and its intervals are derived exactly "
    "as one minus total variation distance"
)
BEHAVIOR_SPEARMAN_SCOPE = (
    "Spearman correlation over every common evaluated word position; zero "
    "Human or OB1 values are observed no-dwell or skipped-word allocations, "
    "not structural padding"
)


def normalized_allocation(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip negative values and normalize a passage to unit allocation mass."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Allocation contains non-finite values")
    clipped_count = int((values < 0).sum())
    nonnegative = np.clip(values, 0.0, None)
    mass = float(nonnegative.sum())
    if mass <= 0:
        raise ValueError("Allocation has zero nonnegative passage mass")
    return nonnegative / mass, clipped_count


def passage_metric_values(
    human: np.ndarray,
    method: np.ndarray,
    ob1: np.ndarray,
    positions: np.ndarray | None = None,
) -> dict:
    """Compute rank and distribution metrics for one passage."""
    human = np.asarray(human, dtype=float)
    method = np.asarray(method, dtype=float)
    ob1 = np.asarray(ob1, dtype=float)
    if not (len(human) == len(method) == len(ob1)):
        raise ValueError("Human, method, and OB1 passage lengths differ")
    if len(human) < 2:
        raise ValueError("At least two passage words are required")
    if not (
        np.isfinite(human).all()
        and np.isfinite(method).all()
        and np.isfinite(ob1).all()
    ):
        raise ValueError("Passage metric input contains non-finite values")
    if bool((human < 0).any()):
        raise ValueError("Human TRT contains negative values")
    if bool((ob1 < 0).any()):
        raise ValueError("OB1 TVT contains negative values")

    human_distribution, human_clipped = normalized_allocation(human)
    method_distribution, method_clipped = normalized_allocation(method)
    ob1_distribution, ob1_clipped = normalized_allocation(ob1)
    if positions is None:
        positions = np.linspace(0.0, 1.0, len(human))
    else:
        positions = np.asarray(positions, dtype=float)
        if (
            len(positions) != len(human)
            or not np.isfinite(positions).all()
            or bool((np.diff(positions) <= 0).any())
        ):
            raise ValueError(
                "Word positions must be finite, strictly increasing, "
                "and match passage length"
            )
    human_spearman = float(spearmanr(human, method).statistic)
    ob1_spearman = float(spearmanr(ob1, method).statistic)
    if not np.isfinite(human_spearman) or not np.isfinite(ob1_spearman):
        raise ValueError("Spearman correlation is undefined for a passage")
    human_distribution_metrics = distribution_similarity_metrics(
        human_distribution,
        method_distribution,
    )
    ob1_distribution_metrics = distribution_similarity_metrics(
        ob1_distribution,
        method_distribution,
    )
    return {
        "human_spearman": human_spearman,
        "js_divergence": float(
            jensenshannon(
                human_distribution,
                method_distribution,
                base=2.0,
            )
            ** 2
        ),
        **human_distribution_metrics,
        "word_order_wasserstein": float(
            wasserstein_distance(
                positions,
                positions,
                u_weights=human_distribution,
                v_weights=method_distribution,
            )
        ),
        "ob1_spearman": ob1_spearman,
        "ob1_js_divergence": float(
            jensenshannon(
                ob1_distribution,
                method_distribution,
                base=2.0,
            )
            ** 2
        ),
        "ob1_hellinger_distance": ob1_distribution_metrics["hellinger_distance"],
        "ob1_total_variation_distance": ob1_distribution_metrics[
            "total_variation_distance"
        ],
        "ob1_overlap_coefficient": ob1_distribution_metrics["overlap_coefficient"],
        "ob1_word_order_wasserstein": float(
            wasserstein_distance(
                positions,
                positions,
                u_weights=ob1_distribution,
                v_weights=method_distribution,
            )
        ),
        "human_clipped_values": human_clipped,
        "method_clipped_values": method_clipped,
        "ob1_clipped_values": ob1_clipped,
    }


def merge_word_values(
    canonical_words: pd.DataFrame,
    et1_words: pd.DataFrame,
    ob1_words: pd.DataFrame,
) -> pd.DataFrame:
    """Join canonical Human, checkpoint-specific ET1, and OB1 word grids."""
    keys = ["passage_id_zero_based", "word_id_zero_based"]
    canonical_columns = keys + [
        "passage_id_raw",
        "word_raw",
        "human_trt_unconditional",
        "human_trt_conditional",
    ]
    canonical_columns.extend(
        column
        for column in (
            "cluster_id",
            "article_id",
            "paragraph_id",
            "ob1_evaluable",
            "ob1_word_normalized",
        )
        if column in canonical_words.columns
    )
    et1_columns = (
        ["checkpoint_id"]
        + keys
        + [
            "et1_raw_word_trt",
            "et1_symmetric_word_trt",
            "et1_rms_side_scale_symmetric_word_trt",
            "et1_asymmetric_word_trt",
        ]
    )
    ob1_columns = keys + ["ob1_tvt"]
    et1_coordinate_columns = ["checkpoint_id", *keys]
    duplicate_canonical = canonical_words.duplicated(keys, keep=False)
    if bool(duplicate_canonical.any()):
        examples = (
            canonical_words.loc[duplicate_canonical, keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate canonical word coordinates: {examples}")
    if et1_words["checkpoint_id"].isna().any():
        raise ValueError("ET1 checkpoint IDs must not be missing")
    duplicate_et1 = et1_words.duplicated(
        et1_coordinate_columns,
        keep=False,
    )
    if bool(duplicate_et1.any()):
        examples = (
            et1_words.loc[duplicate_et1, et1_coordinate_columns]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate ET1 checkpoint-word coordinates: {examples}")
    duplicate_ob1 = ob1_words.duplicated(keys, keep=False)
    if bool(duplicate_ob1.any()):
        examples = (
            ob1_words.loc[duplicate_ob1, keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate OB1 word coordinates: {examples}")

    canonical_coordinates = set(
        canonical_words[keys].itertuples(index=False, name=None)
    )
    ob1_coordinates = set(ob1_words[keys].itertuples(index=False, name=None))
    missing_ob1 = canonical_coordinates - ob1_coordinates
    extra_ob1 = ob1_coordinates - canonical_coordinates
    if missing_ob1 or extra_ob1:
        raise ValueError(
            "OB1 coordinate grid differs from canonical grid: "
            f"missing={len(missing_ob1)}, extra={len(extra_ob1)}, "
            f"missing_examples={sorted(missing_ob1)[:10]}, "
            f"extra_examples={sorted(extra_ob1)[:10]}"
        )

    checkpoint_ids = et1_words["checkpoint_id"].drop_duplicates().tolist()
    if not checkpoint_ids:
        raise ValueError("ET1 word table contains no checkpoints")
    for checkpoint_id, checkpoint_words in et1_words.groupby(
        "checkpoint_id",
        sort=False,
    ):
        checkpoint_coordinates = set(
            checkpoint_words[keys].itertuples(index=False, name=None)
        )
        missing_et1 = canonical_coordinates - checkpoint_coordinates
        extra_et1 = checkpoint_coordinates - canonical_coordinates
        if missing_et1 or extra_et1:
            raise ValueError(
                "ET1 coordinate grid differs from canonical grid for "
                f"checkpoint {checkpoint_id}: "
                f"missing={len(missing_et1)}, extra={len(extra_et1)}, "
                f"missing_examples={sorted(missing_et1)[:10]}, "
                f"extra_examples={sorted(extra_et1)[:10]}"
            )

    merged = canonical_words[canonical_columns].merge(
        et1_words[et1_columns],
        on=keys,
        how="inner",
        validate="one_to_many",
    )
    merged = merged.merge(
        ob1_words[ob1_columns],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    checkpoints = merged["checkpoint_id"].nunique()
    expected = len(canonical_words) * checkpoints
    if len(merged) != expected:
        raise ValueError(f"Expected {expected} joined word rows, found {len(merged)}")
    raw_variation = merged.groupby(keys)["et1_raw_word_trt"].nunique(dropna=False).max()
    if raw_variation != 1:
        raise ValueError("Raw ET1 values differ across RM checkpoints")
    return merged.sort_values(
        ["checkpoint_id", "passage_id_zero_based", "word_id_zero_based"]
    ).reset_index(drop=True)


def evaluate_passages(
    word_values: pd.DataFrame,
    human_column: str = "human_trt_unconditional",
) -> pd.DataFrame:
    """Compute every method metric per checkpoint and passage."""
    if human_column not in word_values:
        raise ValueError(f"Unknown Human TRT target: {human_column}")
    records = []
    for checkpoint_id, checkpoint_frame in word_values.groupby(
        "checkpoint_id",
        sort=True,
    ):
        unavailable_method_columns = {
            column
            for method, column in METHOD_COLUMNS.items()
            if method != "ob1" and checkpoint_frame[column].isna().all()
        }
        for passage_id, passage in checkpoint_frame.groupby(
            "passage_id_zero_based",
            sort=True,
        ):
            passage = passage.sort_values("word_id_zero_based")
            original_word_count = len(passage)
            passage_context = {}
            for column in ("cluster_id", "article_id", "paragraph_id"):
                if column not in passage:
                    continue
                values = passage[column].drop_duplicates()
                if len(values) != 1:
                    raise ValueError(
                        f"Passage {passage_id} has multiple {column} values"
                    )
                passage_context[column] = values.iloc[0]
            if "ob1_evaluable" in passage:
                eligibility = passage["ob1_evaluable"]
                if eligibility.dtype == bool:
                    ob1_evaluable = eligibility
                else:
                    normalized = eligibility.astype(str).str.lower()
                    if not normalized.isin({"true", "false"}).all():
                        raise ValueError(
                            f"Passage {passage_id} has invalid ob1_evaluable values"
                        )
                    ob1_evaluable = normalized.eq("true")
            else:
                ob1_evaluable = pd.Series(
                    True,
                    index=passage.index,
                    dtype=bool,
                )
            ob1_incompatible_words_excluded = int((~ob1_evaluable).sum())
            compatible_passage = passage.loc[ob1_evaluable].copy()
            if len(compatible_passage) < 2:
                raise ValueError(
                    f"OB1 compatibility leaves fewer than two words "
                    f"in passage {passage_id}"
                )

            human_all = compatible_passage[human_column].to_numpy(
                dtype=float,
                na_value=np.nan,
            )
            if human_column == "human_trt_conditional":
                if np.isinf(human_all).any():
                    raise ValueError(
                        f"Conditional Human TRT contains infinity in passage "
                        f"{passage_id}"
                    )
                missing_human = np.isnan(human_all)
                evaluated_passage = compatible_passage.loc[~missing_human]
                human_missing_words_excluded = int(missing_human.sum())
                if len(evaluated_passage) < 2:
                    raise ValueError(
                        f"Conditional Human TRT leaves fewer than two words "
                        f"in passage {passage_id}"
                    )
            else:
                if not np.isfinite(human_all).all():
                    raise ValueError(
                        f"Human TRT target {human_column} contains missing or "
                        f"non-finite values in passage {passage_id}"
                    )
                evaluated_passage = compatible_passage
                human_missing_words_excluded = 0

            human = evaluated_passage[human_column].to_numpy(dtype=float)
            ob1 = evaluated_passage["ob1_tvt"].to_numpy(dtype=float)
            maximum_word_id = float(passage["word_id_zero_based"].max())
            if maximum_word_id <= 0:
                raise ValueError(
                    f"Passage {passage_id} has no positive word coordinate"
                )
            positions = (
                evaluated_passage["word_id_zero_based"].to_numpy(dtype=float)
                / maximum_word_id
            )
            for method, column in METHOD_COLUMNS.items():
                if column in unavailable_method_columns:
                    continue
                if method != "ob1" and evaluated_passage[column].isna().any():
                    raise ValueError(
                        f"ET1 method {method} is only partially missing in "
                        f"checkpoint {checkpoint_id}, passage {passage_id}"
                    )
                method_values = (
                    ob1
                    if method == "ob1"
                    else evaluated_passage[column].to_numpy(dtype=float)
                )
                metrics = passage_metric_values(
                    human,
                    method_values,
                    ob1,
                    positions=positions,
                )
                records.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_zero_based": int(passage_id),
                        "method": method,
                        "human_target": human_column,
                        "original_word_count": original_word_count,
                        "ob1_compatible_word_count": len(compatible_passage),
                        "ob1_incompatible_words_excluded": (
                            ob1_incompatible_words_excluded
                        ),
                        "word_count": len(evaluated_passage),
                        "human_missing_words_excluded": (human_missing_words_excluded),
                        **passage_context,
                        **metrics,
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No passage metrics were computed")
    return frame


def select_ob1_clean_passages(
    passage_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Select passages unaffected by OB1 runtime token surrogates."""
    passage_column = "passage_id_zero_based"
    excluded_column = "ob1_incompatible_words_excluded"
    required = {passage_column, excluded_column}
    missing = sorted(required - set(passage_metrics.columns))
    if missing:
        raise ValueError(f"OB1 clean-passage sensitivity requires columns: {missing}")
    if passage_metrics.empty:
        raise ValueError("OB1 clean-passage sensitivity requires passage metrics")

    counts = pd.to_numeric(
        passage_metrics[excluded_column],
        errors="coerce",
    )
    if counts.isna().any() or bool((counts < 0).any()) or bool((counts % 1 != 0).any()):
        raise ValueError("OB1 incompatible-word counts must be nonnegative integers")
    working = passage_metrics.copy()
    working[excluded_column] = counts.astype(int)
    variation = working.groupby(passage_column, sort=False)[excluded_column].nunique(
        dropna=False
    )
    if bool((variation != 1).any()):
        inconsistent = variation[variation != 1].index.tolist()
        raise ValueError(
            f"OB1 incompatible-word counts differ within passages: {inconsistent[:10]}"
        )

    passage_counts = (
        working[[passage_column, excluded_column]]
        .drop_duplicates()
        .sort_values(passage_column)
    )
    clean_ids = passage_counts.loc[
        passage_counts[excluded_column].eq(0),
        passage_column,
    ].tolist()
    excluded = passage_counts.loc[passage_counts[excluded_column].gt(0)]
    excluded_ids = excluded[passage_column].tolist()
    clean_metrics = working.loc[working[passage_column].isin(clean_ids)].copy()
    if clean_metrics.empty:
        raise ValueError("OB1 clean-passage sensitivity has no eligible passages")

    audit = {
        "sensitivity_policy": OB1_CLEAN_PASSAGE_POLICY,
        "sensitivity_filter_column": excluded_column,
        "source_passages": int(len(passage_counts)),
        "included_passages": int(len(clean_ids)),
        "excluded_passages": int(len(excluded_ids)),
        "excluded_passage_ids": [int(passage_id) for passage_id in excluded_ids],
        "source_ob1_incompatible_words": int(passage_counts[excluded_column].sum()),
        "excluded_passage_ob1_incompatible_words": int(excluded[excluded_column].sum()),
        "source_passage_metric_rows": int(len(working)),
        "included_passage_metric_rows": int(len(clean_metrics)),
    }
    return clean_metrics.reset_index(drop=True), audit


def bootstrap_mean(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Bootstrap a mean and percentile interval by row or optional cluster."""
    values = np.asarray(values, dtype=float)
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite with length at least two")
    cluster_codes, cluster_count = resampling_cluster_codes(values, clusters)
    if clusters is None:
        indices = rng.integers(0, len(values), size=(samples, len(values)))
        bootstrap_values = values[indices].mean(axis=1)
    else:
        cluster_sums = np.bincount(cluster_codes, weights=values)
        cluster_sizes = np.bincount(cluster_codes)
        indices = rng.integers(
            0,
            cluster_count,
            size=(samples, cluster_count),
        )
        resampled_sums = cluster_sums[indices].sum(axis=1)
        resampled_observation_counts = cluster_sizes[indices].sum(axis=1)
        bootstrap_values = resampled_sums / resampled_observation_counts
    lower, upper = np.percentile(bootstrap_values, [2.5, 97.5])
    return float(values.mean()), float(lower), float(upper)


def resampling_cluster_codes(
    values: np.ndarray,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Validate optional cluster labels and return one code per observation."""
    if clusters is None:
        return np.arange(len(values), dtype=int), len(values)
    cluster_values = np.asarray(clusters)
    if cluster_values.ndim != 1 or len(cluster_values) != len(values):
        raise ValueError("cluster labels must be one-dimensional and match values")
    if bool(pd.isna(cluster_values).any()):
        raise ValueError("cluster labels must not contain missing values")
    cluster_codes, unique_clusters = pd.factorize(cluster_values, sort=False)
    if len(unique_clusters) < 2:
        raise ValueError("at least two resampling clusters are required")
    return cluster_codes.astype(int), len(unique_clusters)


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> float:
    """Compute an exact or Monte Carlo two-sided paired sign-flip p-value."""
    differences = np.asarray(differences, dtype=float)
    if samples < 1:
        raise ValueError("permutation samples must be positive")
    if len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("paired differences must be finite with length at least two")
    cluster_codes, unit_count = resampling_cluster_codes(
        differences,
        clusters,
    )
    observed = abs(float(differences.mean()))
    exact_count = 1 << unit_count if unit_count <= 20 else None
    if exact_count is not None and exact_count <= samples:
        patterns = np.arange(exact_count, dtype=np.uint64)[:, None]
        offsets = np.arange(unit_count, dtype=np.uint64)[None, :]
        signs = ((patterns >> offsets) & 1).astype(float) * 2.0 - 1.0
        permuted = (signs[:, cluster_codes] * differences[None, :]).mean(axis=1)
        return float(
            np.count_nonzero(np.abs(permuted) >= observed - 1e-15) / exact_count
        )
    signs = rng.integers(0, 2, size=(samples, unit_count), dtype=np.int8) * 2 - 1
    permuted = (signs[:, cluster_codes] * differences[None, :]).mean(axis=1)
    extreme = int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
    return float((extreme + 1) / (samples + 1))


def passage_cluster_mapping(
    passage_metrics: pd.DataFrame,
    cluster_column: str | None,
) -> pd.DataFrame | None:
    """Return one validated optional cluster label for every passage."""
    if cluster_column is None:
        return None
    if cluster_column not in passage_metrics:
        raise ValueError(f"Unknown cluster column: {cluster_column}")
    columns = ["passage_id_zero_based", cluster_column]
    mapping = passage_metrics[columns].drop_duplicates()
    if mapping[cluster_column].isna().any():
        raise ValueError("Passage cluster labels must not be missing")
    if mapping["passage_id_zero_based"].duplicated().any():
        raise ValueError("Each passage must map to exactly one cluster")
    return mapping


def summarize_methods(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Average RM seeds within passage and bootstrap method-level means."""
    metric_columns = list(INFERENTIAL_METRIC_DIRECTIONS)
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    cluster_mapping = passage_cluster_mapping(
        passage_metrics,
        cluster_column,
    )
    if cluster_mapping is not None:
        per_passage = per_passage.merge(
            cluster_mapping,
            on="passage_id_zero_based",
            how="left",
            validate="many_to_one",
        )
    rng = np.random.default_rng(seed)
    records = []
    for method, group in per_passage.groupby("method", sort=True):
        record = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "passages": len(group),
        }
        clusters = (
            group[cluster_column].to_numpy() if cluster_column is not None else None
        )
        if clusters is not None:
            record["clusters"] = int(pd.Series(clusters).nunique())
        for metric in INFERENTIAL_METRIC_DIRECTIONS:
            mean, lower, upper = bootstrap_mean(
                group[metric].to_numpy(),
                bootstrap_samples,
                rng,
                clusters=clusters,
            )
            record[metric] = mean
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
        for overlap_metric, distance_metric in OVERLAP_COMPLEMENTS.items():
            record[overlap_metric] = 1.0 - record[distance_metric]
            record[f"{overlap_metric}_ci_low"] = (
                1.0 - record[f"{distance_metric}_ci_high"]
            )
            record[f"{overlap_metric}_ci_high"] = (
                1.0 - record[f"{distance_metric}_ci_low"]
            )
        records.append(record)
    return pd.DataFrame(records)


def summarize_methods_by_checkpoint(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Produce a separate method summary for every RM checkpoint."""
    frames = []
    for offset, (checkpoint_id, checkpoint_metrics) in enumerate(
        passage_metrics.groupby("checkpoint_id", sort=True)
    ):
        summary = summarize_methods(
            checkpoint_metrics,
            bootstrap_samples,
            seed + offset,
            cluster_column=cluster_column,
        )
        summary.insert(0, "checkpoint_id", checkpoint_id)
        frames.append(summary)
    if not frames:
        raise ValueError("No checkpoint-level method summaries were computed")
    return pd.concat(frames, ignore_index=True)


def paired_contrasts(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Bootstrap paired intervals and sign-flip tests for improvements."""
    metric_directions = INFERENTIAL_METRIC_DIRECTIONS
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )[list(metric_directions)]
        .mean()
        .reset_index()
    )
    cluster_mapping = passage_cluster_mapping(
        passage_metrics,
        cluster_column,
    )
    if cluster_mapping is not None:
        per_passage = per_passage.merge(
            cluster_mapping,
            on="passage_id_zero_based",
            how="left",
            validate="many_to_one",
        )
    bootstrap_rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed)
    records = []
    comparisons = (
        ("et1_symmetric", "et1_raw"),
        ("et1_rms_side_scale_symmetric", "et1_raw"),
        ("et1_asymmetric", "et1_raw"),
        ("et1_asymmetric", "et1_symmetric"),
        ("et1_asymmetric", "et1_rms_side_scale_symmetric"),
        ("et1_rms_side_scale_symmetric", "et1_symmetric"),
    )
    for method, baseline_method in comparisons:
        candidate = per_passage[per_passage["method"] == method].set_index(
            "passage_id_zero_based"
        )
        baseline = per_passage[per_passage["method"] == baseline_method].set_index(
            "passage_id_zero_based"
        )
        common = baseline.index.intersection(candidate.index)
        if len(common) == 0:
            continue
        clusters = (
            candidate.loc[common, cluster_column].to_numpy()
            if cluster_column is not None
            else None
        )
        for metric, direction in metric_directions.items():
            if direction == "higher":
                differences = (
                    candidate.loc[common, metric] - baseline.loc[common, metric]
                ).to_numpy()
            else:
                differences = (
                    baseline.loc[common, metric] - candidate.loc[common, metric]
                ).to_numpy()
            mean, lower, upper = bootstrap_mean(
                differences,
                bootstrap_samples,
                bootstrap_rng,
                clusters=clusters,
            )
            permutation_p = paired_sign_flip_pvalue(
                differences,
                bootstrap_samples,
                permutation_rng,
                clusters=clusters,
            )
            record = {
                "candidate": method,
                "baseline": baseline_method,
                "metric": metric,
                "positive_means_improvement": True,
                "passages": len(common),
                "mean_paired_improvement": mean,
                "ci_low": lower,
                "ci_high": upper,
                "permutation_p_two_sided": permutation_p,
            }
            if clusters is not None:
                record["clusters"] = int(pd.Series(clusters).nunique())
            records.append(record)
            derived_metric = next(
                (
                    overlap_metric
                    for overlap_metric, distance_metric in OVERLAP_COMPLEMENTS.items()
                    if distance_metric == metric
                ),
                None,
            )
            if derived_metric is not None:
                derived_record = dict(record)
                derived_record["metric"] = derived_metric
                records.append(derived_record)
    return pd.DataFrame(records)


def paired_contrasts_by_checkpoint(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Compute paired contrasts without averaging distinct sigma records."""
    frames = []
    for offset, (checkpoint_id, checkpoint_metrics) in enumerate(
        passage_metrics.groupby("checkpoint_id", sort=True)
    ):
        contrasts = paired_contrasts(
            checkpoint_metrics,
            bootstrap_samples,
            seed + offset,
            cluster_column=cluster_column,
        )
        contrasts.insert(0, "checkpoint_id", checkpoint_id)
        frames.append(contrasts)
    if not frames:
        raise ValueError("No checkpoint-level paired contrasts were computed")
    return pd.concat(frames, ignore_index=True)


def attach_checkpoint_metadata(
    frame: pd.DataFrame,
    checkpoint_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join one validated sigma-provenance row to each checkpoint result."""
    if checkpoint_metadata is None:
        return frame
    if "checkpoint_id" not in frame:
        raise ValueError("Checkpoint result is missing checkpoint_id")
    if "checkpoint_id" not in checkpoint_metadata:
        raise ValueError("Sigma metadata is missing checkpoint_id")
    if checkpoint_metadata["checkpoint_id"].duplicated().any():
        raise ValueError("Sigma metadata checkpoint IDs must be unique")
    metadata_columns = [
        column
        for column in (
            "checkpoint_id",
            "source_accuracy",
            "sigma_left",
            "sigma_right",
            "sigma_symmetric",
            "sigma_symmetric_fixed",
            "sigma_symmetric_rms_scale",
            "right_left_ratio",
        )
        if column in checkpoint_metadata
    ]
    metadata = checkpoint_metadata[metadata_columns].copy()
    result = frame.merge(
        metadata,
        on="checkpoint_id",
        how="left",
        validate="many_to_one",
    )
    missing = sorted(set(frame["checkpoint_id"]) - set(metadata["checkpoint_id"]))
    if missing:
        raise ValueError(f"Sigma metadata is missing checkpoint IDs: {missing}")
    return result


def build_sigma_sweep_summary(
    checkpoint_summary: pd.DataFrame,
    checkpoint_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Build one analysis-ready row per learned sigma configuration."""
    required_summary = {"checkpoint_id", "method"}
    missing_summary = sorted(required_summary - set(checkpoint_summary))
    if missing_summary:
        raise ValueError(f"Checkpoint summary is missing columns: {missing_summary}")
    metric_columns = [
        metric for metric in METRIC_DIRECTIONS if metric in checkpoint_summary
    ]
    metadata_columns = [
        column
        for column in (
            "source_accuracy",
            "sigma_left",
            "sigma_right",
            "sigma_symmetric",
            "sigma_symmetric_fixed",
            "sigma_symmetric_rms_scale",
            "right_left_ratio",
        )
        if column in checkpoint_summary
    ]
    metadata = (
        checkpoint_summary[["checkpoint_id", *metadata_columns]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if metadata["checkpoint_id"].duplicated().any():
        raise ValueError("Checkpoint metadata is inconsistent across method rows")
    point_estimates = checkpoint_summary.pivot(
        index="checkpoint_id",
        columns="method",
        values=metric_columns,
    )
    point_estimates.columns = [
        f"{method}__{metric}" for metric, method in point_estimates.columns
    ]
    point_estimates = point_estimates.reset_index()
    matched = matched_asymmetry_contrast_table(checkpoint_contrasts)
    if matched.empty:
        contrast_wide = pd.DataFrame({"checkpoint_id": metadata["checkpoint_id"]})
    else:
        contrast_fields = [
            field
            for field in (
                "mean_paired_improvement",
                "ci_low",
                "ci_high",
                "permutation_p_two_sided",
            )
            if field in matched
        ]
        contrast_wide = matched.pivot(
            index="checkpoint_id",
            columns="metric",
            values=contrast_fields,
        )
        contrast_wide.columns = [
            f"asym_minus_symmetric__{metric}__{field}"
            for field, metric in contrast_wide.columns
        ]
        contrast_wide = contrast_wide.reset_index()
    return (
        metadata.merge(
            point_estimates,
            on="checkpoint_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            contrast_wide,
            on="checkpoint_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["source_accuracy", "checkpoint_id"]
            if "source_accuracy" in metadata_columns
            else ["checkpoint_id"],
            ascending=(
                [False, True] if "source_accuracy" in metadata_columns else True
            ),
        )
        .reset_index(drop=True)
    )


def cognitive_result_table(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Select ET1-to-OB1 alignment estimates for the reviewer-facing table."""
    identifier_columns = ["method", "display_name", "passages"]
    if "clusters" in method_summary.columns:
        identifier_columns.append("clusters")
    metric_columns = []
    for metric in COGNITIVE_METRIC_DIRECTIONS:
        metric_columns.extend([metric, f"{metric}_ci_low", f"{metric}_ci_high"])
    required = set(identifier_columns + metric_columns)
    missing = sorted(required - set(method_summary.columns))
    if missing:
        raise ValueError(f"Cognitive result table requires columns: {missing}")
    result = method_summary.loc[
        method_summary["method"].ne("ob1"),
        identifier_columns + metric_columns,
    ].copy()
    if result.empty:
        raise ValueError("Cognitive result table contains no non-OB1 methods")
    result.insert(2, "reference_model", "ob1_tvt")
    method_order = {
        method: index for index, method in enumerate(METHOD_COLUMNS) if method != "ob1"
    }
    result["_method_order"] = result["method"].map(method_order)
    return (
        result.sort_values("_method_order")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def cognitive_contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Select paired ET1 contrasts whose reference allocation is OB1 TVT."""
    if contrasts.empty:
        columns = [
            "candidate",
            "baseline",
            "reference_model",
            "metric",
            "positive_means_improvement",
            "passages",
            "mean_paired_improvement",
            "ci_low",
            "ci_high",
            "permutation_p_two_sided",
        ]
        if "clusters" in contrasts.columns:
            columns.insert(6, "clusters")
        return pd.DataFrame(columns=columns)
    if "metric" not in contrasts:
        raise ValueError("Cognitive contrast table requires a metric column")
    result = contrasts.loc[contrasts["metric"].isin(COGNITIVE_METRIC_DIRECTIONS)].copy()
    if result.empty:
        raise ValueError("Cognitive contrast table contains no OB1-referenced metrics")
    result.insert(2, "reference_model", "ob1_tvt")
    return result.reset_index(drop=True)


def matched_asymmetry_contrast_table(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Select learned asymmetric versus fixed sigma-one SymGaussian."""
    required = {"candidate", "baseline", "metric"}
    missing = sorted(required - set(contrasts.columns))
    if missing:
        raise ValueError(f"Matched asymmetry table requires columns: {missing}")
    result = contrasts.loc[
        contrasts["candidate"].eq("et1_asymmetric")
        & contrasts["baseline"].eq("et1_symmetric")
    ].copy()
    if result.empty:
        columns = list(contrasts.columns)
        for index, column in enumerate(
            [
                "contrast_interpretation",
                "human_reference",
                "ob1_reference",
            ],
            start=2,
        ):
            if column not in columns:
                columns.insert(index, column)
        return pd.DataFrame(columns=columns)
    result.insert(
        2,
        "contrast_interpretation",
        "learned_asymmetric_vs_fixed_symgaussian_sigma1",
    )
    result.insert(
        3,
        "human_reference",
        result["metric"].isin(HUMAN_METRIC_DIRECTIONS),
    )
    result.insert(
        4,
        "ob1_reference",
        result["metric"].isin(COGNITIVE_METRIC_DIRECTIONS),
    )
    return result.reset_index(drop=True)


def plot_human_spearman(
    passage_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot passage-level Human Spearman distributions with visible points."""
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )["human_spearman"]
        .mean()
        .reset_index()
    )
    methods = [
        method for method in METHOD_COLUMNS if method in per_passage["method"].unique()
    ]
    values = [
        per_passage.loc[
            per_passage["method"] == method,
            "human_spearman",
        ].to_numpy()
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    label_argument = (
        {"tick_labels": [DISPLAY_NAMES[item] for item in methods]}
        if matplotlib.__version_info__[:2] >= (3, 9)
        else {"labels": [DISPLAY_NAMES[item] for item in methods]}
    )
    axis.boxplot(values, **label_argument)
    rng = np.random.default_rng(0)
    for index, method_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(method_values))
        axis.scatter(
            np.full(len(method_values), index) + jitter,
            method_values,
            s=14,
            alpha=0.55,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("Human word-level TRT Spearman")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_evaluation_outputs(
    output_dir: Path,
    word_values: pd.DataFrame,
    passage_metrics: pd.DataFrame,
    method_summary: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    audit: dict,
    checkpoint_contrasts: pd.DataFrame | None = None,
    checkpoint_metadata: pd.DataFrame | None = None,
) -> None:
    """Write machine-readable metrics, rebuttal table, plot, and audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_summary = attach_checkpoint_metadata(
        checkpoint_summary,
        checkpoint_metadata,
    )
    if checkpoint_contrasts is not None:
        checkpoint_contrasts = attach_checkpoint_metadata(
            checkpoint_contrasts,
            checkpoint_metadata,
        )
    cognitive_summary = cognitive_result_table(method_summary)
    cognitive_contrasts = cognitive_contrast_table(contrasts)
    matched_asymmetry = matched_asymmetry_contrast_table(contrasts)
    word_values.to_csv(output_dir / "word_level_values.csv", index=False)
    passage_metrics.to_csv(output_dir / "passage_metrics.csv", index=False)
    method_summary.to_csv(output_dir / "result_table.csv", index=False)
    cognitive_summary.to_csv(
        output_dir / "cognitive_result_table.csv",
        index=False,
    )
    checkpoint_summary.to_csv(
        output_dir / "seed_result_table.csv",
        index=False,
    )
    contrasts.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    cognitive_contrasts.to_csv(
        output_dir / "cognitive_bootstrap_summary.csv",
        index=False,
    )
    matched_asymmetry.to_csv(
        output_dir / "matched_asymmetry_contrasts.csv",
        index=False,
    )
    if checkpoint_contrasts is not None:
        checkpoint_contrasts.to_csv(
            output_dir / "checkpoint_bootstrap_summary.csv",
            index=False,
        )
        cognitive_contrast_table(checkpoint_contrasts).to_csv(
            output_dir / "checkpoint_cognitive_bootstrap_summary.csv",
            index=False,
        )
        matched_asymmetry_contrast_table(checkpoint_contrasts).to_csv(
            output_dir / "checkpoint_matched_asymmetry_contrasts.csv",
            index=False,
        )
        build_sigma_sweep_summary(
            checkpoint_summary,
            checkpoint_contrasts,
        ).to_csv(
            output_dir / "sigma_sweep_summary.csv",
            index=False,
        )
    plot_human_spearman(
        passage_metrics,
        output_dir / "human_spearman_by_passage.png",
    )
    with (output_dir / "evaluation_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
