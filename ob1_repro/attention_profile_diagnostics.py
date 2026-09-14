"""Build reviewer-facing diagnostics for OB1 kernel-profile comparisons."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


PROFILE_PLOT_METRICS = (
    ("profile_spearman", "Spearman rank correlation", "higher"),
    ("js_divergence", "Jensen-Shannon divergence", "lower"),
    ("hellinger_distance", "Hellinger distance", "lower"),
    ("total_variation_distance", "Total variation distance", "lower"),
)
PROFILE_REGION_COLUMNS = (
    "left_mass",
    "center_mass",
    "adjacent_right_mass",
    "distant_right_mass",
)
PROFILE_REGION_LABELS = (
    "Left (< 0)",
    "Center (= 0)",
    "Adjacent right (= +1)",
    "Distant right (>= +2)",
)
PROFILE_REGION_COLORS = (
    "#4477AA",
    "#BBBBBB",
    "#66CCEE",
    "#EE6677",
)


def safe_identifier(value: object) -> str:
    """Return a filesystem-safe identifier for one checkpoint label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def summarize_profile_regions(
    profiles: pd.DataFrame,
    profile_columns: tuple[str, ...],
    display_names: dict[str, str],
) -> pd.DataFrame:
    """Partition every normalized profile into four exhaustive regions."""
    required = {
        "checkpoint_id",
        "ob1_attention_skew",
        "relative_t5_token_offset",
        *profile_columns,
    }
    missing = sorted(required - set(profiles.columns))
    if missing:
        raise ValueError(f"Profile-region input is missing columns: {missing}")
    records = []
    for (checkpoint_id, skew), group in profiles.groupby(
        ["checkpoint_id", "ob1_attention_skew"],
        sort=True,
    ):
        offsets = group["relative_t5_token_offset"].to_numpy(dtype=int)
        for method in profile_columns:
            values = group[method].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Profile-region mass must be finite and nonnegative")
            total = float(values.sum())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("Every profile-region source must sum to one")
            record = {
                "checkpoint_id": checkpoint_id,
                "ob1_attention_skew": float(skew),
                "method": method,
                "display_name": display_names[method],
                "left_mass": float(values[offsets < 0].sum()),
                "center_mass": float(values[offsets == 0].sum()),
                "adjacent_right_mass": float(values[offsets == 1].sum()),
                "distant_right_mass": float(values[offsets >= 2].sum()),
            }
            record["region_mass_sum"] = sum(
                float(record[column]) for column in PROFILE_REGION_COLUMNS
            )
            records.append(record)
    result = pd.DataFrame(records)
    if not np.allclose(
        result["region_mass_sum"].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Profile-region partition does not preserve mass")
    return result


def plot_metric_comparison(
    result_table: pd.DataFrame,
    output_path: Path,
    method_order: tuple[str, ...],
    short_display_names: dict[str, str],
) -> None:
    """Plot every profile metric with passage-bootstrap confidence intervals."""
    required = {
        "ob1_attention_skew",
        "method",
        *[
            column
            for metric, _, _ in PROFILE_PLOT_METRICS
            for column in (
                metric,
                f"{metric}_ci_low",
                f"{metric}_ci_high",
            )
        ],
    }
    missing = sorted(required - set(result_table.columns))
    if missing:
        raise ValueError(f"Metric-plot input is missing columns: {missing}")
    methods = [
        method for method in method_order if method in set(result_table["method"])
    ]
    if not methods:
        raise ValueError("Metric plot has no recognized methods")
    skews = sorted(result_table["ob1_attention_skew"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.7, len(skews)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    x = np.arange(len(methods), dtype=float)
    for axis, (metric, title, direction) in zip(
        axes.ravel(),
        PROFILE_PLOT_METRICS,
    ):
        for skew_index, (skew, color) in enumerate(zip(skews, colors)):
            table = result_table.loc[
                result_table["ob1_attention_skew"].eq(skew)
            ].set_index("method")
            table = table.loc[methods]
            values = table[metric].to_numpy(dtype=float)
            lows = table[f"{metric}_ci_low"].to_numpy(dtype=float)
            highs = table[f"{metric}_ci_high"].to_numpy(dtype=float)
            errors = np.maximum(
                np.vstack((values - lows, highs - values)),
                0.0,
            )
            shift = (skew_index - (len(skews) - 1) / 2.0) * 0.13
            axis.errorbar(
                x + shift,
                values,
                yerr=errors,
                color=color,
                marker="o",
                markersize=5,
                linewidth=1.2,
                capsize=2.5,
                label=f"OB1 skew {skew:g}",
            )
        axis.set_title(f"{title} ({'↑' if direction == 'higher' else '↓'})")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [short_display_names[method] for method in methods],
            rotation=28,
            ha="right",
            fontsize=8,
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "OB1 kernel correspondence by metric "
        "(points: passage means; bars: 95% passage-bootstrap CI)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_profile_regions(
    region_table: pd.DataFrame,
    output_path: Path,
    method_order: tuple[str, ...],
    short_display_names: dict[str, str],
) -> None:
    """Plot center preservation and near-versus-distant right-tail mass."""
    skews = sorted(region_table["ob1_attention_skew"].unique())
    methods = [
        method for method in method_order if method in set(region_table["method"])
    ]
    figure, axes = plt.subplots(
        1,
        len(skews),
        figsize=(7.2 * len(skews), 5.2),
        sharey=True,
        squeeze=False,
    )
    for axis, skew in zip(axes[0], skews):
        table = region_table.loc[region_table["ob1_attention_skew"].eq(skew)].set_index(
            "method"
        )
        table = table.loc[methods]
        bottom = np.zeros(len(methods), dtype=float)
        for column, label, color in zip(
            PROFILE_REGION_COLUMNS,
            PROFILE_REGION_LABELS,
            PROFILE_REGION_COLORS,
        ):
            values = table[column].to_numpy(dtype=float)
            axis.bar(
                np.arange(len(methods)),
                values,
                bottom=bottom,
                label=label,
                color=color,
                width=0.72,
            )
            bottom += values
        axis.set_title(f"OB1 attention skew = {skew:g}")
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(
            [short_display_names[method] for method in methods],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Normalized allocation mass")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper right")
    figure.suptitle(
        "Where each kernel places mass: center preservation and right tail",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def build_sigma_landscape(
    reference_profiles: dict[float, np.ndarray],
    support: np.ndarray,
    candidate_builder: Callable[[float, float], np.ndarray],
    sigma_values: np.ndarray,
    metric_function: Callable[[np.ndarray, np.ndarray, np.ndarray], dict],
) -> pd.DataFrame:
    """Evaluate pooled OB1 correspondence over a two-dimensional sigma grid."""
    grid = np.asarray(sigma_values, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 3
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("Sigma landscape values must be positive and increasing")
    records = []
    offsets = np.asarray(support, dtype=int)
    for sigma_left in grid:
        for sigma_right in grid:
            candidate = candidate_builder(
                float(sigma_left),
                float(sigma_right),
            )
            center = float(candidate[offsets == 0].sum())
            left = float(candidate[offsets < 0].sum())
            right = float(candidate[offsets > 0].sum())
            distant_right = float(candidate[offsets >= 2].sum())
            for skew, reference in sorted(reference_profiles.items()):
                records.append(
                    {
                        "ob1_attention_skew": float(skew),
                        "sigma_left": float(sigma_left),
                        "sigma_right": float(sigma_right),
                        "right_left_ratio": float(sigma_right / sigma_left),
                        "quadratic_side_scale": float(
                            math.sqrt((sigma_left**2 + sigma_right**2) / 2.0)
                        ),
                        **metric_function(reference, candidate, offsets),
                        "left_mass": left,
                        "center_mass": center,
                        "right_mass": right,
                        "distant_right_mass": distant_right,
                        "right_share_of_noncenter_mass": (
                            right / (left + right) if left + right > 0 else np.nan
                        ),
                        "metric_scope": (
                            "descriptive fixation-weighted pooled profile; "
                            "no passage bootstrap"
                        ),
                    }
                )
    return pd.DataFrame(records)


def plot_sigma_landscape(
    landscape: pd.DataFrame,
    parameter_points: pd.DataFrame,
    output_dir: Path,
    checkpoint_id: str,
) -> None:
    """Plot every metric over sigma space with learned and fitted landmarks."""
    checkpoint_points = parameter_points.loc[
        parameter_points["checkpoint_id"].eq(checkpoint_id)
    ]
    if checkpoint_points.empty:
        raise ValueError("Sigma-landscape plot has no checkpoint landmarks")
    sigma_min = float(
        min(
            landscape["sigma_left"].min(),
            landscape["sigma_right"].min(),
        )
    )
    sigma_max = float(
        max(
            landscape["sigma_left"].max(),
            landscape["sigma_right"].max(),
        )
    )
    marked_points = checkpoint_points.loc[checkpoint_points["landscape_marker"].notna()]
    outside = marked_points.loc[
        marked_points["sigma_left"].lt(sigma_min)
        | marked_points["sigma_left"].gt(sigma_max)
        | marked_points["sigma_right"].lt(sigma_min)
        | marked_points["sigma_right"].gt(sigma_max)
    ]
    if not outside.empty:
        labels = ", ".join(outside["short_label"].astype(str).unique())
        raise ValueError(
            "Sigma landscape does not contain every plotted landmark: "
            f"{labels}; expand the landscape sigma range"
        )
    metric_limits = {
        metric: (
            float(landscape[metric].min()),
            float(landscape[metric].max()),
        )
        for metric, _, _ in PROFILE_PLOT_METRICS
    }
    for skew, skew_landscape in landscape.groupby(
        "ob1_attention_skew",
        sort=True,
    ):
        figure, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
        x_values = np.sort(skew_landscape["sigma_left"].unique())
        y_values = np.sort(skew_landscape["sigma_right"].unique())
        for axis, (metric, title, direction) in zip(
            axes.ravel(),
            PROFILE_PLOT_METRICS,
        ):
            matrix = (
                skew_landscape.pivot(
                    index="sigma_right",
                    columns="sigma_left",
                    values=metric,
                )
                .reindex(index=y_values, columns=x_values)
                .to_numpy(dtype=float)
            )
            color_map = "viridis" if direction == "higher" else "viridis_r"
            image = axis.pcolormesh(
                x_values,
                y_values,
                matrix,
                shading="nearest",
                cmap=color_map,
                vmin=metric_limits[metric][0],
                vmax=metric_limits[metric][1],
            )
            axis.plot(
                x_values,
                x_values,
                color="white",
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                label="ratio 1:1",
            )
            for ratio, line_style in ((3.0, ":"), (4.0, "-.")):
                axis.plot(
                    x_values,
                    ratio * x_values,
                    color="white",
                    linewidth=0.9,
                    linestyle=line_style,
                    alpha=0.8,
                    label=f"ratio 1:{ratio:g}",
                )
            points = checkpoint_points.loc[
                checkpoint_points["ob1_attention_skew"].eq(skew)
                & checkpoint_points["landscape_marker"].notna()
            ]
            for point in points.itertuples():
                axis.scatter(
                    point.sigma_left,
                    point.sigma_right,
                    marker=point.landscape_marker,
                    s=float(point.landscape_marker_size),
                    facecolor=point.landscape_facecolor,
                    edgecolor=point.landscape_edgecolor,
                    linewidth=1.1,
                    label=point.short_label,
                    zorder=5,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(x_values.min(), x_values.max())
            axis.set_ylim(y_values.min(), y_values.max())
            axis.set_title(f"{title} ({'↑' if direction == 'higher' else '↓'})")
            axis.set_xlabel("sigma_left (T5-token offsets)")
            axis.set_ylabel("sigma_right (T5-token offsets)")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        figure.legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            ncol=4,
            loc="lower center",
            fontsize=8,
        )
        figure.suptitle(
            f"Sigma landscape against OB1 skew {skew:g}: "
            "pooled descriptive diagnostics; color limits shared across skews",
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.96))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (f"sigma_landscape_skew_{float(skew):g}.png")
        figure.savefig(output_path, dpi=220)
        plt.close(figure)


def plot_parameter_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot pooled metric values for fixed and OB1-fitted parameter controls."""
    row_order = [
        "fixed_symmetric_sigma1",
        "fixed_same_rms_symmetric",
        "fixed_ratio4_same_rms",
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
        "learned_fixed",
        "mirrored_learned",
        "ob1_fit_symmetric",
        "ob1_fit_ratio3",
        "ob1_fit_ratio4",
        "ob1_fit_learned_ratio",
        "ob1_fit_unconstrained",
    ]
    present = [row for row in row_order if row in set(diagnostics["model_id"])]
    skews = sorted(diagnostics["ob1_attention_skew"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.7, len(skews)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    x = np.arange(len(present), dtype=float)
    for axis, (metric, title, direction) in zip(
        axes.ravel(),
        PROFILE_PLOT_METRICS,
    ):
        for skew_index, (skew, color) in enumerate(zip(skews, colors)):
            table = diagnostics.loc[
                diagnostics["ob1_attention_skew"].eq(skew)
            ].set_index("model_id")
            values = table.loc[present, metric].to_numpy(dtype=float)
            shift = (skew_index - (len(skews) - 1) / 2.0) * 0.16
            axis.scatter(
                x + shift,
                values,
                color=color,
                s=32,
                label=f"OB1 skew {skew:g}",
            )
        axis.set_title(f"{title} ({'↑' if direction == 'higher' else '↓'})")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                diagnostics.loc[
                    diagnostics["model_id"].eq(model_id),
                    "short_label",
                ].iloc[0]
                for model_id in present
            ],
            rotation=32,
            ha="right",
            fontsize=7.5,
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Fixed controls versus OB1-fitted Gaussian diagnostics\n"
        "Fixation-weighted pooled descriptive metrics; no passage bootstrap",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
