"""Plot six-checkpoint mean OB1 and redistribution attention profiles."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(
    r"(?:^|/)llama3_8b_attention/(s\d{2})/"
    r"(kernel_profiles|reviewer_kernel_summary)\.csv$"
)
PROFILE_COLUMNS = {
    "OB1 Gaussian attention": "ob1_attention_profile",
    "No redistribution": "raw_delta",
    "Paired symmetric control": "support_centered_sd_symmetric",
    "Learned asymmetric redistribution": "learned_asymmetric",
}
METRIC_COLUMNS = {
    "Spearman": "profile_spearman",
    "JS": "js_divergence",
}
METHOD_ROWS = {
    "No redistribution": "raw_delta",
    "Paired symmetric control": "support_centered_sd_symmetric",
    "Learned asymmetric redistribution": "learned_asymmetric",
}
COLORS = {
    "OB1 Gaussian attention": "#0072B2",
    "No redistribution": "#6E6E6E",
    "Paired symmetric control": "#E69F00",
    "Learned asymmetric redistribution": "#009E73",
}
MARKERS = {
    "OB1 Gaussian attention": "o",
    "No redistribution": "D",
    "Paired symmetric control": "s",
    "Learned asymmetric redistribution": "^",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Average the six Llama-3-8B attention analyses and plot "
            "OB1, no redistribution, symmetric redistribution, and "
            "learned asymmetric redistribution for skew 3 and 4."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Verification ZIP or extracted directory containing "
            "llama3_8b_attention/s01 through s06."
        ),
    )
    parser.add_argument(
        "--skew3-analysis-dir",
        type=Path,
        help=(
            "Attention-profile output computed from trajectories generated "
            "with OB1 attention_skew=3."
        ),
    )
    parser.add_argument(
        "--skew4-analysis-dir",
        type=Path,
        help=(
            "Attention-profile output computed from trajectories generated "
            "with OB1 attention_skew=4."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG, PDF, CSV, and caption outputs.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=6,
        help="Required number of checkpoint runs. Default: 6.",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=-3,
        help="Minimum displayed relative T5-token offset. Default: -3.",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=6,
        help="Maximum displayed relative T5-token offset. Default: 6.",
    )
    return parser


def read_tables_from_zip(
    path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read per-run profile and metric tables from a verification ZIP."""
    profiles: dict[str, pd.DataFrame] = {}
    metrics: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            match = RUN_PATTERN.search(member)
            if match is None:
                continue
            run_id, table_name = match.groups()
            with archive.open(member) as handle:
                frame = pd.read_csv(io.BytesIO(handle.read()))
            target = profiles if table_name == "kernel_profiles" else metrics
            if run_id in target:
                raise ValueError(f"Duplicate {table_name} table for {run_id} in {path}")
            target[run_id] = frame
    return profiles, metrics


def read_tables_from_directory(
    path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read per-run profile and metric tables from an extracted directory."""
    profiles: dict[str, pd.DataFrame] = {}
    metrics: dict[str, pd.DataFrame] = {}
    for profile_path in sorted(
        path.glob("**/llama3_8b_attention/s??/kernel_profiles.csv")
    ):
        run_id = profile_path.parent.name
        profiles[run_id] = pd.read_csv(profile_path)
        metric_path = profile_path.parent / "reviewer_kernel_summary.csv"
        if not metric_path.is_file():
            raise FileNotFoundError(metric_path)
        metrics[run_id] = pd.read_csv(metric_path)
    return profiles, metrics


def load_run_tables(
    path: Path,
    expected_runs: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load and validate the requested number of run-level source tables."""
    source = path.expanduser().resolve()
    if source.is_file() and zipfile.is_zipfile(source):
        profiles, metrics = read_tables_from_zip(source)
    elif source.is_dir():
        profiles, metrics = read_tables_from_directory(source)
    else:
        raise FileNotFoundError(
            f"Input must be a verification ZIP or extracted directory: {source}"
        )
    expected_ids = [f"s{index:02d}" for index in range(1, expected_runs + 1)]
    if sorted(profiles) != expected_ids:
        raise ValueError(
            f"Expected profile runs {expected_ids}, found {sorted(profiles)}"
        )
    if sorted(metrics) != expected_ids:
        raise ValueError(
            f"Expected metric runs {expected_ids}, found {sorted(metrics)}"
        )
    return profiles, metrics


def read_trajectory_matched_analysis(
    path: Path,
    expected_skew: float,
    expected_runs: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Load one analysis whose saved trajectories match its attention skew."""
    source = path.expanduser().resolve()
    profile_path = source / "kernel_profiles.csv"
    metric_path = source / "reviewer_kernel_summary.csv"
    audit_path = source / "attention_profile_audit.json"
    for required in (profile_path, metric_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    trajectory_skew = float(audit["trajectory_attention_skew"])
    if not math.isclose(
        trajectory_skew,
        expected_skew,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{source} contains attention_skew={trajectory_skew:g}, "
            f"expected {expected_skew:g}"
        )
    if int(audit["checkpoint_count"]) != expected_runs:
        raise ValueError(
            f"{source} contains {audit['checkpoint_count']} checkpoints, "
            f"expected {expected_runs}"
        )
    if int(audit["seed_count"]) != 100 or int(audit["passage_count"]) != 55:
        raise ValueError(f"{source} must contain 100 readers and 55 Provo passages")

    profile_frame = pd.read_csv(profile_path)
    metric_frame = pd.read_csv(metric_path)
    expected_ids = [f"s{index:02d}" for index in range(1, expected_runs + 1)]
    profiles: dict[str, pd.DataFrame] = {}
    metrics: dict[str, pd.DataFrame] = {}
    for run_id in expected_ids:
        checkpoint_profile = profile_frame.loc[
            profile_frame["checkpoint_id"].astype(str).str.startswith(run_id)
            & profile_frame["ob1_attention_skew"].eq(expected_skew)
        ].copy()
        checkpoint_metric = metric_frame.loc[
            metric_frame["checkpoint_id"].astype(str).str.startswith(run_id)
            & metric_frame["ob1_attention_skew"].eq(expected_skew)
        ].copy()
        if checkpoint_profile.empty or checkpoint_metric.empty:
            raise ValueError(
                f"{source} is missing trajectory-matched rows for {run_id}"
            )
        for frame, label in (
            (checkpoint_profile, "profile"),
            (checkpoint_metric, "metric"),
        ):
            if "requested_skew_matches_trajectory" not in frame:
                raise ValueError(f"{source} {label} table lacks skew provenance")
            matched = frame["requested_skew_matches_trajectory"].astype(str).str.lower()
            if not matched.eq("true").all():
                raise ValueError(
                    f"{source} {label} rows for {run_id} are not trajectory matched"
                )
        profiles[run_id] = checkpoint_profile
        metrics[run_id] = checkpoint_metric
    return profiles, metrics, audit


def load_trajectory_matched_tables(
    skew3_path: Path,
    skew4_path: Path,
    expected_runs: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Combine skew-specific analyses generated from matching trajectories."""
    profiles3, metrics3, audit3 = read_trajectory_matched_analysis(
        skew3_path,
        3.0,
        expected_runs,
    )
    profiles4, metrics4, audit4 = read_trajectory_matched_analysis(
        skew4_path,
        4.0,
        expected_runs,
    )
    profiles = {
        run_id: pd.concat(
            [profiles3[run_id], profiles4[run_id]],
            ignore_index=True,
        )
        for run_id in sorted(profiles3)
    }
    metrics = {
        run_id: pd.concat(
            [metrics3[run_id], metrics4[run_id]],
            ignore_index=True,
        )
        for run_id in sorted(metrics3)
    }
    source_audit = {
        "comparison_design": "trajectory_matched_skew3_and_skew4",
        "skew3_analysis_dir": str(skew3_path.expanduser().resolve()),
        "skew4_analysis_dir": str(skew4_path.expanduser().resolve()),
        "skew3_ob1_fixations_sha256": audit3["ob1_fixations_sha256"],
        "skew4_ob1_fixations_sha256": audit4["ob1_fixations_sha256"],
        "skew3_ob1_worker_manifest_sha256": audit3["ob1_worker_manifest_sha256"],
        "skew4_ob1_worker_manifest_sha256": audit4["ob1_worker_manifest_sha256"],
    }
    if (
        source_audit["skew3_ob1_fixations_sha256"]
        == source_audit["skew4_ob1_fixations_sha256"]
    ):
        raise ValueError(
            "Skew 3 and skew 4 analyses unexpectedly use the same fixation table"
        )
    return profiles, metrics, source_audit


def validate_profile_frame(frame: pd.DataFrame, run_id: str) -> None:
    """Validate one normalized profile table before aggregation."""
    required = {
        "checkpoint_id",
        "ob1_attention_skew",
        "relative_t5_token_offset",
        *PROFILE_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{run_id} profile table is missing {missing}")
    if sorted(frame["ob1_attention_skew"].unique().tolist()) != [3.0, 4.0]:
        raise ValueError(f"{run_id} must contain only skew 3 and skew 4")
    checkpoint_ids = frame["checkpoint_id"].astype(str).unique().tolist()
    if len(checkpoint_ids) != 1 or not checkpoint_ids[0].startswith(run_id):
        raise ValueError(f"{run_id} has unexpected checkpoint IDs: {checkpoint_ids}")
    for skew, group in frame.groupby("ob1_attention_skew"):
        offsets = group["relative_t5_token_offset"].astype(int)
        if offsets.duplicated().any():
            raise ValueError(f"{run_id} skew {skew:g} has duplicate offsets")
        for column in PROFILE_COLUMNS.values():
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{run_id} skew {skew:g} {column} is not a valid mass")
            if not math.isclose(
                float(values.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(f"{run_id} skew {skew:g} {column} does not sum to one")


def aggregate_profiles(
    profiles: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute equal-checkpoint mean profiles and left-center-right masses."""
    long_records: list[dict] = []
    region_records: list[dict] = []
    for run_id, frame in sorted(profiles.items()):
        validate_profile_frame(frame, run_id)
        for skew, group in frame.groupby("ob1_attention_skew", sort=True):
            ordered = group.sort_values("relative_t5_token_offset")
            offsets = ordered["relative_t5_token_offset"].to_numpy(dtype=int)
            for condition, column in PROFILE_COLUMNS.items():
                values = ordered[column].to_numpy(dtype=float)
                for offset, value in zip(offsets, values):
                    long_records.append(
                        {
                            "run_id": run_id,
                            "skew": float(skew),
                            "offset": int(offset),
                            "condition": condition,
                            "normalized_mass": float(value),
                        }
                    )
                for region, mask in (
                    ("Left", offsets < 0),
                    ("Center", offsets == 0),
                    ("Right", offsets > 0),
                ):
                    region_records.append(
                        {
                            "run_id": run_id,
                            "skew": float(skew),
                            "condition": condition,
                            "region": region,
                            "mass": float(values[mask].sum()),
                        }
                    )
    long = pd.DataFrame(long_records)
    region = pd.DataFrame(region_records)
    profile_summary = (
        long.groupby(["skew", "offset", "condition"], sort=True)["normalized_mass"]
        .agg(mean="mean", checkpoint_sd="std", checkpoint_count="count")
        .reset_index()
    )
    region_summary = (
        region.groupby(["skew", "condition", "region"], sort=True)["mass"]
        .agg(mean="mean", checkpoint_sd="std", checkpoint_count="count")
        .reset_index()
    )
    return profile_summary, region_summary


def aggregate_metrics(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute equal-checkpoint means for Spearman correlation and JS divergence."""
    records: list[dict] = []
    for run_id, frame in sorted(metrics.items()):
        required = {
            "ob1_attention_skew",
            "method",
            *METRIC_COLUMNS.values(),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{run_id} metric table is missing {missing}")
        for condition, method in METHOD_ROWS.items():
            selected = frame.loc[frame["method"].eq(method)]
            for row in selected.itertuples(index=False):
                for metric, column in METRIC_COLUMNS.items():
                    records.append(
                        {
                            "run_id": run_id,
                            "skew": float(row.ob1_attention_skew),
                            "condition": condition,
                            "metric": metric,
                            "value": float(getattr(row, column)),
                        }
                    )
    long = pd.DataFrame(records)
    return (
        long.groupby(["skew", "condition", "metric"], sort=True)["value"]
        .agg(mean="mean", checkpoint_sd="std", checkpoint_count="count")
        .reset_index()
    )


def plot_profiles_and_metrics(
    profile_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    x_min: int,
    x_max: int,
) -> None:
    """Plot mean offset profiles with the two reported comparison metrics."""
    skews = [3.0, 4.0]
    conditions = list(PROFILE_COLUMNS)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.4, 8.2),
        gridspec_kw={"height_ratios": [1.65, 1.0]},
    )
    for column_index, skew in enumerate(skews):
        profile_axis = axes[0, column_index]
        skew_profiles = profile_summary.loc[profile_summary["skew"].eq(skew)]
        for condition in conditions:
            rows = (
                skew_profiles.loc[skew_profiles["condition"].eq(condition)]
                .sort_values("offset")
                .reset_index(drop=True)
            )
            shown = rows.loc[rows["offset"].between(x_min, x_max)]
            x = shown["offset"].to_numpy(dtype=float)
            y = shown["mean"].to_numpy(dtype=float)
            profile_axis.plot(
                x,
                y,
                color=COLORS[condition],
                marker=MARKERS[condition],
                linewidth=2.2,
                markersize=4.4,
                label=condition,
            )
            if condition in {
                "Paired symmetric control",
                "Learned asymmetric redistribution",
            }:
                sd = shown["checkpoint_sd"].fillna(0).to_numpy(dtype=float)
                profile_axis.fill_between(
                    x,
                    np.maximum(0.0, y - sd),
                    np.minimum(1.0, y + sd),
                    color=COLORS[condition],
                    alpha=0.13,
                    linewidth=0,
                )
        profile_axis.axvline(0, color="#777777", linewidth=0.9, alpha=0.7)
        setting = "released code" if skew == 3.0 else "human-motivated 4:1"
        profile_axis.set_title(f"OB1 attention skew = {skew:g} ({setting})")
        profile_axis.set_xlim(x_min, x_max)
        profile_axis.set_ylim(-0.02, 1.04)
        profile_axis.set_xticks(range(x_min, x_max + 1))
        profile_axis.set_ylabel("Normalized weight")
        profile_axis.set_xlabel(
            "Relative T5-token position (0 = fixation-aligned token)"
        )
        profile_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.6)

    metric_specs = (
        (
            "Spearman",
            "Spearman correlation with OB1 (higher = closer)",
            "Same ordering of high- and low-weight token positions",
            (0.0, 1.04),
        ),
        (
            "JS",
            "Jensen–Shannon divergence from OB1 (lower = closer)",
            "Same complete normalized allocation shape",
            (0.0, 0.32),
        ),
    )
    metric_conditions = list(METHOD_ROWS)
    x_positions = np.arange(len(skews), dtype=float)
    width = 0.23
    for axis, (metric, title, subtitle, y_limits) in zip(axes[1], metric_specs):
        for condition_index, condition in enumerate(metric_conditions):
            rows = (
                metric_summary.loc[
                    metric_summary["metric"].eq(metric)
                    & metric_summary["condition"].eq(condition)
                ]
                .set_index("skew")
                .reindex(skews)
            )
            values = rows["mean"].to_numpy(dtype=float)
            errors = rows["checkpoint_sd"].fillna(0).to_numpy(dtype=float)
            positions = (
                x_positions
                + (condition_index - (len(metric_conditions) - 1) / 2) * width
            )
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[condition],
                yerr=errors,
                error_kw={"elinewidth": 0.8, "capsize": 2.0},
                alpha=0.9,
            )
            label_padding = 0.018 if metric == "Spearman" else 0.007
            for bar, value, error in zip(bars, values, errors):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + label_padding,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        axis.set_xticks(x_positions, ["skew=3", "skew=4"])
        axis.set_ylim(*y_limits)
        axis.set_ylabel("Correlation" if metric == "Spearman" else "Divergence")
        axis.set_xlabel("OB1 attention setting")
        axis.set_title(f"{title}\n{subtitle}", fontsize=10.5)
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.7,
            alpha=0.6,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "OB1 attention and ET1 redistribution averaged over six Llama-3-8B checkpoints",
        y=1.055,
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.035,
        "Lines and bars are equal-checkpoint means; shading and error bars "
        "show ±1 SD across six checkpoints.\n"
        "Each panel uses 100 OB1 trajectories generated under the indicated "
        "attention skew. Within each panel, all conditions use the same "
        "mapped fixation contexts.\n"
        "The paired symmetric and learned asymmetric kernels have matched "
        "overall token-space spread checkpoint by checkpoint and differ in "
        "left–right directionality.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def plot_region_mass(
    region_summary: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Plot left-center-right pooled mass as a descriptive directionality check."""
    skews = [3.0, 4.0]
    regions = ["Left", "Center", "Right"]
    conditions = list(PROFILE_COLUMNS)
    x_positions = np.arange(len(regions), dtype=float)
    width = 0.19
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), sharey=True)
    for axis, skew in zip(axes, skews):
        for condition_index, condition in enumerate(conditions):
            rows = (
                region_summary.loc[
                    region_summary["skew"].eq(skew)
                    & region_summary["condition"].eq(condition)
                ]
                .set_index("region")
                .reindex(regions)
            )
            values = rows["mean"].to_numpy(dtype=float)
            errors = rows["checkpoint_sd"].fillna(0).to_numpy(dtype=float)
            positions = (
                x_positions + (condition_index - (len(conditions) - 1) / 2) * width
            )
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[condition],
                yerr=errors,
                error_kw={"elinewidth": 0.8, "capsize": 2.0},
                alpha=0.9,
                label=condition,
            )
            for bar, value, error in zip(bars, values, errors):
                if value < 0.005:
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + 0.025,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                )
        setting = "released code" if skew == 3.0 else "human-motivated 4:1"
        axis.set_title(f"OB1 attention skew = {skew:g} ({setting})")
        axis.set_xticks(x_positions, regions)
        axis.set_ylim(0.0, 1.12)
        axis.set_xlabel("Relative-position region")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.6)
    axes[0].set_ylabel("Pooled normalized weight")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "Direction of OB1 attention and ET1 redistribution "
        "averaged over six Llama-3-8B checkpoints",
        y=1.075,
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.035,
        "Left, center, and right weights are sums over relative T5-token "
        "positions below, equal to, and above zero; error bars show ±1 SD "
        "across six checkpoints.\n"
        "This is a descriptive directionality summary, not a distribution-"
        "similarity metric. Equal Gaussian scales do not imply equal pooled "
        "left/right mass on OB1's right-heavy mapped window.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def write_caption(path: Path) -> None:
    """Write a reviewer-ready descriptive caption for the generated figure."""
    caption = (
        "Top: mean normalized token-position weights for OB1 Gaussian "
        "attention, no redistribution, the paired symmetric control, and "
        "learned asymmetric redistribution under OB1 attention skew 3 and 4. "
        "Bottom: Spearman rank correspondence, which measures agreement in "
        "the ordering of high- and low-weight token positions, and "
        "Jensen–Shannon divergence, which measures the difference between "
        "the complete normalized allocation shapes. Higher Spearman and lower "
        "Jensen–Shannon values indicate closer correspondence with OB1. "
        "Results were computed separately for six independently learned "
        "Llama-3-8B checkpoints and then averaged with equal checkpoint "
        "weight; shading and error bars show one standard deviation. For each "
        "checkpoint, the paired symmetric control was matched to the frozen "
        "learned kernel's centered token-position standard deviation using "
        "only learned parameters and mapped token geometry. The two Gaussian "
        "conditions therefore share the same overall spread and differ mainly "
        "in left–right directionality, explaining their relatively small "
        "metric differences. The skew=3 and skew=4 panels use separate sets "
        "of 100 OB1 trajectories generated under their corresponding skew. "
        "Within each panel, all conditions were evaluated on the same "
        "OB1–Provo five-word window (n−1 to n+3) mapped to ET1 T5 tokens."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def write_region_caption(path: Path) -> None:
    """Write a caption for the descriptive left-center-right mass figure."""
    caption = (
        "Pooled normalized weight to the left of, at, and to the right of "
        "the fixation-aligned T5 token under OB1 attention skew 3 and 4. "
        "Results for the redistribution conditions were computed separately "
        "for six Llama-3-8B checkpoints and then averaged with equal "
        "checkpoint weight; error bars show one standard deviation. This "
        "regional pooling is a descriptive summary of directionality rather "
        "than a distribution-similarity metric because it discards the exact "
        "token positions within each region. The paired symmetric control can "
        "have unequal pooled left and right mass because every condition is "
        "restricted to the same right-heavy OB1–Provo window mapped to T5 "
        "tokens. Each skew panel uses trajectories generated under that "
        "same skew."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def write_reviewer_metric_table(
    metric_summary: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Write the four reviewer metrics as equal-checkpoint mean values."""
    rows = []
    for condition in METHOD_ROWS:
        record: dict[str, object] = {"Condition": condition}
        for skew in (3.0, 4.0):
            for metric in ("Spearman", "JS"):
                selected = metric_summary.loc[
                    metric_summary["condition"].eq(condition)
                    & metric_summary["skew"].eq(skew)
                    & metric_summary["metric"].eq(metric),
                    "mean",
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one {condition}, skew={skew:g}, {metric} aggregate"
                    )
                record[f"Skew={int(skew)} {metric}"] = float(selected.item())
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)
    lines = [
        "| Condition | Skew=3 Spearman ↑ | Skew=3 JS ↓ | "
        "Skew=4 Spearman ↑ | Skew=4 JS ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    """Generate the requested six-checkpoint figure and summary tables."""
    legacy_mode = args.input is not None
    matched_mode = (
        args.skew3_analysis_dir is not None or args.skew4_analysis_dir is not None
    )
    if legacy_mode == matched_mode:
        raise ValueError(
            "Pass either --input or both --skew3-analysis-dir and --skew4-analysis-dir"
        )
    if legacy_mode:
        profiles, metrics = load_run_tables(args.input, args.expected_runs)
        source_audit = {
            "comparison_design": "legacy_combined_input",
            "input": str(args.input.expanduser().resolve()),
        }
    else:
        if args.skew3_analysis_dir is None or args.skew4_analysis_dir is None:
            raise ValueError(
                "Both trajectory-matched analysis directories are required"
            )
        profiles, metrics, source_audit = load_trajectory_matched_tables(
            args.skew3_analysis_dir,
            args.skew4_analysis_dir,
            args.expected_runs,
        )
    profile_summary, region_summary = aggregate_profiles(profiles)
    metric_summary = aggregate_metrics(metrics)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = output_dir / "mean_offset_profiles.csv"
    region_csv = output_dir / "mean_region_mass.csv"
    metric_csv = output_dir / "mean_metric_table.csv"
    png_path = output_dir / "ob1_mean_profiles.png"
    pdf_path = output_dir / "ob1_mean_profiles.pdf"
    region_png_path = output_dir / "ob1_region_mass.png"
    region_pdf_path = output_dir / "ob1_region_mass.pdf"
    caption_path = output_dir / "figure_caption.txt"
    region_caption_path = output_dir / "region_mass_caption.txt"
    reviewer_table_path = output_dir / "reviewer_metric_table.csv"
    reviewer_markdown_path = output_dir / "reviewer_metric_table.md"
    profile_summary.to_csv(profile_csv, index=False)
    region_summary.to_csv(region_csv, index=False)
    metric_summary.to_csv(metric_csv, index=False)
    plot_profiles_and_metrics(
        profile_summary,
        metric_summary,
        png_path,
        pdf_path,
        args.x_min,
        args.x_max,
    )
    plot_region_mass(
        region_summary,
        region_png_path,
        region_pdf_path,
    )
    write_caption(caption_path)
    write_region_caption(region_caption_path)
    write_reviewer_metric_table(
        metric_summary,
        reviewer_table_path,
        reviewer_markdown_path,
    )
    result = {
        **source_audit,
        "run_ids": sorted(profiles),
        "checkpoint_count": len(profiles),
        "checkpoint_weighting": "equal",
        "displayed_skews": [3, 4],
        "displayed_conditions": list(PROFILE_COLUMNS),
        "x_axis_range": [int(args.x_min), int(args.x_max)],
        "profile_source_rows": len(profile_summary),
        "region_source_rows": len(region_summary),
        "metric_source_rows": len(metric_summary),
        "profile_png": str(png_path),
        "profile_pdf": str(pdf_path),
        "region_mass_png": str(region_png_path),
        "region_mass_pdf": str(region_pdf_path),
        "profile_csv": str(profile_csv),
        "region_csv": str(region_csv),
        "metric_csv": str(metric_csv),
        "caption": str(caption_path),
        "region_mass_caption": str(region_caption_path),
        "reviewer_metric_table_csv": str(reviewer_table_path),
        "reviewer_metric_table_markdown": str(reviewer_markdown_path),
    }
    (output_dir / "figure_audit.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Run the figure generator."""
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
