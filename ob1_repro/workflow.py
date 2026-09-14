"""Coordinate and validate every stage of the standalone OB1 analysis."""

from __future__ import annotations

import json
import math
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import load_manifest, sha256_file, verify_all
from .attention_profile import (
    compare_attention_profiles,
    write_attention_profile_outputs,
)
from .et1_inference import ET1NativePredictor, run_et1_inference
from .figures import run as run_figure_builder
from .ob1_runner import (
    aggregate_ob1_tvt,
    derived_cache_filenames,
    prepare_ob1_runtime,
    run_ob1_subprocess,
    stimulus_word_coordinates,
    write_ob1_aggregation,
)
from .prepare_provo import (
    build_canonical_tables,
    validate_canonical_tables,
    write_canonical_tables,
)
from .tokenizer_fingerprint import tokenizer_fingerprint


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
SIGMA_PATH = ROOT / "config/llama3_8b_six_checkpoints.json"
EXPECTED_PATH = ROOT / "expected/expected_run.json"
EXPECTED_METRIC_PATH = ROOT / "expected/reviewer_metric_table.csv"
PYTHON_HASH_SEED = 20260725
EXPECTED_SEEDS = list(range(100))
TOKEN_GRID_COLUMNS = [
    "checkpoint_id",
    "passage_id_zero_based",
    "token_index",
    "token_id",
    "token",
    "character_start",
    "character_end",
    "is_special",
    "attention_mask",
    "word_id_zero_based",
]


def load_expected() -> dict:
    """Load the checked-in experiment contract."""
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


def load_sigma_records() -> list[dict]:
    """Load and validate the frozen six-checkpoint parameter table."""
    records = json.loads(SIGMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Expected six Llama-3-8B checkpoint records")
    for index, record in enumerate(records, start=1):
        required = {
            "checkpoint_id",
            "source_model",
            "source_dataset",
            "source_accuracy",
            "sigma_left",
            "sigma_right",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"Sigma record {index} is missing {missing}")
        if not str(record["checkpoint_id"]).startswith(f"s{index:02d}_"):
            raise ValueError(f"Unexpected checkpoint ID in record {index}")
        if record["source_model"] != "Llama-3-8B":
            raise ValueError(f"Unexpected model in record {index}")
        for key in ("source_accuracy", "sigma_left", "sigma_right"):
            value = float(record[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} in record {index}")
        if float(record["sigma_left"]) <= 0 or float(record["sigma_right"]) <= 0:
            raise ValueError(f"Non-positive sigma in record {index}")
    return records


def write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically write a deterministic JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_file_hashes() -> dict[str, str]:
    """Return hashes for every checked-in file that defines the analysis."""
    paths = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "asset_manifest.json",
        *sorted(ROOT.glob("*.py")),
        *sorted((ROOT / "ob1_repro").glob("*.py")),
        *sorted((ROOT / "config").glob("*.json")),
        *sorted((ROOT / "expected").glob("*")),
    ]
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def require_empty_or_valid(path: Path, validator, *args) -> dict | None:
    """Reuse a valid stage and reject any nonempty partial stage."""
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"Expected a directory: {path}")
    if not any(path.iterdir()):
        return None
    try:
        return validator(path, *args)
    except Exception as error:
        raise RuntimeError(
            f"Existing output is incomplete or incompatible: {path}. "
            "Move it aside or pass a different --output-root."
        ) from error


def validate_prepared(path: Path = PROCESSED_DIR) -> dict:
    """Validate the canonical Provo tables and their audit."""
    passage_path = path / "provo_passages.csv"
    word_path = path / "provo_words.csv"
    exclusion_path = path / "provo_excluded_positions.csv"
    audit_path = path / "provo_prepare_audit.json"
    for required in (passage_path, word_path, exclusion_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    passages = pd.read_csv(passage_path)
    words = pd.read_csv(word_path)
    exclusions = pd.read_csv(exclusion_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_canonical_tables(passages, words, exclusions, audit)
    return {
        "passages": len(passages),
        "evaluable_words": len(words),
        "excluded_positions": len(exclusions),
        "audit_sha256": sha256_file(audit_path),
    }


def prepare_provo(path: Path = PROCESSED_DIR) -> dict:
    """Build or safely reuse the 55-passage canonical Provo tables."""
    reused = require_empty_or_valid(path, validate_prepared)
    if reused is not None:
        print("reusing verified Provo preparation", flush=True)
        return reused
    assets = load_manifest()
    eye_path = ROOT / assets["provo_eye_tracking"]["destination"]
    predictability_path = ROOT / assets["provo_predictability"]["destination"]
    artifacts = build_canonical_tables(eye_path, predictability_path)
    write_canonical_tables(path, *artifacts)
    result = validate_prepared(path)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def load_prepared(path: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical passage and word tables after full validation."""
    validate_prepared(path)
    return (
        pd.read_csv(path / "provo_passages.csv"),
        pd.read_csv(path / "provo_words.csv"),
    )


def validate_et1_output(path: Path) -> dict:
    """Validate full ET1 predictions and the independent token grid."""
    prediction_path = path / "et1_predictions.csv"
    grid_path = path / "t5_token_grid.csv"
    audit_path = path / "et1_inference_audit.json"
    for required in (prediction_path, grid_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    predictions = pd.read_csv(prediction_path)
    grid = pd.read_csv(grid_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = load_expected()
    if (
        len(predictions) != expected["et1_token_rows"]
        or len(grid) != expected["et1_token_rows"]
    ):
        raise ValueError("ET1 output does not contain the expected 3,715 tokens")
    if list(grid.columns) != TOKEN_GRID_COLUMNS:
        raise ValueError("T5 token-grid schema changed")
    raw = pd.to_numeric(predictions["et1_raw_token_trt"], errors="coerce")
    if raw.isna().any() or not bool(np.isfinite(raw).all()):
        raise ValueError("ET1 predictions contain non-finite values")
    if int(grid["passage_id_zero_based"].nunique()) != expected["passages"]:
        raise ValueError("ET1 output does not contain all 55 passages")
    expected_grid = predictions[TOKEN_GRID_COLUMNS].copy()
    if not expected_grid.equals(grid):
        raise ValueError("ET1 prediction geometry and token grid differ")
    required_audit = {
        "passages": expected["passages"],
        "token_rows": expected["et1_token_rows"],
        "tokenizer_fingerprint": expected["tokenizer_fingerprint"],
        "actual_et1_trt_magnitudes_used": False,
    }
    mismatches = {
        key: {"expected": value, "found": audit.get(key)}
        for key, value in required_audit.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"ET1 audit mismatch: {mismatches}")
    return {
        "passages": expected["passages"],
        "token_rows": len(grid),
        "prediction_sha256": sha256_file(prediction_path),
        "token_grid_sha256": sha256_file(grid_path),
        "audit_sha256": sha256_file(audit_path),
    }


def run_et1(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    """Run ET1 on all passages and save predictions apart from geometry."""
    output_dir = output_root / "et1"
    reused = require_empty_or_valid(output_dir, validate_et1_output)
    if reused is not None:
        print("reusing verified ET1 inference", flush=True)
        return reused
    verify_all()
    passages, words = load_prepared()
    manifest = load_manifest()
    checkpoint_path = ROOT / manifest["et1_checkpoint"]["destination"]
    tokenizer_path = ROOT / "models/t5_tokenizer"
    predictor = ET1NativePredictor(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        tokenizer_cache_dir=ROOT / "models/tokenizer_cache",
    )
    token_frame, _, _, base_audit = run_et1_inference(
        passages,
        words,
        [],
        predictor,
    )
    token_grid = token_frame[TOKEN_GRID_COLUMNS].copy()
    prediction_columns = [*TOKEN_GRID_COLUMNS, "et1_raw_token_trt"]
    predictions = token_frame[prediction_columns].copy()
    raw_values = predictions["et1_raw_token_trt"].to_numpy(dtype=float)
    if not np.isfinite(raw_values).all():
        raise ValueError("ET1 generated non-finite TRT predictions")
    fingerprint = tokenizer_fingerprint(predictor.predictor.fixTokenizer)
    audit = {
        **base_audit,
        "tokenizer_repository": manifest["et1_tokenizer"]["repository"],
        "tokenizer_revision": manifest["et1_tokenizer"]["revision"],
        "tokenizer_fingerprint": fingerprint,
        "et1_checkpoint_commit": manifest["et1_checkpoint"]["commit"],
        "et1_checkpoint_sha256": sha256_file(checkpoint_path),
        "actual_et1_trt_magnitudes_used": False,
        "comparison_consumes": "t5_token_grid.csv geometry only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "et1_predictions.csv", index=False)
    token_grid.to_csv(output_dir / "t5_token_grid.csv", index=False)
    write_json_atomic(output_dir / "et1_inference_audit.json", audit)
    return validate_et1_output(output_dir)


def validate_ob1_output(path: Path, expected_skew: float) -> dict:
    """Validate one complete trajectory set against its worker manifest."""
    fixation_path = path / "ob1_fixations.csv"
    manifest_path = path / "ob1_worker_manifest.json"
    aggregation_path = path / "ob1_aggregation_audit.json"
    for required in (fixation_path, manifest_path, aggregation_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregation = json.loads(aggregation_path.read_text(encoding="utf-8"))
    actual_skew = float(manifest["parameters"]["attention_skew"])
    if not math.isclose(actual_skew, expected_skew, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Trajectory skew is {actual_skew}, expected {expected_skew}")
    if float(aggregation["trajectory_attention_skew"]) != expected_skew:
        raise ValueError("Aggregation skew differs from the trajectory skew")
    seeds = [int(value) for value in manifest["seeds"]]
    if seeds != EXPECTED_SEEDS:
        raise ValueError("OB1 output must contain ordered seeds 0 through 99")
    if int(manifest["n_trials"]) != 55:
        raise ValueError("OB1 output must contain all 55 passages")
    expected_rows = load_expected()[f"skew{int(expected_skew)}_fixation_rows"]
    fixations = pd.read_csv(
        fixation_path,
        usecols=["simulation_id", "seed", "text_id"],
    )
    if (
        len(fixations) != expected_rows
        or int(manifest["fixation_rows"]) != expected_rows
    ):
        raise ValueError(
            f"Expected {expected_rows} skew={expected_skew:g} fixation rows, "
            f"found {len(fixations)}"
        )
    if sorted(fixations["simulation_id"].unique().tolist()) != EXPECTED_SEEDS:
        raise ValueError("OB1 output is missing simulation IDs")
    if sorted(fixations["seed"].unique().tolist()) != EXPECTED_SEEDS:
        raise ValueError("OB1 output is missing seeds")
    if sorted(fixations["text_id"].unique().tolist()) != list(range(55)):
        raise ValueError("OB1 output is missing passage IDs")
    if str(manifest.get("python_hash_seed")) != str(PYTHON_HASH_SEED):
        raise ValueError("OB1 PYTHONHASHSEED differs from the frozen setting")
    return {
        "attention_skew": actual_skew,
        "virtual_readers": 100,
        "passages": 55,
        "fixation_rows": len(fixations),
        "fixations_sha256": sha256_file(fixation_path),
        "worker_manifest_sha256": sha256_file(manifest_path),
    }


def run_ob1(
    skew: int,
    workers: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    """Simulate 100 independently seeded OB1 readers for one skew."""
    if skew not in (3, 4):
        raise ValueError("OB1 skew must be 3 or 4")
    if workers < 1:
        raise ValueError("Worker count must be positive")
    output_dir = output_root / "ob1" / f"skew{skew}"
    reused = require_empty_or_valid(output_dir, validate_ob1_output, float(skew))
    if reused is not None:
        print(f"reusing verified skew={skew} OB1 trajectories", flush=True)
        return reused
    verify_all()
    passages, words = load_prepared()
    runtime_dir = ROOT / "data/ob1_runtime" / f"skew{skew}"
    preparation = prepare_ob1_runtime(
        passages,
        runtime_dir,
        python_hash_seed=PYTHON_HASH_SEED,
        stimulus_name="Provo_Corpus",
    )
    removed_caches = []
    for filename in derived_cache_filenames("Provo_Corpus"):
        cache_path = runtime_dir / "data/processed" / filename
        if cache_path.is_file():
            cache_path.unlink()
            removed_caches.append(filename)
    preparation["removed_preexisting_derived_caches"] = removed_caches
    output_dir.mkdir(parents=True, exist_ok=True)
    transformations = pd.read_csv(preparation["token_transformations_path"])
    transformations.to_csv(output_dir / "ob1_token_transformations.csv", index=False)
    write_json_atomic(output_dir / "ob1_runtime_preparation.json", preparation)
    run_ob1_subprocess(
        runtime_dir,
        output_dir,
        seeds=EXPECTED_SEEDS,
        n_trials=55,
        python_hash_seed=PYTHON_HASH_SEED,
        workers=min(workers, len(EXPECTED_SEEDS)),
        stimulus_name="Provo_Corpus",
        attention_skew=float(skew),
    )
    fixations = pd.read_csv(output_dir / "ob1_fixations.csv")
    artifacts = aggregate_ob1_tvt(
        fixations,
        words,
        valid_fixation_coordinates=stimulus_word_coordinates(passages),
        expected_seeds=EXPECTED_SEEDS,
    )
    aggregation_audit = {
        **artifacts[-1],
        "corpus": "provo",
        "stimulus_name": "Provo_Corpus",
        "n_trials": 55,
        "trajectory_attention_skew": float(skew),
    }
    write_ob1_aggregation(output_dir, *artifacts[:-1], aggregation_audit)
    return validate_ob1_output(output_dir, float(skew))


def _validate_manifest_against_fixations(
    manifest: dict,
    fixations: pd.DataFrame,
    passages: pd.DataFrame,
) -> dict:
    """Cross-check trajectory rows, seed mapping, passages, and skew."""
    required = {"condition", "fixation_rows", "parameters", "runtimes", "seeds"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"OB1 manifest is missing {missing}")
    if int(manifest["fixation_rows"]) != len(fixations):
        raise ValueError("OB1 manifest fixation count differs from the CSV")
    seeds = [int(value) for value in manifest["seeds"]]
    if seeds != sorted(fixations["seed"].astype(int).unique().tolist()):
        raise ValueError("OB1 manifest seed set differs from the CSV")
    runtime_pairs = {
        (int(item["simulation_id"]), int(item["seed"])) for item in manifest["runtimes"]
    }
    fixation_pairs = {
        tuple(map(int, row))
        for row in fixations[["simulation_id", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if runtime_pairs != fixation_pairs:
        raise ValueError("OB1 manifest simulation-to-seed mapping differs from the CSV")
    passage_ids = sorted(passages["passage_id_zero_based"].astype(int).unique())
    if sorted(fixations["text_id"].astype(int).unique()) != passage_ids:
        raise ValueError("OB1 fixation passage IDs differ from canonical Provo")
    if int(manifest["n_trials"]) != len(passage_ids):
        raise ValueError("OB1 manifest passage count differs from canonical Provo")
    skew = float(manifest["parameters"]["attention_skew"])
    if not math.isfinite(skew) or skew < 1:
        raise ValueError("Invalid trajectory attention skew")
    return {
        "trajectory_attention_skew": skew,
        "validated_trial_count": len(passage_ids),
        "trial_count_validation_source": "manifest and exact fixation grid",
        "legacy_parallel_manifest_missing_n_trials": False,
    }


def validate_analysis_output(path: Path, expected_skew: float) -> dict:
    """Validate one six-checkpoint trajectory-matched analysis."""
    profile_path = path / "kernel_profiles.csv"
    metric_path = path / "reviewer_kernel_summary.csv"
    audit_path = path / "attention_profile_audit.json"
    for required in (profile_path, metric_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    profiles = pd.read_csv(profile_path)
    metrics = pd.read_csv(metric_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_audit = {
        "checkpoint_count": 6,
        "passage_count": 55,
        "seed_count": 100,
        "candidate_support_policy": "fixation_matched",
        "profile_component": "focused",
        "fixation_weighting": "duration",
        "actual_et1_trt_magnitudes_used": False,
        "support_rms_displacement_controls_enabled": False,
        "support_centered_sd_controls_enabled": True,
    }
    mismatches = {
        key: {"expected": value, "found": audit.get(key)}
        for key, value in expected_audit.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Attention-analysis audit mismatch: {mismatches}")
    spread_matches = audit.get("support_centered_sd_matches")
    if not isinstance(spread_matches, dict) or len(spread_matches) != 6:
        raise ValueError("Attention analysis lacks six spread-matching records")
    for checkpoint_id, record in spread_matches.items():
        symmetric = record.get("symmetric_ratio1", {})
        if symmetric.get("right_left_ratio") != 1.0:
            raise ValueError(f"Symmetric control ratio changed for {checkpoint_id}")
        if symmetric.get("absolute_match_error", math.inf) > 1e-6:
            raise ValueError(f"Spread match exceeds tolerance for {checkpoint_id}")
        if record.get("target_profile_uses_ob1_attention_weights") is not False:
            raise ValueError(f"Spread target improperly uses OB1 for {checkpoint_id}")
    if float(audit["trajectory_attention_skew"]) != expected_skew:
        raise ValueError("Attention analysis uses the wrong trajectory skew")
    for frame in (profiles, metrics):
        if set(frame["ob1_attention_skew"].astype(float)) != {expected_skew}:
            raise ValueError("Attention analysis includes a nonmatching formula skew")
        matched = frame["requested_skew_matches_trajectory"].astype(str).str.lower()
        if not matched.eq("true").all():
            raise ValueError("Attention analysis is not trajectory matched")
        if frame["checkpoint_id"].astype(str).nunique() != 6:
            raise ValueError("Attention analysis does not contain six checkpoints")
    return {
        "attention_skew": expected_skew,
        "checkpoint_count": 6,
        "profile_sha256": sha256_file(profile_path),
        "metric_sha256": sha256_file(metric_path),
        "audit_sha256": sha256_file(audit_path),
    }


def run_analysis(skew: int, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    """Compare three kernels against matching-skew OB1 trajectories."""
    if skew not in (3, 4):
        raise ValueError("Analysis skew must be 3 or 4")
    output_dir = output_root / "analysis" / f"skew{skew}"
    reused = require_empty_or_valid(
        output_dir,
        validate_analysis_output,
        float(skew),
    )
    if reused is not None:
        print(f"reusing verified skew={skew} attention analysis", flush=True)
        return reused
    passages, _ = load_prepared()
    validate_et1_output(output_root / "et1")
    validate_ob1_output(output_root / "ob1" / f"skew{skew}", float(skew))
    token_path = output_root / "et1/t5_token_grid.csv"
    fixation_path = output_root / "ob1" / f"skew{skew}/ob1_fixations.csv"
    manifest_path = output_root / "ob1" / f"skew{skew}/ob1_worker_manifest.json"
    fixations = pd.read_csv(fixation_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_validation = _validate_manifest_against_fixations(
        manifest,
        fixations,
        passages,
    )
    trajectory_skew = float(manifest_validation["trajectory_attention_skew"])
    if trajectory_skew != float(skew):
        raise ValueError("Requested analysis skew differs from trajectory skew")
    artifacts = compare_attention_profiles(
        passages,
        pd.read_csv(token_path),
        fixations,
        load_sigma_records(),
        attention_skews=(float(skew),),
        fixation_weighting="duration",
        profile_component="focused",
        bootstrap_samples=10000,
        seed=PYTHON_HASH_SEED,
        trajectory_attention_skew=trajectory_skew,
        candidate_support_policy="fixation_matched",
        skip_support_rms_displacement_controls=True,
        include_support_centered_sd_controls=True,
    )
    artifacts["audit"].update(
        {
            "et1_token_grid_path": str(token_path.resolve()),
            "et1_token_grid_sha256": sha256_file(token_path),
            "actual_et1_trt_magnitudes_used": False,
            "ob1_fixations_path": str(fixation_path.resolve()),
            "ob1_fixations_sha256": sha256_file(fixation_path),
            "ob1_worker_manifest_path": str(manifest_path.resolve()),
            "ob1_worker_manifest_sha256": sha256_file(manifest_path),
            "ob1_condition": manifest.get("condition"),
            "ob1_manifest_validation": manifest_validation,
            "sigma_json_path": str(SIGMA_PATH.resolve()),
            "sigma_json_sha256": sha256_file(SIGMA_PATH),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_attention_profile_outputs(output_dir, artifacts)
    pd.DataFrame(load_sigma_records()).to_csv(
        output_dir / "checkpoint_sigmas.csv",
        index=False,
    )
    (output_dir / "checkpoint_sigmas.json").write_text(
        SIGMA_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return validate_analysis_output(output_dir, float(skew))


def validate_final_output(path: Path, tolerance: float = 1e-6) -> dict:
    """Validate final tables, figure inputs, and metric regression values."""
    required = (
        "reviewer_metric_table.csv",
        "reviewer_metric_table.md",
        "mean_metric_table.csv",
        "mean_offset_profiles.csv",
        "mean_region_mass.csv",
        "ob1_mean_profiles.png",
        "ob1_mean_profiles.pdf",
        "ob1_region_mass.png",
        "ob1_region_mass.pdf",
        "figure_audit.json",
    )
    for name in required:
        file_path = path / name
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise FileNotFoundError(file_path)
    observed = pd.read_csv(path / "reviewer_metric_table.csv")
    expected = pd.read_csv(EXPECTED_METRIC_PATH)
    if observed.columns.tolist() != expected.columns.tolist():
        raise ValueError("Reviewer metric-table schema changed")
    if observed["Condition"].tolist() != expected["Condition"].tolist():
        raise ValueError("Reviewer metric-table condition order changed")
    numeric_columns = [column for column in expected if column != "Condition"]
    error = np.abs(
        observed[numeric_columns].to_numpy(dtype=float)
        - expected[numeric_columns].to_numpy(dtype=float)
    )
    max_error = float(error.max())
    if max_error > tolerance:
        raise ValueError(
            f"Final metrics differ from the frozen result by {max_error:.12g}"
        )
    profiles = pd.read_csv(path / "mean_offset_profiles.csv")
    regions = pd.read_csv(path / "mean_region_mass.csv")
    metrics = pd.read_csv(path / "mean_metric_table.csv")
    if set(profiles["skew"].astype(float)) != {3.0, 4.0}:
        raise ValueError("Mean profiles do not contain skew 3 and 4")
    if set(regions["region"]) != {"Left", "Center", "Right"}:
        raise ValueError("Region table does not contain left, center, and right")
    if set(metrics["metric"]) != {"Spearman", "JS"}:
        raise ValueError("Metric table must contain only Spearman and JS")
    audit = json.loads((path / "figure_audit.json").read_text(encoding="utf-8"))
    if audit.get("checkpoint_count") != 6:
        raise ValueError("Figure audit does not contain six checkpoints")
    expected_figure_audit = {
        "displayed_skews": [3, 4],
        "displayed_conditions": [
            "OB1 Gaussian attention",
            "No redistribution",
            "Paired symmetric control",
            "Learned asymmetric redistribution",
        ],
        "x_axis_range": [-3, 6],
        "profile_source_rows": len(profiles),
        "region_source_rows": len(regions),
        "metric_source_rows": len(metrics),
    }
    mismatches = {
        key: {"expected": value, "found": audit.get(key)}
        for key, value in expected_figure_audit.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Figure audit mismatch: {mismatches}")
    return {
        "maximum_metric_absolute_error": max_error,
        "profile_rows": len(profiles),
        "region_rows": len(regions),
        "metric_rows": len(metrics),
        "figure_audit_sha256": sha256_file(path / "figure_audit.json"),
    }


def make_figures(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    """Create final figures and reviewer tables from matched analyses."""
    final_dir = output_root / "final"
    reused = require_empty_or_valid(final_dir, validate_final_output)
    if reused is not None:
        print("reusing verified final figures and tables", flush=True)
        return reused
    validate_analysis_output(output_root / "analysis/skew3", 3.0)
    validate_analysis_output(output_root / "analysis/skew4", 4.0)
    result = run_figure_builder(
        Namespace(
            input=None,
            skew3_analysis_dir=output_root / "analysis/skew3",
            skew4_analysis_dir=output_root / "analysis/skew4",
            output_dir=final_dir,
            expected_runs=6,
            x_min=-3,
            x_max=6,
        )
    )
    write_json_atomic(final_dir / "figure_audit.json", result)
    return validate_final_output(final_dir)


def verify_complete(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    """Validate assets, all stages, independent skews, and final regression."""
    assets = verify_all()
    prepared = validate_prepared()
    et1 = validate_et1_output(output_root / "et1")
    skew3 = validate_ob1_output(output_root / "ob1/skew3", 3.0)
    skew4 = validate_ob1_output(output_root / "ob1/skew4", 4.0)
    if skew3["fixations_sha256"] == skew4["fixations_sha256"]:
        raise ValueError("Skew 3 and skew 4 fixation files must differ")
    analysis3 = validate_analysis_output(output_root / "analysis/skew3", 3.0)
    analysis4 = validate_analysis_output(output_root / "analysis/skew4", 4.0)
    final = validate_final_output(output_root / "final")
    execution_log = output_root / "execution.log"
    if not execution_log.is_file() or execution_log.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing complete execution log: {execution_log}. Use run_all.py."
        )
    result = {
        "status": "verified",
        "root": str(ROOT),
        "assets": assets,
        "provo": prepared,
        "et1": et1,
        "ob1_skew3": skew3,
        "ob1_skew4": skew4,
        "analysis_skew3": analysis3,
        "analysis_skew4": analysis4,
        "final": final,
        "actual_et1_trt_magnitudes_used": False,
        "asset_manifest_sha256": sha256_file(ROOT / "asset_manifest.json"),
        "sigma_manifest_sha256": sha256_file(SIGMA_PATH),
        "expected_contract_sha256": sha256_file(EXPECTED_PATH),
        "expected_metrics_sha256": sha256_file(EXPECTED_METRIC_PATH),
        "execution_log": str(execution_log.resolve()),
        "execution_log_sha256_at_verification": sha256_file(execution_log),
        "source_file_sha256": source_file_hashes(),
    }
    write_json_atomic(output_root / "final/provenance_manifest.json", result)
    if not (output_root / "final/execution.log").is_file():
        raise FileNotFoundError(output_root / "final/execution.log")
    return result


def default_workers() -> int:
    """Leave four logical CPUs free and cap OB1 concurrency at forty."""
    return min(40, max(1, (os.cpu_count() or 1) - 4))
