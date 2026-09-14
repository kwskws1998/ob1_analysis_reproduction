"""Prepare, execute, and aggregate the pinned OB1 reading baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "third_party/ob1_reader_provo_2024"
OB1_SOURCE_COMMIT = "56b8d6401d1c2c1886a9c6ff9df4a143c6f2c12d"
WORKER_PATH = ROOT / "ob1_repro/ob1_worker.py"
DEFAULT_RUNTIME_DIR = ROOT / "data/ob1_runtime"
DEFAULT_ONESTOP_RUNTIME_DIR = ROOT / "data/ob1_runtime_onestop"
SUBTLEX_PATH = ROOT / "data/raw/SUBTLEX_UK.txt"
TOKEN_TRANSFORMATION_REASON = "punctuation_only_ob1_empty_token"


def validate_stimulus_name(stimulus_name: str) -> str:
    """Validate a stable filename stem passed into the isolated OB1 runtime."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", stimulus_name):
        raise ValueError(
            "OB1 stimulus_name may contain only letters, digits, and underscores"
        )
    return stimulus_name


def derived_cache_filenames(stimulus_name: str) -> tuple[str, ...]:
    """Return every upstream cache derived from one named stimulus table."""
    stimulus_name = validate_stimulus_name(stimulus_name)
    return (
        "lexicon.pkl",
        (f"frequency_map_{stimulus_name}_continuous_reading_english.json"),
        (f"prediction_map_{stimulus_name}__continuous_reading_english.json"),
        "inhibition_matrix_previous.pkl",
        "inhibition_matrix_parameters_previous.pkl",
    )


DERIVED_CACHE_FILENAMES = derived_cache_filenames("Provo_Corpus")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one runtime input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_ob1_passage(
    passage_id_zero_based: int,
    text: str,
) -> tuple[str, list[dict]]:
    """Replace OB1-empty whitespace tokens while preserving text coordinates."""
    records = []
    word_id = -1

    def replace(match: re.Match) -> str:
        """Return one unchanged lexical token or a length-preserving surrogate."""
        nonlocal word_id
        word_id += 1
        token = match.group(0)
        normalized = re.sub(r"[^\w\s]", "", token).lower().strip()
        if normalized:
            return token
        surrogate = "x" * max(1, len(token))
        records.append(
            {
                "passage_id_zero_based": int(passage_id_zero_based),
                "word_id_zero_based": int(word_id),
                "word_raw": token,
                "ob1_word": surrogate,
                "reason": TOKEN_TRANSFORMATION_REASON,
            }
        )
        return surrogate

    transformed = re.sub(r"\S+", replace, str(text))
    if len(transformed) != len(str(text)):
        raise ValueError("OB1 token transformation changed passage length")
    if len(transformed.split()) != len(str(text).split()):
        raise ValueError("OB1 token transformation changed whitespace token count")
    return transformed, records


def validate_ob1_passages(passages: pd.DataFrame) -> pd.DataFrame:
    """Require the row-index coordinate system used by upstream OB1."""
    required = {"passage_id_zero_based", "passage_text"}
    missing = required.difference(passages.columns)
    if missing:
        raise ValueError(f"OB1 passages are missing columns: {sorted(missing)}")
    if passages.empty:
        raise ValueError("OB1 passages must not be empty")
    if passages[list(required)].isna().any().any():
        raise ValueError("OB1 passages contain missing IDs or text")

    passage_ids = pd.to_numeric(
        passages["passage_id_zero_based"],
        errors="coerce",
    )
    if (
        passage_ids.isna().any()
        or not bool(np.isfinite(passage_ids).all())
        or not bool((passage_ids == np.floor(passage_ids)).all())
    ):
        raise ValueError("OB1 passage IDs must be finite integers")
    passage_ids = passage_ids.astype(int)
    expected_ids = list(range(len(passages)))
    if sorted(passage_ids.tolist()) != expected_ids:
        raise ValueError(
            "OB1 passage IDs must be unique and contiguous from zero because "
            "upstream OB1 emits row-position text IDs"
        )
    passage_text = passages["passage_text"].astype(str)
    if bool(passage_text.map(lambda text: len(text.split()) == 0).any()):
        raise ValueError("OB1 passages must contain at least one word")

    validated = passages.copy()
    validated["passage_id_zero_based"] = passage_ids
    return validated.sort_values("passage_id_zero_based").reset_index(drop=True)


def stimulus_word_coordinates(passages: pd.DataFrame) -> pd.DataFrame:
    """Build the complete whitespace-token coordinate grid simulated by OB1."""
    validated = validate_ob1_passages(passages)
    records = []
    for passage in validated.itertuples():
        passage_id = int(passage.passage_id_zero_based)
        records.extend(
            {
                "passage_id_zero_based": passage_id,
                "word_id_zero_based": word_id,
            }
            for word_id, _ in enumerate(str(passage.passage_text).split())
        )
    return pd.DataFrame(
        records,
        columns=["passage_id_zero_based", "word_id_zero_based"],
    )


def prepare_ob1_runtime(
    passages: pd.DataFrame,
    runtime_dir: Path,
    subtlex_path: Path = SUBTLEX_PATH,
    python_hash_seed: int = 20260725,
    stimulus_name: str = "Provo_Corpus",
) -> dict:
    """Create the exact file layout required by the unmodified OB1 code."""
    stimulus_name = validate_stimulus_name(stimulus_name)
    runtime_dir = runtime_dir.resolve()
    source_working_dir = runtime_dir / "src"
    raw_dir = runtime_dir / "data/raw"
    processed_dir = runtime_dir / "data/processed"
    source_working_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    subtlex_destination = raw_dir / "SUBTLEX_UK.txt"
    if not subtlex_destination.is_file() or sha256_file(
        subtlex_destination
    ) != sha256_file(subtlex_path):
        shutil.copyfile(subtlex_path, subtlex_destination)

    stimuli = validate_ob1_passages(passages)
    transformed_texts = []
    transformation_records = []
    for passage in stimuli.itertuples():
        transformed_text, records = transform_ob1_passage(
            int(passage.passage_id_zero_based),
            str(passage.passage_text),
        )
        transformed_texts.append(transformed_text)
        transformation_records.extend(records)
    transformation_columns = [
        "passage_id_zero_based",
        "word_id_zero_based",
        "word_raw",
        "ob1_word",
        "reason",
    ]
    transformations = pd.DataFrame(
        transformation_records,
        columns=transformation_columns,
    )
    transformations_path = processed_dir / f"{stimulus_name}_token_transformations.csv"
    transformations.to_csv(transformations_path, index=False)
    stimuli_frame = pd.DataFrame(
        {
            "id": stimuli["passage_id_zero_based"].astype(int),
            "all": transformed_texts,
            "words": [
                [item for item in str(text).split()] for text in transformed_texts
            ],
            "word_ids": [
                list(range(len(str(text).split()))) for text in transformed_texts
            ],
        }
    )
    stimuli_path = processed_dir / f"{stimulus_name}.csv"
    stimuli_frame.to_csv(stimuli_path, sep="\t", index=False)
    input_manifest_path = processed_dir / "runtime_input_manifest.json"
    input_manifest = {
        "stimuli_sha256": sha256_file(stimuli_path),
        "subtlex_sha256": sha256_file(subtlex_destination),
        "passages": len(stimuli_frame),
        "python_hash_seed": int(python_hash_seed),
        "vendor_commit": OB1_SOURCE_COMMIT,
        "stimulus_name": stimulus_name,
        "token_transformations_filename": transformations_path.name,
        "token_transformations_sha256": sha256_file(transformations_path),
        "token_transformations": len(transformations),
    }
    previous_manifest = None
    if input_manifest_path.is_file():
        with input_manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
    removed_caches = []
    if previous_manifest != input_manifest:
        for filename in derived_cache_filenames(stimulus_name):
            cache_path = processed_dir / filename
            if cache_path.is_file():
                cache_path.unlink()
                removed_caches.append(filename)
    with input_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(input_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "runtime_dir": str(runtime_dir),
        "source_working_dir": str(source_working_dir),
        "stimuli_path": str(stimuli_path),
        "token_transformations_path": str(transformations_path),
        "token_transformations_sha256": input_manifest["token_transformations_sha256"],
        "subtlex_path": str(subtlex_destination),
        "passages": len(stimuli_frame),
        "token_transformations": len(transformations),
        "stimulus_name": stimulus_name,
        "vendor_source": str(VENDOR_ROOT.resolve()),
        "vendor_commit": OB1_SOURCE_COMMIT,
        "input_manifest": input_manifest,
        "removed_stale_caches": removed_caches,
    }


def run_ob1_subprocess(
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int = 55,
    python_hash_seed: int = 20260725,
    workers: int = 1,
    stimulus_name: str = "Provo_Corpus",
    attention_skew: float | None = None,
) -> None:
    """Run fixed-hash OB1 workers and merge their deterministic outputs."""
    stimulus_name = validate_stimulus_name(stimulus_name)
    if attention_skew is not None and (
        not math.isfinite(attention_skew) or attention_skew < 1
    ):
        raise ValueError("OB1 attention skew must be finite and at least one")
    if not seeds:
        raise ValueError("At least one OB1 virtual-reader seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("OB1 virtual-reader seeds must be unique")
    if workers < 1:
        raise ValueError("OB1 worker count must be positive")
    workers = min(workers, len(seeds))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers == 1:
        run_ob1_worker(
            runtime_dir,
            output_dir,
            seeds,
            n_trials,
            python_hash_seed,
            stimulus_name,
            attention_skew,
        )
        return

    cache_ready = all(
        (runtime_dir / "data/processed" / filename).is_file()
        for filename in derived_cache_filenames(stimulus_name)
    )
    completed_chunks = []
    remaining_seeds = list(seeds)
    worker_root = output_dir / "_parallel_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    if not cache_ready:
        warmup_seed = remaining_seeds.pop(0)
        warmup_dir = worker_root / "warmup"
        run_ob1_worker(
            runtime_dir,
            warmup_dir,
            [warmup_seed],
            n_trials,
            python_hash_seed,
            stimulus_name,
            attention_skew,
        )
        completed_chunks.append(([warmup_seed], warmup_dir))

    chunks = split_seed_chunks(remaining_seeds, workers)
    parallel_chunks = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_dir = worker_root / f"worker_{chunk_index:03d}"
        parallel_chunks.append((chunk, chunk_dir))

    with ThreadPoolExecutor(max_workers=len(parallel_chunks) or 1) as executor:
        futures = {
            executor.submit(
                run_ob1_worker,
                runtime_dir,
                chunk_dir,
                chunk,
                n_trials,
                python_hash_seed,
                stimulus_name,
                attention_skew,
            ): chunk
            for chunk, chunk_dir in parallel_chunks
        }
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            future.result()
            print(
                "OB1 worker chunk completed "
                f"{completed}/{len(futures)} "
                f"(seeds={','.join(map(str, futures[future]))})",
                flush=True,
            )

    completed_chunks.extend(parallel_chunks)
    merge_ob1_worker_outputs(
        output_dir,
        seeds,
        completed_chunks,
        workers_requested=workers,
        python_hash_seed=python_hash_seed,
        n_trials=n_trials,
        stimulus_name=stimulus_name,
        requested_attention_skew=attention_skew,
    )


def split_seed_chunks(seeds: list[int], workers: int) -> list[list[int]]:
    """Distribute seeds evenly across a bounded number of worker processes."""
    if workers < 1:
        raise ValueError("OB1 worker count must be positive")
    worker_count = min(workers, len(seeds))
    if worker_count == 0:
        return []
    chunks = [[] for _ in range(worker_count)]
    for index, seed in enumerate(seeds):
        chunks[index % worker_count].append(seed)
    return [chunk for chunk in chunks if chunk]


def ob1_worker_environment(python_hash_seed: int) -> dict[str, str]:
    """Create a deterministic single-thread environment for one OB1 process."""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(python_hash_seed)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    return environment


def run_ob1_worker(
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int,
    python_hash_seed: int,
    stimulus_name: str = "Provo_Corpus",
    attention_skew: float | None = None,
) -> None:
    """Run one isolated OB1 subprocess for an explicit seed chunk."""
    if not seeds:
        raise ValueError("An OB1 worker cannot receive an empty seed chunk")
    stimulus_name = validate_stimulus_name(stimulus_name)
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--vendor-src",
        str(VENDOR_ROOT / "src"),
        "--runtime-dir",
        str(runtime_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--seeds",
        ",".join(map(str, seeds)),
        "--n-trials",
        str(n_trials),
        "--stimuli-filename",
        f"{stimulus_name}.csv",
    ]
    if attention_skew is not None:
        command.extend(["--attention-skew", str(float(attention_skew))])
    subprocess.run(
        command,
        check=True,
        env=ob1_worker_environment(python_hash_seed),
    )


def merge_ob1_worker_outputs(
    output_dir: Path,
    seeds: list[int],
    chunks: list[tuple[list[int], Path]],
    workers_requested: int,
    python_hash_seed: int,
    n_trials: int | None = None,
    stimulus_name: str = "Provo_Corpus",
    requested_attention_skew: float | None = None,
) -> None:
    """Merge worker CSVs and manifests into the serial output contract."""
    stimulus_name = validate_stimulus_name(stimulus_name)
    seed_to_simulation = {seed: index for index, seed in enumerate(seeds)}
    fixation_frames = []
    runtimes = []
    parameters = None
    chunk_records = []

    for chunk_seeds, chunk_dir in chunks:
        fixation_path = chunk_dir / "ob1_fixations.csv"
        manifest_path = chunk_dir / "ob1_worker_manifest.json"
        if not fixation_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing OB1 worker output under {chunk_dir}")
        fixation_frame = pd.read_csv(fixation_path)
        observed_seeds = set(fixation_frame["seed"].astype(int).unique())
        if observed_seeds != set(chunk_seeds):
            raise ValueError(
                f"OB1 worker seed mismatch in {chunk_dir}: "
                f"{sorted(observed_seeds)} versus {sorted(chunk_seeds)}"
            )
        fixation_frame["simulation_id"] = fixation_frame["seed"].map(seed_to_simulation)
        if fixation_frame["simulation_id"].isna().any():
            raise ValueError(f"Unknown OB1 seed in {chunk_dir}")
        fixation_frame["simulation_id"] = fixation_frame["simulation_id"].astype(int)
        fixation_frames.append(fixation_frame)

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        observed_stimulus = manifest.get(
            "stimuli_filename",
            "Provo_Corpus.csv",
        )
        if observed_stimulus != f"{stimulus_name}.csv":
            raise ValueError(
                f"OB1 worker stimulus mismatch in {chunk_dir}: {observed_stimulus}"
            )
        if parameters is None:
            parameters = manifest["parameters"]
        elif manifest["parameters"] != parameters:
            raise ValueError("Parallel OB1 workers used different parameters")
        for runtime in manifest["runtimes"]:
            runtime = dict(runtime)
            runtime["simulation_id"] = seed_to_simulation[int(runtime["seed"])]
            runtimes.append(runtime)
        chunk_records.append(
            {
                "seeds": [int(seed) for seed in chunk_seeds],
                "output_dir": str(chunk_dir.resolve()),
            }
        )

    fixations = pd.concat(fixation_frames, ignore_index=True)
    fixations = fixations.sort_values(
        ["simulation_id", "text_id", "fixation_counter"]
    ).reset_index(drop=True)
    observed_all_seeds = fixations["seed"].astype(int).drop_duplicates().tolist()
    if observed_all_seeds != seeds:
        raise ValueError(
            f"Merged OB1 seed order mismatch: {observed_all_seeds} versus {seeds}"
        )
    observed_trial_count = int(fixations["text_id"].nunique())
    merged_trial_count = observed_trial_count if n_trials is None else int(n_trials)
    if merged_trial_count != observed_trial_count:
        raise ValueError(
            "Requested OB1 n_trials disagrees with merged fixation passages"
        )
    if parameters is None:
        raise ValueError("Parallel OB1 output is missing model parameters")
    actual_attention_skew = parameters.get("attention_skew")
    if requested_attention_skew is not None:
        if actual_attention_skew is None or not math.isclose(
            float(actual_attention_skew),
            float(requested_attention_skew),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Merged OB1 attention skew differs from the requested value: "
                f"{actual_attention_skew} versus {requested_attention_skew}"
            )
    fixations.to_csv(output_dir / "ob1_fixations.csv", index=False)
    runtimes = sorted(runtimes, key=lambda item: item["simulation_id"])
    with (output_dir / "ob1_worker_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "condition": "baseline_no_predictability",
                "python_hash_seed": str(python_hash_seed),
                "seeds": seeds,
                "parameters": parameters,
                "runtimes": runtimes,
                "fixation_rows": len(fixations),
                "n_trials": merged_trial_count,
                "parallel": True,
                "workers_requested": workers_requested,
                "worker_chunks": chunk_records,
                "stimuli_filename": f"{stimulus_name}.csv",
                "requested_attention_skew": requested_attention_skew,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def aggregate_ob1_tvt(
    fixations: pd.DataFrame,
    canonical_words: pd.DataFrame,
    valid_fixation_coordinates: pd.DataFrame | None = None,
    expected_seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate TVT after validating full simulation and evaluation grids."""
    required = {
        "simulation_id",
        "seed",
        "text_id",
        "word_id",
        "fixation_duration",
    }
    missing = required.difference(fixations.columns)
    if missing:
        raise ValueError(f"OB1 fixation table is missing columns: {sorted(missing)}")
    canonical_required = {
        "passage_id_raw",
        "passage_id_zero_based",
        "word_id_zero_based",
        "word_raw",
    }
    missing_canonical = canonical_required.difference(canonical_words.columns)
    if missing_canonical:
        raise ValueError(
            f"Canonical word table is missing columns: {sorted(missing_canonical)}"
        )
    if fixations.empty:
        raise ValueError("OB1 fixation table must not be empty")
    if canonical_words.empty:
        raise ValueError("Canonical word table must not be empty")
    if fixations[list(required)].isna().any().any():
        raise ValueError("OB1 fixation table contains missing required values")
    durations = pd.to_numeric(
        fixations["fixation_duration"],
        errors="coerce",
    )
    if durations.isna().any() or not bool(np.isfinite(durations).all()):
        raise ValueError("OB1 fixation durations must be finite")
    if bool((durations < 0).any()):
        raise ValueError("OB1 fixation durations must be nonnegative")
    fixations = fixations.copy()
    fixations["fixation_duration"] = durations.astype(float)
    durations = fixations["fixation_duration"]

    canonical_coordinates = canonical_words[
        ["passage_id_zero_based", "word_id_zero_based"]
    ]
    if canonical_coordinates.duplicated().any():
        raise ValueError("Canonical OB1 word coordinates must be unique")
    coordinate_columns = [
        "passage_id_zero_based",
        "word_id_zero_based",
    ]
    if valid_fixation_coordinates is None:
        validation_coordinates = canonical_coordinates
    else:
        missing_valid_columns = set(coordinate_columns).difference(
            valid_fixation_coordinates.columns
        )
        if missing_valid_columns:
            raise ValueError(
                "Valid fixation coordinate table is missing columns: "
                f"{sorted(missing_valid_columns)}"
            )
        validation_coordinates = valid_fixation_coordinates[coordinate_columns]
        if validation_coordinates.duplicated().any():
            raise ValueError("Valid OB1 fixation coordinates must be unique")
        missing_output_coordinates = canonical_coordinates.merge(
            validation_coordinates,
            on=coordinate_columns,
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        missing_output_coordinates = missing_output_coordinates.loc[
            missing_output_coordinates["_merge"] == "left_only",
            coordinate_columns,
        ]
        if not missing_output_coordinates.empty:
            examples = missing_output_coordinates.head(10).to_dict("records")
            raise ValueError(
                "Canonical evaluation coordinates are absent from the full "
                f"OB1 stimulus grid: {examples}"
            )
    fixation_coordinates = (
        fixations[["text_id", "word_id"]]
        .drop_duplicates()
        .rename(
            columns={
                "text_id": "passage_id_zero_based",
                "word_id": "word_id_zero_based",
            }
        )
    )
    unknown_coordinates = fixation_coordinates.merge(
        validation_coordinates,
        on=coordinate_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unknown_coordinates = unknown_coordinates.loc[
        unknown_coordinates["_merge"] == "left_only",
        ["passage_id_zero_based", "word_id_zero_based"],
    ]
    if not unknown_coordinates.empty:
        examples = unknown_coordinates.head(10).to_dict("records")
        raise ValueError(
            "OB1 fixations reference coordinates outside the canonical "
            f"word grid: {examples}"
        )
    output_coordinate_index = pd.MultiIndex.from_frame(canonical_coordinates)
    fixation_coordinate_index = pd.MultiIndex.from_arrays(
        [
            fixations["text_id"].to_numpy(),
            fixations["word_id"].to_numpy(),
        ],
        names=coordinate_columns,
    )
    fixation_in_output_grid = fixation_coordinate_index.isin(output_coordinate_index)
    excluded_fixation_mask = ~np.asarray(fixation_in_output_grid, dtype=bool)

    simulation_ids = sorted(fixations["simulation_id"].unique().tolist())
    simulation_seed_counts = fixations.groupby(
        "simulation_id",
        sort=False,
    )["seed"].nunique(dropna=False)
    if bool((simulation_seed_counts != 1).any()):
        raise ValueError("Every simulation ID must map to exactly one seed")
    seed_simulation_counts = fixations.groupby(
        "seed",
        sort=False,
    )["simulation_id"].nunique(dropna=False)
    if bool((seed_simulation_counts != 1).any()):
        raise ValueError("Every OB1 seed must map to exactly one simulation ID")
    seed_map = (
        fixations[["simulation_id", "seed"]]
        .drop_duplicates()
        .set_index("simulation_id")["seed"]
        .to_dict()
    )
    if len(seed_map) != len(simulation_ids):
        raise ValueError("Every simulation ID must map to exactly one seed")
    if expected_seeds is not None:
        requested_seeds = [int(seed) for seed in expected_seeds]
        if not requested_seeds:
            raise ValueError("Expected OB1 seeds must not be empty")
        if len(set(requested_seeds)) != len(requested_seeds):
            raise ValueError("Expected OB1 seeds must be unique")
        expected_simulation_ids = list(range(len(requested_seeds)))
        if simulation_ids != expected_simulation_ids:
            raise ValueError(
                "OB1 output simulation IDs do not cover every requested "
                f"virtual reader: expected {expected_simulation_ids}, "
                f"found {simulation_ids}"
            )
        observed_seeds = [
            int(seed_map[simulation_id]) for simulation_id in expected_simulation_ids
        ]
        if observed_seeds != requested_seeds:
            raise ValueError(
                "OB1 output seed order differs from the requested virtual "
                f"readers: expected {requested_seeds}, found {observed_seeds}"
            )

    canonical_passages = (
        canonical_coordinates[["passage_id_zero_based"]]
        .drop_duplicates()
        .sort_values("passage_id_zero_based")
    )
    expected_coverage = pd.MultiIndex.from_product(
        [simulation_ids, canonical_passages["passage_id_zero_based"]],
        names=["simulation_id", "passage_id_zero_based"],
    ).to_frame(index=False)
    observed_coverage = (
        fixations[["simulation_id", "text_id"]]
        .drop_duplicates()
        .rename(columns={"text_id": "passage_id_zero_based"})
    )
    coverage = expected_coverage.merge(
        observed_coverage,
        on=["simulation_id", "passage_id_zero_based"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing_coverage = coverage.loc[
        coverage["_merge"] == "left_only",
        ["simulation_id", "passage_id_zero_based"],
    ]
    if not missing_coverage.empty:
        examples = missing_coverage.head(10).to_dict("records")
        raise ValueError(
            "OB1 output is missing every fixation for canonical "
            f"simulation-passage pairs: {examples}"
        )

    grids = []
    canonical = canonical_words[
        [
            "passage_id_raw",
            "passage_id_zero_based",
            "word_id_zero_based",
            "word_raw",
        ]
    ]
    for simulation_id in simulation_ids:
        grid = canonical.copy()
        grid["simulation_id"] = simulation_id
        grid["seed"] = seed_map[simulation_id]
        grids.append(grid)
    complete = pd.concat(grids, ignore_index=True)
    summed = (
        fixations.groupby(
            ["simulation_id", "text_id", "word_id"],
            sort=True,
        )["fixation_duration"]
        .sum()
        .reset_index(name="ob1_tvt")
        .rename(
            columns={
                "text_id": "passage_id_zero_based",
                "word_id": "word_id_zero_based",
            }
        )
    )
    per_simulation = complete.merge(
        summed,
        on=[
            "simulation_id",
            "passage_id_zero_based",
            "word_id_zero_based",
        ],
        how="left",
        validate="one_to_one",
    )
    per_simulation["ob1_tvt"] = per_simulation["ob1_tvt"].fillna(0.0)
    mean_values = (
        per_simulation.groupby(
            [
                "passage_id_raw",
                "passage_id_zero_based",
                "word_id_zero_based",
                "word_raw",
            ],
            sort=True,
        )["ob1_tvt"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "ob1_tvt",
                "std": "ob1_tvt_reader_std",
            }
        )
    )
    audit = {
        "virtual_readers": len(simulation_ids),
        "seeds": [int(seed_map[item]) for item in simulation_ids],
        "fixations": len(fixations),
        "fixations_outside_evaluation_grid": int(excluded_fixation_mask.sum()),
        "fixation_duration_outside_evaluation_grid": float(
            durations.loc[excluded_fixation_mask].sum()
        ),
        "fixation_coordinates_outside_evaluation_grid": int(
            fixation_coordinates.merge(
                canonical_coordinates,
                on=coordinate_columns,
                how="left",
                indicator=True,
                validate="one_to_one",
            )["_merge"]
            .eq("left_only")
            .sum()
        ),
        "per_simulation_word_rows": len(per_simulation),
        "mean_word_rows": len(mean_values),
        "zero_tvt_rows": int((per_simulation["ob1_tvt"] == 0).sum()),
        "regressive_fixations": (
            int((fixations["saccade_type"] == "regression").sum())
            if "saccade_type" in fixations
            else None
        ),
    }
    expected_rows = len(canonical_words) * len(simulation_ids)
    if len(per_simulation) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} OB1 word rows, found {len(per_simulation)}"
        )
    if len(mean_values) != len(canonical_words):
        raise ValueError("OB1 mean grid does not match the canonical word grid")
    return per_simulation, mean_values, audit


def write_ob1_aggregation(
    output_dir: Path,
    per_simulation: pd.DataFrame,
    mean_values: pd.DataFrame,
    audit: dict,
) -> None:
    """Write word-level OB1 values and aggregation audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_simulation.to_csv(
        output_dir / "ob1_word_values_by_simulation.csv",
        index=False,
    )
    mean_values.to_csv(output_dir / "ob1_word_values.csv", index=False)
    with (output_dir / "ob1_aggregation_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
