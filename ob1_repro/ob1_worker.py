"""Execute deterministic virtual readers using the pinned OB1 source tree."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PARAMETER_NAMES = (
    "cycle_size",
    "bigram_to_word_excitation",
    "bigram_to_word_inhibition",
    "word_inhibition",
    "min_activity",
    "max_activity",
    "decay",
    "discounted_Ngrams",
    "bigram_gap",
    "max_threshold",
    "freq_weight",
    "frequency_flag",
    "attend_width",
    "max_attend_width",
    "min_attend_width",
    "attention_skew",
    "letPerDeg",
    "refix_size",
    "salience_position",
    "sacc_optimal_distance",
    "saccErr_scaler",
    "saccErr_sigma",
    "saccErr_sigma_scaler",
    "mu",
    "sigma",
    "recog_speeding",
    "use_saccade_error",
    "prediction_flag",
    "pred_weight",
)


def flatten_simulation(
    simulation_data: dict,
    simulation_id: int,
    seed: int,
) -> list[dict]:
    """Flatten one OB1 virtual reader to essential fixation records."""
    records = []
    texts = simulation_data[0]
    for text_id, fixations in texts.items():
        for fixation_counter, fixation in fixations.items():
            records.append(
                {
                    "simulation_id": simulation_id,
                    "seed": seed,
                    "text_id": int(text_id),
                    "fixation_counter": int(fixation_counter),
                    "word_id": int(fixation["foveal_word_index"]),
                    "word": fixation["foveal_word"],
                    "fixation_duration": float(fixation["fixation_duration"]),
                    "saccade_type": fixation["saccade_type"],
                    "attentional_width": float(fixation["attentional_width"]),
                    "eye_position": float(fixation["eye_position"]),
                    "saccade_distance": float(fixation["saccade_distance"]),
                    "saccade_error": float(fixation["saccade_error"]),
                    "saccade_cause": fixation["saccade_cause"],
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    """Parse pinned source, isolated runtime, output, and reader seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-src", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--n-trials", type=int, default=55)
    parser.add_argument(
        "--stimuli-filename",
        default="Provo_Corpus.csv",
    )
    parser.add_argument("--attention-skew", type=float)
    parser.add_argument("--initialize-cache-only", action="store_true")
    return parser.parse_args()


def build_global_parameters(
    stimuli_path: Path,
    n_trials: int,
) -> dict:
    """Build the frozen upstream OB1 execution configuration."""
    return {
        "task_to_run": "continuous_reading",
        "stimuli_filepath": str(stimuli_path),
        "stimuli_separator": "\t",
        "language": "english",
        "number_of_simulations": 1,
        "n_trials": n_trials,
        "prediction_flag": "",
        "results_identifier": "",
        "run_exp": True,
        "analyze_results": False,
        "results_filepath": "",
        "parameters_filepath": "",
        "eye_tracking_filepath": "",
        "experiment_parameters_filepath": "",
        "optimize": False,
        "print_process": False,
        "plotting": False,
    }


def capture_parameter_record(parameters) -> dict:
    """Freeze the OB1 parameters relevant to trajectory provenance."""
    return {name: getattr(parameters, name) for name in PARAMETER_NAMES}


def initialize_runtime_cache(
    upstream_simulation,
    return_params,
    global_parameters: dict,
    attention_skew: float | None,
    output_dir: Path,
) -> None:
    """Build upstream derived caches without simulating a virtual reader."""
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    parameters = return_params(global_parameters)
    if attention_skew is not None:
        parameters.attention_skew = float(attention_skew)
    parameters.number_of_simulations = 0
    started = time.perf_counter()
    simulation_data = upstream_simulation.simulate_experiment(parameters)
    elapsed = time.perf_counter() - started
    if simulation_data:
        raise RuntimeError("Cache-only OB1 initialization unexpectedly simulated data")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cache_initialization_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "mode": "cache_initialization_only",
                "python_hash_seed": os.environ["PYTHONHASHSEED"],
                "simulated_readers": 0,
                "simulated_passages": 0,
                "seconds": elapsed,
                "parameters": capture_parameter_record(parameters),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def main() -> None:
    """Run all requested OB1 baseline virtual readers."""
    args = parse_args()
    vendor_src = args.vendor_src.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_dir = args.output_dir.resolve()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if not seeds and not args.initialize_cache_only:
        raise ValueError("No OB1 seeds were provided")
    if seeds and args.initialize_cache_only:
        raise ValueError("Cache-only OB1 initialization must not receive seeds")
    if os.environ.get("PYTHONHASHSEED") is None:
        raise RuntimeError("PYTHONHASHSEED must be fixed by the parent process")
    if args.attention_skew is not None and (
        not np.isfinite(args.attention_skew) or args.attention_skew < 1
    ):
        raise ValueError("--attention-skew must be finite and at least one")
    stimuli_filename = Path(args.stimuli_filename)
    if (
        stimuli_filename.name != args.stimuli_filename
        or stimuli_filename.suffix != ".csv"
    ):
        raise ValueError("--stimuli-filename must be one CSV basename")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    sys.path.insert(0, str(vendor_src))
    os.chdir(runtime_dir / "src")
    from parameters import return_params
    import simulate_experiment as upstream_simulation

    upstream_simulation.sleep = lambda _: None
    stimuli_path = runtime_dir / "data/processed" / stimuli_filename
    if not stimuli_path.is_file():
        raise FileNotFoundError(stimuli_path)
    global_parameters = build_global_parameters(
        stimuli_path,
        args.n_trials,
    )

    if args.initialize_cache_only:
        initialize_runtime_cache(
            upstream_simulation,
            return_params,
            global_parameters,
            args.attention_skew,
            output_dir,
        )
        return

    all_records = []
    runtimes = []
    parameter_record = None
    for simulation_id, seed in enumerate(seeds):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        parameters = return_params(global_parameters)
        if args.attention_skew is not None:
            parameters.attention_skew = float(args.attention_skew)
        if parameters.prediction_flag:
            raise ValueError("The primary OB1 condition must disable predictability")
        if parameter_record is None:
            parameter_record = capture_parameter_record(parameters)
        started = time.perf_counter()
        simulation_data = upstream_simulation.simulate_experiment(parameters)
        runtimes.append(
            {
                "simulation_id": simulation_id,
                "seed": seed,
                "seconds": time.perf_counter() - started,
            }
        )
        all_records.extend(flatten_simulation(simulation_data, simulation_id, seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(
        output_dir / "ob1_fixations.csv",
        index=False,
    )
    with (output_dir / "ob1_worker_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "condition": "baseline_no_predictability",
                "python_hash_seed": os.environ["PYTHONHASHSEED"],
                "seeds": seeds,
                "n_trials": args.n_trials,
                "stimuli_filename": args.stimuli_filename,
                "parameters": parameter_record,
                "runtimes": runtimes,
                "fixation_rows": len(all_records),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
