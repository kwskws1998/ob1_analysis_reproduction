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
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--n-trials", type=int, default=55)
    parser.add_argument(
        "--stimuli-filename",
        default="Provo_Corpus.csv",
    )
    parser.add_argument("--attention-skew", type=float)
    return parser.parse_args()


def main() -> None:
    """Run all requested OB1 baseline virtual readers."""
    args = parse_args()
    vendor_src = args.vendor_src.resolve()
    runtime_dir = args.runtime_dir.resolve()
    output_dir = args.output_dir.resolve()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if not seeds:
        raise ValueError("No OB1 seeds were provided")
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
    global_parameters = {
        "task_to_run": "continuous_reading",
        "stimuli_filepath": str(stimuli_path),
        "stimuli_separator": "\t",
        "language": "english",
        "number_of_simulations": 1,
        "n_trials": args.n_trials,
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
            parameter_record = {
                name: getattr(parameters, name) for name in PARAMETER_NAMES
            }
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
