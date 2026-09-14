"""Tests for cache-only initialization and OB1 process concurrency."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ob1_repro.ob1_runner import (
    ROOT,
    initialize_ob1_cache,
    run_ob1_subprocess,
    split_seed_chunks,
    validate_derived_caches,
)


class ParallelOB1Tests(unittest.TestCase):
    """Verify that cache creation does not consume a simulation worker."""

    def test_one_hundred_seeds_split_across_forty_workers(self) -> None:
        """Assign every reader exactly once across forty balanced chunks."""
        chunks = split_seed_chunks(list(range(100)), 40)
        self.assertEqual(len(chunks), 40)
        self.assertEqual(
            sorted(seed for chunk in chunks for seed in chunk), list(range(100))
        )
        self.assertEqual({len(chunk) for chunk in chunks}, {2, 3})

    def test_parallel_run_initializes_cache_then_launches_all_seeds(self) -> None:
        """Launch seed zero concurrently instead of serializing it as warm-up."""
        cache_audit = {"mode": "initialized_without_simulation"}
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            with (
                patch(
                    "ob1_repro.ob1_runner.initialize_ob1_cache",
                    return_value=cache_audit,
                ) as initialize_mock,
                patch("ob1_repro.ob1_runner.run_ob1_worker") as worker_mock,
                patch("ob1_repro.ob1_runner.merge_ob1_worker_outputs") as merge_mock,
            ):
                run_ob1_subprocess(
                    temporary_root / "runtime",
                    temporary_root / "output",
                    seeds=list(range(100)),
                    n_trials=55,
                    python_hash_seed=20260725,
                    workers=40,
                    attention_skew=3.0,
                )

        initialize_mock.assert_called_once()
        self.assertEqual(worker_mock.call_count, 40)
        worker_chunks = [list(call.args[2]) for call in worker_mock.call_args_list]
        self.assertEqual(
            sorted(seed for chunk in worker_chunks for seed in chunk),
            list(range(100)),
        )
        self.assertTrue(any(0 in chunk for chunk in worker_chunks))
        self.assertFalse(
            any(call.args[1].name == "warmup" for call in worker_mock.call_args_list)
        )
        self.assertEqual(
            merge_mock.call_args.kwargs["cache_initialization"],
            cache_audit,
        )

    def test_cache_validation_checks_content_and_matrix_shape(self) -> None:
        """Accept a complete cache set and reject an incompatible matrix."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            processed = Path(temporary_dir) / "data/processed"
            processed.mkdir(parents=True)
            with (processed / "lexicon.pkl").open("wb") as handle:
                pickle.dump(["a", "b"], handle)
            (
                processed / "frequency_map_Provo_Corpus_continuous_reading_english.json"
            ).write_text(
                json.dumps({"a": 5.0, "b": 4.0}),
                encoding="utf-8",
            )
            (
                processed
                / "prediction_map_Provo_Corpus__continuous_reading_english.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )
            with (processed / "inhibition_matrix_previous.pkl").open("wb") as handle:
                pickle.dump(np.zeros((2, 2)), handle)
            with (processed / "inhibition_matrix_parameters_previous.pkl").open(
                "wb"
            ) as handle:
                pickle.dump("frozen-parameters", handle)

            audit = validate_derived_caches(Path(temporary_dir), "Provo_Corpus")
            self.assertEqual(audit["lexicon_entries"], 2)
            self.assertEqual(audit["inhibition_matrix_shape"], [2, 2])
            self.assertEqual(len(audit["files"]), 5)

            with (processed / "inhibition_matrix_previous.pkl").open("wb") as handle:
                pickle.dump(np.zeros((1, 1)), handle)
            with self.assertRaisesRegex(ValueError, "shape differs"):
                validate_derived_caches(Path(temporary_dir), "Provo_Corpus")

    @unittest.skipUnless(
        os.environ.get("OB1_RUN_INTEGRATION_TESTS") == "1",
        "Set OB1_RUN_INTEGRATION_TESTS=1 after downloading assets",
    )
    def test_real_cache_initialization_simulates_no_reader(self) -> None:
        """Build upstream caches without entering the passage-reading loop."""
        from ob1_repro.ob1_runner import prepare_ob1_runtime
        from ob1_repro.prepare_provo import build_canonical_tables

        passages, _, _, _ = build_canonical_tables(
            ROOT / "data/raw/Provo_Corpus-Eyetracking_Data.csv",
            ROOT / "data/raw/Provo_Corpus-Predictability_Norms.csv",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            runtime = temporary_root / "runtime"
            prepare_ob1_runtime(
                passages,
                runtime,
                subtlex_path=ROOT / "data/raw/SUBTLEX_UK.txt",
            )
            audit = initialize_ob1_cache(
                runtime,
                temporary_root / "initialization",
                n_trials=55,
                python_hash_seed=20260725,
                stimulus_name="Provo_Corpus",
                attention_skew=3.0,
            )
            self.assertEqual(audit["mode"], "initialized_without_simulation")
            self.assertEqual(audit["manifest"]["simulated_readers"], 0)
            self.assertEqual(audit["manifest"]["simulated_passages"], 0)
            self.assertFalse(
                (temporary_root / "initialization/ob1_fixations.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
