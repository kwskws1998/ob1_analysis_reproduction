"""Tests for frozen checkpoint provenance and package independence."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ob1_repro.assets import load_manifest
from ob1_repro.workflow import ROOT, load_expected, load_sigma_records


class ContractTests(unittest.TestCase):
    """Exercise frozen constants and isolation from the parent repository."""

    def test_frozen_llama_checkpoint_table(self) -> None:
        """Require six positive checkpoint-specific sigma pairs and known means."""
        records = load_sigma_records()
        self.assertEqual(len(records), 6)
        self.assertTrue(
            np.isclose(
                np.mean([item["source_accuracy"] for item in records]),
                0.7481433333333333,
            )
        )
        self.assertTrue(
            np.isclose(np.mean([item["sigma_left"] for item in records]), 0.80385)
        )
        self.assertTrue(
            np.isclose(np.mean([item["sigma_right"] for item in records]), 3.47126)
        )

    def test_frozen_expected_counts(self) -> None:
        """Require the checked-in dataset and trajectory regression contract."""
        expected = load_expected()
        self.assertEqual(expected["passages"], 55)
        self.assertEqual(expected["evaluable_words"], 2686)
        self.assertEqual(expected["excluded_positions"], 59)
        self.assertEqual(expected["et1_token_rows"], 3715)
        self.assertEqual(expected["skew3_fixation_rows"], 275032)
        self.assertEqual(expected["skew4_fixation_rows"], 279109)

    def test_external_asset_provenance_is_complete(self) -> None:
        """Require URL, revision, byte count, and digest provenance."""
        manifest = load_manifest()
        for name, asset in manifest.items():
            self.assertIn("revision", asset, name)
            self.assertIn("bytes", asset, name)
            if name == "et1_tokenizer":
                self.assertIn("files_sha256", asset)
                self.assertEqual(len(asset["files"]), 3)
                for file_asset in asset["files"]:
                    self.assertIn("source", file_asset)
                    self.assertIn("bytes", file_asset)
                    self.assertIn("sha256", file_asset)
            else:
                self.assertIn("source", asset, name)
                self.assertIn("sha256", asset, name)

    def test_package_source_has_no_parent_project_reference(self) -> None:
        """Reject imports or absolute paths into the original project."""
        forbidden = (
            "cognitive_model_comparsion",
            "/workspace/but",
            "from models",
            "import models",
        )
        for path in ROOT.rglob("*.py"):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase!r} found in {path}")

    def test_package_imports_after_standalone_copy(self) -> None:
        """Import the package after copying it outside its parent repository."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            copied = Path(temporary_dir) / "ob1_analysis_reproduction"
            shutil.copytree(
                ROOT,
                copied,
                ignore=shutil.ignore_patterns(
                    ".venv",
                    "outputs",
                    "data",
                    "models",
                    "third_party",
                    "__pycache__",
                ),
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(copied)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                [sys.executable, "-c", "import ob1_repro.workflow"],
                cwd=copied,
                env=environment,
                check=True,
            )

    @unittest.skipUnless(
        os.environ.get("OB1_RUN_INTEGRATION_TESTS") == "1",
        "Set OB1_RUN_INTEGRATION_TESTS=1 after downloading assets",
    )
    def test_et1_checkpoint_smoke(self) -> None:
        """Load the pinned ET1 checkpoint and produce finite token predictions."""
        from ob1_repro.et1_inference import ET1NativePredictor

        predictor = ET1NativePredictor(
            checkpoint_path=ROOT / "models/T5-tokenizer-BiLSTM-TRT-12-concat-3",
            tokenizer_path=ROOT / "models/t5_tokenizer",
        )
        result = predictor.predict("A short Provo-style sentence.")
        self.assertTrue(np.isfinite(result["values"].detach().cpu().numpy()).all())

    @unittest.skipUnless(
        os.environ.get("OB1_RUN_INTEGRATION_TESTS") == "1",
        "Set OB1_RUN_INTEGRATION_TESTS=1 after downloading assets",
    )
    def test_ob1_simulation_smoke(self) -> None:
        """Simulate one reader on one passage with the pinned OB1 source."""
        import pandas as pd

        from ob1_repro.ob1_runner import prepare_ob1_runtime, run_ob1_subprocess
        from ob1_repro.prepare_provo import build_canonical_tables

        passages, _, _, _ = build_canonical_tables(
            ROOT / "data/raw/Provo_Corpus-Eyetracking_Data.csv",
            ROOT / "data/raw/Provo_Corpus-Predictability_Norms.csv",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            runtime = temporary_root / "runtime"
            output = temporary_root / "output"
            prepare_ob1_runtime(
                passages.iloc[:1],
                runtime,
                subtlex_path=ROOT / "data/raw/SUBTLEX_UK.txt",
            )
            run_ob1_subprocess(
                runtime,
                output,
                seeds=[0],
                n_trials=1,
                python_hash_seed=20260725,
                workers=1,
                attention_skew=3.0,
            )
            fixations = pd.read_csv(output / "ob1_fixations.csv")
            self.assertFalse(fixations.empty)
            self.assertEqual(fixations["seed"].unique().tolist(), [0])
            self.assertEqual(fixations["text_id"].unique().tolist(), [0])


if __name__ == "__main__":
    unittest.main()
