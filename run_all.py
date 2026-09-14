"""Execute the entire standalone analysis while preserving a combined log."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ob1_repro.workflow import (
    DEFAULT_OUTPUT_ROOT,
    PYTHON_HASH_SEED,
    default_workers,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parent


def run_logged(command: list[str], log_handle, environment: dict[str, str]) -> None:
    """Stream one child process to both the terminal and the execution log."""
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    log_handle.write(f"$ {rendered}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture child-process output")
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    """Run acquisition, inference, simulations, analysis, figures, and checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "execution.log"
    audit_path = output_root / "execution_manifest.json"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["MPLCONFIGDIR"] = str(output_root / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "workers": args.workers,
        "python_hash_seed": PYTHON_HASH_SEED,
        "output_root": str(output_root),
    }
    write_json_atomic(audit_path, audit)
    commands = [
        [sys.executable, "-u", "download_assets.py"],
        [
            sys.executable,
            "-u",
            "run_et1_prediction.py",
            "--output-root",
            str(output_root),
        ],
        [
            sys.executable,
            "-u",
            "run_ob1_simulations.py",
            "--workers",
            str(args.workers),
            "--output-root",
            str(output_root),
        ],
        [
            sys.executable,
            "-u",
            "analyze_attention.py",
            "--output-root",
            str(output_root),
        ],
        [sys.executable, "-u", "make_figures.py", "--output-root", str(output_root)],
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            for command in commands:
                run_logged(command, log_handle, environment)
            final_dir = output_root / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            log_handle.flush()
            shutil.copyfile(log_path, final_dir / "execution.log")
            verification = [
                sys.executable,
                "-u",
                "verify_results.py",
                "--output-root",
                str(output_root),
            ]
            run_logged(verification, log_handle, environment)
        shutil.copyfile(log_path, output_root / "final/execution.log")
    except Exception as error:
        audit.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(audit_path, audit)
        raise
    audit.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_log": str(log_path),
        }
    )
    write_json_atomic(audit_path, audit)
    shutil.copyfile(audit_path, output_root / "final/execution_manifest.json")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
