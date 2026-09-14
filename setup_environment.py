"""Create the local Python 3.12 environment and install pinned packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def environment_python() -> Path:
    """Return the platform-specific interpreter inside the local environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts/python.exe"
    return VENV_DIR / "bin/python"


def main() -> None:
    """Create or update the environment, then run dependency validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required; found {sys.version.split()[0]}")
    interpreter = environment_python()
    if not interpreter.is_file():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--requirement",
            str(REQUIREMENTS),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(interpreter), "-m", "pip", "check"],
        cwd=ROOT,
        check=True,
    )
    freeze = subprocess.run(
        [str(interpreter), "-m", "pip", "freeze", "--all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    requirements_sha256 = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    (VENV_DIR / "environment_manifest.json").write_text(
        json.dumps(
            {
                "python": subprocess.run(
                    [str(interpreter), "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "platform": platform.platform(),
                "requirements_sha256": requirements_sha256,
                "installed_packages": freeze,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.skip_tests:
        subprocess.run(
            [
                str(interpreter),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"environment ready: {interpreter}")


if __name__ == "__main__":
    main()
