"""Prepare Provo and run the frozen ET1 model on all 55 passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ob1_repro.assets import download_all
from ob1_repro.workflow import DEFAULT_OUTPUT_ROOT, prepare_provo, run_et1


def main() -> None:
    """Download inputs, prepare Provo, and write ET1 predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    download_all()
    prepare_provo()
    result = run_et1(args.output_root.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
