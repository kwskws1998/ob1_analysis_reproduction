"""Verify every standalone input, trajectory, metric, and final artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ob1_repro.workflow import DEFAULT_OUTPUT_ROOT, verify_complete


def main() -> None:
    """Run the complete offline validation suite for produced results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = verify_complete(args.output_root.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
