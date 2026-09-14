"""Run trajectory-matched attention analyses for the six checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ob1_repro.workflow import DEFAULT_OUTPUT_ROOT, run_analysis


def main() -> None:
    """Analyze one or both independently simulated skew conditions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skew", choices=("3", "4", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    skews = (3, 4) if args.skew == "all" else (int(args.skew),)
    result = {str(skew): run_analysis(skew, output_root) for skew in skews}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
