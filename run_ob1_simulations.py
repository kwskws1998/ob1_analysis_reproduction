"""Generate independent skew-3 and skew-4 OB1 virtual-reader trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ob1_repro.assets import download_all
from ob1_repro.workflow import (
    DEFAULT_OUTPUT_ROOT,
    default_workers,
    prepare_provo,
    run_ob1,
)


def main() -> None:
    """Run one or both fixed 100-reader OB1 simulation conditions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--skew", choices=("3", "4", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    download_all()
    prepare_provo()
    output_root = args.output_root.expanduser().resolve()
    skews = (3, 4) if args.skew == "all" else (int(args.skew),)
    result = {str(skew): run_ob1(skew, args.workers, output_root) for skew in skews}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
