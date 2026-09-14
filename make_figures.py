"""Generate final reviewer tables and the two publication figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ob1_repro.workflow import DEFAULT_OUTPUT_ROOT, make_figures


def main() -> None:
    """Build and validate the final table and figure directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = make_figures(args.output_root.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
