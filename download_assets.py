"""Download or verify all standalone OB1-analysis assets."""

from __future__ import annotations

import argparse
import json

from ob1_repro.assets import download_all, verify_all


def main() -> None:
    """Run the checksum-enforced asset operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_all() if args.verify_only else download_all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
