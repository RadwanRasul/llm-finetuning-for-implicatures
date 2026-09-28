"""Command-line entry point for preparing PUB Task 3 for manual annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.pub_processing import prepare_pub_task3

DEFAULT_OUTPUT = Path("data/processed/pub_processed.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PUB Task 3 for manual annotation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    pub_df = prepare_pub_task3()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pub_df.to_csv(args.output, sep=";", index=False)

    print(f"Prepared {len(pub_df)} PUB Task 3 rows.")
    print(f"Saved annotation input to {args.output}.")


if __name__ == "__main__":
    main()
