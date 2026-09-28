"""Command-line entry point for aggregating completed Gemma 3 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution from the repository root:
# poetry run python scripts/aggregate_results.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.aggregation import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate fold-level and summary metrics from Gemma 3 result ZIPs."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    fold_metrics, summaries = aggregate_results(args.results_root, args.output_dir)
    output_dir = args.output_dir or args.results_root / "aggregated"

    print(f"Wrote {len(fold_metrics)} fold-level rows to {output_dir / 'fold_metrics.csv'}")
    print(f"Wrote {len(summaries)} summary rows to {output_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
