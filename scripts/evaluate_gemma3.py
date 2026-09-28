"""CLI entry point for one Gemma 3 evaluation run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.inference import main


if __name__ == "__main__":
    main()