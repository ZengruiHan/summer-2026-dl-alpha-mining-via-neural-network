#!/usr/bin/env python
"""Evaluate M0 OOS Rank IC, Pearson IC, and accuracy."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from alpha_mining_neural_network.prediction_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
