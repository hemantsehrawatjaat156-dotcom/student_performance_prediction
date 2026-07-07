"""Convenience launcher for model training.

Run from the project root with:
    python train.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from train_model import main


if __name__ == "__main__":
    main()
