"""Thin command-line entry point for the kernel-checked formal gate."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_verification import write_formal_verification  # noqa: E402


if __name__ == "__main__":
    print(write_formal_verification(PROJECT_ROOT))
