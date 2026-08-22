"""Regenerate and validate analysis outputs."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manuscript_variables import generate_variables
from src.validation import write_validation_report


def main() -> int:
    # The manuscript-variable generator is the canonical source-bound producer:
    # it regenerates the base trace, advanced evaluation, native RxInfer output,
    # figures, evidence ledgers, and the initial validation report together.
    generate_variables()
    report = write_validation_report()
    print("validation_passed:", report.summary["passed"])
    print("validation_failed:", report.summary["failed"])
    print("validation_total:", report.summary["total"])
    for path in report.artifact_paths.values():
        print("artifact:", PROJECT_ROOT / path)
    return 0 if report.summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
