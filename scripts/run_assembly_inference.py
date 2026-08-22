"""Run the end-to-end assembly observation and inference experiment."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_experiment


def main() -> int:
    result = run_experiment()
    print("sample:", result.sample_name)
    print("instructions:", len(result.instructions))
    print("final_rax:", result.trace.final_rax)
    print("final_state:", result.inference.final_state)
    for path in [*result.data_paths.values(), *result.figure_paths.values()]:
        print("artifact:", PROJECT_ROOT / path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
