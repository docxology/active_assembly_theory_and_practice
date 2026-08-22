#!/usr/bin/env python3
"""Generate, verify, or score the frozen local replication bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.replication import (  # noqa: E402
    score_replication_predictions,
    validate_replication_bundle,
    write_replication_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("generate", "verify", "score"), help="replication action"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSON predictions file for score mode",
    )
    args = parser.parse_args()
    if args.mode == "generate":
        print(write_replication_bundle(PROJECT_ROOT))
        return 0
    if args.mode == "verify":
        valid, detail = validate_replication_bundle(PROJECT_ROOT)
        print(detail)
        return 0 if valid else 1
    if args.predictions is None:
        parser.error("--predictions is required for score mode")
    print(score_replication_predictions(args.predictions, PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
