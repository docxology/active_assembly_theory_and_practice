#!/usr/bin/env python3
"""Check authored documentation links, fences, citations and index closure."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docs_validation import validate_documentation


def main() -> int:
    errors = validate_documentation(PROJECT_ROOT)
    for error in errors:
        print(error)
    print(f"documentation_errors: {len(errors)}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
