#!/usr/bin/env python3
"""Render the already hydrated manuscript through the canonical template."""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_rendering import render_manuscript


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--skip-manuscript-hydration", action="store_true", default=True
    )
    args = parser.parse_args()
    render_manuscript(args.project_root)
