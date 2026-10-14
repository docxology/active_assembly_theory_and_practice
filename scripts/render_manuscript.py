#!/usr/bin/env python3
"""Render the already hydrated manuscript through the canonical template."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_rendering import render_manuscript


if __name__ == "__main__":
    render_manuscript(PROJECT_ROOT)
