#!/usr/bin/env python3
"""Complete the configured DOCX cover using the canonical template renderer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_rendering import render_docx_with_cover  # noqa: E402

if __name__ == "__main__":
    print(render_docx_with_cover())
