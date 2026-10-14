#!/usr/bin/env python3
"""Check actual VeraPDF UA-2 compliance and retain the PDF-bound JSON report."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_ua import audit_pdf_ua


if __name__ == "__main__":
    errors = audit_pdf_ua(PROJECT_ROOT)
    for error in errors:
        print(error)
    print(f"pdf_ua_errors: {len(errors)}")
    raise SystemExit(bool(errors))
