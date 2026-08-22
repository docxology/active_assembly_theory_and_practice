#!/usr/bin/env python3
"""Run the post-render structural accessibility audit for the manuscript PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_accessibility import (  # noqa: E402
    DEFAULT_PDF,
    DEFAULT_REPORT,
    write_pdf_accessibility_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = write_pdf_accessibility_report(
        args.pdf,
        args.report,
        project_root=PROJECT_ROOT,
    )
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    print(
        f"PDF accessibility audit: {summary['passed']}/{summary['passed'] + summary['failed']} checks passed; "
        f"report={report_path}"
    )
    for check in report["checks"]:
        print(
            f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}"
        )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
