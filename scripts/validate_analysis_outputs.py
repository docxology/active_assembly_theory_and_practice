"""Regenerate and validate analysis outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_provenance import write_build_provenance
from src.manuscript_variables import generate_variables
from src.artifact_contract import write_json_artifact
from src.validation import build_validation_report, write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Validate current outputs without regenerating scientific inputs or figures; use after rendering.",
    )
    args = parser.parse_args()
    # The manuscript-variable generator is the canonical source-bound producer:
    # it regenerates the base trace, advanced evaluation, native RxInfer output,
    # figures, evidence ledgers, and the initial validation report together.
    if not args.existing_only:
        generate_variables()
    # This is the source-side regeneration gate: it runs before the sibling
    # template render, so the optional DOCX lane must keep the documented
    # pre-render deferral (docs/analysis_validation.md). An operator can force
    # post-render strictness here by exporting ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=0;
    # the fail-closed post-render gate remains scripts/validate_docx_render.py.
    if not args.existing_only:
        os.environ.setdefault("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER", "1")
    # Bind the actual source before checking it. This mode never refits models
    # or rewrites substantive PNGs after the document renderer consumed them.
    write_build_provenance(PROJECT_ROOT)
    if args.existing_only:
        report = build_validation_report()
        write_json_artifact(
            PROJECT_ROOT / "output/data/validation_report.json",
            report.to_dict(),
            project_root=PROJECT_ROOT,
            producer_path=PROJECT_ROOT / "src/validation.py",
        )
        write_build_provenance(PROJECT_ROOT)
    else:
        report = write_validation_report()
    print("validation_passed:", report.summary["passed"])
    print("validation_failed:", report.summary["failed"])
    print("validation_total:", report.summary["total"])
    for path in report.artifact_paths.values():
        print("artifact:", PROJECT_ROOT / path)
    return 0 if report.summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
