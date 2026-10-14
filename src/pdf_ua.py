"""Fail-closed VeraPDF compliance checks for the actual rendered PDF/UA-2 file."""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .artifact_contract import write_json_atomic


def validate_verapdf_report(payload: Any) -> list[str]:
    """Require one normally completed UA-2 job with actual passing rule counts."""
    errors: list[str] = []
    report = payload.get("report") if isinstance(payload, dict) else None
    if not isinstance(report, dict):
        return ["missing VeraPDF report"]
    jobs = report.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
        return ["expected exactly one VeraPDF job"]
    results = jobs[0].get("validationResult")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        return ["expected exactly one VeraPDF validation result"]
    result = results[0]
    if result.get("compliant") is not True:
        errors.append("VeraPDF reports noncompliance")
    if result.get("jobEndStatus") != "normal":
        errors.append("VeraPDF job did not complete normally")
    if "PDF/UA-2" not in str(result.get("profileName", "")):
        errors.append("VeraPDF did not use the PDF/UA-2 profile")
    details = result.get("details")
    if not isinstance(details, dict):
        errors.append("missing VeraPDF rule/check counts")
    else:
        for key in ("failedRules", "failedChecks"):
            if type(details.get(key)) is not int or details[key] != 0:
                errors.append(f"VeraPDF {key} is not zero")
        for key in ("passedRules", "passedChecks"):
            if type(details.get(key)) is not int or details[key] <= 0:
                errors.append(f"VeraPDF {key} is not positive")
    return errors


def audit_pdf_ua(project_root: Path) -> list[str]:
    """Run native VeraPDF and retain a PDF-digest-bound report before returning."""
    root = Path(project_root)
    pdf = root / "output/pdf/active_assembly_theory_and_practice_combined.pdf"
    before = hashlib.sha256(pdf.read_bytes()).hexdigest()
    completed = subprocess.run(
        ["verapdf", "--format", "json", "--flavour", "ua2", str(pdf)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    payload = json.loads(completed.stdout)
    errors = validate_verapdf_report(payload)
    if completed.returncode != 0:
        errors.append(f"VeraPDF process exit code {completed.returncode}")
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != before:
        errors.append("PDF changed while VeraPDF was running")
    write_json_atomic(
        root / "output/reports/verapdf_ua2.json",
        {
            "pdf_path": str(pdf.relative_to(root)),
            "pdf_sha256": before,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "errors": errors,
            "verapdf": payload,
        },
    )
    return errors
