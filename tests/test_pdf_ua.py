"""VeraPDF report negative controls and real PDF/UA native validation."""

from copy import deepcopy
import json

from pypdf import PdfWriter
import pytest

from src.pdf_ua import audit_pdf_ua, validate_verapdf_report


def _report():
    return {
        "report": {
            "jobs": [
                {
                    "validationResult": [
                        {
                            "compliant": True,
                            "jobEndStatus": "normal",
                            "profileName": "PDF/UA-2 profile",
                            "details": {
                                "failedRules": 0,
                                "failedChecks": 0,
                                "passedRules": 10,
                                "passedChecks": 100,
                            },
                        }
                    ]
                }
            ]
        }
    }


def test_complete_report_is_required():
    assert validate_verapdf_report(_report()) == []
    for malformed in (
        None,
        {},
        {"report": {"jobs": []}},
        {"report": {"jobs": [None]}},
        {"report": {"jobs": [{"validationResult": []}]}},
        {"report": {"jobs": [{"validationResult": [None]}]}},
    ):
        assert validate_verapdf_report(malformed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("compliant", False),
        ("compliant", "true"),
        ("jobEndStatus", "failed"),
        ("profileName", "PDF/A-2b"),
        ("details", None),
    ],
)
def test_noncompliance_or_wrong_profile_is_rejected(field, value):
    report = _report()
    report["report"]["jobs"][0]["validationResult"][0][field] = value
    assert validate_verapdf_report(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("failedRules", 1),
        ("failedChecks", 1),
        ("failedRules", False),
        ("passedRules", 0),
        ("passedChecks", "100"),
    ],
)
def test_failed_or_invalid_counts_are_rejected(field, value):
    report = deepcopy(_report())
    report["report"]["jobs"][0]["validationResult"][0]["details"][field] = value
    assert validate_verapdf_report(report)


def test_native_verapdf_rejects_an_untagged_pdf(tmp_path):
    pdf = tmp_path / "output/pdf/active_assembly_theory_and_practice_combined.pdf"
    pdf.parent.mkdir(parents=True)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(pdf)
    errors = audit_pdf_ua(tmp_path)
    assert any("noncompliance" in error for error in errors)
    retained = json.loads((tmp_path / "output/reports/verapdf_ua2.json").read_text())
    assert retained["errors"] == errors
    assert retained["pdf_sha256"]
