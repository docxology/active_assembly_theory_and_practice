"""Coverage for the read-only promote-evidence summary helper."""

from __future__ import annotations

import json
from pathlib import Path

from src.external_sources import EXTERNAL_RECEIPT_ARTIFACT
from src.receipt import promote_evidence_summary

SURFACE_NAMES = {
    "P13_external_validation",
    "P13_source_bound_labels",
    "P13_static_only",
    "P13_capabilities",
    "P13_acquisition",
    "P13_preparation_audit",
    "P15_local_replay_harness",
    "P15_independent_replay",
}


def _external_validation_path(root: Path) -> Path:
    return root / EXTERNAL_RECEIPT_ARTIFACT


def test_promote_summary_reports_missing_receipts_as_not_yet_observed(
    tmp_path: Path,
) -> None:
    summary = promote_evidence_summary(tmp_path)

    assert summary["status"] == "read_only_summary"
    surfaces = summary["surfaces"]
    assert set(surfaces) == SURFACE_NAMES
    for row in surfaces.values():
        assert row["status"] == "not_yet_observed"


def test_promote_summary_projects_known_status_fields(tmp_path: Path) -> None:
    path = _external_validation_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"status": "validated", "endpoint_status": "pilot_receipt"}),
        encoding="utf-8",
    )

    row = promote_evidence_summary(tmp_path)["surfaces"]["P13_external_validation"]
    assert row == {
        "status": "validated",
        "endpoint_status": "pilot_receipt",
        "path": str(path),
    }


def test_promote_summary_defaults_missing_status_field(tmp_path: Path) -> None:
    path = _external_validation_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"endpoint_status": "frozen_local_harness"}), encoding="utf-8"
    )

    row = promote_evidence_summary(tmp_path)["surfaces"]["P13_external_validation"]
    assert row["status"] == "not_yet_observed"
    assert row["endpoint_status"] == "frozen_local_harness"


def test_promote_summary_falls_back_to_failed_on_unreadable_receipt(
    tmp_path: Path,
) -> None:
    path = _external_validation_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    row = promote_evidence_summary(tmp_path)["surfaces"]["P13_external_validation"]
    assert row["status"] == "failed"
    assert row["path"] == str(path)
    assert isinstance(row["detail"], str) and row["detail"]
