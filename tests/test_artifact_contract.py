"""Regression tests for the protocol-v2 generated-artifact boundary."""

from __future__ import annotations

import json
from pathlib import Path

from src.artifact_contract import (
    artifact_domain_view,
    build_artifact_envelope,
    seal_data_artifact_surface,
    seal_existing_json_artifact,
    validate_artifact_envelope,
    validate_data_artifact_surface,
    write_json_atomic,
    write_json_artifact,
)
from src.scientific_protocol import scientific_protocol_payload


def test_writer_preserves_domain_schema_and_seals_sequence_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output" / "data" / "sequence.json"
    write_json_artifact(
        path,
        ["decoded", "unsupported"],
        project_root=tmp_path,
        producer_path=Path("src/artifact_contract.py"),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "artifact-payload-v1"
    assert payload["payload"] == ["decoded", "unsupported"]
    assert payload["receipt_schema_version"] == "2.0.0"
    assert validate_artifact_envelope(
        payload, expected_protocol=scientific_protocol_payload()
    ) == (True, "ok")
    assert not list(path.parent.glob("*.tmp"))


def test_raw_interface_writer_is_atomic_without_changing_fixture_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output" / "replication" / "public.json"
    fixture = [{"record_id": "a", "sequence": ["x"], "features": [0.0]}]

    write_json_atomic(path, fixture)

    assert json.loads(path.read_text(encoding="utf-8")) == fixture
    assert not list(path.parent.glob("*.tmp"))


def test_legacy_migration_is_explicit_and_domain_view_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "output" / "data" / "legacy.json"
    path.parent.mkdir(parents=True)
    legacy = {"schema_version": "1.0", "rows": [{"value": 1}]}
    path.write_text(json.dumps(legacy), encoding="utf-8")

    sealed = seal_existing_json_artifact(
        path,
        project_root=tmp_path,
        producer_path=Path("src/artifact_contract.py"),
    )
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["rows"] == legacy["rows"]
    assert artifact_domain_view(stored) == legacy
    assert sealed["artifact_id"] == "output/data/legacy.json"
    assert validate_data_artifact_surface(tmp_path) == []


def test_surface_validator_rejects_mixed_or_unsealed_protocol(tmp_path: Path) -> None:
    path = tmp_path / "output" / "data" / "unsealed.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "1.0", "rows": []}), encoding="utf-8")

    errors = validate_data_artifact_surface(tmp_path)

    assert len(errors) == 1
    assert errors[0]["path"] == "output/data/unsealed.json"
    assert "fields missing" in errors[0]["detail"]


def test_artifact_protocol_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "output" / "data" / "receipt.json"
    sealed = write_json_artifact(
        path,
        {"schema_version": "1.0", "rows": []},
        project_root=tmp_path,
        producer_path=Path("src/artifact_contract.py"),
    )
    sealed["protocol_sha256"] = "f" * 64

    valid, detail = validate_artifact_envelope(
        sealed, expected_protocol=scientific_protocol_payload()
    )

    assert not valid
    assert "protocol hash mismatch" in detail


def test_artifact_envelope_preserves_existing_receipts_and_fallbacks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output/data/custom.json"
    payload = {
        "schema_version": "domain-v2",
        "status": "validated_bounded",
        "valid": False,
        "limitations": ["domain limitation"],
        "unsupported_cases": ("unsupported case",),
        "underpowered_cases": ["underpowered case"],
        "input_hashes": {"existing": "digest"},
        "toolchain_versions": {"custom": "1.0"},
    }
    sealed = build_artifact_envelope(
        payload,
        artifact_path=path,
        project_root=tmp_path,
        producer_path=tmp_path / "not-a-real-producer.py",
        input_paths=(tmp_path / "missing-input",),
    )
    assert sealed["schema_version"] == "domain-v2"
    assert sealed["status"] == "validated_bounded"
    assert sealed["limitations"] == ["domain limitation"]
    assert sealed["unsupported_cases"] == ["unsupported case"]
    assert sealed["underpowered_cases"] == ["underpowered case"]
    assert "existing" in sealed["input_hashes"]
    assert artifact_domain_view(sealed)["schema_version"] == "domain-v2"
    assert artifact_domain_view("not-an-object") == "not-an-object"


def test_artifact_validator_rejects_nonobjects_and_missing_receipt_fields() -> None:
    assert validate_artifact_envelope(None)[0] is False
    valid, detail = validate_artifact_envelope({})
    assert not valid
    assert "fields missing" in detail
    valid, detail = validate_artifact_envelope({"receipt_schema_version": "0.0.0"})
    assert not valid
    assert "fields missing" in detail


def test_data_surface_reports_missing_directory_and_malformed_json(
    tmp_path: Path,
) -> None:
    assert seal_data_artifact_surface(tmp_path) == []
    diagnostics = validate_data_artifact_surface(tmp_path)
    assert diagnostics == [
        {"path": "output/data", "detail": "data directory is missing"}
    ]

    data_dir = tmp_path / "output/data"
    data_dir.mkdir(parents=True)
    (data_dir / "broken.json").write_text("{", encoding="utf-8")
    (data_dir / "legacy.json").write_text(
        json.dumps({"schema_version": "1.0", "rows": []}), encoding="utf-8"
    )
    changed = seal_data_artifact_surface(tmp_path)
    assert "output/data/legacy.json" in changed
    diagnostics = validate_data_artifact_surface(tmp_path)
    assert len(diagnostics) == 1
    assert diagnostics[0]["path"] == "output/data/broken.json"
