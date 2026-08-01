"""Tests for the RxInfer bridge version pin and Julia-availability contract.

No mocks: the Julia-unavailable path is exercised with a genuinely absent
executable name resolved through the real ``shutil.which``, and the version
contract is exercised with plain string arguments.
"""

from __future__ import annotations

import json

import pytest

from src.rxinfer_bridge import (
    EXPECTED_JULIA_VERSION,
    EXPECTED_RXINFER_VERSION,
    julia_available,
    load_free_energy_records,
    resolve_julia_executable,
    validate_backend_versions,
    validate_free_energy_records,
    validate_rxinfer_outputs,
)

# A name that cannot exist on PATH, used to drive the unavailable-toolchain path.
_MISSING_JULIA = "definitely-not-a-real-julia-executable-xyz"


def test_expected_versions_are_pinned_nonempty_strings() -> None:
    """The pinned toolchain versions are explicit, documented string constants."""
    assert isinstance(EXPECTED_JULIA_VERSION, str) and EXPECTED_JULIA_VERSION
    assert isinstance(EXPECTED_RXINFER_VERSION, str) and EXPECTED_RXINFER_VERSION


def test_validate_backend_versions_accepts_pinned_versions() -> None:
    # Must not raise when the reported versions match the pins.
    validate_backend_versions(EXPECTED_JULIA_VERSION, EXPECTED_RXINFER_VERSION)


def test_validate_backend_versions_rejects_julia_drift() -> None:
    with pytest.raises(RuntimeError, match="Julia"):
        validate_backend_versions("0.0.0", EXPECTED_RXINFER_VERSION)


def test_validate_backend_versions_rejects_rxinfer_drift() -> None:
    with pytest.raises(RuntimeError, match="RxInfer"):
        validate_backend_versions(EXPECTED_JULIA_VERSION, "0.0.0")


def test_validate_backend_versions_reports_both_mismatches() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        validate_backend_versions("0.0.0", "0.0.0")
    message = str(excinfo.value)
    assert "Julia" in message
    assert "RxInfer" in message


def test_resolve_julia_executable_raises_when_unavailable() -> None:
    """Missing Julia is a hard, clearly-labeled failure, not a silent None."""
    with pytest.raises(RuntimeError, match="Julia executable not found"):
        resolve_julia_executable(_MISSING_JULIA)


def test_julia_available_is_false_when_unavailable() -> None:
    """The availability contract returns False (never raises) for a missing tool."""
    assert julia_available(_MISSING_JULIA) is False


def test_julia_available_returns_bool_for_default() -> None:
    """Whether or not Julia is installed, the contract answers with a bool."""
    assert isinstance(julia_available(), bool)


def _valid_free_energy_record(iteration: int = 1) -> dict[str, object]:
    return {
        "run_id": "run",
        "sample_id": "sample",
        "model_name": "gaussian_event_code_model",
        "mode": "static",
        "iteration": iteration,
        "observation_index": None,
        "value": 1.0,
        "source_field": "result.free_energy",
        "is_finite": True,
        "delta": None,
        "rxinfer_version": "5.4.0",
        "julia_version": "1.12.6",
        "data_hash": "a" * 64,
        "config_hash": "b" * 64,
    }


def test_load_free_energy_records_empty_trace_raises(tmp_path) -> None:
    path = tmp_path / "free_energy_trace.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_free_energy_records(path)


def test_load_free_energy_records_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        load_free_energy_records(tmp_path / "does_not_exist.jsonl")


def test_validate_free_energy_records_rejects_unknown_source() -> None:
    record = _valid_free_energy_record()
    record["source_field"] = "result.something_else"
    with pytest.raises(ValueError, match="source"):
        validate_free_energy_records([record])


def test_validate_free_energy_records_rejects_missing_fields() -> None:
    record = _valid_free_energy_record()
    record.pop("value")
    with pytest.raises(ValueError, match="missing fields"):
        validate_free_energy_records([record])


def test_validate_free_energy_records_rejects_non_increasing_iterations() -> None:
    records = [_valid_free_energy_record(i) for i in (1, 2, 1)]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_free_energy_records(records)


def test_validate_free_energy_records_rejects_data_hash_mismatch() -> None:
    record = _valid_free_energy_record()
    with pytest.raises(ValueError, match="data_hash"):
        validate_free_energy_records([record], expected_data_hash="c" * 64)


def test_validate_rxinfer_outputs_missing_companion_raises(tmp_path) -> None:
    output_dir = tmp_path / "rxinfer"
    output_dir.mkdir()
    (output_dir / "free_energy_trace.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="posteriors"):
        validate_rxinfer_outputs(output_dir)


def test_validate_rxinfer_outputs_round_trips_records(tmp_path) -> None:
    output_dir = tmp_path / "rxinfer"
    output_dir.mkdir()
    (output_dir / "free_energy_trace.jsonl").write_text(
        json.dumps(_valid_free_energy_record()) + "\n", encoding="utf-8"
    )
    (output_dir / "posteriors.json").write_text("{}", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    (output_dir / "model_config.json").write_text("{}", encoding="utf-8")
    records = validate_rxinfer_outputs(output_dir)
    assert len(records) == 1
    assert records[0]["value"] == 1.0
