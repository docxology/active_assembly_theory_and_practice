"""Tests for the RxInfer bridge version pin and Julia-availability contract.

No mocks: the Julia-unavailable path is exercised with a genuinely absent
executable name resolved through the real ``shutil.which``, and the version
contract is exercised with plain string arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rxinfer_bridge import (
    EXPECTED_JULIA_VERSION,
    EXPECTED_RXINFER_VERSION,
    JULIA_MANIFEST,
    JULIA_PROJECT,
    _cached_rxinfer_outputs,
    julia_available,
    load_free_energy_records,
    load_categorical_prefix_records,
    resolve_julia_executable,
    run_rxinfer_backend,
    run_rxinfer_categorical_backend,
    validate_backend_versions,
    validate_free_energy_records,
    validate_julia_manifest,
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


def _manifest_text(
    *,
    julia_version: str = EXPECTED_JULIA_VERSION,
    rxinfer_version: str = EXPECTED_RXINFER_VERSION,
) -> str:
    return f'''julia_version = "{julia_version}"
manifest_format = "2.0"

[[deps.RxInfer]]
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "{rxinfer_version}"
'''


def test_committed_julia_manifest_matches_native_pins() -> None:
    validate_julia_manifest(JULIA_MANIFEST)


def test_validate_julia_manifest_accepts_matching_temp_manifest(tmp_path) -> None:
    manifest = tmp_path / "Manifest.toml"
    manifest.write_text(_manifest_text(), encoding="utf-8")
    validate_julia_manifest(manifest)


def test_validate_julia_manifest_rejects_julia_drift(tmp_path) -> None:
    manifest = tmp_path / "Manifest.toml"
    manifest.write_text(_manifest_text(julia_version="1.12.1"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest version mismatch"):
        validate_julia_manifest(manifest)


def test_validate_julia_manifest_rejects_rxinfer_drift(tmp_path) -> None:
    manifest = tmp_path / "Manifest.toml"
    manifest.write_text(_manifest_text(rxinfer_version="5.3.0"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RxInfer version mismatch"):
        validate_julia_manifest(manifest)


def test_validate_julia_manifest_fails_closed_for_missing_or_malformed_file(
    tmp_path,
) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        validate_julia_manifest(tmp_path / "missing-Manifest.toml")

    malformed = tmp_path / "malformed-Manifest.toml"
    malformed.write_text("julia_version = [", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        validate_julia_manifest(malformed)


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
        "julia_version": EXPECTED_JULIA_VERSION,
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


def test_cached_rxinfer_outputs_requires_matching_lineage(tmp_path) -> None:
    """A valid local native receipt is reusable only for the same trace hash."""

    trace = tmp_path / "trace.json"
    trace.write_text('{"sample": "fixture"}\n', encoding="utf-8")
    from src.rxinfer_bridge import file_sha256

    data_hash = file_sha256(trace)
    output_dir = tmp_path / "rxinfer"
    output_dir.mkdir()
    record = _valid_free_energy_record()
    record["data_hash"] = data_hash
    (output_dir / "free_energy_trace.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    for name in ("posteriors.json", "diagnostics.json", "model_config.json"):
        (output_dir / name).write_text("{}", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "data_hash": data_hash,
                "iteration_count": 1,
                "model_name": "gaussian_event_code_model",
                "mode": "static",
                "julia_version": EXPECTED_JULIA_VERSION,
                "rxinfer_version": EXPECTED_RXINFER_VERSION,
                "native_source_sha256": file_sha256(
                    JULIA_PROJECT / "src/ActiveAssemblyRxInfer.jl"
                ),
            }
        ),
        encoding="utf-8",
    )
    cached = _cached_rxinfer_outputs(trace, output_dir, iterations=1, categorical=False)
    assert cached is not None
    trace.write_text('{"sample": "changed"}\n', encoding="utf-8")
    assert (
        _cached_rxinfer_outputs(trace, output_dir, iterations=1, categorical=False)
        is None
    )


def test_native_backends_execute_real_julia_in_isolated_output_roots(tmp_path) -> None:
    """Exercise both native execution paths, not only cached receipt reuse."""

    if not julia_available():
        pytest.skip("pinned Julia unavailable")
    trace = Path("output/data/trace_bundle.json").resolve()
    gaussian = run_rxinfer_backend(
        trace, output_dir=tmp_path / "gaussian", iterations=1, instantiate=False
    )
    categorical = run_rxinfer_categorical_backend(
        trace, output_dir=tmp_path / "categorical", iterations=1, instantiate=False
    )
    assert gaussian.mode == "static"
    assert categorical.mode == "categorical"
    assert len(gaussian.free_energy_values) == 1
    assert len(categorical.free_energy_values) == 1
    assert str(tmp_path) in gaussian.artifact_paths["diagnostics"]
    directory = tmp_path / "categorical"
    prefixes = load_categorical_prefix_records(directory)
    assert prefixes[-1]["value"] == pytest.approx(categorical.free_energy_values[-1])
    assert len({row["value"] for row in prefixes}) > 1
    path = directory / "categorical_evidence_by_prefix.json"
    original = path.read_text()
    for key, bad in (
        ("value", prefixes[0]["value"] + 0.1),
        ("value", True),
        ("source_field", "invented"),
        ("conditioning", "iteration"),
        ("observation_count", 999),
        ("backend_iterations", 8),
        ("model_digest", "foreign"),
    ):
        rows = json.loads(original)
        rows[0][key] = bad
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError):
            load_categorical_prefix_records(directory)
    path.write_text(original)
    config_path = directory / "categorical_model_config.json"
    config = json.loads(config_path.read_text())
    config["emission_matrix"][0][0] = -0.1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="malformed distributions"):
        load_categorical_prefix_records(directory)
