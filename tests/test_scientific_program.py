"""Tests for the executable long-horizon scientific-program contract."""

from __future__ import annotations

import json

from src.scientific_program import (
    SCIENTIFIC_PROGRAM_SCHEMA_VERSION,
    scientific_program_payload,
    validate_scientific_program_payload,
    write_scientific_program,
)


def test_scientific_program_is_falsifiable_and_dependency_closed() -> None:
    payload = scientific_program_payload()

    valid, detail = validate_scientific_program_payload(payload)

    assert valid, detail
    assert payload["schema_version"] == SCIENTIFIC_PROGRAM_SCHEMA_VERSION
    milestones = payload["milestones"]
    milestone_ids = {row["milestone_id"] for row in milestones}
    assert len(milestone_ids) == len(milestones)
    assert payload["status_counts"] == {
        "implemented_local": 12,
        "active": 3,
        "blocked_by_evidence": 2,
        "planned": 1,
    }
    assert milestone_ids >= {
        "P9",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
        "P15",
        "P16",
        "P17",
    }
    for row in milestones:
        assert row["falsifier"]
        assert row["primary_endpoint"]
        assert row["sampling_unit"]
        assert set(row["dependencies"]).issubset(milestone_ids)


def test_scientific_program_rejects_unknown_dependencies() -> None:
    payload = scientific_program_payload()
    first = dict(payload["milestones"][0])
    first["dependencies"] = ["P999"]
    payload["milestones"] = [first, *payload["milestones"][1:]]

    valid, detail = validate_scientific_program_payload(payload)

    assert not valid
    assert "unknown-dependencies" in detail


def test_scientific_program_rejects_dependency_cycles() -> None:
    payload = scientific_program_payload()
    first = dict(payload["milestones"][0])
    second = dict(payload["milestones"][1])
    first["dependencies"] = [second["milestone_id"]]
    second["dependencies"] = [first["milestone_id"]]
    payload["milestones"] = [first, second, *payload["milestones"][2:]]

    valid, detail = validate_scientific_program_payload(payload)

    assert not valid
    assert "dependency-cycle" in detail


def test_scientific_program_write_is_deterministic(tmp_path) -> None:
    path = tmp_path / write_scientific_program(tmp_path)
    first = path.read_text(encoding="utf-8")
    write_scientific_program(tmp_path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first)["program_id"] == "active-assembly-quotient-inference"
