"""Tests for the executable long-horizon scientific-program contract."""

from __future__ import annotations

import json

from src.scientific_program import (
    SCIENTIFIC_PROGRAM_SCHEMA_VERSION,
    scientific_program_payload,
    validate_scientific_program_payload,
    write_scientific_program,
)
from src.scientific_protocol import (
    PROTOCOL_ID,
    PROTOCOL_SCHEMA_VERSION,
    build_scientific_protocol,
    scientific_protocol_payload,
    validate_scientific_protocol_payload,
)
from src.corpus import CORPUS_FAMILY_SPLIT, corpus_family_split


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


def test_scientific_protocol_pins_temporal_corpus_revision() -> None:
    """MED-01 freeze-first: protocol 2.1.0 pins the frozen temporal split."""

    assert PROTOCOL_SCHEMA_VERSION == "2.1.0"
    assert PROTOCOL_ID == "active-assembly-open-roadmap-v2-2026-09-17"
    payload = scientific_protocol_payload()
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert validate_scientific_protocol_payload(payload) == (True, "ok")
    block = build_scientific_protocol().split_contract["temporal_corpus_split"]
    assert block["sampling_unit"] == "source_program_family"
    assert block["frozen_before_scoring"] is True
    assert block["holdout_scoring"] == (
        "leave_one_fixture_out_over_complete_fixtures"
    )
    assert block["fixture_split"] == corpus_family_split()
    assert block["fixture_split"] == CORPUS_FAMILY_SPLIT
    assert set(block["fixture_families"].values()) == set(CORPUS_FAMILY_SPLIT)
    assert set(block["fixture_bytes_sha256"]) == set(CORPUS_FAMILY_SPLIT)
    assignment = block["family_split_assignment"]
    assert set(assignment.values()) == {"training", "development", "held_out"}
    assert assignment["x86_guard"] == "held_out"
    assert assignment["x86_guard_taken"] == "held_out"


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
