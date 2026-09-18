"""MED-02 expansion: Lean-column correspondence receipt tests.

The correspondence receipt gained per-fixture Lean columns
(``lean_bridge_status``) and a dedicated domain schema version.  These tests
lock the new surface and its fail-closed validator alongside the real
kernel-checked bridge fixtures.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from src.formal_correspondence import (
    FORMAL_CORRESPONDENCE_SCHEMA_VERSION,
    formal_correspondence_payload,
    validate_formal_correspondence_payload,
)

EXPECTED_BRIDGE_IDS = (
    "push_pop",
    "cmp_jcc_taken",
    "cmp_jcc_not_taken",
    "mov_width_32",
    "mov_width_64",
    "mov_width_16",
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return formal_correspondence_payload()


def test_lean_columns_are_present_and_valid(payload: dict[str, Any]) -> None:
    assert FORMAL_CORRESPONDENCE_SCHEMA_VERSION == "2.1.0"
    assert payload["schema_version"] == FORMAL_CORRESPONDENCE_SCHEMA_VERSION
    status = payload["lean_bridge_status"]
    assert {row["bridge_id"] for row in status} == set(EXPECTED_BRIDGE_IDS)
    for row in status:
        assert row["kernel_checked"] is True
        assert row["trace_matches_lean_expected"] is True
        assert isinstance(row["theorem_count"], int)
        assert row["theorem_count"] > 0
    assert payload["lean_prerequisite"]["axiom_report_clean"] is True
    assert payload["lean_prerequisite"]["theorems"]


def test_receipt_validates_with_lean_columns(payload: dict[str, Any]) -> None:
    valid, detail = validate_formal_correspondence_payload(payload)
    assert valid, detail


def test_bridge_fixture_traces_match_real_execution(
    payload: dict[str, Any],
) -> None:
    fixtures = payload["cross_language_bridge"]["fixtures"]
    assert {row["bridge_id"] for row in fixtures} == set(EXPECTED_BRIDGE_IDS)
    for row in fixtures:
        assert row["kernel_checked"] is True
        assert row["trace_matches_lean_expected"] is True
        assert row["boundary_matches_lean_expected"] is True
        assert row["final_matches_lean_expected"] is True
        assert row["memory_matches_lean_expected"] is True
        assert row["reference_boundary_matches"] is True
        assert row["reference_final_matches"] is True


def test_fail_closed_mutation_matrix(payload: dict[str, Any]) -> None:
    """Every Lean-column tamper must be rejected by the validator."""

    def reject(mutation) -> tuple[bool, str]:
        candidate = copy.deepcopy(payload)
        mutation(candidate)
        return validate_formal_correspondence_payload(candidate)

    def tamper_lean_prerequisite_theorem(candidate: dict[str, Any]) -> None:
        candidate["lean_prerequisite"]["theorems"] = [
            "ActiveAssembly.totally_made_up_theorem"
        ]

    def tamper_bridge_fixture_theorem(candidate: dict[str, Any]) -> None:
        candidate["cross_language_bridge"]["fixtures"][0]["theorems"][0] = (
            "ActiveAssembly.totally_made_up_theorem"
        )

    def kernel_checked_false(candidate: dict[str, Any]) -> None:
        candidate["lean_bridge_status"][0]["kernel_checked"] = False

    def trace_mismatch(candidate: dict[str, Any]) -> None:
        candidate["lean_bridge_status"][1]["trace_matches_lean_expected"] = False

    def empty_lean_status(candidate: dict[str, Any]) -> None:
        candidate["lean_bridge_status"] = []

    def duplicate_bridge_row(candidate: dict[str, Any]) -> None:
        candidate["lean_bridge_status"].append(
            dict(candidate["lean_bridge_status"][0])
        )

    def zero_theorem_count(candidate: dict[str, Any]) -> None:
        candidate["lean_bridge_status"][0]["theorem_count"] = 0

    def axiom_flag_false(candidate: dict[str, Any]) -> None:
        candidate["lean_prerequisite"]["axiom_report_clean"] = False

    def schema_drift(candidate: dict[str, Any]) -> None:
        candidate["schema_version"] = "0.0.0"

    def drop_fixture(candidate: dict[str, Any]) -> None:
        candidate["cross_language_bridge"]["fixtures"] = []

    mutations = (
        tamper_lean_prerequisite_theorem,
        tamper_bridge_fixture_theorem,
        kernel_checked_false,
        trace_mismatch,
        empty_lean_status,
        duplicate_bridge_row,
        zero_theorem_count,
        axiom_flag_false,
        schema_drift,
        drop_fixture,
    )
    for mutation in mutations:
        valid, detail = reject(mutation)
        assert not valid, f"{mutation.__name__} accepted: {detail}"


def test_payload_builds_are_deterministic(payload: dict[str, Any]) -> None:
    """Two fresh payload builds must be byte-identical domain content."""

    fresh = formal_correspondence_payload()
    assert json.dumps(payload, sort_keys=True) == json.dumps(fresh, sort_keys=True)


def test_written_artifact_is_fresh_on_recompute() -> None:
    """The writer and the recompute must agree through the freshness gate."""

    from src.artifact_contract import artifact_domain_view
    from src.formal_correspondence import write_formal_correspondence
    from src.validation import _domain_payload_equal

    path = write_formal_correspondence()
    disk = json.loads(path.read_text(encoding="utf-8"))
    fresh = formal_correspondence_payload()
    assert artifact_domain_view(disk) == artifact_domain_view(fresh)
    assert _domain_payload_equal(disk, fresh) is True
