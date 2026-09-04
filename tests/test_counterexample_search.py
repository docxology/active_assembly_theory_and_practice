"""Tests for the bounded formal counterexample search receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.counterexample_search import (
    COUNTEREXAMPLE_SEARCH_ARTIFACT,
    COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION,
    _CANDIDATE_SPECS,
    _evaluate_dec_inc_restores_of,
    _inputs_from_serialized,
    _run_fresh,
    counterexample_search_payload,
    validate_counterexample_search_payload,
    write_counterexample_search,
)


def _assert_predicate_fails(property_id: str, serialized_inputs: object) -> None:
    inputs = _inputs_from_serialized(serialized_inputs)
    assert inputs is not None, "witness inputs must deserialize"
    spec = next(item for item in _CANDIDATE_SPECS if item.property_id == property_id)
    holds, _ = spec.evaluate(inputs, _run_fresh)
    assert not holds, f"witness for {property_id} must fail its predicate"


def test_every_falsified_witness_fails_its_predicate_on_rerun() -> None:
    payload = counterexample_search_payload()
    falsified = [
        row
        for row in payload["candidates"]
        if row["status"] == "falsified_by_counterexample"
    ]
    assert falsified, "the declared search must falsify at least one candidate"
    for row in falsified:
        witness = row["witness"]
        assert witness["reverified"] is True
        _assert_predicate_fails(row["property_id"], witness["inputs"])
        # The independent fresh run must reproduce the recorded projections.
        inputs = _inputs_from_serialized(witness["inputs"])
        assert inputs is not None
        before, after = _run_fresh(
            bytes.fromhex(witness["runs"][0]["code_hex"]), inputs
        )
        assert before == witness["before"]
        assert after == witness["after"]


def test_expected_counterexamples_are_found_smallest_first() -> None:
    payload = counterexample_search_payload()
    by_id = {row["property_id"]: row for row in payload["candidates"]}
    sub = by_id["sub_equals_add_of_negation_incl_cf"]
    assert sub["status"] == "falsified_by_counterexample"
    assert sub["witness"]["inputs"]["registers"] == {"rax": 0, "rbx": 1, "rcx": 0}
    sar = by_id["sar_equals_signed_division"]
    assert sar["witness"]["inputs"]["registers"]["rax"] == 2**64 - 1
    assert sar["witness"]["inputs"]["registers"]["rcx"] == 1
    dec_inc = by_id["dec_then_inc_restores_of"]
    assert dec_inc["witness"]["inputs"]["registers"]["rax"] == 2**63
    mov32 = by_id["mov32_preserves_upper_bits"]
    assert mov32["status"] == "falsified_by_counterexample"
    holds_ids = {
        row["property_id"]
        for row in payload["candidates"]
        if row["status"] == "holds_in_searched_domain"
    }
    assert holds_ids == {
        "add_cf_symmetry",
        "inc_dec_preserve_cf_and_definedness",
        "adc_zero_carry_equals_add",
        "shl_count_masked_mod_64",
    }
    assert payload["falsified_count"] == 4
    assert payload["holds_count"] == 4
    assert payload["status"] == "validated_bounded"


def test_holds_rows_carry_positive_searched_domain_sizes() -> None:
    payload = counterexample_search_payload()
    for row in payload["candidates"]:
        assert isinstance(row["searched_domain_size"], int)
        assert row["searched_domain_size"] > 0
        if row["status"] == "holds_in_searched_domain":
            assert row["witness"] is None
        else:
            assert row["witness"] is not None
            assert row["witness"]["reverified"] is True


def test_dec_inc_counterexample_witness_projects_of_flip() -> None:
    witness_inputs = _inputs_from_serialized(
        {
            "registers": {"rax": 2**63, "rbx": 0, "rcx": 0},
            "initial_eflags": 0,
            "initial_flags_defined": {
                "CF": True,
                "PF": True,
                "AF": True,
                "ZF": True,
                "SF": True,
                "OF": True,
            },
        }
    )
    assert witness_inputs is not None
    holds, evidence = _evaluate_dec_inc_restores_of(witness_inputs, _run_fresh)
    assert not holds
    run = evidence["runs"][0]
    before = run["before"]
    after = run["after"]
    of_bit = 1 << 11
    assert before["eflags"] & of_bit != after["eflags"] & of_bit
    assert before["registers"]["rax"] == 2**63
    assert after["registers"]["rax"] == 2**63


def test_validator_rejects_wrong_schema_version() -> None:
    payload = counterexample_search_payload()
    payload["schema_version"] = "0.0.1"
    valid, detail = validate_counterexample_search_payload(payload)
    assert not valid
    assert "schema" in detail


def test_validator_rejects_witness_that_does_not_fail_predicate() -> None:
    payload = counterexample_search_payload()
    sub = next(
        row
        for row in payload["candidates"]
        if row["property_id"] == "sub_equals_add_of_negation_incl_cf"
    )
    fabricated = {
        "registers": {"rax": 3, "rbx": 0, "rcx": 0},
        "initial_eflags": 0,
        "initial_flags_defined": {
            "CF": True,
            "PF": True,
            "AF": True,
            "ZF": True,
            "SF": True,
            "OF": True,
        },
    }
    with pytest.raises(AssertionError):
        _assert_predicate_fails("sub_equals_add_of_negation_incl_cf", fabricated)
    sub["witness"]["inputs"] = fabricated
    valid, detail = validate_counterexample_search_payload(payload)
    assert not valid
    assert "does not fail its predicate" in detail


def test_validator_rejects_missing_input_hashes() -> None:
    payload = counterexample_search_payload()
    payload["input_hashes"].pop("src/formal_correspondence.py")
    valid, detail = validate_counterexample_search_payload(payload)
    assert not valid
    assert "input hash missing" in detail


def test_validator_rejects_unresolved_and_unexpected_statuses() -> None:
    payload = counterexample_search_payload()
    payload["candidates"][0]["status"] = "unresolved"
    valid, detail = validate_counterexample_search_payload(payload)
    assert not valid
    assert "unresolved" in detail

    payload = counterexample_search_payload()
    for row in payload["candidates"]:
        if row["expected"] == "falsified":
            row["status"] = "holds_in_searched_domain"
            row["witness"] = None
    valid, detail = validate_counterexample_search_payload(payload)
    assert not valid
    assert "contradicts" in detail


def test_writer_round_trip_seals_envelope_under_tmp_path(tmp_path: Path) -> None:
    path = write_counterexample_search(tmp_path)
    assert path == tmp_path / COUNTEREXAMPLE_SEARCH_ARTIFACT
    sealed = json.loads(path.read_text(encoding="utf-8"))
    assert sealed["schema_version"] == COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION
    assert sealed["artifact"] == COUNTEREXAMPLE_SEARCH_ARTIFACT
    assert sealed["receipt_schema_version"] == "2.0.0"
    assert sealed["status"] == "validated_bounded"
    assert sealed["protocol_sha256"] in {
        entry for entry in sealed["input_hashes"].values()
    }
    for key in (
        "research_protocol.json",
        "src/counterexample_search.py",
        "src/formal_correspondence.py",
    ):
        assert key in sealed["input_hashes"]
    valid, detail = validate_counterexample_search_payload(sealed)
    assert valid, detail


def test_payload_is_deterministic_across_calls() -> None:
    first = counterexample_search_payload()
    second = counterexample_search_payload()
    assert first["candidates"] == second["candidates"]
    assert first["schema_version"] == COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION


def test_validator_rejects_structural_drift() -> None:
    """Roster, count, and witness-structure drift all fail closed."""

    def falsified_row(payload: dict[str, object]) -> dict[str, object]:
        return next(
            row
            for row in payload["candidates"]  # type: ignore[index,union-attr]
            if row["status"] == "falsified_by_counterexample"
        )

    base = counterexample_search_payload()

    drifted = counterexample_search_payload()
    drifted["candidates"] = drifted["candidates"][:7]
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "candidates are missing" in detail

    drifted = counterexample_search_payload()
    drifted["candidates"] = list(reversed(drifted["candidates"]))
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "roster drift" in detail

    drifted = counterexample_search_payload()
    drifted["falsified_count"] += 1
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "falsified_count drift" in detail

    drifted = counterexample_search_payload()
    drifted["holds_count"] += 1
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "holds_count drift" in detail

    drifted = counterexample_search_payload()
    drifted["enumeration_rule"] = None
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "enumeration_rule is missing" in detail

    drifted = counterexample_search_payload()
    holds_row = next(
        row
        for row in drifted["candidates"]
        if row["status"] == "holds_in_searched_domain"
    )
    holds_row["witness"] = falsified_row(base)["witness"]
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "carries a witness" in detail

    drifted = counterexample_search_payload()
    row = falsified_row(drifted)
    row["witness"] = {**row["witness"], "reverified": False}
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "was not re-verified" in detail

    drifted = counterexample_search_payload()
    row = falsified_row(drifted)
    row["witness"] = {
        key: value for key, value in row["witness"].items() if key != "before"
    }
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "lacks state projections" in detail

    drifted = counterexample_search_payload()
    row = falsified_row(drifted)
    row["searched_domain_size"] = 0
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "positive searched_domain_size" in detail

    drifted = counterexample_search_payload()
    row = falsified_row(drifted)
    row["witness"] = {
        **row["witness"],
        "inputs": {"registers": "not-a-mapping"},
    }
    valid, detail = validate_counterexample_search_payload(drifted)
    assert not valid
    assert "inputs" in detail and "malformed" in detail
