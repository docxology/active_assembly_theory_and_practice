"""Tests for the bounded quotient/invariance protocol."""

from __future__ import annotations

from copy import deepcopy

import pytest

import src.quotient_invariance as quotient_module
from src.quotient_invariance import (
    INTERVENTION_INITIAL_RAX,
    TransformationPair,
    evaluate_transformation_pair,
    quotient_invariance_payload,
    validate_quotient_invariance_payload,
    write_quotient_invariance_protocol,
)


def test_quotient_protocol_has_equivalent_pairs_and_a_near_miss() -> None:
    payload = quotient_invariance_payload()
    valid, detail = validate_quotient_invariance_payload(payload)

    assert valid, detail
    assert payload["output_relation"] == "final_RAX"
    assert payload["initial_state_interventions"] == list(INTERVENTION_INITIAL_RAX)
    assert payload["observed_equivalent_count"] == payload["expected_equivalent_count"]
    assert payload["near_miss_separated_count"] == payload["near_miss_count"]
    assert any(row["observation_surface_changes"] for row in payload["pairs"])
    near_miss = next(
        row for row in payload["pairs"] if row["pair_id"] == "near_miss_substitution"
    )
    # The near miss has the same coarse observation surface but differs in the
    # declared output relation; this is exactly the separation pressure needed
    # for a future quotient-state model.
    assert near_miss["observation_surface_changes"] is False


def test_quotient_protocol_records_real_intervention_witnesses() -> None:
    payload = quotient_invariance_payload()

    for pair in payload["pairs"]:
        assert len(pair["interventions"]) == len(INTERVENTION_INITIAL_RAX)
        assert all(
            row["reference"]["static_instruction_count"] > 0
            and row["variant"]["static_instruction_count"] > 0
            for row in pair["interventions"]
        )


def test_quotient_protocol_rejects_equivalent_witness_failure() -> None:
    payload = quotient_invariance_payload()
    first = dict(payload["pairs"][0])
    first["observed_equivalent"] = False
    payload["pairs"] = [first, *payload["pairs"][1:]]

    valid, detail = validate_quotient_invariance_payload(payload)

    assert not valid
    assert "equivalent-witness-failed" in detail


def test_quotient_protocol_records_custom_intervention_catalog() -> None:
    pair = TransformationPair(
        pair_id="custom_catalog_pair",
        motif="constant_accumulator_update",
        expected_equivalent=True,
        reference_code=bytes.fromhex("b8 01 00 00 00 83 c0 02"),
        variant_code=bytes.fromhex("b8 01 00 00 00 83 c0 02"),
        relation="final_RAX_equal_for_each_declared_initial_state",
    )

    row = evaluate_transformation_pair(pair, initial_rax_values=(0, 7))

    assert row["initial_state_interventions"] == [0, 7]
    assert [item["initial_rax"] for item in row["interventions"]] == [0, 7]
    assert row["observed_equivalent"] is True


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        ((), "must not be empty"),
        ((0, 0), "must be unique"),
        ((-1,), "unsigned 64-bit"),
        ((1 << 64,), "unsigned 64-bit"),
        ((True,), "not booleans"),
    ],
)
def test_quotient_protocol_rejects_invalid_intervention_catalog(
    catalog, message
) -> None:
    pair = TransformationPair(
        pair_id="invalid_catalog_pair",
        motif="constant_accumulator_update",
        expected_equivalent=True,
        reference_code=bytes.fromhex("b8 01 00 00 00"),
        variant_code=bytes.fromhex("b8 01 00 00 00"),
        relation="final_RAX_equal_for_each_declared_initial_state",
    )

    with pytest.raises(ValueError, match=message):
        evaluate_transformation_pair(pair, initial_rax_values=catalog)


def test_quotient_protocol_rejects_intervention_catalog_drift() -> None:
    payload = quotient_invariance_payload()
    first = dict(payload["pairs"][0])
    first["initial_state_interventions"] = [0]
    payload["pairs"] = [first, *payload["pairs"][1:]]

    valid, detail = validate_quotient_invariance_payload(payload)

    assert not valid
    assert "intervention-catalog" in detail


def test_quotient_protocol_rejects_empty_and_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="at least one transformation pair"):
        quotient_invariance_payload([])

    payload = quotient_invariance_payload()
    payload["schema_version"] = "0.0.0"
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "schema=" in detail

    payload = quotient_invariance_payload()
    payload["initial_state_interventions"] = []
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "must be non-empty" in detail

    payload = quotient_invariance_payload()
    payload["initial_state_interventions"] = [0]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "differs from canonical" in detail

    payload = quotient_invariance_payload()
    payload["pairs"] = []
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "pairs must be a non-empty list" in detail

    payload = quotient_invariance_payload()
    payload["pairs"] = ["not-a-pair"]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "non-object pair" in detail


def test_quotient_protocol_rejects_pair_schema_and_alignment_errors() -> None:
    payload = quotient_invariance_payload()
    first = dict(payload["pairs"][0])
    first.pop("initial_state_interventions")
    payload["pairs"] = [first, *payload["pairs"][1:]]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "missing" in detail

    payload = quotient_invariance_payload()
    first = deepcopy(payload["pairs"][0])
    first["interventions"] = []
    payload["pairs"] = [first, *payload["pairs"][1:]]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "intervention-row-count" in detail

    payload = quotient_invariance_payload()
    first = deepcopy(payload["pairs"][0])
    first["interventions"][0]["initial_rax"] = 999
    payload["pairs"] = [first, *payload["pairs"][1:]]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "intervention-alignment" in detail

    payload = quotient_invariance_payload()
    first = dict(payload["pairs"][0])
    first["pair_id"] = payload["pairs"][1]["pair_id"]
    payload["pairs"] = [first, *payload["pairs"][1:]]
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "duplicate pair IDs" in detail


def test_quotient_protocol_rejects_witness_and_count_drift() -> None:
    payload = quotient_invariance_payload()
    near_miss_index = next(
        index
        for index, row in enumerate(payload["pairs"])
        if not row["expected_equivalent"]
    )
    near_miss = dict(payload["pairs"][near_miss_index])
    near_miss["near_miss_separated"] = False
    payload["pairs"][near_miss_index] = near_miss
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "near-miss-not-separated" in detail

    payload = quotient_invariance_payload()
    payload["pair_count"] += 1
    payload["expected_equivalent_count"] += 1
    valid, detail = validate_quotient_invariance_payload(payload)
    assert not valid
    assert "pair_count mismatch" in detail
    assert "expected_equivalent_count mismatch" in detail


def test_quotient_writer_refuses_invalid_canonical_payload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        quotient_module,
        "validate_quotient_invariance_payload",
        lambda payload: (False, "forced-invalid-test"),
    )

    with pytest.raises(ValueError, match="forced-invalid-test"):
        quotient_module.write_quotient_invariance_protocol(tmp_path)


def test_quotient_protocol_writes_under_requested_project_root(tmp_path) -> None:
    relative_path = write_quotient_invariance_protocol(tmp_path)

    assert (tmp_path / relative_path).is_file()
