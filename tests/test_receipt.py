"""Direct tests for the shared source-bound receipt envelope."""

from __future__ import annotations

import copy

import pytest

from src.receipt import (
    RECEIPT_SCHEMA_VERSION,
    build_receipt_envelope,
    validate_receipt_envelope,
)


def _envelope() -> dict[str, object]:
    return build_receipt_envelope(
        protocol={"protocol_id": "protocol-v2", "protocol_sha256": "a" * 64},
        generator_revision="generator-revision",
        input_hashes={"input.json": "b" * 64},
        toolchain_versions={"python": "3.13"},
        status="validated_bounded",
        limitations=("bounded fixture",),
        unsupported_cases=("unsupported instruction",),
        underpowered_cases=("small pilot",),
    )


def test_receipt_build_round_trip_and_expected_protocol_match() -> None:
    payload = _envelope()
    assert payload["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert validate_receipt_envelope(
        payload,
        expected_protocol={
            "protocol_id": "protocol-v2",
            "protocol_sha256": "a" * 64,
        },
    ) == (True, "ok")
    assert payload["limitations"] == ["bounded fixture"]


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (lambda payload: payload.pop("status"), "fields missing"),
        (lambda payload: payload.update(schema_version="1.0.0"), "schema"),
        (lambda payload: payload.update(status="unknown"), "unknown evidence"),
        (lambda payload: payload.update(protocol_id=""), "protocol_id is empty"),
        (lambda payload: payload.update(input_hashes={"x": 1}), "string mapping"),
        (lambda payload: payload.update(limitations=[""]), "string list"),
    ],
)
def test_receipt_validator_rejects_structural_drift(mutation, detail: str) -> None:
    payload = copy.deepcopy(_envelope())
    mutation(payload)
    valid, message = validate_receipt_envelope(payload)
    assert not valid
    assert detail in message


def test_receipt_validator_rejects_protocol_mismatch_and_non_object() -> None:
    payload = _envelope()
    assert validate_receipt_envelope(None)[0] is False
    valid, detail = validate_receipt_envelope(
        payload,
        expected_protocol={
            "protocol_id": "other",
            "protocol_sha256": "c" * 64,
        },
    )
    assert not valid
    assert "protocol id mismatch" in detail


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("protocol_sha256", "not-a-hash", "protocol_sha256"),
        ("input_hashes", {"input.json": "not-a-hash"}, "input_hashes"),
    ],
)
def test_receipt_validator_rejects_noncanonical_hashes(
    field: str, value: object, detail: str
) -> None:
    payload = _envelope()
    payload[field] = value
    valid, message = validate_receipt_envelope(payload)
    assert not valid
    assert detail in message


def test_receipt_builder_fails_closed_for_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown evidence"):
        build_receipt_envelope(
            protocol={"protocol_id": "p", "protocol_sha256": "a" * 64},
            generator_revision="g",
            input_hashes={},
            toolchain_versions={},
            status="not-a-status",
        )
