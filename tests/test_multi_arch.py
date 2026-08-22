from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from src.multi_arch import (
    CROSS_ARCHITECTURE_SCHEMA_VERSION,
    architecture_fixtures,
    cross_architecture_payload,
    validate_cross_architecture_payload,
    write_cross_architecture_payload,
)
import src.multi_arch as multi_arch_module


def test_cross_architecture_payload_uses_real_decoder_and_emulator() -> None:
    payload = cross_architecture_payload()
    valid, detail = validate_cross_architecture_payload(payload)
    assert valid, detail
    assert payload["schema_version"] == CROSS_ARCHITECTURE_SCHEMA_VERSION
    assert payload["architectures"] == ["x86_64", "aarch64", "riscv64"]
    assert payload["summary"] == {
        "architecture_count": 3,
        "all_decoded_three_instructions": True,
        "all_bounded_executions_passed": True,
        "same_numeric_result": True,
    }
    rows = cast(list[dict[str, object]], payload["rows"])
    for row in rows:
        instructions = cast(list[object], row["instructions"])
        execution = cast(dict[str, object], row["execution"])
        provenance = cast(dict[str, object], row["api_provenance"])
        assert len(instructions) == 3
        assert execution["final_accumulator"] == 8
        assert execution["executed_instruction_count"] == 2
        assert cast(str, provenance["decoder"]).startswith("Capstone")
        assert cast(str, provenance["emulator"]).startswith("Unicorn")


def test_cross_architecture_validator_rejects_row_and_summary_drift() -> None:
    payload = cross_architecture_payload()
    rows = cast(list[dict[str, object]], payload["rows"])
    cast(dict[str, object], rows[0]["execution"])["bounded_execution_passed"] = False
    valid, detail = validate_cross_architecture_payload(payload)
    assert not valid
    assert "execution" in detail

    payload = cross_architecture_payload()
    cast(dict[str, object], payload["summary"])["same_numeric_result"] = False
    valid, detail = validate_cross_architecture_payload(payload)
    assert not valid
    assert "summary" in detail


def test_cross_architecture_writer_round_trips(tmp_path) -> None:
    path = tmp_path / write_cross_architecture_payload(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert validate_cross_architecture_payload(payload)[0]
    assert len(architecture_fixtures()) == 3


def test_cross_architecture_decoder_and_executor_fail_closed_on_bad_fixture() -> None:
    fixture = architecture_fixtures()[0]
    with pytest.raises(ValueError, match="decoded"):
        multi_arch_module._decode(replace(fixture, code=b"\x90"))

    with pytest.raises(ValueError, match="unsupported fixture architecture"):
        multi_arch_module._execute(replace(fixture, architecture="unsupported"))


def test_cross_architecture_validator_rejects_each_structural_contract() -> None:
    payload = cross_architecture_payload()
    payload["rows"] = ["not-an-object", *cast(list[object], payload["rows"])[1:]]
    assert "non-object" in validate_cross_architecture_payload(payload)[1]

    payload = cross_architecture_payload()
    rows = cast(list[dict[str, object]], payload["rows"])
    rows[0]["instructions"] = []
    assert "instruction contract" in validate_cross_architecture_payload(payload)[1]

    payload = cross_architecture_payload()
    rows = cast(list[dict[str, object]], payload["rows"])
    cast(dict[str, object], rows[0]["execution"])["bounded_execution_passed"] = False
    assert "execution contract" in validate_cross_architecture_payload(payload)[1]

    payload = cross_architecture_payload()
    rows = cast(list[dict[str, object]], payload["rows"])
    rows[0]["code_hex"] = ""
    assert "missing code hash" in validate_cross_architecture_payload(payload)[1]
