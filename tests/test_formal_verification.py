from __future__ import annotations

import json
import shutil

from src.formal_verification import (
    FORMAL_SCHEMA_VERSION,
    FORMAL_THEOREMS,
    FORMAL_TOOLCHAIN,
    formal_verification_payload,
    validate_formal_verification_payload,
    write_formal_verification,
)
from src.lean_bridge import BRIDGE_FIXTURES, BRIDGE_FIXTURE_THEOREMS
import src.formal_verification as formal_module
import pytest


def test_formal_project_compiles_and_receipt_is_source_bound(tmp_path) -> None:
    shutil.copytree("formal", tmp_path / "formal")
    path = tmp_path / write_formal_verification(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid, detail = validate_formal_verification_payload(payload)
    assert valid, detail
    assert payload["schema_version"] == FORMAL_SCHEMA_VERSION
    assert payload["toolchain"] == FORMAL_TOOLCHAIN
    assert set(payload["theorems"]) == set(FORMAL_THEOREMS)
    assert payload["build"]["build_completed"] is True


def test_formal_validator_fails_closed_on_scope_and_manifest_drift() -> None:
    payload = formal_verification_payload()
    payload["status"] = "planned"
    assert not validate_formal_verification_payload(payload)[0]

    payload = formal_verification_payload()
    payload["theorems"] = []
    assert not validate_formal_verification_payload(payload)[0]

    payload = formal_verification_payload()
    payload["scope"] = "complete x86 semantics"
    assert not validate_formal_verification_payload(payload)[0]

    payload = formal_verification_payload()
    payload["toolchain"] = "unmanaged"
    assert "toolchain" in validate_formal_verification_payload(payload)[1]

    payload = formal_verification_payload()
    payload["source_hashes"] = {}
    assert "source hashes" in validate_formal_verification_payload(payload)[1]


def test_formal_build_requires_a_project_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing formal project"):
        formal_module.build_formal_project(tmp_path)


def test_formal_theorem_manifest_covers_expansion_families() -> None:
    """MED-02: every new state-transformer theorem is pinned by exact id."""

    names = {name.removeprefix("ActiveAssembly.") for name in FORMAL_THEOREMS}
    for theorem in (
        "cmp_matches_sub_flags",
        "add_sub_inverse",
        "movzx32_to_64_zero_extension",
        "push_pop_roundtrip",
        "jcc_taken_vs_not_taken_distinct",
        "flag_definedness_logic_ops",
        "arithmetic_flags_defined_after_add_sub",
        "logic_flags_af_undefined",
        "logic_flags_cf_defined_and_clear",
        "cmp64N_preserves_registers_and_memory",
        "mov8N_preserves_upper_bits",
        "mov16N_preserves_upper_bits",
        "movzx32N_zero_extension",
        "movsx32N_sign_extension",
        "read_write_little_endian_roundtrip",
        "push64N_pop64N_roundtrip",
        "push64N_out_of_bounds_is_explicit",
        "pop64N_out_of_bounds_is_explicit",
        "jcc_undefined_flag_is_exceptional",
        "jcc_zero_flag_branch_distinct",
    ):
        assert theorem in names
    bridge_theorems = {
        theorem.removeprefix("ActiveAssembly.") for theorem in BRIDGE_FIXTURE_THEOREMS
    }
    assert bridge_theorems <= names
    assert len(FORMAL_THEOREMS) == len(set(FORMAL_THEOREMS))


def test_checked_fragment_prose_names_expanded_families() -> None:
    payload = formal_verification_payload()
    operations = " ".join(payload["checked_fragment"]["operations"]).lower()
    for family in (
        "cmp",
        "af undefined",
        "push/pop",
        "jcc",
        "zero-extend",
        "movzx/movsx",
    ):
        assert family in operations, family


def test_bridge_fixture_theorems_are_pinned_in_the_receipt(tmp_path) -> None:
    shutil.copytree("formal", tmp_path / "formal")
    path = tmp_path / write_formal_verification(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    bridge = payload["cross_language_bridge"]
    fixture_rows = {row["bridge_id"]: row["theorems"] for row in bridge["fixtures"]}
    for fixture in BRIDGE_FIXTURES:
        assert fixture_rows[fixture["bridge_id"]] == list(fixture["theorems"])
    for theorem in BRIDGE_FIXTURE_THEOREMS:
        assert theorem in payload["theorems"]
    assert bridge["kernel_checked"] is True
    for row in bridge["fixtures"]:
        assert row["trace_matches_lean_expected"] is True
