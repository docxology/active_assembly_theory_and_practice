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
