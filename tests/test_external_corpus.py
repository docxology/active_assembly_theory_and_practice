"""Regression tests for source-bound external label contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.external_corpus import (
    label_contract_summary,
    load_label_contracts,
    validate_label_contracts,
)


def _valid_contract_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "example_id": "example-1",
        "project_id": "project-a",
        "revision_label": "r1",
        "source_family": "family-a",
        "source_path": "src/a.c",
        "function_id": "f",
        "role": "equivalent",
        "transformation_family": "algebraic_rewrite",
        "preconditions": "none",
        "equivalence_argument": "same source contract",
        "source_sha256": "a" * 64,
        "pair_source_sha256": "b" * 64,
        "object_sha256": "c" * 64,
        "disassembly_sha256": "d" * 64,
        "split": "test",
    }
    row.update(overrides)
    return row


def test_empty_label_contract_is_visible_as_underpowered(tmp_path: Path) -> None:
    path = tmp_path / "contracts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "status": "not_yet_observed",
                "minimum_labelable_examples": 100,
                "contracts": [],
                "limitations": ["no independently reviewed pairs"],
            }
        ),
        encoding="utf-8",
    )
    payload = load_label_contracts(path)
    valid, detail = validate_label_contracts(payload)
    assert valid, detail
    assert label_contract_summary.__name__ == "label_contract_summary"
    assert payload["contracts"] == []
    assert payload["status"] == "not_yet_observed"


def test_label_contract_validation_rejects_duplicate_ids_and_bad_splits() -> None:
    row = _valid_contract_row(split="holdout")
    valid, detail = validate_label_contracts({"contracts": [row]})
    assert not valid
    assert "split drift" in detail or "unknown split" in detail


def test_valid_label_contract_loads_and_reports_source_bound_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contracts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "status": "validated",
                "minimum_labelable_examples": 1,
                "contracts": [
                    _valid_contract_row(),
                    _valid_contract_row(
                        example_id="example-2",
                        role="near_miss",
                        split="development",
                    ),
                    _valid_contract_row(
                        example_id="example-3",
                        role="negative_control",
                        revision_label="r2",
                        split="train",
                    ),
                ],
                "limitations": ["reviewed source pair only"],
            }
        ),
        encoding="utf-8",
    )
    payload = load_label_contracts(path)
    valid, detail = validate_label_contracts(payload)
    assert valid, detail

    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs/external_label_contracts.yaml").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    summary = label_contract_summary(project_root)
    assert summary["status"] == "validated"
    assert summary["labelable_example_count"] == 3
    assert summary["project_count"] == 1
    assert summary["project_disjoint_test"] is True
    assert summary["roles"] == {
        "equivalent": 1,
        "near_miss": 1,
        "negative_control": 1,
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not: [valid", "unreadable"),
        ("[]", "must be an object"),
        ("schema_version: 0.0.0\ncontracts: []\n", "schema mismatch"),
        ("schema_version: 1.0.0\ncontracts: {}\n", "must be a list"),
    ],
)
def test_label_contract_loader_rejects_unreadable_and_structural_inputs(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_label_contracts(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("source_path"), "missing"),
        (lambda row: row.update(example_id=""), "no example_id"),
        (lambda row: row.update(role="unknown"), "unknown role"),
        (lambda row: row.update(split="holdout"), "unknown split"),
        (lambda row: row.update(project_id=""), "field project_id is empty"),
        (lambda row: row.update(source_sha256="A" * 64), "not a SHA-256"),
    ],
)
def test_label_contract_loader_rejects_row_level_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    row = _valid_contract_row()
    mutation(row)
    path = tmp_path / "row.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": "1.0.0", "contracts": [row]}, sort_keys=False
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_label_contracts(path)


def test_label_contract_validator_has_fail_closed_payload_matrix() -> None:
    row = _valid_contract_row()
    cases = [
        (None, "must be an object"),
        ({}, "must be a list"),
        ({"contracts": [None]}, "row is not an object"),
        ({"contracts": [{**row, "project_id": ""}]}, "project_id is empty"),
        ({"contracts": [row, row]}, "not unique"),
        ({"contracts": [{**row, "split": "holdout"}]}, "split drift"),
    ]
    for payload, message in cases:
        valid, detail = validate_label_contracts(payload)
        assert not valid
        assert message in detail
