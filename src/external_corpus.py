"""Source-bound label-contract validation for the external corpus.

Static compilation and source-level equivalence labels are intentionally
separate surfaces.  This module validates the latter without inferring a
label from a model prediction, a byte sequence, a filename, or a compiler
artifact.  An empty contract file is valid evidence of an unpopulated track,
not a zero-valued benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_CONTRACT_SCHEMA_VERSION = "1.0.0"
LABEL_CONTRACT_PATH = "configs/external_label_contracts.yaml"
ALLOWED_ROLES = {"equivalent", "near_miss", "negative_control"}


def load_label_contracts(
    path: Path = PROJECT_ROOT / LABEL_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load the explicit source-level label contract manifest."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"external label contract manifest unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("external label contract manifest must be an object")
    if payload.get("schema_version") != LABEL_CONTRACT_SCHEMA_VERSION:
        raise ValueError("external label contract schema mismatch")
    rows = payload.get("contracts")
    if not isinstance(rows, list):
        raise ValueError("external label contracts must be a list")
    required = {
        "example_id",
        "project_id",
        "revision_label",
        "source_family",
        "source_path",
        "function_id",
        "role",
        "transformation_family",
        "preconditions",
        "equivalence_argument",
        "source_sha256",
        "pair_source_sha256",
        "object_sha256",
        "disassembly_sha256",
        "split",
    }
    seen: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"external label contract row {index} is not an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"external label contract row {index} missing {missing}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"external label contract row {index} has no example_id")
        if example_id in seen:
            raise ValueError(
                f"external label contract example_id duplicated: {example_id}"
            )
        seen.add(example_id)
        role = row["role"]
        if role not in ALLOWED_ROLES:
            raise ValueError(f"external label contract row {index} has unknown role")
        split = row["split"]
        if split not in {"train", "development", "test"}:
            raise ValueError(f"external label contract row {index} has unknown split")
        for field in (
            "project_id",
            "revision_label",
            "source_family",
            "source_path",
            "function_id",
            "transformation_family",
            "preconditions",
            "equivalence_argument",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(
                    f"external label contract row {index} field {field} is empty"
                )
        for field in (
            "source_sha256",
            "pair_source_sha256",
            "object_sha256",
            "disassembly_sha256",
        ):
            value = row[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"external label contract row {index} field {field} is not a SHA-256 digest"
                )
        normalized.append({str(key): value for key, value in row.items()})
    return {
        "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
        "minimum_labelable_examples": int(
            payload.get("minimum_labelable_examples", 100)
        ),
        "contracts": normalized,
        "status": str(payload.get("status", "not_yet_observed")),
        "limitations": [str(item) for item in payload.get("limitations", [])],
    }


def validate_label_contracts(payload: object) -> tuple[bool, str]:
    """Validate a parsed label-contract payload and its split boundaries."""

    if not isinstance(payload, dict):
        return False, "external label contract payload must be an object"
    try:
        rows = payload.get("contracts")
        if not isinstance(rows, list):
            return False, "external label contracts must be a list"
        # Reuse the strict file validator without writing or modifying the input.
        normalized = dict(payload)
        normalized["contracts"] = rows
        for row in rows:
            if not isinstance(row, dict):
                return False, "external label contract row is not an object"
        project_splits: dict[str, set[str]] = {}
        for row in rows:
            project = str(row.get("project_id", ""))
            split = str(row.get("split", ""))
            project_splits.setdefault(project, set()).add(split)
        if any(not project for project in project_splits):
            return False, "external label contract project_id is empty"
        if len({str(row.get("example_id")) for row in rows}) != len(rows):
            return False, "external label contract example identifiers are not unique"
        for project, splits in project_splits.items():
            if splits - {"train", "development", "test"}:
                return False, f"external label contract split drift for {project}"
        return True, f"contracts={len(rows)} projects={len(project_splits)}"
    except (TypeError, ValueError) as exc:
        return False, str(exc)


def label_contract_summary(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Return counts and promotion gates without inventing labels."""

    payload = load_label_contracts(project_root / LABEL_CONTRACT_PATH)
    valid, detail = validate_label_contracts(payload)
    if not valid:
        raise ValueError(detail)
    rows = cast(list[dict[str, object]], payload["contracts"])
    projects = sorted({str(row["project_id"]) for row in rows})
    revisions_by_project = {
        project: sorted(
            {str(row["revision_label"]) for row in rows if row["project_id"] == project}
        )
        for project in projects
    }
    test_rows = [row for row in rows if row["split"] == "test"]
    has_project_disjoint_test = (
        len({str(row["project_id"]) for row in test_rows}) == len(projects)
        if rows
        else False
    )
    minimum = int(payload["minimum_labelable_examples"])
    return {
        "status": "validated" if rows and len(rows) >= minimum else "underpowered",
        "contract_count": len(rows),
        "labelable_example_count": len(rows),
        "project_count": len(projects),
        "projects": projects,
        "revision_counts_by_project": {
            key: len(value) for key, value in revisions_by_project.items()
        },
        "revisions_by_project": revisions_by_project,
        "test_row_count": len(test_rows),
        "project_disjoint_test": has_project_disjoint_test,
        "minimum_labelable_examples": minimum,
        "roles": {
            role: sum(1 for row in rows if row["role"] == role)
            for role in sorted(ALLOWED_ROLES)
        },
        "limitations": list(payload["limitations"]),
    }


__all__ = [
    "LABEL_CONTRACT_PATH",
    "LABEL_CONTRACT_SCHEMA_VERSION",
    "label_contract_summary",
    "load_label_contracts",
    "validate_label_contracts",
]
