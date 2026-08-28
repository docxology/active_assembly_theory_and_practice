"""Source-bound label-contract validation for the external corpus.

Static compilation and source-level equivalence labels are intentionally
separate surfaces.  This module validates the latter without inferring a
label from a model prediction, a byte sequence, a filename, or a compiler
artifact.  An empty contract file is valid evidence of an unpopulated track,
not a zero-valued benchmark.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_CONTRACT_SCHEMA_VERSION = "2.0.0"
LEGACY_LABEL_CONTRACT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_LABEL_CONTRACT_SCHEMA_VERSIONS = frozenset(
    {LEGACY_LABEL_CONTRACT_SCHEMA_VERSION, LABEL_CONTRACT_SCHEMA_VERSION}
)
LABEL_CONTRACT_PATH = "configs/external_label_contracts.yaml"
ALLOWED_ROLES = {"equivalent", "near_miss", "negative_control"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASE_REQUIRED_FIELDS = {
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
_V2_REQUIRED_FIELDS = {
    "source_id",
    "contract_basis",
    "base_object_sha256",
    "pair_object_sha256",
    "base_disassembly_sha256",
    "pair_disassembly_sha256",
    "base_source_artifact",
    "pair_source_artifact",
    "base_object_artifact",
    "pair_object_artifact",
    "base_disassembly_artifact",
    "pair_disassembly_artifact",
    "compiler",
    "target_triple",
    "optimization",
    "static_only",
    "execution_performed",
    "labelability_status",
}


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
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_LABEL_CONTRACT_SCHEMA_VERSIONS:
        raise ValueError("external label contract schema mismatch")
    rows = payload.get("contracts")
    if not isinstance(rows, list):
        raise ValueError("external label contracts must be a list")
    valid, detail = validate_label_contracts(payload)
    if not valid:
        raise ValueError(detail)
    normalized = [cast(dict[str, object], row) for row in rows]
    try:
        minimum = int(payload.get("minimum_labelable_examples", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_labelable_examples must be an integer") from exc
    return {
        "schema_version": str(schema_version),
        "minimum_labelable_examples": minimum,
        "contracts": normalized,
        "status": str(payload.get("status", "not_yet_observed")),
        "limitations": [str(item) for item in payload.get("limitations", [])],
        "exclusions": [
            cast(dict[str, object], item)
            for item in payload.get("exclusions", [])
            if isinstance(item, dict)
        ],
        **{
            str(key): value
            for key, value in payload.items()
            if key
            not in {
                "schema_version",
                "minimum_labelable_examples",
                "contracts",
                "status",
                "limitations",
                "exclusions",
            }
        },
    }


def validate_label_contracts(payload: object) -> tuple[bool, str]:
    """Validate a parsed label-contract payload and its split boundaries."""

    if not isinstance(payload, dict):
        return False, "external label contract payload must be an object"
    try:
        schema_version = payload.get(
            "schema_version", LEGACY_LABEL_CONTRACT_SCHEMA_VERSION
        )
        if schema_version not in SUPPORTED_LABEL_CONTRACT_SCHEMA_VERSIONS:
            return False, "external label contract schema mismatch"
        rows = payload.get("contracts")
        if not isinstance(rows, list):
            return False, "external label contracts must be a list"
        required = set(_BASE_REQUIRED_FIELDS)
        if schema_version == LABEL_CONTRACT_SCHEMA_VERSION:
            required |= _V2_REQUIRED_FIELDS
        seen: set[str] = set()
        project_splits: dict[str, set[str]] = {}
        family_splits: dict[str, set[str]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return False, "external label contract row is not an object"
            missing = sorted(required - set(row))
            if missing:
                return False, f"external label contract row {index} missing {missing}"
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id.strip():
                return False, "external label contract row has no example_id"
            if example_id in seen:
                return (
                    False,
                    "external label contract example identifiers are not unique",
                )
            seen.add(example_id)
            role = row.get("role")
            if role not in ALLOWED_ROLES:
                return False, "external label contract row has unknown role"
            split = row.get("split")
            if split not in {"train", "development", "test"}:
                return False, "external label contract unknown split (split drift)"
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
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    return False, f"external label contract field {field} is empty"
            for field in (
                "source_sha256",
                "pair_source_sha256",
                "object_sha256",
                "disassembly_sha256",
            ):
                if not _SHA256_RE.fullmatch(str(row.get(field, ""))):
                    return (
                        False,
                        f"external label contract field {field} is not a SHA-256 digest",
                    )
            project = str(row["project_id"])
            family = str(row["source_family"])
            project_splits.setdefault(project, set()).add(str(split))
            family_splits.setdefault(family, set()).add(str(split))
            if schema_version == LABEL_CONTRACT_SCHEMA_VERSION:
                for field in _V2_REQUIRED_FIELDS:
                    value = row.get(field)
                    if field in {
                        "static_only",
                        "execution_performed",
                    }:
                        if not isinstance(value, bool):
                            return (
                                False,
                                f"external label contract field {field} is not boolean",
                            )
                    elif field.endswith("_artifact"):
                        if (
                            not isinstance(value, str)
                            or not value
                            or Path(value).is_absolute()
                            or ".." in Path(value).parts
                        ):
                            return (
                                False,
                                f"external label contract field {field} is not a safe relative path",
                            )
                    elif field.endswith("_sha256"):
                        if not _SHA256_RE.fullmatch(str(value)):
                            return (
                                False,
                                f"external label contract field {field} is not a SHA-256 digest",
                            )
                    elif not isinstance(value, str) or not value.strip():
                        return False, f"external label contract field {field} is empty"
                for field in (
                    "base_object_sha256",
                    "pair_object_sha256",
                    "base_disassembly_sha256",
                    "pair_disassembly_sha256",
                ):
                    if not _SHA256_RE.fullmatch(str(row.get(field, ""))):
                        return (
                            False,
                            f"external label contract field {field} is not a SHA-256 digest",
                        )
                if row.get("static_only") is not True:
                    return False, "external label contract row is not static-only"
                if row.get("execution_performed") is not False:
                    return (
                        False,
                        "external label contract row execution_performed records execution",
                    )
                if row.get("labelability_status") != "labelable":
                    return False, "external label contract row is not labelable"
        if schema_version == LABEL_CONTRACT_SCHEMA_VERSION:
            if any(len(splits) != 1 for splits in project_splits.values()):
                return False, "external label contract project split leakage"
            if any(len(splits) != 1 for splits in family_splits.values()):
                return False, "external label contract source-family split leakage"
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
    schema_version = str(
        payload.get("schema_version", LEGACY_LABEL_CONTRACT_SCHEMA_VERSION)
    )
    projects = sorted({str(row["project_id"]) for row in rows})
    revisions_by_project = {
        project: sorted(
            {str(row["revision_label"]) for row in rows if row["project_id"] == project}
        )
        for project in projects
    }
    test_rows = [row for row in rows if row["split"] == "test"]
    test_projects = {str(row["project_id"]) for row in test_rows}
    fit_projects = {
        str(row["project_id"])
        for row in rows
        if row["split"] in {"train", "development"}
    }
    project_splits = {
        project: sorted(
            {str(row["split"]) for row in rows if str(row["project_id"]) == project}
        )
        for project in projects
    }
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(str(row["source_family"]), set()).add(
            str(row["split"])
        )
    # A held-out test is meaningful only when there is at least one fitting
    # project to hold it out from. ``set().isdisjoint(set())`` is true, but
    # treating an all-test or empty contract as project-held-out evidence
    # would make a malformed receipt look promoted.
    project_held_out_test = (
        bool(test_projects)
        and bool(fit_projects)
        and test_projects.isdisjoint(fit_projects)
    )
    has_project_disjoint_test = len(test_projects) == len(projects) if rows else False
    minimum = int(payload["minimum_labelable_examples"])
    status = (
        "validated"
        if rows
        and len(rows) >= minimum
        and (
            schema_version == LEGACY_LABEL_CONTRACT_SCHEMA_VERSION
            or project_held_out_test
        )
        else "underpowered"
    )
    return {
        "status": status,
        "schema_version": schema_version,
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
        "project_held_out_test": project_held_out_test,
        "test_projects": sorted(test_projects),
        "fit_projects": sorted(fit_projects),
        "project_splits": project_splits,
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ("train", "development", "test")
        },
        "source_family_split_disjoint": all(
            len(splits) == 1 for splits in family_splits.values()
        ),
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
    "LEGACY_LABEL_CONTRACT_SCHEMA_VERSION",
    "SUPPORTED_LABEL_CONTRACT_SCHEMA_VERSIONS",
    "label_contract_summary",
    "load_label_contracts",
    "validate_label_contracts",
]
