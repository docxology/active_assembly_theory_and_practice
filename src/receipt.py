"""Shared source-bound receipt metadata and fail-closed status vocabulary.

The research package emits many different scientific artifacts, but they all
need the same minimum lineage fields.  This module intentionally contains no
domain logic: it provides a small, dependency-light envelope that producers
can attach to their domain payloads and validators can check consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


RECEIPT_SCHEMA_VERSION = "2.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EVIDENCE_STATUSES = frozenset(
    {
        "validated",
        "validated_bounded",
        "implemented_local",
        "underpowered",
        "unsupported",
        "unavailable",
        "failed",
        "blocked_external",
        "falsified",
        "unresolved",
        "not_yet_observed",
    }
)


@dataclass(frozen=True)
class ReceiptEnvelope:
    """Minimum provenance and claim-boundary metadata for a generated receipt."""

    schema_version: str
    protocol_id: str
    protocol_sha256: str
    generator_revision: str
    input_hashes: dict[str, str]
    toolchain_versions: dict[str, str]
    status: str
    limitations: tuple[str, ...] = ()
    unsupported_cases: tuple[str, ...] = ()
    underpowered_cases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the envelope."""

        return {
            "schema_version": self.schema_version,
            "receipt_schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "generator_revision": self.generator_revision,
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "toolchain_versions": dict(sorted(self.toolchain_versions.items())),
            "status": self.status,
            "limitations": list(self.limitations),
            "unsupported_cases": list(self.unsupported_cases),
            "underpowered_cases": list(self.underpowered_cases),
        }


def build_receipt_envelope(
    *,
    protocol: dict[str, Any],
    generator_revision: str,
    input_hashes: dict[str, str],
    toolchain_versions: dict[str, str],
    status: str,
    limitations: Iterable[str] = (),
    unsupported_cases: Iterable[str] = (),
    underpowered_cases: Iterable[str] = (),
) -> dict[str, Any]:
    """Build validated receipt fields for a domain-specific payload."""

    envelope = ReceiptEnvelope(
        schema_version=RECEIPT_SCHEMA_VERSION,
        protocol_id=str(protocol.get("protocol_id", "")),
        protocol_sha256=str(protocol.get("protocol_sha256", "")),
        generator_revision=str(generator_revision),
        input_hashes={str(key): str(value) for key, value in input_hashes.items()},
        toolchain_versions={
            str(key): str(value) for key, value in toolchain_versions.items()
        },
        status=str(status),
        limitations=tuple(str(value) for value in limitations),
        unsupported_cases=tuple(str(value) for value in unsupported_cases),
        underpowered_cases=tuple(str(value) for value in underpowered_cases),
    )
    valid, detail = validate_receipt_envelope(envelope.to_dict())
    if not valid:
        raise ValueError(f"invalid receipt envelope: {detail}")
    return envelope.to_dict()


def validate_receipt_envelope(
    payload: object,
    *,
    expected_protocol: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate the shared receipt fields without interpreting domain results."""

    if not isinstance(payload, dict):
        return False, "receipt envelope must be an object"
    required = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "generator_revision",
        "input_hashes",
        "toolchain_versions",
        "status",
        "limitations",
        "unsupported_cases",
        "underpowered_cases",
    }
    missing = sorted(required - set(payload))
    if missing:
        return False, f"receipt fields missing: {missing}"
    receipt_schema = payload.get(
        "receipt_schema_version", payload.get("schema_version")
    )
    if receipt_schema != RECEIPT_SCHEMA_VERSION or (
        "receipt_schema_version" in payload
        and payload.get("schema_version") != RECEIPT_SCHEMA_VERSION
    ):
        return False, f"receipt schema={payload.get('schema_version')}"
    status = payload.get("status")
    if status not in EVIDENCE_STATUSES:
        return False, f"unknown evidence status={status!r}"
    for key in ("protocol_id", "protocol_sha256", "generator_revision"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            return False, f"receipt {key} is empty"
    if not _SHA256_RE.fullmatch(str(payload["protocol_sha256"])):
        return False, "receipt protocol_sha256 is not a lowercase SHA-256 digest"
    for key in ("input_hashes", "toolchain_versions"):
        value = payload.get(key)
        if not isinstance(value, dict) or any(
            not isinstance(item_key, str) or not isinstance(item_value, str)
            for item_key, item_value in value.items()
        ):
            return False, f"receipt {key} is not a string mapping"
    if any(
        not _SHA256_RE.fullmatch(value) for value in payload["input_hashes"].values()
    ):
        return False, "receipt input_hashes contains a non-SHA-256 digest"
    for key in ("limitations", "unsupported_cases", "underpowered_cases"):
        value = payload.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return False, f"receipt {key} is not a string list"
    if expected_protocol is not None:
        if payload.get("protocol_id") != expected_protocol.get("protocol_id"):
            return False, "receipt protocol id mismatch"
        if payload.get("protocol_sha256") != expected_protocol.get("protocol_sha256"):
            return False, "receipt protocol hash mismatch"
    return True, "ok"


def promote_evidence_summary(root: Path) -> dict[str, Any]:
    """Summarize the promotion-tranche receipt statuses without executing anything.

    Read-only projection of the evidence surfaces the promotion pipeline
    depends on.  Imports are deferred: this module is imported by nearly every
    producer, and the evidence constants live in modules that import it back.
    """

    from .external_labeling import LABEL_CONTRACT_ARTIFACT
    from .external_sources import (
        EXTERNAL_ACQUISITION_ARTIFACT,
        EXTERNAL_CAPABILITY_ARTIFACT,
        EXTERNAL_RECEIPT_ARTIFACT,
        EXTERNAL_STATIC_RECEIPT_ARTIFACT,
    )
    from .replication import INDEPENDENT_REPLAY_ARTIFACT, REPLICATION_ARTIFACT

    status_paths = {
        "P13_external_validation": root / EXTERNAL_RECEIPT_ARTIFACT,
        "P13_source_bound_labels": root / LABEL_CONTRACT_ARTIFACT,
        "P13_static_only": root / EXTERNAL_STATIC_RECEIPT_ARTIFACT,
        "P13_capabilities": root / EXTERNAL_CAPABILITY_ARTIFACT,
        "P13_acquisition": root / EXTERNAL_ACQUISITION_ARTIFACT,
        "P15_local_replay_harness": root / REPLICATION_ARTIFACT,
        "P15_independent_replay": root / INDEPENDENT_REPLAY_ARTIFACT,
    }
    rows: dict[str, object] = {}
    for name, path in status_paths.items():
        if not path.is_file():
            rows[name] = {"status": "not_yet_observed", "path": str(path)}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows[name] = {"status": "failed", "path": str(path), "detail": str(exc)}
            continue
        rows[name] = {
            "status": payload.get("status", "not_yet_observed"),
            "endpoint_status": payload.get("endpoint_status"),
            "path": str(path),
        }
    return {"status": "read_only_summary", "surfaces": rows}


__all__ = [
    "EVIDENCE_STATUSES",
    "RECEIPT_SCHEMA_VERSION",
    "ReceiptEnvelope",
    "build_receipt_envelope",
    "promote_evidence_summary",
    "validate_receipt_envelope",
]
