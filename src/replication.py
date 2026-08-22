"""Frozen local CLI/manifest replication bundle for the quotient endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .quotient_classifier import _dataset, _metrics, _sequence
from .provenance import generator_revision, toolchain_versions
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLICATION_SCHEMA_VERSION = "1.0.0"
REPLICATION_ARTIFACT = "output/data/replication_manifest.json"
PUBLIC_MANIFEST = "output/replication/public_manifest.json"
GOLD_LABELS = "output/replication/gold_labels.json"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for row_index, row in enumerate(_dataset()):
        sequence = list(_sequence(row))
        public_row = {
            "record_id": _digest(
                {
                    "sequence": sequence,
                    "features": list(row.feature_vector),
                    # The occurrence index is a label-blind disambiguator for
                    # otherwise identical public rows. It is not a target,
                    # family, split, or transformation label.
                    "occurrence_index": row_index,
                }
            ),
            "sequence": sequence,
            "features": list(row.feature_vector),
        }
        public.append(public_row)
        gold.append(
            {
                "record_id": public_row["record_id"],
                "label": int(row.label),
                "family_id": row.family_id,
                "split": row.split,
            }
        )
    return public, gold


def replication_bundle_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the manifest metadata without embedding gold labels."""

    del project_root
    public, gold = _public_rows()
    protocol = scientific_protocol_payload()
    return {
        "schema_version": REPLICATION_SCHEMA_VERSION,
        "artifact": REPLICATION_ARTIFACT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "generator_revision": generator_revision(Path(__file__)),
        "input_hashes": {
            "public_manifest": _digest(public),
            "gold_labels": _digest(gold),
        },
        "toolchain_versions": toolchain_versions(("numpy",)),
        "public_manifest": PUBLIC_MANIFEST,
        "gold_labels": GOLD_LABELS,
        "public_row_count": len(public),
        "gold_row_count": len(gold),
        "public_manifest_sha256": _digest(public),
        "gold_labels_sha256": _digest(gold),
        "label_separated": True,
        "network_required_for_replay": False,
        "arbitrary_external_execution": False,
        "input_contract": {
            "public_fields": ["record_id", "sequence", "features"],
            "forbidden_public_fields": [
                "label",
                "family_id",
                "split",
                "variant_role",
                "quotient_class",
            ],
        },
        "status": "frozen_local_harness",
        "independent_replay": "not_yet_observed",
    }


def validate_replication_bundle(project_root: Path = PROJECT_ROOT) -> tuple[bool, str]:
    """Validate public/gold separation and deterministic hashes."""

    manifest_path = project_root / REPLICATION_ARTIFACT
    public_path = project_root / PUBLIC_MANIFEST
    gold_path = project_root / GOLD_LABELS
    if not manifest_path.exists() or not public_path.exists() or not gold_path.exists():
        return False, "replication bundle files are missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public = json.loads(public_path.read_text(encoding="utf-8"))
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"replication bundle unreadable: {exc}"
    if manifest.get("schema_version") != REPLICATION_SCHEMA_VERSION:
        return False, "replication schema mismatch"
    protocol = scientific_protocol_payload()
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        return False, "replication protocol hash mismatch"
    if (
        not isinstance(manifest.get("generator_revision"), str)
        or not isinstance(manifest.get("input_hashes"), dict)
        or not isinstance(manifest.get("toolchain_versions"), dict)
    ):
        return False, "replication provenance is incomplete"
    if not isinstance(public, list) or not isinstance(gold, list):
        return False, "replication public/gold rows must be lists"
    forbidden = set(manifest["input_contract"]["forbidden_public_fields"])
    if any(forbidden & set(row) for row in public if isinstance(row, dict)):
        return False, "replication public manifest leaks a forbidden label field"
    if len(public) != len(gold) or len(public) != int(manifest["public_row_count"]):
        return False, "replication public/gold row counts differ"
    public_ids = [row.get("record_id") for row in public if isinstance(row, dict)]
    gold_ids = [row.get("record_id") for row in gold if isinstance(row, dict)]
    if len(public_ids) != len(set(public_ids)) or len(gold_ids) != len(set(gold_ids)):
        return False, "replication record identifiers are not unique"
    if manifest.get("public_manifest_sha256") != _digest(public):
        return False, "replication public hash mismatch"
    if manifest.get("gold_labels_sha256") != _digest(gold):
        return False, "replication gold hash mismatch"
    if set(public_ids) != set(gold_ids):
        return False, "replication public/gold identifiers differ"
    return True, "ok"


def write_replication_bundle(project_root: Path = PROJECT_ROOT) -> Path:
    """Write public inputs, separated gold labels, and the manifest receipt."""

    public, gold = _public_rows()
    replication_dir = project_root / "output" / "replication"
    replication_dir.mkdir(parents=True, exist_ok=True)
    (project_root / PUBLIC_MANIFEST).write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (project_root / GOLD_LABELS).write_text(
        json.dumps(gold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path = project_root / REPLICATION_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replication_bundle_payload(project_root), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    valid, detail = validate_replication_bundle(project_root)
    if not valid:
        raise ValueError(detail)
    return path


def score_replication_predictions(
    predictions_path: Path, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Score a prediction JSON against the separated local gold file."""

    valid, detail = validate_replication_bundle(project_root)
    if not valid:
        raise ValueError(detail)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if isinstance(predictions, dict) and "predictions" in predictions:
        predictions = predictions["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list or an object with predictions")
    values = {str(row["record_id"]): float(row["probability"]) for row in predictions}
    gold = json.loads((project_root / GOLD_LABELS).read_text(encoding="utf-8"))
    if set(values) != {str(row["record_id"]) for row in gold}:
        raise ValueError("prediction identifiers do not exactly match gold identifiers")
    labels = np.asarray([int(row["label"]) for row in gold], dtype=float)
    probabilities = np.asarray([values[str(row["record_id"])] for row in gold])
    return _metrics(labels, probabilities)
