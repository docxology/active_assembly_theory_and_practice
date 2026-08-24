"""Frozen local CLI/manifest replication bundle for the quotient endpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .quotient_classifier import _dataset, _metrics, _sequence
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLICATION_SCHEMA_VERSION = "2.0.0"
REPLICATION_ARTIFACT = "output/data/replication_manifest.json"
PUBLIC_MANIFEST = "output/replication/public/public_manifest.json"
GOLD_LABELS = "output/replication/private/gold_labels.json"
STALE_PUBLIC_MANIFEST_PATH = "output/replication/public_manifest.json"
STALE_GOLD_LABELS_PATH = "output/replication/gold_labels.json"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    label_blind_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = {}
    for row in _dataset():
        sequence = list(_sequence(row))
        label_blind_payload = {
            "sequence": sequence,
            "features": list(row.feature_vector),
        }
        label_blind_digest = _digest(label_blind_payload)
        public_row = {
            "record_id": label_blind_digest,
            "sequence": sequence,
            "features": list(row.feature_vector),
        }
        gold_row = {
            "record_id": label_blind_digest,
            "label": int(row.label),
            "family_id": row.family_id,
            "split": row.split,
        }
        label_blind_groups.setdefault(label_blind_digest, []).append(
            (public_row, gold_row, int(row.label))
        )
    for digest, group in label_blind_groups.items():
        labels = {label for _, _, label in group}
        if len(labels) > 1:
            raise ValueError(
                "label-blind replication rows collide with different gold labels: "
                f"{digest}"
            )
        for occurrence_index, (public_row, gold_row, _) in enumerate(group):
            if len(group) > 1:
                # Only rows with identical label-blind content reach this
                # branch, and the label-consistency guard above proves that
                # the ordinal cannot encode a hidden target.
                public_row["record_id"] = _digest(
                    {"label_blind_digest": digest, "occurrence_index": occurrence_index}
                )
                gold_row["record_id"] = public_row["record_id"]
            public.append(public_row)
            gold.append(gold_row)
    # The serialized order is a deterministic digest order over label-blind
    # fields.  The source generator's family/role order must not become a
    # covert target channel for a blind replay participant.
    order = sorted(range(len(public)), key=lambda index: public[index]["record_id"])
    return [public[index] for index in order], [gold[index] for index in order]


def replication_bundle_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the manifest metadata without embedding gold labels."""

    public, gold = _public_rows()
    protocol = scientific_protocol_payload()
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            "public_manifest": _digest(public),
            "gold_labels": _digest(gold),
            "src/replication.py": sha256_path(Path(__file__)),
        },
        toolchain_versions=toolchain_versions(("numpy",)),
        status="implemented_local",
        limitations=(
            "This is a frozen local harness, not independent replication or a completed public challenge.",
            "Gold labels remain private to the scoring boundary and are not part of public inputs.",
        ),
        unsupported_cases=(
            "Network access and arbitrary external binary execution are disabled in replay mode.",
        ),
        underpowered_cases=("Independent external replay has not yet been observed.",),
    )
    return {
        **envelope,
        "schema_version": REPLICATION_SCHEMA_VERSION,
        "artifact": REPLICATION_ARTIFACT,
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
            "ordering": "deterministic_record_id_digest_order; not a target field",
        },
        "endpoint_status": "frozen_local_harness",
        "independent_replay": "not_yet_observed",
        "public_input_directory": "output/replication/public",
        "private_gold_directory": "output/replication/private",
        "labels_outside_public_directory": True,
        "limitations": list(envelope["limitations"]),
        "unsupported_cases": list(envelope["unsupported_cases"]),
        "underpowered_cases": list(envelope["underpowered_cases"]),
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
    receipt_valid, receipt_detail = validate_receipt_envelope(
        manifest, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"replication receipt: {receipt_detail}"
    if manifest.get("status") != "implemented_local":
        return False, "replication evidence status is invalid"
    if (
        not isinstance(manifest.get("generator_revision"), str)
        or not isinstance(manifest.get("input_hashes"), dict)
        or not isinstance(manifest.get("toolchain_versions"), dict)
    ):
        return False, "replication provenance is incomplete"
    if not isinstance(public, list) or not isinstance(gold, list):
        return False, "replication public/gold rows must be lists"
    if manifest.get("labels_outside_public_directory") is not True:
        return False, "replication labels are not outside the public directory"
    if (
        manifest.get("public_manifest") != PUBLIC_MANIFEST
        or manifest.get("gold_labels") != GOLD_LABELS
    ):
        return False, "replication public/private paths drifted"
    forbidden = set(manifest["input_contract"]["forbidden_public_fields"])
    public_fields = set(manifest["input_contract"]["public_fields"])
    for row in public:
        if not isinstance(row, dict):
            return False, "replication public row is not an object"
        if set(row) != public_fields or forbidden & set(row):
            return False, "replication public manifest leaks a forbidden label field"
        if (
            not isinstance(row.get("record_id"), str)
            or not isinstance(row.get("sequence"), list)
            or not isinstance(row.get("features"), list)
        ):
            return False, "replication public row shape is invalid"
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
    expected_public, expected_gold = _public_rows()
    if public != expected_public or gold != expected_gold:
        return (
            False,
            "replication public/gold rows are not the canonical deterministic order",
        )
    return True, "ok"


def write_replication_bundle(project_root: Path = PROJECT_ROOT) -> Path:
    """Write public inputs, separated gold labels, and the manifest receipt."""

    public, gold = _public_rows()
    public_dir = project_root / "output" / "replication" / "public"
    private_dir = project_root / "output" / "replication" / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    (project_root / PUBLIC_MANIFEST).write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (project_root / GOLD_LABELS).write_text(
        json.dumps(gold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for stale_path in (STALE_PUBLIC_MANIFEST_PATH, STALE_GOLD_LABELS_PATH):
        (project_root / stale_path).unlink(missing_ok=True)
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


def run_label_blind_model(
    public_path: Path,
    predictions_path: Path,
    *,
    model: str = "calibrated_interpretable",
) -> Path:
    """Run a deterministic, label-free reference scorer over public rows.

    The CLI deliberately does not read the private gold directory. These
    frozen score functions exercise the public-input/prediction interface and
    are replication plumbing diagnostics, not replacement fits for the
    source-bound co-primary benchmark.
    """

    allowed = {
        "calibrated_interpretable",
        "temporal_probabilistic",
        "neural_sequence",
    }
    if model not in allowed:
        raise ValueError(f"unknown label-blind replication model {model!r}")
    try:
        public = json.loads(public_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"public replication input is unreadable: {exc}") from exc
    if not isinstance(public, list) or not public:
        raise ValueError("public replication input must be a non-empty row list")
    predictions: list[dict[str, object]] = []
    for row in public:
        if not isinstance(row, dict) or set(row) != {
            "record_id",
            "sequence",
            "features",
        }:
            raise ValueError("public replication row is not label blind")
        record_id = row.get("record_id")
        sequence = row.get("sequence")
        features = row.get("features")
        if (
            not isinstance(record_id, str)
            or not isinstance(sequence, list)
            or not isinstance(features, list)
        ):
            raise ValueError("public replication row has invalid fields")
        numeric_features = [float(value) for value in features]
        if model == "calibrated_interpretable":
            score = float(np.mean(numeric_features)) if numeric_features else 0.0
        elif model == "temporal_probabilistic":
            score = float(len(sequence)) / max(float(len(sequence) + 16), 1.0)
        else:
            digest = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12], 16)
            score = (digest % 1000000) / 1000000.0
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
        predictions.append({"record_id": record_id, "probability": probability})
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps(
            {
                "model": model,
                "label_blind": True,
                "predictions": predictions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions_path


def score_replication_predictions(
    predictions_path: Path,
    project_root: Path = PROJECT_ROOT,
    *,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    """Score predictions against canonical or explicitly supplied gold labels."""

    valid, detail = validate_replication_bundle(project_root)
    if not valid:
        raise ValueError(detail)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if isinstance(predictions, dict) and "predictions" in predictions:
        predictions = predictions["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list or an object with predictions")
    if any(
        not isinstance(row, dict) or set(row) != {"record_id", "probability"}
        for row in predictions
    ):
        raise ValueError(
            "each prediction must contain exactly record_id and probability"
        )
    prediction_ids = [str(row["record_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction identifiers are duplicated")
    values = {str(row["record_id"]): float(row["probability"]) for row in predictions}
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()
    ):
        raise ValueError("prediction probabilities must be finite values in [0, 1]")
    selected_gold_path = project_root / GOLD_LABELS if gold_path is None else gold_path
    gold = json.loads(selected_gold_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (project_root / REPLICATION_ARTIFACT).read_text(encoding="utf-8")
    )
    if _digest(gold) != manifest.get("gold_labels_sha256"):
        raise ValueError("explicit gold labels do not match the sealed manifest hash")
    if set(values) != {str(row["record_id"]) for row in gold}:
        raise ValueError("prediction identifiers do not exactly match gold identifiers")
    labels = np.asarray([int(row["label"]) for row in gold], dtype=float)
    probabilities = np.asarray([values[str(row["record_id"])] for row in gold])
    return _metrics(labels, probabilities)
