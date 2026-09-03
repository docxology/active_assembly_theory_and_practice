"""Frozen local CLI/manifest replication bundle for the quotient endpoint."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np

from .artifact_contract import write_json_artifact, write_json_atomic
from .quotient_classifier import _dataset, _metrics, _sequence
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLICATION_SCHEMA_VERSION = "2.0.0"
REPLICATION_ARTIFACT = "output/data/replication_manifest.json"
PUBLIC_MANIFEST = "output/replication/public/public_manifest.json"
GOLD_LABELS = "output/replication/private/gold_labels.json"
PUBLIC_PROTOCOL = "output/replication/public/protocol_manifest.json"
PUBLIC_TOOLCHAIN = "output/replication/public/toolchain_receipt.json"
PUBLIC_INSTRUCTIONS = "output/replication/public/REPLAY_INSTRUCTIONS.md"
INDEPENDENT_REPLAY_ARTIFACT = "output/data/independent_replay_receipt.json"
REPLAY_RECEIPT_SCHEMA_VERSION = "2.0.0"
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STALE_PUBLIC_MANIFEST_PATH = "output/replication/public_manifest.json"
STALE_GOLD_LABELS_PATH = "output/replication/gold_labels.json"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    """Hash a file's exact bytes for cross-machine replay validation."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_rows(path: Path) -> tuple[list[dict[str, object]], str]:
    """Read and validate a label-blind prediction file without gold access."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"prediction file is unreadable: {exc}") from exc
    model = "unspecified"
    if isinstance(payload, dict) and "predictions" in payload:
        model_value = payload.get("model")
        if model_value is not None:
            if not isinstance(model_value, str) or not model_value.strip():
                raise ValueError("prediction model must be a non-empty string")
            model = model_value
        if payload.get("label_blind") is not True:
            raise ValueError("prediction wrapper must declare label_blind=true")
        payload = payload["predictions"]
    if not isinstance(payload, list):
        raise ValueError("predictions must be a list or an object with predictions")
    rows = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"record_id", "probability"}:
            raise ValueError(
                "each prediction must contain exactly record_id and probability"
            )
        record_id = row.get("record_id")
        probability = row.get("probability")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("prediction record_id must be a non-empty string")
        if isinstance(probability, bool) or not isinstance(
            probability, (int, float, str)
        ):
            raise ValueError("prediction probability is not numeric")
        try:
            numeric_probability = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction probability is not numeric") from exc
        if (
            not np.isfinite(numeric_probability)
            or not 0.0 <= numeric_probability <= 1.0
        ):
            raise ValueError("prediction probabilities must be finite values in [0, 1]")
        rows.append({"record_id": record_id, "probability": numeric_probability})
    identifiers = [str(row["record_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prediction identifiers are duplicated")
    return rows, model


def _prediction_identifier_digest(rows: list[dict[str, object]]) -> str:
    """Hash the exact prediction identifier ordering."""

    return _digest([str(row["record_id"]) for row in rows])


def _replay_timestamp(value: str | None) -> str:
    """Return an ISO-8601 timestamp with an explicit UTC offset.

    A receipt that says only ``operator_supplied`` is not auditable: it cannot
    establish when the public bundle was run or support comparison with the
    operator's retained logs.  The CLI accepts recorded timestamps for an
    external run, while the local rehearsal receives a precise producer-side
    timestamp automatically.
    """

    if value is None:
        return datetime.now(UTC).isoformat(timespec="seconds")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("replay timestamps must be non-empty ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("replay timestamps must be valid ISO-8601 values") from exc
    if parsed.tzinfo is None:
        raise ValueError("replay timestamps must include a timezone offset")
    return value.strip()


def _timestamps_ordered(started_at: str, completed_at: str) -> bool:
    """Return whether a replay completion timestamp follows its start."""

    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return completed >= started


def _public_bundle_digest(project_root: Path) -> str:
    """Hash the public replay files, excluding private labels and receipts."""

    files = (
        PUBLIC_MANIFEST,
        PUBLIC_PROTOCOL,
        PUBLIC_TOOLCHAIN,
        PUBLIC_INSTRUCTIONS,
    )
    entries: dict[str, str] = {}
    for relative_path in files:
        path = project_root / relative_path
        if not path.is_file():
            raise ValueError(f"public replay bundle file is missing: {relative_path}")
        entries[relative_path] = _file_digest(path)
    return _digest(entries)


def _write_frozen_public_toolchain(project_root: Path) -> Path:
    """Create the producer toolchain receipt once for a frozen public bundle.

    The public bundle is an input to a separately operated replay.  Rewriting
    its platform-specific capability receipt every time the owner regenerates
    manuscript artifacts would silently invalidate an otherwise identical
    replay (for example after a host OS patch).  Keep the bundle input stable
    once created; the current producer environment remains recorded in the
    owner-side receipt envelope.  An operator who intentionally changes the
    public bundle must remove this file and obtain a new replay receipt.
    """

    path = project_root / PUBLIC_TOOLCHAIN
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "frozen public toolchain receipt is unreadable; repair or remove "
                "the public bundle explicitly before regeneration"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("frozen public toolchain receipt must be an object")
        required = {
            "schema_version",
            "python",
            "platform",
            "architecture",
            "packages",
            "network_required",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"frozen public toolchain receipt is incomplete: missing={missing}"
            )
        if payload.get("network_required") is not False:
            raise ValueError(
                "frozen public toolchain receipt must declare network_required=false"
            )
        return path
    write_json_atomic(
        path,
        {
            "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "packages": toolchain_versions(("numpy",)),
            "network_required": False,
        },
    )
    return path


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
        "public_bundle_files": [
            PUBLIC_MANIFEST,
            PUBLIC_PROTOCOL,
            PUBLIC_TOOLCHAIN,
            PUBLIC_INSTRUCTIONS,
        ],
        "public_bundle_sha256": _public_bundle_digest(project_root),
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
    public_protocol_path = project_root / PUBLIC_PROTOCOL
    public_toolchain_path = project_root / PUBLIC_TOOLCHAIN
    public_instructions_path = project_root / PUBLIC_INSTRUCTIONS
    if not all(
        path.exists()
        for path in (
            manifest_path,
            public_path,
            gold_path,
            public_protocol_path,
            public_toolchain_path,
            public_instructions_path,
        )
    ):
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
    if manifest.get("public_bundle_files") != [
        PUBLIC_MANIFEST,
        PUBLIC_PROTOCOL,
        PUBLIC_TOOLCHAIN,
        PUBLIC_INSTRUCTIONS,
    ]:
        return False, "replication public bundle file list drifted"
    try:
        public_bundle_sha256 = _public_bundle_digest(project_root)
    except (OSError, ValueError) as exc:
        return False, f"replication public bundle hash failed: {exc}"
    if manifest.get("public_bundle_sha256") != public_bundle_sha256:
        return False, "replication public bundle hash mismatch"
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


def validate_public_replay_bundle(
    project_root: Path = PROJECT_ROOT,
) -> tuple[bool, str]:
    """Validate the public replay surface without requiring private gold labels.

    A separately controlled operator must be able to run the frozen replay with
    only the public bundle. This validator intentionally reads no source
    generator, private gold file, family manifest, or split metadata. The
    owner-side :func:`validate_replication_bundle` remains the stronger check
    used before protected scoring.
    """

    public_path = project_root / PUBLIC_MANIFEST
    protocol_path = project_root / PUBLIC_PROTOCOL
    toolchain_path = project_root / PUBLIC_TOOLCHAIN
    instructions_path = project_root / PUBLIC_INSTRUCTIONS
    required_paths = (
        public_path,
        protocol_path,
        toolchain_path,
        instructions_path,
    )
    if not all(path.is_file() for path in required_paths):
        missing = [
            str(path.relative_to(project_root))
            for path in required_paths
            if not path.is_file()
        ]
        return False, f"public replay bundle files are missing: {missing}"
    try:
        public = json.loads(public_path.read_text(encoding="utf-8"))
        public_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"public replay bundle unreadable: {exc}"
    if not isinstance(public, list) or not public:
        return False, "public replay manifest must be a non-empty row list"
    if not isinstance(public_protocol, dict) or not isinstance(toolchain, dict):
        return False, "public replay metadata must be objects"
    protocol = scientific_protocol_payload()
    if public_protocol.get("schema_version") != REPLAY_RECEIPT_SCHEMA_VERSION:
        return False, "public replay protocol schema mismatch"
    if public_protocol.get("protocol_id") != protocol.get(
        "protocol_id"
    ) or public_protocol.get("protocol_sha256") != protocol.get("protocol_sha256"):
        return False, "public replay protocol does not match the current protocol"
    if public_protocol.get("public_manifest") != PUBLIC_MANIFEST:
        return False, "public replay manifest path drifted"
    if public_protocol.get("public_manifest_sha256") != _digest(public):
        return False, "public replay manifest hash mismatch"
    if public_protocol.get("network_required") is not False:
        return False, "public replay protocol requires network access"
    if public_protocol.get("arbitrary_external_execution") is not False:
        return False, "public replay protocol permits arbitrary execution"
    forbidden = {
        "label",
        "family_id",
        "split",
        "variant_role",
        "quotient_class",
        "transformation_name",
        "source_family",
        "project_id",
    }
    expected_fields = {"record_id", "sequence", "features"}
    identifiers: list[str] = []
    for row in public:
        if not isinstance(row, dict) or set(row) != expected_fields:
            return False, "public replay row is not label blind"
        if forbidden & set(row):
            return False, "public replay row contains a forbidden field"
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            return False, "public replay record identifier is invalid"
        if not isinstance(row.get("sequence"), list) or not isinstance(
            row.get("features"), list
        ):
            return False, "public replay row fields are invalid"
        identifiers.append(record_id)
    if len(identifiers) != len(set(identifiers)):
        return False, "public replay record identifiers are duplicated"
    if toolchain.get("network_required") is not False:
        return False, "public replay toolchain receipt requires network access"
    if instructions_path.stat().st_size == 0:
        return False, "public replay instructions are empty"
    return (
        True,
        f"public_rows={len(public)} label_blind=true private_labels_absent_required",
    )


def write_replication_bundle(project_root: Path = PROJECT_ROOT) -> Path:
    """Write public inputs, separated gold labels, and the manifest receipt."""

    public, gold = _public_rows()
    public_dir = project_root / "output" / "replication" / "public"
    private_dir = project_root / "output" / "replication" / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(project_root / PUBLIC_MANIFEST, public)
    write_json_atomic(project_root / GOLD_LABELS, gold)
    protocol = scientific_protocol_payload()
    write_json_atomic(
        project_root / PUBLIC_PROTOCOL,
        {
            "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol["protocol_sha256"],
            "public_manifest": PUBLIC_MANIFEST,
            "public_manifest_sha256": _digest(public),
            "public_fields": ["record_id", "sequence", "features"],
            "forbidden_fields": [
                "label",
                "family_id",
                "split",
                "variant_role",
                "quotient_class",
            ],
            "network_required": False,
            "arbitrary_external_execution": False,
        },
    )
    _write_frozen_public_toolchain(project_root)
    (project_root / PUBLIC_INSTRUCTIONS).write_text(
        """# Offline replay bundle

This directory is the public, label-blind input surface. Run the reference
model with the repository CLI in offline mode, then submit predictions and a
separately operated replay receipt for protected scoring. A local rerun is
not independent replication.

```text
uv run python scripts/run_scientific_program.py verify-public
uv run python scripts/run_scientific_program.py replay --offline
uv run python scripts/run_scientific_program.py verify-separation
```

The scorer must receive gold labels through a protected path; gold labels,
family identifiers, split identifiers, and transformation roles are
intentionally absent from this public directory.
""",
        encoding="utf-8",
    )
    for stale_path in (STALE_PUBLIC_MANIFEST_PATH, STALE_GOLD_LABELS_PATH):
        (project_root / stale_path).unlink(missing_ok=True)
    path = project_root / REPLICATION_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        replication_bundle_payload(project_root),
        project_root=project_root,
        producer_path=Path(__file__),
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
    write_json_atomic(
        predictions_path,
        {
            "model": model,
            "label_blind": True,
            "predictions": predictions,
        },
    )
    return predictions_path


def _expected_public_manifest_digest(project_root: Path) -> str:
    """Read the sealed public-manifest digest from owner or public metadata."""

    owner_manifest_path = project_root / REPLICATION_ARTIFACT
    if owner_manifest_path.is_file():
        try:
            owner_manifest = json.loads(owner_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"replication manifest is unreadable: {exc}") from exc
        digest = owner_manifest.get("public_manifest_sha256")
        if isinstance(digest, str) and digest:
            return digest
    protocol_path = project_root / PUBLIC_PROTOCOL
    try:
        public_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"public replay protocol is unreadable: {exc}") from exc
    digest = public_protocol.get("public_manifest_sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError("public replay manifest digest is missing")
    return digest


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


def replay_receipt_payload(
    public_path: Path,
    predictions_path: Path,
    *,
    operator_id: str,
    machine_id: str,
    replay_bundle_sha256: str,
    model: str,
    seed: int,
    command: list[str],
    network_disabled: bool,
    numerical_tolerance: float = 1e-12,
    project_root: Path = PROJECT_ROOT,
    started_at: str | None = None,
    completed_at: str | None = None,
    source_revision: str | None = None,
    source_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a submission receipt without reading protected gold labels.

    The resulting receipt is deliberately *pending* and cannot assert
    independent replay.  A separately operated party must submit it to the
    ingestion boundary, where the protected scorer records the observed
    result.
    """

    if not operator_id.strip() or not machine_id.strip():
        raise ValueError("operator_id and machine_id must be non-empty")
    if not command or any(not isinstance(token, str) or not token for token in command):
        raise ValueError("replay command must be a non-empty argument vector")
    if not network_disabled:
        raise ValueError("independent replay receipts require network_disabled=true")
    if (
        not isinstance(replay_bundle_sha256, str)
        or len(replay_bundle_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in replay_bundle_sha256
        )
    ):
        raise ValueError("replay_bundle_sha256 must be a lowercase SHA-256 digest")
    if not np.isfinite(float(numerical_tolerance)) or numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be a finite non-negative number")
    if source_revision is not None and not _GIT_REVISION_RE.fullmatch(source_revision):
        raise ValueError("source_revision must be a 40-character lowercase commit hash")
    if source_archive_sha256 is not None and not _SHA256_RE.fullmatch(
        source_archive_sha256
    ):
        raise ValueError("source_archive_sha256 must be a lowercase SHA-256 digest")
    if (source_revision is None) != (source_archive_sha256 is None):
        raise ValueError(
            "source_revision and source_archive_sha256 must be supplied together"
        )
    started_timestamp = _replay_timestamp(started_at)
    completed_timestamp = _replay_timestamp(completed_at)
    if not _timestamps_ordered(started_timestamp, completed_timestamp):
        raise ValueError("replay completed_at must not precede started_at")
    try:
        public = json.loads(public_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"public replay input is unreadable: {exc}") from exc
    if not isinstance(public, list) or not public:
        raise ValueError("public replay input must be a non-empty row list")
    public_ids: list[str] = []
    for row in public:
        if not isinstance(row, dict) or set(row) != {
            "record_id",
            "sequence",
            "features",
        }:
            raise ValueError("public replay input is not label blind")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("public replay record identifiers must be non-empty")
        public_ids.append(record_id)
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("public replay identifiers are duplicated")
    predictions, detected_model = _prediction_rows(predictions_path)
    prediction_ids = [str(row["record_id"]) for row in predictions]
    if prediction_ids != public_ids:
        raise ValueError("prediction identifiers must exactly follow public ordering")
    if model != detected_model and detected_model != "unspecified":
        raise ValueError("receipt model does not match prediction model")
    protocol = scientific_protocol_payload()
    public_protocol_path = project_root / PUBLIC_PROTOCOL
    if not public_protocol_path.is_file():
        raise ValueError("public protocol manifest is missing")
    public_protocol = json.loads(public_protocol_path.read_text(encoding="utf-8"))
    if (
        not isinstance(public_protocol, dict)
        or public_protocol.get("protocol_id") != protocol.get("protocol_id")
        or public_protocol.get("protocol_sha256") != protocol.get("protocol_sha256")
    ):
        raise ValueError("public replay protocol does not match the current protocol")
    receipt = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            "public_manifest": _digest(public),
            "public_bundle": replay_bundle_sha256,
            "predictions": _file_digest(predictions_path),
            "src/replication.py": sha256_path(Path(__file__)),
        },
        toolchain_versions=toolchain_versions(("numpy",)),
        status="implemented_local",
        limitations=(
            "This is a separately operated replay submission receipt; protected scoring and repository ingestion are still required.",
            "Independent replay does not establish an independent reimplementation.",
        ),
        unsupported_cases=(
            "The receipt does not authorize network access or arbitrary external binary execution.",
        ),
        underpowered_cases=(
            "Independent replay is not promoted by local receipt generation.",
        ),
    )
    payload = {
        **receipt,
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "separate_operator_replay_submission",
        "operator_id": operator_id,
        "machine_id": machine_id,
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "public_manifest": PUBLIC_MANIFEST,
        "public_manifest_sha256": _digest(public),
        "public_bundle_sha256": replay_bundle_sha256,
        "model": model,
        "seed": int(seed),
        "command": list(command),
        "network_disabled": True,
        "prediction_file_sha256": _file_digest(predictions_path),
        "prediction_row_count": len(predictions),
        "prediction_identifier_sha256": _prediction_identifier_digest(predictions),
        "numerical_tolerance": float(numerical_tolerance),
        "started_at": started_timestamp,
        "completed_at": completed_timestamp,
        "outcome": "submitted_for_protected_scoring",
        "endpoint_status": "submitted_pending_verification",
        "independent_replay": "not_yet_observed",
        "attestation": {
            "operator_asserted": True,
            "separate_controlled_run": False,
            "external_operator_confirmation": "",
            "network_disabled": True,
            "gold_labels_accessed": False,
            "arbitrary_external_execution": False,
            "independent_reimplementation": False,
        },
    }
    if source_revision is not None and source_archive_sha256 is not None:
        payload["source_revision"] = source_revision
        payload["source_archive_sha256"] = source_archive_sha256
        payload["input_hashes"] = {
            **dict(payload["input_hashes"]),
            "runtime_source_archive": source_archive_sha256,
        }
    return payload


def validate_replay_receipt(
    payload: object,
    project_root: Path = PROJECT_ROOT,
    *,
    predictions_path: Path | None = None,
) -> tuple[bool, str]:
    """Validate a pending external replay receipt against the public bundle."""

    if not isinstance(payload, dict):
        return False, "replay receipt must be an object"
    protocol = scientific_protocol_payload()
    valid, detail = validate_receipt_envelope(payload, expected_protocol=protocol)
    if not valid:
        return False, f"replay receipt envelope: {detail}"
    required = {
        "receipt_kind",
        "operator_id",
        "machine_id",
        "public_manifest_sha256",
        "public_bundle_sha256",
        "model",
        "seed",
        "command",
        "network_disabled",
        "prediction_file_sha256",
        "prediction_row_count",
        "prediction_identifier_sha256",
        "numerical_tolerance",
        "started_at",
        "completed_at",
        "outcome",
        "endpoint_status",
        "independent_replay",
        "attestation",
    }
    missing = sorted(required - set(payload))
    if missing:
        return False, f"replay receipt fields missing: {missing}"
    if payload.get("receipt_kind") != "separate_operator_replay_submission":
        return False, "replay receipt is not a separate-operator submission"
    if payload.get("status") != "implemented_local":
        return False, "replay submission status must remain implemented_local"
    for key in ("operator_id", "machine_id", "model", "started_at", "completed_at"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            return False, f"replay receipt {key} is empty"
    for key in ("started_at", "completed_at"):
        timestamp = str(payload[key]).strip()
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return False, f"replay receipt {key} is not ISO-8601"
        if parsed.tzinfo is None:
            return False, f"replay receipt {key} has no timezone offset"
    if not _timestamps_ordered(
        str(payload["started_at"]), str(payload["completed_at"])
    ):
        return False, "replay receipt completed_at precedes started_at"
    if payload.get("network_disabled") is not True:
        return False, "replay receipt does not confirm network-disabled execution"
    if payload.get("endpoint_status") != "submitted_pending_verification":
        return False, "replay receipt endpoint status is already promoted"
    if payload.get("independent_replay") != "not_yet_observed":
        return False, "replay submission cannot claim independent replay"
    if payload.get("outcome") != "submitted_for_protected_scoring":
        return False, "replay receipt outcome is invalid"
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        return False, "replay receipt attestation is missing"
    if any(
        attestation.get(key) is not expected
        for key, expected in (
            ("operator_asserted", True),
            ("network_disabled", True),
            ("gold_labels_accessed", False),
            ("arbitrary_external_execution", False),
            ("independent_reimplementation", False),
        )
    ):
        return False, "replay receipt attestation safety flags drifted"
    if attestation.get("separate_controlled_run") not in {True, False}:
        return False, "replay receipt separate-control flag is invalid"
    if not isinstance(attestation.get("external_operator_confirmation"), str):
        return False, "replay receipt external confirmation is invalid"
    if (
        not isinstance(payload.get("command"), list)
        or not payload["command"]
        or any(not isinstance(token, str) or not token for token in payload["command"])
    ):
        return False, "replay receipt command is invalid"
    if not isinstance(payload.get("seed"), int) or isinstance(
        payload.get("seed"), bool
    ):
        return False, "replay receipt seed is invalid"
    if (
        not isinstance(payload.get("prediction_row_count"), int)
        or payload["prediction_row_count"] < 0
    ):
        return False, "replay receipt prediction row count is invalid"
    tolerance_value = payload.get("numerical_tolerance")
    if isinstance(tolerance_value, bool) or not isinstance(
        tolerance_value, (int, float, str)
    ):
        return False, "replay receipt numerical tolerance is invalid"
    try:
        tolerance = float(tolerance_value)
    except (TypeError, ValueError):
        return False, "replay receipt numerical tolerance is invalid"
    if not np.isfinite(tolerance) or tolerance < 0:
        return False, "replay receipt numerical tolerance is invalid"
    for key in (
        "public_manifest_sha256",
        "public_bundle_sha256",
        "prediction_file_sha256",
        "prediction_identifier_sha256",
    ):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False, f"replay receipt {key} is not a lowercase SHA-256 digest"
    has_source_revision = "source_revision" in payload
    has_source_archive = "source_archive_sha256" in payload
    if has_source_revision != has_source_archive:
        return False, "replay source revision and archive hash must appear together"
    if has_source_revision and (
        not isinstance(payload.get("source_revision"), str)
        or not _GIT_REVISION_RE.fullmatch(str(payload["source_revision"]))
    ):
        return False, "replay receipt source_revision is invalid"
    if has_source_archive and (
        not isinstance(payload.get("source_archive_sha256"), str)
        or not _SHA256_RE.fullmatch(str(payload["source_archive_sha256"]))
    ):
        return False, "replay receipt source_archive_sha256 is invalid"
    if has_source_archive:
        input_hashes = payload.get("input_hashes")
        if not isinstance(input_hashes, dict) or input_hashes.get(
            "runtime_source_archive"
        ) != payload.get("source_archive_sha256"):
            return (
                False,
                "replay receipt source archive hash is not bound in input_hashes",
            )
    public_path = project_root / PUBLIC_MANIFEST
    try:
        public = json.loads(public_path.read_text(encoding="utf-8"))
        bundle_hash = _public_bundle_digest(project_root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, f"replay public bundle is unreadable: {exc}"
    if not isinstance(public, list):
        return False, "replay public manifest is not a row list"
    if payload.get("public_manifest_sha256") != _digest(public):
        return False, "replay receipt public manifest hash mismatch"
    try:
        expected_public_manifest = _expected_public_manifest_digest(project_root)
    except ValueError as exc:
        return False, str(exc)
    if payload.get("public_manifest_sha256") != expected_public_manifest:
        return False, "replay receipt does not match sealed public manifest"
    if payload.get("public_bundle_sha256") != bundle_hash:
        return False, "replay receipt public bundle hash mismatch"
    if predictions_path is not None:
        try:
            predictions, model = _prediction_rows(predictions_path)
        except ValueError as exc:
            return False, str(exc)
        public_ids = [
            str(row.get("record_id")) for row in public if isinstance(row, dict)
        ]
        prediction_ids = [str(row["record_id"]) for row in predictions]
        if prediction_ids != public_ids:
            return False, "replay predictions do not exactly match public identifiers"
        if payload.get("prediction_file_sha256") != _file_digest(predictions_path):
            return False, "replay receipt prediction file hash mismatch"
        if payload.get("prediction_row_count") != len(predictions):
            return False, "replay receipt prediction row count mismatch"
        if payload.get("prediction_identifier_sha256") != _prediction_identifier_digest(
            predictions
        ):
            return False, "replay receipt prediction identifier digest mismatch"
        if model != "unspecified" and payload.get("model") != model:
            return False, "replay receipt model mismatch"
    return True, "ok"


def write_replay_receipt(
    public_path: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    operator_id: str,
    machine_id: str,
    replay_bundle_sha256: str,
    model: str,
    seed: int,
    command: list[str],
    project_root: Path = PROJECT_ROOT,
    started_at: str | None = None,
    completed_at: str | None = None,
    source_revision: str | None = None,
    source_archive_sha256: str | None = None,
) -> Path:
    """Write a pending replay receipt for an external operator to submit."""

    payload = replay_receipt_payload(
        public_path,
        predictions_path,
        operator_id=operator_id,
        machine_id=machine_id,
        replay_bundle_sha256=replay_bundle_sha256,
        model=model,
        seed=seed,
        command=command,
        network_disabled=True,
        project_root=project_root,
        started_at=started_at,
        completed_at=completed_at,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
    )
    valid, detail = validate_replay_receipt(
        payload, project_root=project_root, predictions_path=predictions_path
    )
    if not valid:
        raise ValueError(detail)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    return output_path


def ingest_independent_replay(
    receipt_path: Path,
    predictions_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Ingest a separately operated receipt and score inside the gold boundary."""

    try:
        external_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"external replay receipt is unreadable: {exc}") from exc
    valid, detail = validate_replay_receipt(
        external_receipt, project_root=project_root, predictions_path=predictions_path
    )
    if not valid:
        raise ValueError(detail)
    attestation = external_receipt.get("attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("separate_controlled_run") is not True
    ):
        raise ValueError(
            "independent replay promotion requires an explicitly attested separately controlled run"
        )
    confirmation = attestation.get("external_operator_confirmation")
    if not isinstance(confirmation, str) or not confirmation.strip():
        raise ValueError(
            "independent replay promotion requires external operator confirmation"
        )
    protected_score = score_replication_predictions(predictions_path, project_root)
    protocol = scientific_protocol_payload()
    output_path = project_root / INDEPENDENT_REPLAY_ARTIFACT
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            "external_replay_receipt": _file_digest(receipt_path),
            "predictions": _file_digest(predictions_path),
            "public_bundle": str(external_receipt["public_bundle_sha256"]),
            "src/replication.py": sha256_path(Path(__file__)),
        },
        toolchain_versions=toolchain_versions(("numpy",)),
        status="validated",
        limitations=(
            "This receipt establishes an observed separately operated replay, not independent reimplementation or new scientific data collection.",
            "The protected scorer uses the repository's sealed gold labels only at this ingestion boundary.",
        ),
        unsupported_cases=(
            "The replay does not validate arbitrary external binaries.",
        ),
    )
    payload = {
        **envelope,
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "artifact": INDEPENDENT_REPLAY_ARTIFACT,
        "endpoint_status": "independent_replay_observed",
        "independent_replay": "observed_external_replay",
        "external_receipt": external_receipt,
        "protected_score": protected_score,
        "gold_labels_exposed_to_operator": False,
        "independent_reimplementation": False,
    }
    write_json_artifact(
        output_path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        status="validated",
        limitations=tuple(str(item) for item in envelope["limitations"]),
        unsupported_cases=tuple(str(item) for item in envelope["unsupported_cases"]),
    )
    return output_path


def validate_independent_replay_artifact(
    payload: object, project_root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    """Validate the protected result produced by replay ingestion."""

    if not isinstance(payload, dict):
        return False, "independent replay artifact must be an object"
    protocol = scientific_protocol_payload()
    valid, detail = validate_receipt_envelope(payload, expected_protocol=protocol)
    if not valid:
        return False, f"independent replay envelope: {detail}"
    if payload.get("schema_version") != REPLAY_RECEIPT_SCHEMA_VERSION:
        return False, "independent replay schema mismatch"
    if payload.get("endpoint_status") != "independent_replay_observed":
        return False, "independent replay endpoint status is missing"
    if payload.get("independent_replay") != "observed_external_replay":
        return False, "independent replay status is missing"
    external_receipt = payload.get("external_receipt")
    if not isinstance(external_receipt, dict):
        return False, "independent replay external receipt is missing"
    receipt_valid, receipt_detail = validate_replay_receipt(
        external_receipt, project_root=project_root
    )
    if not receipt_valid:
        return False, f"independent replay external receipt: {receipt_detail}"
    if payload.get("gold_labels_exposed_to_operator") is not False:
        return False, "independent replay gold-label exposure flag is unsafe"
    if payload.get("independent_reimplementation") is not False:
        return False, "independent replay cannot claim independent reimplementation"
    return True, "ok"
