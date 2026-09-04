"""Protocol-v2 closure for generated JSON artifacts.

The package contains domain-specific receipts (traces, model fits, figures,
manuscript registries, and post-render audits).  They deliberately retain
their local ``schema_version`` because downstream validators use those
versions to interpret the payload.  This module adds a second, orthogonal
receipt layer: ``receipt_schema_version`` and the shared protocol/provenance
fields required by the scientific contract.

Keeping the domain schema and the receipt schema separate is important.  A
trace schema change must not be confused with a provenance-contract change,
and adding lineage fields must not make a valid domain payload unreadable by
the Capstone/Unicorn or manuscript consumers.

The writer is intentionally conservative.  It preserves domain keys, never
turns unavailable evidence into a numeric value, records a non-promotion
limitation for migrated legacy payloads, and writes atomically.  The final
surface validator is fail-closed: a JSON artifact that bypasses this boundary
is reported rather than silently repaired at validation time.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import tempfile
from typing import Any

from .provenance import generator_revision, input_hashes, toolchain_versions
from .receipt import (
    EVIDENCE_STATUSES,
    RECEIPT_SCHEMA_VERSION,
    validate_receipt_envelope,
)
from .scientific_protocol import (
    scientific_protocol_payload,
    validate_scientific_protocol_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_SCHEMA_KEY = "receipt_schema_version"
ARTIFACT_PAYLOAD_SCHEMA_VERSION = "artifact-payload-v1"

# These are project-owned data receipts.  Template-owned reports and JSONL
# streams are intentionally outside this registry because their producer is
# the sibling template or a native backend, not this Python package.
DATA_ARTIFACT_NAMES = frozenset(
    {
        "action_sensing.json",
        "active_information_frontier.json",
        "active_selection.json",
        "counterexample_search.json",
        "assembly_listing.json",
        "build_provenance.json",
        "cfg_evidence.json",
        "claim_register.json",
        "compiler_transfer.json",
        "corpus_coverage.json",
        "cross_architecture_fixtures.json",
        "disassembly_trace.json",
        "docx_render_receipt.json",
        "emulation_trace.json",
        "evaluation_report.json",
        "experiment_summary.json",
        "external_source_validation.json",
        "external_acquisition.json",
        "sensing_stress.json",
        "external_static_validation.json",
        "external_capabilities.json",
        "external_label_contract.json",
        "independent_replay_receipt.json",
        "factorized_observation_model.json",
        "figure_registry.json",
        "formal_trace_correspondence.json",
        "formal_verification.json",
        "generative_adequacy.json",
        "held_out_model.json",
        "inference_result.json",
        "intervention_power.json",
        "manuscript_variable_registry.json",
        "manuscript_variables.json",
        "memory_evidence.json",
        "parameter_learning.json",
        "pdf_accessibility_report.json",
        "phase_segments.json",
        "quotient_classifier.json",
        "quotient_invariance_protocol.json",
        "replication_manifest.json",
        "representation_invariance_benchmark.json",
        "reproducibility_manifest.json",
        "research_protocol.json",
        "scholarship_sources.json",
        "scientific_program.json",
        "sensor_cost_receipt.json",
        "symbolic_execution.json",
        "temporal_inference.json",
        "theorem_or_countertheorem.json",
        "trace_bundle.json",
        "typed_observations.json",
        "validation_report.json",
    }
)

_PRODUCER_MODULES = {
    "action_sensing.json": "src/advanced_inference.py",
    "active_information_frontier.json": "src/advanced_inference.py",
    "active_selection.json": "src/advanced_inference.py",
    "assembly_listing.json": "src/experiments.py",
    "build_provenance.json": "src/build_provenance.py",
    "cfg_evidence.json": "src/advanced_inference.py",
    "claim_register.json": "src/claim_register.py",
    "compiler_transfer.json": "src/compiler_transfer.py",
    "corpus_coverage.json": "src/advanced_inference.py",
    "cross_architecture_fixtures.json": "src/advanced_inference.py",
    "disassembly_trace.json": "src/experiments.py",
    "docx_render_receipt.json": "src/docx_validation.py",
    "emulation_trace.json": "src/experiments.py",
    "evaluation_report.json": "src/advanced_inference.py",
    "experiment_summary.json": "src/experiments.py",
    "external_source_validation.json": "src/external_sources.py",
    "external_acquisition.json": "src/external_sources.py",
    "external_static_validation.json": "src/external_sources.py",
    "external_capabilities.json": "src/external_sources.py",
    "external_label_contract.json": "src/external_labeling.py",
    "independent_replay_receipt.json": "src/replication.py",
    "factorized_observation_model.json": "src/advanced_inference.py",
    "figure_registry.json": "src/figure_contract.py",
    "formal_trace_correspondence.json": "src/formal_correspondence.py",
    "formal_verification.json": "src/formal_verification.py",
    "counterexample_search.json": "src/counterexample_search.py",
    "generative_adequacy.json": "src/advanced_inference.py",
    "held_out_model.json": "src/advanced_inference.py",
    "inference_result.json": "src/experiments.py",
    "intervention_power.json": "src/intervention_analysis.py",
    "manuscript_variable_registry.json": "src/manuscript_registry.py",
    "manuscript_variables.json": "src/manuscript_variables.py",
    "memory_evidence.json": "src/advanced_inference.py",
    "parameter_learning.json": "src/advanced_inference.py",
    "pdf_accessibility_report.json": "src/pdf_accessibility.py",
    "phase_segments.json": "src/advanced_inference.py",
    "quotient_classifier.json": "src/quotient_classifier.py",
    "quotient_invariance_protocol.json": "src/quotient_invariance.py",
    "replication_manifest.json": "src/replication.py",
    "sensing_stress.json": "src/sensing_stress.py",
    "representation_invariance_benchmark.json": "src/representation_invariance.py",
    "reproducibility_manifest.json": "src/reproducibility.py",
    "research_protocol.json": "src/scientific_protocol.py",
    "scholarship_sources.json": "src/scholarship.py",
    "scientific_program.json": "src/scientific_program.py",
    "sensor_cost_receipt.json": "src/sensor_costs.py",
    "symbolic_execution.json": "src/experiments.py",
    "temporal_inference.json": "src/advanced_inference.py",
    "theorem_or_countertheorem.json": "src/theorem_phase.py",
    "trace_bundle.json": "src/experiments.py",
    "typed_observations.json": "src/advanced_inference.py",
    "validation_report.json": "src/validation.py",
}

COMMON_RECEIPT_FIELDS = frozenset(
    {
        RECEIPT_SCHEMA_KEY,
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
)

_DEFAULT_LIMITATION = (
    "This is a source-bound generated artifact. Its domain-specific validator "
    "defines the evidence boundary; the receipt alone is not a scientific claim."
)
_DEFAULT_UNSUPPORTED = (
    "Domain-specific unsupported cases are declared by the producing module or "
    "its validator; no unsupported case was silently imputed here."
)
_DEFAULT_UNDERPOWERED = (
    "This artifact is not, by itself, a promotion of external validity, broad "
    "semantic identifiability, or a general theorem."
)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _protocol_for(project_root: Path) -> dict[str, Any]:
    """Load the current protocol when present, otherwise build it."""

    protocol_path = project_root / "output" / "data" / "research_protocol.json"
    if protocol_path.is_file():
        try:
            candidate = json.loads(protocol_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict) and candidate.get("protocol_sha256"):
                valid, _detail = validate_scientific_protocol_payload(candidate)
                if valid:
                    return candidate
        except (OSError, json.JSONDecodeError):
            pass
    return scientific_protocol_payload()


def _string_list(value: object, fallback: tuple[str, ...]) -> list[str]:
    if isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return [str(item) for item in value]
    if isinstance(value, tuple) and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return [str(item) for item in value]
    return list(fallback)


def _status(payload: dict[str, Any], default: str) -> str:
    value = payload.get("status")
    if isinstance(value, str) and value in EVIDENCE_STATUSES:
        return value
    if payload.get("valid") is False:
        return "failed"
    return default


def build_artifact_envelope(
    payload: object,
    *,
    artifact_path: Path,
    project_root: Path = PROJECT_ROOT,
    producer_path: Path | None = None,
    status: str = "implemented_local",
    limitations: Iterable[str] = (),
    unsupported_cases: Iterable[str] = (),
    underpowered_cases: Iterable[str] = (),
    input_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Attach the shared protocol-v2 receipt fields without changing domain data.

    ``schema_version`` remains the domain payload version.  The common receipt
    version is explicit in ``receipt_schema_version`` so older domain validators
    continue to work while the artifact surface gains one enforceable contract.
    """

    if isinstance(payload, dict):
        result: dict[str, Any] = dict(payload)
    else:
        result = {
            "schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
            "payload": payload,
        }
    result.setdefault("schema_version", ARTIFACT_PAYLOAD_SCHEMA_VERSION)
    protocol = _protocol_for(project_root)
    producer = producer_path if producer_path is not None else Path(__file__)
    producer_hash = (
        generator_revision(producer)
        if producer.is_file()
        else generator_revision(Path(__file__))
    )
    artifact_relative = _relative(artifact_path, project_root)
    hashes = dict(input_hashes(input_paths, project_root))
    hashes.setdefault(
        "research_protocol.json", str(protocol.get("protocol_sha256", ""))
    )
    producer_key = f"producer/{_relative(producer, project_root)}"
    hashes.setdefault(producer_key, producer_hash)
    existing_hashes = result.get("input_hashes")
    if isinstance(existing_hashes, dict):
        hashes = {
            **{str(key): str(value) for key, value in existing_hashes.items()},
            **hashes,
        }
    existing_versions = result.get("toolchain_versions")
    versions = toolchain_versions(("capstone", "numpy", "unicorn"))
    if isinstance(existing_versions, dict):
        versions = {
            **versions,
            **{str(key): str(value) for key, value in existing_versions.items()},
        }
    result[RECEIPT_SCHEMA_KEY] = RECEIPT_SCHEMA_VERSION
    result["protocol_id"] = str(protocol.get("protocol_id", ""))
    result["protocol_sha256"] = str(protocol.get("protocol_sha256", ""))
    result["generator_revision"] = producer_hash
    result["input_hashes"] = dict(sorted(hashes.items()))
    result["toolchain_versions"] = dict(sorted(versions.items()))
    result["status"] = _status(result, status)
    result["limitations"] = _string_list(
        result.get("limitations"), tuple(limitations) or (_DEFAULT_LIMITATION,)
    )
    result["unsupported_cases"] = _string_list(
        result.get("unsupported_cases"),
        tuple(unsupported_cases) or (_DEFAULT_UNSUPPORTED,),
    )
    result["underpowered_cases"] = _string_list(
        result.get("underpowered_cases"),
        tuple(underpowered_cases) or (_DEFAULT_UNDERPOWERED,),
    )
    result.setdefault("artifact_id", artifact_relative)
    result.setdefault("producer", _relative(producer, project_root))
    return result


def validate_artifact_envelope(
    payload: object,
    *,
    expected_protocol: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate the common receipt layer on a domain payload."""

    if not isinstance(payload, dict):
        return False, "artifact receipt must be an object"
    missing = sorted(COMMON_RECEIPT_FIELDS - set(payload))
    if missing:
        return False, f"artifact receipt fields missing: {missing}"
    if payload.get(RECEIPT_SCHEMA_KEY) != RECEIPT_SCHEMA_VERSION:
        return False, "artifact receipt schema mismatch"
    envelope = dict(payload)
    # The domain ``schema_version`` is intentionally not passed to the shared
    # validator.  A dedicated receipt schema key avoids coupling trace/model/
    # document schema versions to the provenance contract.
    envelope["schema_version"] = RECEIPT_SCHEMA_VERSION
    valid, detail = validate_receipt_envelope(
        envelope, expected_protocol=expected_protocol
    )
    if not valid:
        return False, detail
    return True, "ok"


def artifact_domain_view(payload: object) -> object:
    """Return domain content for freshness comparisons, excluding receipt keys."""

    if not isinstance(payload, dict):
        return payload
    view = {
        key: value
        for key, value in payload.items()
        if key not in COMMON_RECEIPT_FIELDS and key not in {"artifact_id", "producer"}
    }
    if view.get("schema_version") == ARTIFACT_PAYLOAD_SCHEMA_VERSION:
        view.pop("schema_version")
    return view


def write_json_artifact(
    path: Path,
    payload: object,
    *,
    project_root: Path = PROJECT_ROOT,
    producer_path: Path | None = None,
    status: str = "implemented_local",
    limitations: Iterable[str] = (),
    unsupported_cases: Iterable[str] = (),
    underpowered_cases: Iterable[str] = (),
    input_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Write an atomically sealed source-bound JSON artifact."""

    sealed = build_artifact_envelope(
        payload,
        artifact_path=path,
        project_root=project_root,
        producer_path=producer_path,
        status=status,
        limitations=limitations,
        unsupported_cases=unsupported_cases,
        underpowered_cases=underpowered_cases,
        input_paths=input_paths,
    )
    write_json_atomic(path, sealed)
    return sealed


def write_json_atomic(path: Path, payload: object) -> None:
    """Write arbitrary JSON atomically for interface fixtures.

    Public label-blind rows, private gold rows, and model predictions are
    intentionally not domain receipts: wrapping them would change their
    frozen interface shape or make the private label boundary ambiguous. They
    still use the same atomic write primitive, while manuscript/data receipts
    continue through :func:`write_json_artifact` and the common envelope.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def seal_existing_json_artifact(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    producer_path: Path | None = None,
) -> dict[str, Any]:
    """Migrate one existing JSON object through the same writer boundary."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=producer_path,
    )


def seal_data_artifact_surface(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Migrate any legacy project data receipts and return changed paths.

    This is an explicit producer-stage migration, not a validation shortcut:
    it is called before the validation report is built, and the resulting
    envelopes are then checked independently by ``validate_data_artifact_surface``.
    """

    data_dir = project_root / "output" / "data"
    changed: list[str] = []
    if not data_dir.is_dir():
        return changed
    for path in sorted(data_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        valid, _ = validate_artifact_envelope(payload)
        if valid:
            continue
        relative_producer = _PRODUCER_MODULES.get(path.name)
        producer = (
            project_root / relative_producer
            if relative_producer is not None
            and (project_root / relative_producer).is_file()
            else PROJECT_ROOT / relative_producer
            if relative_producer is not None
            and (PROJECT_ROOT / relative_producer).is_file()
            else Path(__file__)
        )
        seal_existing_json_artifact(
            path, project_root=project_root, producer_path=producer
        )
        changed.append(_relative(path, project_root))
    return changed


def validate_data_artifact_surface(
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, str]]:
    """Return one diagnostic per unsealed or malformed project data artifact."""

    data_dir = project_root / "output" / "data"
    diagnostics: list[dict[str, str]] = []
    if not data_dir.is_dir():
        return [{"path": "output/data", "detail": "data directory is missing"}]
    # A receipt can be structurally complete while still belonging to an
    # older protocol.  The surface validator is the fail-closed boundary for
    # manuscript evidence, so it must compare every artifact with the current
    # canonical protocol rather than accepting any valid historical envelope.
    expected_protocol = scientific_protocol_payload()
    for path in sorted(data_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            diagnostics.append(
                {"path": _relative(path, project_root), "detail": str(exc)}
            )
            continue
        valid, detail = validate_artifact_envelope(
            payload, expected_protocol=expected_protocol
        )
        if not valid:
            diagnostics.append(
                {"path": _relative(path, project_root), "detail": detail}
            )
    return diagnostics


__all__ = [
    "ARTIFACT_PAYLOAD_SCHEMA_VERSION",
    "COMMON_RECEIPT_FIELDS",
    "DATA_ARTIFACT_NAMES",
    "RECEIPT_SCHEMA_KEY",
    "artifact_domain_view",
    "build_artifact_envelope",
    "seal_existing_json_artifact",
    "seal_data_artifact_surface",
    "validate_artifact_envelope",
    "validate_data_artifact_surface",
    "write_json_atomic",
    "write_json_artifact",
]
