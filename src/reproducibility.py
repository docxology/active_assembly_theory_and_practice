"""Reproducibility manifest for generated analysis artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import build_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ArtifactManifestTarget:
    """Expected generated artifact tracked by the reproducibility manifest."""

    path: str
    artifact_type: str
    generated_by: str
    claim_ids: tuple[str, ...]
    validation_checks: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ArtifactManifestRow:
    """One manifest row with the current hash and file size."""

    path: str
    artifact_type: str
    sha256: str | None
    bytes: int
    required: bool
    generated_by: str
    claim_ids: tuple[str, ...]
    validation_checks: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe row."""

        data = asdict(self)
        data["claim_ids"] = list(self.claim_ids)
        data["validation_checks"] = list(self.validation_checks)
        return data


# Every corpus fixture emits its own temporal-belief heatmap; track all of them
# so a missing or replaced variant figure is caught by the manifest and the
# pixel-variance figure oracle (derived from the fixtures, not hard-coded).
_TEMPORAL_BELIEF_TARGETS: tuple[ArtifactManifestTarget, ...] = tuple(
    ArtifactManifestTarget(
        path=f"output/figures/temporal_beliefs_{fixture.name}.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_temporal_belief_figures",
        claim_ids=("CLM-010", "CLM-012"),
        validation_checks=(f"figure_temporal_beliefs_{fixture.name}",),
    )
    for fixture in build_corpus()
)


MANIFEST_TARGETS: tuple[ArtifactManifestTarget, ...] = (
    ArtifactManifestTarget(
        path="output/data/disassembly_trace.json",
        artifact_type="trace",
        generated_by="src.experiments.run_experiment",
        claim_ids=("CLM-001",),
        validation_checks=("trace_instruction_observation_alignment",),
    ),
    ArtifactManifestTarget(
        path="output/data/emulation_trace.json",
        artifact_type="trace",
        generated_by="src.experiments.run_experiment",
        claim_ids=("CLM-002",),
        validation_checks=("trace_dynamic_register_contract",),
    ),
    ArtifactManifestTarget(
        path="output/data/inference_result.json",
        artifact_type="inference",
        generated_by="src.experiments.run_experiment",
        claim_ids=("CLM-003", "CLM-005"),
        validation_checks=("trace_observation_semantics",),
    ),
    ArtifactManifestTarget(
        path="output/data/trace_bundle.json",
        artifact_type="trace",
        generated_by="src.experiments.run_experiment",
        claim_ids=("CLM-001", "CLM-002", "CLM-003", "CLM-006"),
        validation_checks=("trace_schema_version", "trace_binary_hash"),
    ),
    ArtifactManifestTarget(
        path="output/data/scholarship_sources.json",
        artifact_type="scholarship",
        generated_by="src.scholarship.write_scholarship_sources",
        claim_ids=("CLM-006", "CLM-008"),
        validation_checks=(
            "scholarship_source_quality",
            "scholarship_evidence_partition",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/claim_register.json",
        artifact_type="claim_register",
        generated_by="src.claim_register.write_claim_register",
        claim_ids=("CLM-009",),
        validation_checks=("claim_register_contract",),
    ),
    ArtifactManifestTarget(
        path="output/data/redteam_report.json",
        artifact_type="redteam",
        generated_by="src.redteam_report.write_redteam_report",
        claim_ids=("CLM-007",),
        validation_checks=("redteam_findings_remediated",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer/free_energy_trace.jsonl",
        artifact_type="rxinfer",
        generated_by="src.rxinfer_bridge.run_rxinfer_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_free_energy_records",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer/posteriors.json",
        artifact_type="rxinfer",
        generated_by="src.rxinfer_bridge.run_rxinfer_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_companion_artifacts",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer/diagnostics.json",
        artifact_type="rxinfer",
        generated_by="src.rxinfer_bridge.run_rxinfer_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_diagnostics_consistent",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer/model_config.json",
        artifact_type="rxinfer",
        generated_by="src.rxinfer_bridge.run_rxinfer_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_companion_artifacts",),
    ),
    ArtifactManifestTarget(
        path="output/figures/observation_pipeline.png",
        artifact_type="figure",
        generated_by="src.figures.plot_observation_pipeline",
        claim_ids=("CLM-003",),
        validation_checks=("figure_observation_pipeline",),
    ),
    ArtifactManifestTarget(
        path="output/figures/posterior_beliefs.png",
        artifact_type="figure",
        generated_by="src.figures.plot_posterior_beliefs",
        claim_ids=("CLM-003",),
        validation_checks=("figure_posterior_beliefs",),
    ),
    ArtifactManifestTarget(
        path="output/figures/package_provenance.png",
        artifact_type="figure",
        generated_by="src.figures.plot_package_provenance",
        claim_ids=("CLM-006",),
        validation_checks=("figure_package_provenance",),
    ),
    ArtifactManifestTarget(
        path="output/figures/trace_feature_matrix.png",
        artifact_type="figure",
        generated_by="src.figures.plot_trace_feature_matrix",
        claim_ids=("CLM-003",),
        validation_checks=("figure_trace_feature_matrix",),
    ),
    ArtifactManifestTarget(
        path="output/figures/rxinfer_free_energy.png",
        artifact_type="figure",
        generated_by="src.figures.plot_rxinfer_free_energy",
        claim_ids=("CLM-004",),
        validation_checks=("figure_rxinfer_free_energy",),
    ),
    ArtifactManifestTarget(
        path="output/figures/inference_diagnostics.png",
        artifact_type="figure",
        generated_by="src.figures.plot_inference_diagnostics",
        claim_ids=("CLM-005",),
        validation_checks=("figure_inference_diagnostics",),
    ),
    ArtifactManifestTarget(
        path="output/figures/scholarship_source_roles.png",
        artifact_type="figure",
        generated_by="src.figures.plot_scholarship_source_roles",
        claim_ids=("CLM-008",),
        validation_checks=("figure_scholarship_source_roles",),
    ),
    ArtifactManifestTarget(
        path="output/figures/scholarship_evidence_matrix.png",
        artifact_type="figure",
        generated_by="src.figures.plot_scholarship_evidence_matrix",
        claim_ids=("CLM-008",),
        validation_checks=("figure_scholarship_evidence_matrix",),
    ),
    ArtifactManifestTarget(
        path="output/figures/claim_evidence_map.png",
        artifact_type="figure",
        generated_by="src.figures.plot_claim_evidence_map",
        claim_ids=("CLM-009",),
        validation_checks=("figure_claim_evidence_map",),
    ),
    ArtifactManifestTarget(
        path="output/figures/redteam_risk_register.png",
        artifact_type="figure",
        generated_by="src.figures.plot_redteam_risk_register",
        claim_ids=("CLM-007",),
        validation_checks=("figure_redteam_risk_register",),
    ),
    ArtifactManifestTarget(
        path="output/figures/model_comparison.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_advanced_figures",
        claim_ids=("CLM-010", "CLM-011"),
        validation_checks=("figure_model_comparison",),
    ),
    ArtifactManifestTarget(
        path="output/figures/active_efe.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_advanced_figures",
        claim_ids=("CLM-013",),
        validation_checks=("figure_active_efe",),
    ),
    ArtifactManifestTarget(
        path="output/figures/calibration_curve.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-011",),
        validation_checks=("figure_calibration_curve",),
    ),
    ArtifactManifestTarget(
        path="output/figures/confusion_matrix.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-011",),
        validation_checks=("figure_confusion_matrix",),
    ),
    ArtifactManifestTarget(
        path="output/figures/ambiguity_ablation.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-014",),
        validation_checks=("figure_ambiguity_ablation",),
    ),
    ArtifactManifestTarget(
        path="output/figures/segmentation_ribbons.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-015",),
        validation_checks=("figure_segmentation_ribbons",),
    ),
    ArtifactManifestTarget(
        path="output/figures/corpus_stability.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-010",),
        validation_checks=("figure_corpus_stability",),
    ),
    ArtifactManifestTarget(
        path="output/figures/phase_recovery.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-010",),
        validation_checks=("figure_phase_recovery",),
    ),
    ArtifactManifestTarget(
        path="output/figures/decoder_calibration.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-016",),
        validation_checks=("figure_decoder_calibration",),
    ),
    ArtifactManifestTarget(
        path="output/figures/generative_model.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-002",),
        validation_checks=("figure_generative_model",),
    ),
    ArtifactManifestTarget(
        path="output/data/temporal_inference.json",
        artifact_type="inference",
        generated_by="src.advanced_inference.run_advanced_inference",
        claim_ids=("CLM-010",),
        validation_checks=("advanced_temporal_artifacts_present",),
    ),
    ArtifactManifestTarget(
        path="output/data/evaluation_report.json",
        artifact_type="evaluation",
        generated_by="src.advanced_inference.run_advanced_inference",
        claim_ids=("CLM-010", "CLM-011"),
        validation_checks=("advanced_temporal_beats_static",),
    ),
    ArtifactManifestTarget(
        path="output/data/phase_segments.json",
        artifact_type="inference",
        generated_by="src.advanced_inference.run_advanced_inference",
        claim_ids=("CLM-012",),
        validation_checks=("advanced_phase_segmentation_partition",),
    ),
    ArtifactManifestTarget(
        path="output/data/active_selection.json",
        artifact_type="inference",
        generated_by="src.advanced_inference.run_advanced_inference",
        claim_ids=("CLM-013",),
        validation_checks=("advanced_active_never_worse",),
    ),
) + _TEMPORAL_BELIEF_TARGETS


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(project_root: Path = PROJECT_ROOT) -> tuple[ArtifactManifestRow, ...]:
    """Return manifest rows for the current generated artifact surface."""

    rows: list[ArtifactManifestRow] = []
    for target in MANIFEST_TARGETS:
        path = project_root / target.path
        exists = path.exists()
        rows.append(
            ArtifactManifestRow(
                path=target.path,
                artifact_type=target.artifact_type,
                sha256=sha256_file(path) if exists else None,
                bytes=path.stat().st_size if exists else 0,
                required=target.required,
                generated_by=target.generated_by,
                claim_ids=target.claim_ids,
                validation_checks=target.validation_checks,
                status="present" if exists else "missing",
            )
        )
    return tuple(rows)


def reproducibility_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the complete reproducibility manifest payload."""

    rows = manifest_rows(project_root)
    missing_required = [
        row.path for row in rows if row.required and row.status != "present"
    ]
    artifact_type_counts: dict[str, int] = {}
    for row in rows:
        artifact_type_counts[row.artifact_type] = (
            artifact_type_counts.get(row.artifact_type, 0) + 1
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_count": len(rows),
        "missing_required_count": len(missing_required),
        "all_required_present": not missing_required,
        "artifact_type_counts": artifact_type_counts,
        "excluded_self_referential_artifacts": [
            "output/data/reproducibility_manifest.json",
            "output/data/reproducibility_manifest.md",
            "output/figures/artifact_hash_manifest.png",
            "output/data/validation_report.json",
            "output/figures/validation_summary.png",
            "output/data/manuscript_variables.json",
        ],
        "artifacts": [row.to_dict() for row in rows],
    }


def verify_reproducibility_manifest(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return verification errors for the current manifest, or an empty list."""

    manifest_path = project_root / "output" / "data" / "reproducibility_manifest.json"
    if not manifest_path.exists():
        return ["missing output/data/reproducibility_manifest.json"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    rows = payload.get("artifacts", [])
    if payload.get("artifact_count") != len(MANIFEST_TARGETS):
        errors.append(
            f"artifact_count={payload.get('artifact_count')} expected={len(MANIFEST_TARGETS)}"
        )
    observed_paths = {row.get("path") for row in rows}
    expected_paths = {target.path for target in MANIFEST_TARGETS}
    missing_rows = sorted(expected_paths - observed_paths)
    if missing_rows:
        errors.append(f"missing_rows={missing_rows}")
    for row in rows:
        relative_path = str(row.get("path"))
        path = project_root / relative_path
        if not path.exists():
            errors.append(f"missing_artifact={relative_path}")
            continue
        expected_hash = row.get("sha256")
        current_hash = sha256_file(path)
        if expected_hash != current_hash:
            errors.append(f"hash_mismatch={relative_path}")
        if int(row.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"size_mismatch={relative_path}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"bad_sha256={relative_path}")
        if path.stat().st_size <= 0:
            errors.append(f"empty_artifact={relative_path}")
    return errors


def write_reproducibility_manifest(output_dir: Path = OUTPUT_DIR) -> str:
    """Write manifest JSON/Markdown and its figure; return the JSON relative path."""

    from .figures import plot_artifact_hash_manifest

    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = reproducibility_payload(output_dir.parent)

    json_path = data_dir / "reproducibility_manifest.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    markdown_rows = [
        "# Reproducibility Manifest",
        "",
        "| Artifact | Type | Bytes | SHA-256 | Status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in payload["artifacts"]:
        digest = str(row["sha256"])[:12] if row["sha256"] else "missing"
        markdown_rows.append(
            f"| `{row['path']}` | {row['artifact_type']} | {row['bytes']} | `{digest}` | {row['status']} |"
        )
    (data_dir / "reproducibility_manifest.md").write_text(
        "\n".join(markdown_rows) + "\n",
        encoding="utf-8",
    )
    plot_artifact_hash_manifest(
        list(payload["artifacts"]),
        figure_dir / "artifact_hash_manifest.png",
    )
    return str(json_path.relative_to(output_dir.parent))
