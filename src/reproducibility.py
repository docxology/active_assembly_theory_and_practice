"""Reproducibility manifest for generated analysis artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifact_contract import write_json_artifact
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
        path="output/data/symbolic_execution.json",
        artifact_type="semantics",
        generated_by="src.symbolic_semantics.symbolic_execute",
        claim_ids=("CLM-001", "CLM-019"),
        validation_checks=(
            "symbolic_semantics_contract",
            "trace_symbolic_facts_contract",
            "symbolic_stepwise_trace_refinement",
        ),
    ),
    ArtifactManifestTarget(
        path="output/figures/trace_refinement.png",
        artifact_type="figure",
        generated_by="src.figures.plot_trace_refinement",
        claim_ids=("CLM-019",),
        validation_checks=("figure_trace_refinement",),
    ),
    ArtifactManifestTarget(
        path="output/data/held_out_model.json",
        artifact_type="evaluation",
        generated_by="src.model_fitting.leave_one_fixture_out_evaluation",
        claim_ids=("CLM-018",),
        validation_checks=("held_out_model_contract", "held_out_model_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/typed_observations.json",
        artifact_type="observation_schema",
        generated_by="src.advanced_inference.typed_observation_payload",
        claim_ids=("CLM-021",),
        validation_checks=(
            "typed_observation_schema_contract",
            "typed_observation_schema_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/factorized_observation_model.json",
        artifact_type="evaluation",
        generated_by="src.factorized_model.leave_one_fixture_out_factorized_evaluation",
        claim_ids=("CLM-021",),
        validation_checks=(
            "factorized_observation_contract",
            "factorized_observation_freshness",
            "factorized_observation_leakage_guard",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/corpus_coverage.json",
        artifact_type="corpus_design",
        generated_by="src.corpus_coverage.corpus_coverage_payload",
        claim_ids=("CLM-022",),
        validation_checks=("corpus_coverage_contract", "corpus_coverage_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/action_sensing.json",
        artifact_type="action_sensing",
        generated_by="src.action_sensing.action_conditioned_sensing_payload",
        claim_ids=("CLM-023",),
        validation_checks=(
            "action_sensing_contract",
            "action_sensing_freshness",
            "action_sensing_leakage_guard",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/parameter_learning.json",
        artifact_type="parameter_learning",
        generated_by="src.parameter_learning.parameter_learning_payload",
        claim_ids=("CLM-024",),
        validation_checks=(
            "parameter_learning_contract",
            "parameter_learning_freshness",
            "parameter_learning_leakage_guard",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/scientific_program.json",
        artifact_type="research_program",
        generated_by="src.scientific_program.write_scientific_program",
        claim_ids=("CLM-025",),
        validation_checks=(
            "scientific_program_contract",
            "scientific_program_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/research_protocol.json",
        artifact_type="research_protocol",
        generated_by="src.scientific_protocol.write_scientific_protocol",
        claim_ids=("CLM-025",),
        validation_checks=(
            "scientific_protocol_contract",
            "scientific_protocol_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/quotient_classifier.json",
        artifact_type="quotient_classifier",
        generated_by="src.quotient_classifier.write_quotient_classifier_report",
        claim_ids=("CLM-027",),
        validation_checks=(
            "quotient_classifier_contract",
            "quotient_classifier_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/quotient_invariance_protocol.json",
        artifact_type="research_protocol",
        generated_by="src.quotient_invariance.write_quotient_invariance_protocol",
        claim_ids=("CLM-026",),
        validation_checks=(
            "quotient_invariance_contract",
            "quotient_invariance_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/representation_invariance_benchmark.json",
        artifact_type="research_benchmark",
        generated_by=(
            "src.representation_invariance.write_representation_invariance_benchmark"
        ),
        claim_ids=("CLM-027",),
        validation_checks=(
            "representation_invariance_contract",
            "representation_invariance_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/intervention_power.json",
        artifact_type="intervention_power",
        generated_by="src.intervention_analysis.write_intervention_power_payload",
        claim_ids=("CLM-034",),
        validation_checks=(
            "intervention_power_contract",
            "intervention_power_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/cross_architecture_fixtures.json",
        artifact_type="cross_architecture_benchmark",
        generated_by="src.advanced_inference.cross_architecture_payload_current",
        claim_ids=("CLM-029",),
        validation_checks=(
            "cross_architecture_contract",
            "cross_architecture_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/formal_verification.json",
        artifact_type="formal_verification",
        generated_by="src.formal_verification.write_formal_verification",
        claim_ids=("CLM-030",),
        validation_checks=(
            "formal_verification_contract",
            "formal_verification_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/formal_trace_correspondence.json",
        artifact_type="formal_correspondence",
        generated_by="src.formal_correspondence.write_formal_correspondence",
        claim_ids=("CLM-030",),
        validation_checks=(
            "formal_correspondence_contract",
            "formal_correspondence_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/compiler_transfer.json",
        artifact_type="compiler_transfer",
        generated_by="src.compiler_transfer.write_compiler_transfer",
        claim_ids=("CLM-029",),
        validation_checks=("compiler_transfer_contract", "compiler_transfer_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/active_information_frontier.json",
        artifact_type="information_policy",
        generated_by="src.advanced_inference.active_information_payload_current",
        claim_ids=("CLM-031",),
        validation_checks=(
            "active_information_contract",
            "active_information_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/sensor_cost_receipt.json",
        artifact_type="sensor_cost",
        generated_by="src.sensor_costs.write_sensor_cost_payload",
        claim_ids=("CLM-031",),
        validation_checks=("sensor_cost_contract", "sensor_cost_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/replication_manifest.json",
        artifact_type="replication_manifest",
        generated_by="src.replication.write_replication_bundle",
        claim_ids=("CLM-032",),
        validation_checks=("replication_contract", "replication_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/external_source_validation.json",
        artifact_type="external_source_validation",
        generated_by="src.external_sources.write_external_source_validation_payload",
        claim_ids=("CLM-036",),
        validation_checks=("external_source_contract", "external_source_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/external_static_validation.json",
        artifact_type="external_static_validation",
        generated_by="src.external_sources.write_external_static_validation",
        claim_ids=("CLM-036", "CLM-038"),
        validation_checks=("external_static_contract", "external_static_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/external_capabilities.json",
        artifact_type="external_capabilities",
        generated_by="src.external_sources.write_compiler_capabilities",
        claim_ids=("CLM-036", "CLM-038"),
        validation_checks=(
            "external_capability_contract",
            "external_capability_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/data/independent_replay_receipt.json",
        artifact_type="independent_replay_receipt",
        generated_by="src.replication.ingest_independent_replay",
        claim_ids=("CLM-032", "CLM-039"),
        validation_checks=(
            "independent_replay_contract",
            "independent_replay_freshness",
        ),
        required=False,
    ),
    ArtifactManifestTarget(
        path="output/data/theorem_or_countertheorem.json",
        artifact_type="theorem_phase",
        generated_by="src.theorem_phase.write_theorem_or_countertheorem",
        claim_ids=("CLM-033",),
        validation_checks=(
            "theorem_countertheorem_contract",
            "theorem_countertheorem_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="output/replication/public/public_manifest.json",
        artifact_type="replication_public_input",
        generated_by="src.replication.write_replication_bundle",
        claim_ids=("CLM-032",),
        validation_checks=("replication_contract",),
    ),
    ArtifactManifestTarget(
        path="output/replication/private/gold_labels.json",
        artifact_type="replication_gold_labels",
        generated_by="src.replication.write_replication_bundle",
        claim_ids=("CLM-032",),
        validation_checks=("replication_contract",),
    ),
    ArtifactManifestTarget(
        path="output/data/cfg_evidence.json",
        artifact_type="cfg_evidence",
        generated_by="src.advanced_inference.cfg_evidence_payload",
        claim_ids=("CLM-001", "CLM-019"),
        validation_checks=("cfg_evidence_contract", "cfg_evidence_freshness"),
    ),
    ArtifactManifestTarget(
        path="output/data/memory_evidence.json",
        artifact_type="memory_evidence",
        generated_by="src.advanced_inference.memory_evidence_payload",
        claim_ids=("CLM-001", "CLM-002", "CLM-019"),
        validation_checks=(
            "memory_evidence_contract",
            "memory_evidence_freshness",
        ),
    ),
    ArtifactManifestTarget(
        path="manuscript/assets/active_assembly_cover.png",
        artifact_type="cover_image",
        generated_by="imagegen: academic assembly-to-belief cover asset",
        claim_ids=("CLM-009",),
        validation_checks=("cover_image_contract",),
    ),
    ArtifactManifestTarget(
        path="output/data/assembly_listing.json",
        artifact_type="trace_projection",
        generated_by="src.assembly_formalism.assembly_listing_payload",
        claim_ids=("CLM-001", "CLM-003"),
        validation_checks=("assembly_listing_contract",),
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
        path="output/rxinfer_categorical/categorical_free_energy_trace.jsonl",
        artifact_type="rxinfer_categorical",
        generated_by="src.rxinfer_bridge.run_rxinfer_categorical_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_categorical_free_energy_records",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer_categorical/categorical_posteriors.json",
        artifact_type="rxinfer_categorical",
        generated_by="src.rxinfer_bridge.run_rxinfer_categorical_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_categorical_companion_artifacts",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer_categorical/categorical_diagnostics.json",
        artifact_type="rxinfer_categorical",
        generated_by="src.rxinfer_bridge.run_rxinfer_categorical_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_categorical_diagnostics_consistent",),
    ),
    ArtifactManifestTarget(
        path="output/rxinfer_categorical/categorical_model_config.json",
        artifact_type="rxinfer_categorical",
        generated_by="src.rxinfer_bridge.run_rxinfer_categorical_backend",
        claim_ids=("CLM-004",),
        validation_checks=("rxinfer_categorical_companion_artifacts",),
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
        path="output/figures/assembly_cfg.png",
        artifact_type="figure",
        generated_by="src.figures.plot_assembly_cfg",
        claim_ids=("CLM-001", "CLM-019"),
        validation_checks=("figure_assembly_cfg",),
    ),
    ArtifactManifestTarget(
        path="output/figures/assembly_memory_timeline.png",
        artifact_type="figure",
        generated_by="src.figures.plot_memory_access_timeline",
        claim_ids=("CLM-001", "CLM-002", "CLM-019"),
        validation_checks=("figure_assembly_memory_timeline",),
    ),
    ArtifactManifestTarget(
        path="output/figures/rxinfer_categorical_free_energy.png",
        artifact_type="figure",
        generated_by="src.figures.plot_rxinfer_categorical_free_energy",
        claim_ids=("CLM-004",),
        validation_checks=("figure_rxinfer_categorical_free_energy",),
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
        path="output/figures/observation_informativeness.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-002",),
        validation_checks=("figure_observation_informativeness",),
    ),
    ArtifactManifestTarget(
        path="output/figures/model_evidence.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-017",),
        validation_checks=("figure_model_evidence",),
    ),
    ArtifactManifestTarget(
        path="output/figures/transition_ablation.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-014",),
        validation_checks=("figure_transition_ablation",),
    ),
    ArtifactManifestTarget(
        path="output/figures/generative_adequacy.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_stats_figures",
        claim_ids=("CLM-020",),
        validation_checks=("figure_generative_adequacy",),
    ),
    ArtifactManifestTarget(
        path="output/figures/typed_observation_matrix.png",
        artifact_type="figure",
        generated_by="src.figures.plot_typed_observation_matrix",
        claim_ids=("CLM-021",),
        validation_checks=("figure_typed_observation_matrix",),
    ),
    ArtifactManifestTarget(
        path="output/figures/factorized_observation_comparison.png",
        artifact_type="figure",
        generated_by="src.figures.plot_factorized_observation_comparison",
        claim_ids=("CLM-021",),
        validation_checks=("figure_factorized_observation_comparison",),
    ),
    ArtifactManifestTarget(
        path="output/figures/corpus_coverage.png",
        artifact_type="figure",
        generated_by="src.figures.plot_corpus_coverage",
        claim_ids=("CLM-022",),
        validation_checks=("figure_corpus_coverage",),
    ),
    ArtifactManifestTarget(
        path="output/figures/action_sensing.png",
        artifact_type="figure",
        generated_by="src.figures.plot_action_sensing",
        claim_ids=("CLM-023",),
        validation_checks=("figure_action_sensing",),
    ),
    ArtifactManifestTarget(
        path="output/figures/parameter_learning.png",
        artifact_type="figure",
        generated_by="src.figures.plot_parameter_learning",
        claim_ids=("CLM-024",),
        validation_checks=("figure_parameter_learning",),
    ),
    ArtifactManifestTarget(
        path="output/figures/representation_invariance_benchmark.png",
        artifact_type="figure",
        generated_by="src.figures.plot_representation_invariance_benchmark",
        claim_ids=("CLM-027", "CLM-028"),
        validation_checks=("figure_representation_invariance_benchmark",),
    ),
    ArtifactManifestTarget(
        path="output/figures/intervention_power.png",
        artifact_type="figure",
        generated_by="src.figures.plot_intervention_power",
        claim_ids=("CLM-034",),
        validation_checks=("figure_intervention_power",),
    ),
    ArtifactManifestTarget(
        path="output/figures/cross_architecture_benchmark.png",
        artifact_type="figure",
        generated_by="src.figures.plot_cross_architecture_benchmark",
        claim_ids=("CLM-029",),
        validation_checks=("figure_cross_architecture_benchmark",),
    ),
    ArtifactManifestTarget(
        path="output/figures/formal_verification_receipt.png",
        artifact_type="figure",
        generated_by="src.figures.plot_formal_verification_receipt",
        claim_ids=("CLM-030",),
        validation_checks=("figure_formal_verification_receipt",),
    ),
    ArtifactManifestTarget(
        path="output/figures/active_information_frontier.png",
        artifact_type="figure",
        generated_by="src.figures.plot_active_information_frontier",
        claim_ids=("CLM-031",),
        validation_checks=("figure_active_information_frontier",),
    ),
    ArtifactManifestTarget(
        path="output/figures/external_source_provenance.png",
        artifact_type="figure",
        generated_by="src.advanced_figures.write_advanced_figures",
        claim_ids=("CLM-036",),
        validation_checks=("figure_external_source_provenance",),
    ),
    ArtifactManifestTarget(
        path="output/figures/artifact_receipt_closure.png",
        artifact_type="figure",
        generated_by="src.figures.plot_artifact_receipt_closure",
        claim_ids=("CLM-037",),
        validation_checks=("figure_artifact_receipt_closure",),
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
        validation_checks=("advanced_temporal_beats_identity_transition",),
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
    ArtifactManifestTarget(
        path="output/data/generative_adequacy.json",
        artifact_type="evaluation",
        generated_by="src.advanced_inference.generative_adequacy_payload",
        claim_ids=("CLM-020",),
        validation_checks=(
            "generative_adequacy_contract",
            "generative_adequacy_freshness",
        ),
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
            "output/data/manuscript_variable_registry.json",
            "output/data/figure_registry.json",
            "output/data/build_provenance.json",
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
            if bool(row.get("required", True)):
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
    write_json_artifact(
        json_path,
        payload,
        project_root=output_dir.parent,
        producer_path=Path(__file__),
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
