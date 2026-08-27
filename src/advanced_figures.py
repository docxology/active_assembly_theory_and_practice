"""Generated figures for the temporal, active, and evaluation layers.

The manuscript describes a temporal HMM, an active EFE policy, and a
quantitative decoder comparison. This module provides the source-bound figure
suite for those analyses and the original observation/RxInfer surfaces, with a
source-bound figure suite and a single caption ledger that documents every
figure the project emits.

All numbers plotted here come from artifacts already produced by
:mod:`advanced_inference` and :mod:`evaluation`; nothing is hand-stamped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifact_contract import (
    seal_data_artifact_surface,
    validate_data_artifact_surface,
)
from .figures import (
    plot_active_efe_trajectory,
    plot_action_sensing,
    plot_assembly_cfg,
    plot_ambiguity_ablation,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_cross_architecture_benchmark,
    plot_compiler_transfer,
    plot_corpus_coverage,
    plot_corpus_stability,
    plot_decoder_calibration,
    plot_generative_model,
    plot_generative_adequacy,
    plot_factorized_observation_comparison,
    plot_formal_verification_receipt,
    plot_formal_correspondence,
    plot_external_source_provenance,
    plot_intervention_power,
    plot_active_information_frontier,
    plot_artifact_receipt_closure,
    plot_model_comparison_bars,
    plot_model_evidence,
    plot_memory_access_timeline,
    plot_observation_informativeness,
    plot_parameter_learning,
    plot_phase_recovery,
    plot_representation_invariance_benchmark,
    plot_quotient_classifier,
    plot_replication_manifest,
    plot_segmentation_ribbons,
    plot_transition_ablation,
    plot_sensor_costs,
    plot_temporal_beliefs_heatmap,
    plot_typed_observation_matrix,
    write_figure_captions,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def write_temporal_belief_figures(output_dir: Path = OUTPUT_DIR) -> list[str]:
    """Write one temporal-belief heatmap per corpus fixture; return relative paths."""

    figure_dir = output_dir / "figures"
    temporal_path = output_dir / "data" / "temporal_inference.json"
    if not temporal_path.exists():
        raise FileNotFoundError(f"missing temporal inference artifact: {temporal_path}")
    payload = json.loads(temporal_path.read_text(encoding="utf-8"))
    hidden_state_labels = payload["model"]["hidden_state_labels"]
    relative_paths: list[str] = []
    for fixture_name, fixture in payload["per_fixture"].items():
        figure_path = figure_dir / f"temporal_beliefs_{fixture_name}.png"
        plot_temporal_beliefs_heatmap(
            fixture_name,
            list(fixture["observation_ids"]),
            [list(row) for row in fixture["filter"]["filtered_posteriors"]],
            list(hidden_state_labels),
            figure_path,
        )
        relative_paths.append(str(figure_path.relative_to(PROJECT_ROOT)))
    return relative_paths


def write_stats_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the evaluation, calibration, ablation, and evidence figures.

    Returns:
        A mapping from figure stem to its project-relative PNG path.
    """

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Keep the cross-cutting receipt figure on the same explicit producer
    # boundary as the JSON surface.  Validation remains read-only: this stage
    # is the only place where legacy envelopes may be migrated.
    seal_data_artifact_surface(output_dir.parent)
    # Keep the external-source figure on the same producer boundary as its
    # receipt.  This makes direct figure regeneration deterministic and avoids
    # accidentally plotting a stale or missing safety-status artifact.
    external_manifest = output_dir.parent / "configs" / "external_sources.yaml"
    if external_manifest.is_file():
        from .external_sources import write_external_source_validation_payload

        write_external_source_validation_payload(output_dir.parent)
    evaluation_path = output_dir / "data" / "evaluation_report.json"
    evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))

    calibration_curve_path = figure_dir / "calibration_curve.png"
    plot_calibration_curve(
        evaluation_report["pooled_calibration_bins"], calibration_curve_path
    )

    confusion_path = figure_dir / "confusion_matrix.png"
    plot_confusion_matrix(evaluation_report["pooled_confusion"], confusion_path)

    ablation_path = figure_dir / "ambiguity_ablation.png"
    plot_ambiguity_ablation(evaluation_report["ambiguity_ablation"], ablation_path)

    ribbons_path = figure_dir / "segmentation_ribbons.png"
    plot_segmentation_ribbons(evaluation_report, ribbons_path)

    stability_path = figure_dir / "corpus_stability.png"
    plot_corpus_stability(evaluation_report["corpus_stability"], stability_path)

    phase_recovery_path = figure_dir / "phase_recovery.png"
    plot_phase_recovery(evaluation_report["phase_recovery"], phase_recovery_path)

    decoder_calibration_path = figure_dir / "decoder_calibration.png"
    plot_decoder_calibration(
        evaluation_report["decoder_calibration"], decoder_calibration_path
    )

    generative_path = figure_dir / "generative_model.png"
    plot_generative_model(generative_path)

    informativeness_path = figure_dir / "observation_informativeness.png"
    plot_observation_informativeness(
        evaluation_report["observation_informativeness"], informativeness_path
    )

    evidence_path = figure_dir / "model_evidence.png"
    plot_model_evidence(evaluation_report["model_evidence"], evidence_path)

    transition_path = figure_dir / "transition_ablation.png"
    plot_transition_ablation(evaluation_report["transition_ablation"], transition_path)

    adequacy_path = figure_dir / "generative_adequacy.png"
    plot_generative_adequacy(evaluation_report["generative_adequacy"], adequacy_path)

    typed_path = figure_dir / "typed_observation_matrix.png"
    typed_payload = json.loads(
        (output_dir / "data" / "typed_observations.json").read_text(encoding="utf-8")
    )
    plot_typed_observation_matrix(typed_payload, typed_path)

    factorized_path = figure_dir / "factorized_observation_comparison.png"
    factorized_payload = json.loads(
        (output_dir / "data" / "factorized_observation_model.json").read_text(
            encoding="utf-8"
        )
    )
    plot_factorized_observation_comparison(factorized_payload, factorized_path)

    coverage_path = figure_dir / "corpus_coverage.png"
    coverage_payload = json.loads(
        (output_dir / "data" / "corpus_coverage.json").read_text(encoding="utf-8")
    )
    plot_corpus_coverage(coverage_payload, coverage_path)

    action_sensing_path = figure_dir / "action_sensing.png"
    action_sensing_payload = json.loads(
        (output_dir / "data" / "action_sensing.json").read_text(encoding="utf-8")
    )
    plot_action_sensing(action_sensing_payload, action_sensing_path)

    parameter_learning_path = figure_dir / "parameter_learning.png"
    parameter_learning_payload = json.loads(
        (output_dir / "data" / "parameter_learning.json").read_text(encoding="utf-8")
    )
    plot_parameter_learning(parameter_learning_payload, parameter_learning_path)

    representation_path = figure_dir / "representation_invariance_benchmark.png"
    representation_payload = json.loads(
        (output_dir / "data" / "representation_invariance_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    plot_representation_invariance_benchmark(
        representation_payload, representation_path
    )

    intervention_power_path = figure_dir / "intervention_power.png"
    intervention_power_payload = json.loads(
        (output_dir / "data" / "intervention_power.json").read_text(encoding="utf-8")
    )
    plot_intervention_power(intervention_power_payload, intervention_power_path)

    cross_architecture_path = figure_dir / "cross_architecture_benchmark.png"
    cross_architecture_payload = json.loads(
        (output_dir / "data" / "cross_architecture_fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    plot_cross_architecture_benchmark(
        cross_architecture_payload, cross_architecture_path
    )

    formal_path = figure_dir / "formal_verification_receipt.png"
    formal_payload = json.loads(
        (output_dir / "data" / "formal_verification.json").read_text(encoding="utf-8")
    )
    plot_formal_verification_receipt(formal_payload, formal_path)

    quotient_path = figure_dir / "quotient_classifier.png"
    quotient_payload = json.loads(
        (output_dir / "data" / "quotient_classifier.json").read_text(encoding="utf-8")
    )
    plot_quotient_classifier(quotient_payload, quotient_path)

    compiler_path = figure_dir / "compiler_transfer.png"
    compiler_payload = json.loads(
        (output_dir / "data" / "compiler_transfer.json").read_text(encoding="utf-8")
    )
    plot_compiler_transfer(compiler_payload, compiler_path)

    correspondence_path = figure_dir / "formal_correspondence.png"
    correspondence_payload = json.loads(
        (output_dir / "data" / "formal_trace_correspondence.json").read_text(
            encoding="utf-8"
        )
    )
    plot_formal_correspondence(correspondence_payload, correspondence_path)

    sensor_path = figure_dir / "sensor_costs.png"
    sensor_payload = json.loads(
        (output_dir / "data" / "sensor_cost_receipt.json").read_text(encoding="utf-8")
    )
    plot_sensor_costs(sensor_payload, sensor_path)

    replication_path = figure_dir / "replication_manifest.png"
    replication_payload = json.loads(
        (output_dir / "data" / "replication_manifest.json").read_text(encoding="utf-8")
    )
    plot_replication_manifest(replication_payload, replication_path)

    external_path = figure_dir / "external_source_provenance.png"
    external_payload = json.loads(
        (output_dir / "data" / "external_source_validation.json").read_text(
            encoding="utf-8"
        )
    )
    plot_external_source_provenance(external_payload, external_path)

    receipt_path = figure_dir / "artifact_receipt_closure.png"
    program_payload = json.loads(
        (output_dir / "data" / "scientific_program.json").read_text(encoding="utf-8")
    )
    artifact_errors = validate_data_artifact_surface(output_dir.parent)
    plot_artifact_receipt_closure(
        program_payload,
        receipt_path,
        artifact_count=len(tuple(sorted((output_dir / "data").glob("*.json")))),
        artifact_error_count=len(artifact_errors),
    )

    information_path = figure_dir / "active_information_frontier.png"
    information_payload = json.loads(
        (output_dir / "data" / "active_information_frontier.json").read_text(
            encoding="utf-8"
        )
    )
    plot_active_information_frontier(information_payload, information_path)

    return {
        "calibration_curve": str(calibration_curve_path.relative_to(PROJECT_ROOT)),
        "confusion_matrix": str(confusion_path.relative_to(PROJECT_ROOT)),
        "ambiguity_ablation": str(ablation_path.relative_to(PROJECT_ROOT)),
        "segmentation_ribbons": str(ribbons_path.relative_to(PROJECT_ROOT)),
        "corpus_stability": str(stability_path.relative_to(PROJECT_ROOT)),
        "phase_recovery": str(phase_recovery_path.relative_to(PROJECT_ROOT)),
        "decoder_calibration": str(decoder_calibration_path.relative_to(PROJECT_ROOT)),
        "generative_model": str(generative_path.relative_to(PROJECT_ROOT)),
        "observation_informativeness": str(
            informativeness_path.relative_to(PROJECT_ROOT)
        ),
        "model_evidence": str(evidence_path.relative_to(PROJECT_ROOT)),
        "transition_ablation": str(transition_path.relative_to(PROJECT_ROOT)),
        "generative_adequacy": str(adequacy_path.relative_to(PROJECT_ROOT)),
        "typed_observation_matrix": str(typed_path.relative_to(PROJECT_ROOT)),
        "factorized_observation_comparison": str(
            factorized_path.relative_to(PROJECT_ROOT)
        ),
        "corpus_coverage": str(coverage_path.relative_to(PROJECT_ROOT)),
        "action_sensing": str(action_sensing_path.relative_to(PROJECT_ROOT)),
        "parameter_learning": str(parameter_learning_path.relative_to(PROJECT_ROOT)),
        "representation_invariance_benchmark": str(
            representation_path.relative_to(PROJECT_ROOT)
        ),
        "intervention_power": str(intervention_power_path.relative_to(PROJECT_ROOT)),
        "cross_architecture_benchmark": str(
            cross_architecture_path.relative_to(PROJECT_ROOT)
        ),
        "formal_verification_receipt": str(formal_path.relative_to(PROJECT_ROOT)),
        "quotient_classifier": str(quotient_path.relative_to(PROJECT_ROOT)),
        "compiler_transfer": str(compiler_path.relative_to(PROJECT_ROOT)),
        "formal_correspondence": str(correspondence_path.relative_to(PROJECT_ROOT)),
        "sensor_costs": str(sensor_path.relative_to(PROJECT_ROOT)),
        "replication_manifest": str(replication_path.relative_to(PROJECT_ROOT)),
        "external_source_provenance": str(external_path.relative_to(PROJECT_ROOT)),
        "artifact_receipt_closure": str(receipt_path.relative_to(PROJECT_ROOT)),
        "active_information_frontier": str(information_path.relative_to(PROJECT_ROOT)),
    }


def write_advanced_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the current figure set and the caption ledger; return relative paths.

    Returns:
        A mapping from figure stem to its project-relative PNG path. The
        ``temporal_beliefs`` entry is the first fixture's path used by the
        manuscript token; every fixture-specific heatmap remains in the output
        registry and caption ledger.
    """

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = output_dir / "data" / "evaluation_report.json"
    evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))

    temporal_paths = write_temporal_belief_figures(output_dir)
    model_comparison_path = figure_dir / "model_comparison.png"
    plot_model_comparison_bars(evaluation_report, model_comparison_path)

    active_path = output_dir / "data" / "active_selection.json"
    active_report = json.loads(active_path.read_text(encoding="utf-8"))
    first_fixture = next(iter(active_report["per_fixture"]))
    active_efe_path = figure_dir / "active_efe.png"
    plot_active_efe_trajectory(
        active_report["per_fixture"][first_fixture]["selection"],
        active_efe_path,
        active_report["per_fixture"][first_fixture].get("predictive_selection"),
    )

    cfg_path = output_dir / "data" / "cfg_evidence.json"
    cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assembly_cfg_path = figure_dir / "assembly_cfg.png"
    plot_assembly_cfg(cfg_payload, assembly_cfg_path)

    memory_path = output_dir / "data" / "memory_evidence.json"
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    memory_figure_path = figure_dir / "assembly_memory_timeline.png"
    plot_memory_access_timeline(memory_payload, memory_figure_path)

    stats_paths = write_stats_figures(output_dir)
    captions_path = write_figure_captions(output_dir)

    temporal_figure_paths = {Path(path).stem: path for path in temporal_paths}
    return {
        "temporal_beliefs": temporal_paths[0] if temporal_paths else "",
        **temporal_figure_paths,
        "model_comparison": str(model_comparison_path.relative_to(PROJECT_ROOT)),
        "active_efe": str(active_efe_path.relative_to(PROJECT_ROOT)),
        "assembly_cfg": str(assembly_cfg_path.relative_to(PROJECT_ROOT)),
        "assembly_memory_timeline": str(memory_figure_path.relative_to(PROJECT_ROOT)),
        "calibration_curve": stats_paths["calibration_curve"],
        "confusion_matrix": stats_paths["confusion_matrix"],
        "ambiguity_ablation": stats_paths["ambiguity_ablation"],
        "segmentation_ribbons": stats_paths["segmentation_ribbons"],
        "corpus_stability": stats_paths["corpus_stability"],
        "phase_recovery": stats_paths["phase_recovery"],
        "decoder_calibration": stats_paths["decoder_calibration"],
        "generative_model": stats_paths["generative_model"],
        "observation_informativeness": stats_paths["observation_informativeness"],
        "model_evidence": stats_paths["model_evidence"],
        "transition_ablation": stats_paths["transition_ablation"],
        "generative_adequacy": stats_paths["generative_adequacy"],
        "typed_observation_matrix": stats_paths["typed_observation_matrix"],
        "factorized_observation_comparison": stats_paths[
            "factorized_observation_comparison"
        ],
        "corpus_coverage": stats_paths["corpus_coverage"],
        "action_sensing": stats_paths["action_sensing"],
        "parameter_learning": stats_paths["parameter_learning"],
        "representation_invariance_benchmark": stats_paths[
            "representation_invariance_benchmark"
        ],
        "intervention_power": stats_paths["intervention_power"],
        "cross_architecture_benchmark": stats_paths["cross_architecture_benchmark"],
        "formal_verification_receipt": stats_paths["formal_verification_receipt"],
        "quotient_classifier": stats_paths["quotient_classifier"],
        "compiler_transfer": stats_paths["compiler_transfer"],
        "formal_correspondence": stats_paths["formal_correspondence"],
        "sensor_costs": stats_paths["sensor_costs"],
        "replication_manifest": stats_paths["replication_manifest"],
        "external_source_provenance": stats_paths["external_source_provenance"],
        "artifact_receipt_closure": stats_paths["artifact_receipt_closure"],
        "active_information_frontier": stats_paths["active_information_frontier"],
        "captions": str(captions_path.relative_to(PROJECT_ROOT)),
    }
