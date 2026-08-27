"""Generate manuscript variables from live experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .advanced_inference import OUTPUT_DIR as ADVANCED_OUTPUT_DIR
from .advanced_inference import run_advanced_inference
from .artifact_contract import (
    COMMON_RECEIPT_FIELDS,
    seal_data_artifact_surface,
    validate_data_artifact_surface,
    write_json_artifact,
)
from .assembly_formalism import assembly_listing_markdown
from .claim_register import CLAIM_RECORDS, write_claim_register
from .build_provenance import write_build_provenance
from .docx_validation import docx_render_payload
from .evaluation import CALIBRATION_BIN_COUNT, WILSON_CONFIDENCE_LEVEL
from .experiments import OUTPUT_DIR, run_experiment
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .figures import plot_inference_diagnostics
from .manuscript_registry import write_variable_registry
from .reproducibility import MANIFEST_TARGETS, write_reproducibility_manifest
from .rxinfer_bridge import (
    run_rxinfer_backend,
    run_rxinfer_categorical_backend,
)
from .scholarship import SCHOLARSHIP_SOURCES, write_scholarship_sources
from .typed_observations import (
    STATIC_TYPED_OBSERVATION_FIELDS,
    TYPED_FIELD_GROUPS,
    TYPED_OBSERVATION_FIELDS,
    TYPED_OBSERVATION_SCHEMA_VERSION,
)
from .validation import write_validation_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_variables() -> dict[str, str]:
    """Run the source-bound experiment and return manuscript variables."""

    result = run_experiment()
    docx_payload = docx_render_payload(PROJECT_ROOT)
    assembly_listing_payload = json.loads(
        (OUTPUT_DIR / "data" / "assembly_listing.json").read_text(encoding="utf-8")
    )
    advanced_paths = run_advanced_inference()
    evaluation_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    active_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "active_selection.json").read_text(
            encoding="utf-8"
        )
    )
    held_out_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "held_out_model.json").read_text(
            encoding="utf-8"
        )
    )
    typed_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "typed_observations.json").read_text(
            encoding="utf-8"
        )
    )
    factorized_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "factorized_observation_model.json").read_text(
            encoding="utf-8"
        )
    )
    corpus_coverage_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "corpus_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    action_sensing_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "action_sensing.json").read_text(
            encoding="utf-8"
        )
    )
    parameter_learning_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "parameter_learning.json").read_text(
            encoding="utf-8"
        )
    )
    scientific_program_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "scientific_program.json").read_text(
            encoding="utf-8"
        )
    )
    representation_benchmark_report = json.loads(
        (
            ADVANCED_OUTPUT_DIR / "data" / "representation_invariance_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    intervention_power_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "intervention_power.json").read_text(
            encoding="utf-8"
        )
    )
    cross_architecture_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "cross_architecture_fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    formal_verification_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "formal_verification.json").read_text(
            encoding="utf-8"
        )
    )
    active_information_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "active_information_frontier.json").read_text(
            encoding="utf-8"
        )
    )
    research_protocol_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "research_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    quotient_classifier_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "quotient_classifier.json").read_text(
            encoding="utf-8"
        )
    )
    formal_correspondence_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "formal_trace_correspondence.json").read_text(
            encoding="utf-8"
        )
    )
    compiler_transfer_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "compiler_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    sensor_cost_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "sensor_cost_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    replication_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "replication_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    theorem_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "theorem_or_countertheorem.json").read_text(
            encoding="utf-8"
        )
    )
    external_source_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "external_source_validation.json").read_text(
            encoding="utf-8"
        )
    )
    external_static_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "external_static_validation.json").read_text(
            encoding="utf-8"
        )
    )
    external_capabilities_report = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "external_capabilities.json").read_text(
            encoding="utf-8"
        )
    )
    independent_replay_path = (
        ADVANCED_OUTPUT_DIR / "data" / "independent_replay_receipt.json"
    )
    independent_replay_report = (
        json.loads(independent_replay_path.read_text(encoding="utf-8"))
        if independent_replay_path.is_file()
        else {}
    )
    memory_evidence = json.loads(
        (ADVANCED_OUTPUT_DIR / "data" / "memory_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation_summary = evaluation_report["summary"]
    ablation = evaluation_report["ambiguity_ablation"]
    ablation_rows = list(ablation["rows"])
    max_blend_row = max(ablation_rows, key=lambda row: row["blend"])
    seg_pooled = evaluation_report["segmentation"]["pooled"]
    stability = evaluation_report["corpus_stability"]
    phase_recovery = evaluation_report["phase_recovery"]
    phase_metrics = phase_recovery["per_phase"]
    decoder_cal = evaluation_report["decoder_calibration"]
    obs_info = evaluation_report["observation_informativeness"]
    evidence = evaluation_report["model_evidence"]
    transition = evaluation_report["transition_ablation"]
    adequacy = evaluation_report["generative_adequacy"]
    first_fixture_evaluation = evaluation_report["per_fixture"][result.sample_name]
    first_fixture_active = active_report["per_fixture"][result.sample_name]
    held_out_summary = held_out_report["summary"]
    factorized_summary = factorized_report["summary"]
    corpus_coverage_summary = corpus_coverage_report["summary"]
    action_sensing_summary = action_sensing_report["summary"]
    parameter_learning_summary = parameter_learning_report["summary"]
    scientific_program_status_counts = scientific_program_report["status_counts"]
    representation_benchmark_summary = representation_benchmark_report["summary"]
    representation_negative_control_summary = representation_benchmark_report[
        "negative_control"
    ]["summary"]
    cross_architecture_summary = cross_architecture_report["summary"]
    representation_negative_control_unseparated_count = sum(
        not bool(row["near_miss_separated"])
        for row in representation_benchmark_report["negative_control"]["pairs"]
        if row["variant_role"] == "near_miss"
    )
    intervention_negative_control = intervention_power_report["negative_control"]
    temporal_phase_sequence = " -> ".join(
        str(label).replace("_", " ")
        for label in first_fixture_evaluation["temporal_labels"]
    )
    rxinfer_result = run_rxinfer_backend(
        PROJECT_ROOT / result.data_paths["trace_bundle"]
    )
    categorical_rxinfer_result = run_rxinfer_categorical_backend(
        PROJECT_ROOT / result.data_paths["trace_bundle"]
    )
    inference_diagnostic_path = plot_inference_diagnostics(
        list(result.inference.free_energy_by_step),
        list(rxinfer_result.free_energy_values),
        OUTPUT_DIR / "figures" / "inference_diagnostics.png",
    )
    scholarship_path = write_scholarship_sources()
    claim_register_path = write_claim_register()
    from .advanced_figures import write_advanced_figures

    advanced_figure_paths = write_advanced_figures()
    reproducibility_path = write_reproducibility_manifest()
    # Close the project-owned JSON surface before the first validation pass.
    # This migrates only legacy metadata wrappers; domain payloads and their
    # local schemas remain untouched and are then validated independently.
    seal_data_artifact_surface(PROJECT_ROOT)
    artifact_receipt_errors = validate_data_artifact_surface(PROJECT_ROOT)
    artifact_receipt_count = len(
        tuple(sorted((PROJECT_ROOT / "output" / "data").glob("*.json")))
    )
    # Post-render receipts are regenerated when their rendered artifact exists.
    # Their hashes depend on the now-closed figure/manuscript surface and must
    # therefore be written after the migration, never hand-edited in place.
    from .docx_validation import DOCX_ARTIFACT, write_docx_render_receipt
    from .pdf_accessibility import DEFAULT_PDF, write_pdf_accessibility_report

    if (PROJECT_ROOT / DOCX_ARTIFACT).is_file():
        write_docx_render_receipt(PROJECT_ROOT)
    if DEFAULT_PDF.is_file():
        write_pdf_accessibility_report(project_root=PROJECT_ROOT)
    reproducibility_path = write_reproducibility_manifest()
    variables = {
        "PROJECT_TITLE": "Active Assembly Theory and Practice",
        "SAMPLE_NAME": result.sample_name,
        "INSTRUCTION_COUNT": str(len(result.instructions)),
        "OBSERVATION_COUNT": str(len(result.observations)),
        "ASSEMBLY_ARCHITECTURE": str(assembly_listing_payload["architecture"]),
        "ASSEMBLY_BASE_ADDRESS": f"0x{int(assembly_listing_payload['base_address']):x}",
        "ASSEMBLY_LISTING": assembly_listing_markdown(assembly_listing_payload),
        "ASSEMBLY_LISTING_ARTIFACT": "../output/data/assembly_listing.json",
        "SYMBOLIC_SEMANTICS_ARTIFACT": "../output/data/symbolic_execution.json",
        "SYMBOLIC_SUPPORTED_COUNT": str(
            result.symbolic_execution.supported_instruction_count
        ),
        "SYMBOLIC_UNSUPPORTED_COUNT": str(
            result.symbolic_execution.unsupported_instruction_count
        ),
        "SYMBOLIC_FINAL_RAX_MATCH": (
            "yes" if result.symbolic_execution.final_rax_matches else "no"
        ),
        "SYMBOLIC_STEPWISE_RAX_AGREEMENT_COUNT": str(
            result.symbolic_execution.stepwise_rax_agreement_count
        ),
        "SYMBOLIC_STEPWISE_RAX_COMPARABLE_COUNT": str(
            result.symbolic_execution.stepwise_rax_comparable_count
        ),
        "SYMBOLIC_STEPWISE_FLAG_AGREEMENT_COUNT": str(
            result.symbolic_execution.modeled_flag_agreements
        ),
        "SYMBOLIC_STEPWISE_FLAG_COMPARISON_COUNT": str(
            result.symbolic_execution.modeled_flag_comparisons
        ),
        "CFG_BLOCK_COUNT": str(len(result.trace_bundle.basic_blocks)),
        "CFG_EDGE_COUNT": str(len(result.trace_bundle.cfg_edges)),
        "MEMORY_STATIC_INSTRUCTION_COUNT": str(
            memory_evidence["summary"]["static_memory_instruction_count"]
        ),
        "MEMORY_DYNAMIC_EVENT_COUNT": str(
            memory_evidence["summary"]["dynamic_event_count"]
        ),
        "MEMORY_READ_COUNT": str(memory_evidence["summary"]["read_count"]),
        "MEMORY_WRITE_COUNT": str(memory_evidence["summary"]["write_count"]),
        "MEMORY_DISTINCT_ADDRESS_COUNT": str(
            memory_evidence["summary"]["distinct_addresses"]
        ),
        "OBSERVATION_CATEGORY_COUNT": str(len(OBSERVATION_LABELS)),
        "HIDDEN_STATE_COUNT": str(len(HIDDEN_STATE_LABELS)),
        "CHANCE_ACCURACY": f"{1.0 / len(HIDDEN_STATE_LABELS):.3f}",
        "CI_LEVEL_PERCENT": str(round(100 * WILSON_CONFIDENCE_LEVEL)),
        "CALIBRATION_BIN_COUNT": str(CALIBRATION_BIN_COUNT),
        "CALIBRATION_NONEMPTY_BIN_COUNT": str(
            len(evaluation_report["pooled_calibration_bins"])
        ),
        "DECODER_COUNT": str(
            len(evaluation_report["compare_models"][result.sample_name])
        ),
        "CORPUS_OBSERVATION_COUNT": str(evidence["observation_count"]),
        "TEMPORAL_PHASE_SEQUENCE": temporal_phase_sequence,
        "TEMPORAL_BELIEFS_BITWISE_FIGURE": (
            f"../{advanced_figure_paths['temporal_beliefs_x86_bitwise']}"
        ),
        "TEMPORAL_BELIEFS_GUARD_FIGURE": (
            f"../{advanced_figure_paths['temporal_beliefs_x86_guard']}"
        ),
        "TEMPORAL_BELIEFS_LONG_ACCUMULATOR_FIGURE": (
            f"../{advanced_figure_paths['temporal_beliefs_x86_long_accumulator']}"
        ),
        "TEMPORAL_BELIEFS_MEMORY_FIGURE": (
            f"../{advanced_figure_paths['temporal_beliefs_x86_memory']}"
        ),
        "TEMPORAL_BELIEFS_GUARD_TAKEN_FIGURE": (
            f"../{advanced_figure_paths['temporal_beliefs_x86_guard_taken']}"
        ),
        "FINAL_RAX": str(result.trace.final_rax),
        "FINAL_STATE": result.inference.final_state.replace("_", " "),
        "FREE_ENERGY_STEPS": str(len(result.inference.free_energy_by_step)),
        "TRACE_SCHEMA_VERSION": result.trace_bundle.schema_version,
        "BINARY_SHA256": result.trace_bundle.binary_sha256,
        "TOTAL_PACKAGE_COUNT": str(len(result.trace_bundle.package_provenance)),
        "AVAILABLE_PACKAGE_COUNT": str(
            sum(
                provenance.available
                for provenance in result.trace_bundle.package_provenance
            )
        ),
        "RXINFER_MODEL_NAME": rxinfer_result.model_name.replace("_", " "),
        "RXINFER_FREE_ENERGY_COUNT": str(len(rxinfer_result.free_energy_values)),
        "RXINFER_FINAL_FREE_ENERGY": f"{rxinfer_result.final_free_energy:.6f}",
        "RXINFER_VERSION": rxinfer_result.rxinfer_version,
        "JULIA_VERSION": rxinfer_result.julia_version,
        "RXINFER_CATEGORICAL_MODEL_NAME": categorical_rxinfer_result.model_name.replace(
            "_", " "
        ),
        "RXINFER_CATEGORICAL_FREE_ENERGY_COUNT": str(
            len(categorical_rxinfer_result.free_energy_values)
        ),
        "RXINFER_CATEGORICAL_FINAL_FREE_ENERGY": f"{categorical_rxinfer_result.final_free_energy:.6f}",
        "RXINFER_CATEGORICAL_ARTIFACT": "../output/rxinfer_categorical/categorical_diagnostics.json",
        "SCHOLARSHIP_SOURCE_COUNT": str(len(SCHOLARSHIP_SOURCES)),
        "SCHOLARSHIP_SOURCE_LEDGER": f"../{scholarship_path}",
        "CLAIM_COUNT": str(len(CLAIM_RECORDS)),
        "VALIDATED_CLAIM_COUNT": str(
            sum(claim.status == "validated" for claim in CLAIM_RECORDS)
        ),
        "CLAIM_REGISTER": f"../{claim_register_path}",
        "REPRODUCIBILITY_MANIFEST": f"../{reproducibility_path}",
        "REPRODUCIBILITY_ARTIFACT_COUNT": str(len(MANIFEST_TARGETS)),
        "CORPUS_FIXTURE_COUNT": str(evaluation_summary["fixture_count"]),
        "TEMPORAL_MEAN_ACCURACY": f"{evaluation_summary['mean_temporal_accuracy']:.3f}",
        "IDENTITY_TRANSITION_MEAN_ACCURACY": f"{evaluation_summary['mean_identity_transition_accuracy']:.3f}",
        "MEMORYLESS_MEAN_ACCURACY": f"{evaluation_summary['mean_memoryless_accuracy']:.3f}",
        "TEMPORAL_ACC_CI": (
            f"[{evaluation_summary['temporal_ci95'][0]:.3f}, "
            f"{evaluation_summary['temporal_ci95'][1]:.3f}]"
        ),
        "IDENTITY_TRANSITION_ACC_CI": (
            f"[{evaluation_summary['identity_transition_ci95'][0]:.3f}, "
            f"{evaluation_summary['identity_transition_ci95'][1]:.3f}]"
        ),
        "MEMORYLESS_ACC_CI": (
            f"[{evaluation_summary['memoryless_ci95'][0]:.3f}, "
            f"{evaluation_summary['memoryless_ci95'][1]:.3f}]"
        ),
        "TEMPORAL_MEMORYLESS_DIFF": f"{evaluation_summary['temporal_memoryless_diff']:.3f}",
        "POOLED_TEMPORAL_ACCURACY": f"{evaluation_summary['pooled_temporal_accuracy']:.3f}",
        "POOLED_MEMORYLESS_ACCURACY": f"{evaluation_summary['pooled_memoryless_accuracy']:.3f}",
        "SHUFFLE_CONTROL_ACCURACY": f"{evaluation_report['shuffled_label_control']['mean_accuracy']:.3f}",
        "UNIFORM_CONTROL_ACCURACY": f"{evaluation_report['uniform_belief_control']['balanced_accuracy']:.3f}",
        "ACTIVE_FIXED_TIE_COUNT": str(active_report["summary"]["ties"]),
        "PREDICTIVE_ACTIVE_STEPS": str(
            first_fixture_active["predictive_steps_to_confidence"]
        ),
        "EXPECTED_OBSERVATION_INFO_GAIN": f"{obs_info['expected_information_gain']:.3f}",
        "HELD_OUT_MEAN_ACCURACY": f"{held_out_summary['mean_held_out_temporal_accuracy']:.3f}",
        "HELD_OUT_FIXED_ACCURACY": f"{held_out_summary['mean_fixed_full_corpus_accuracy']:.3f}",
        "HELD_OUT_ACCURACY_MIN": f"{held_out_summary['held_out_temporal_accuracy_min']:.3f}",
        "HELD_OUT_ACCURACY_MAX": f"{held_out_summary['held_out_temporal_accuracy_max']:.3f}",
        "TYPED_OBSERVATION_SCHEMA_VERSION": TYPED_OBSERVATION_SCHEMA_VERSION,
        "TYPED_FIELD_COUNT": str(len(TYPED_OBSERVATION_FIELDS)),
        "TYPED_STATIC_FIELD_COUNT": str(len(STATIC_TYPED_OBSERVATION_FIELDS)),
        "TYPED_FIELD_GROUP_COUNT": str(len(TYPED_FIELD_GROUPS)),
        "TYPED_FIELD_NAMES": ", ".join(
            field.replace("_", " ") for field in TYPED_OBSERVATION_FIELDS
        ),
        "TYPED_FIXTURE_COUNT": str(len(typed_report["per_fixture"])),
        "CORPUS_COVERAGE_ARTIFACT": f"../{advanced_paths['corpus_coverage']}",
        "CORPUS_MEMORY_FIXTURE_COUNT": str(
            corpus_coverage_summary["memory_fixture_count"]
        ),
        "CORPUS_ALTERNATE_PATH_FIXTURE_COUNT": str(
            corpus_coverage_summary["alternate_path_fixture_count"]
        ),
        "CORPUS_REPEATED_PHASE_FIXTURE_COUNT": str(
            corpus_coverage_summary["repeated_phase_fixture_count"]
        ),
        "CORPUS_UNIQUE_INSTRUCTION_FAMILY_COUNT": str(
            corpus_coverage_summary["unique_instruction_family_count"]
        ),
        "CORPUS_UNIQUE_MEMORY_MODE_COUNT": str(
            corpus_coverage_summary["unique_memory_mode_count"]
        ),
        "CORPUS_UNIQUE_BRANCH_CLASS_COUNT": str(
            corpus_coverage_summary["unique_branch_target_class_count"]
        ),
        "ACTION_SENSING_ARTIFACT": f"../{advanced_paths['action_sensing']}",
        "ACTION_SENSING_ACTIVE_MEAN_ACTIONS": f"{action_sensing_summary['mean_actions_used']['active']:.3f}",
        "ACTION_SENSING_FIXED_MEAN_ACTIONS": f"{action_sensing_summary['mean_actions_used']['fixed']:.3f}",
        "ACTION_SENSING_RANDOM_CONFIDENCE_RATE": f"{action_sensing_summary['confidence_rate']['random']:.3f}",
        "ACTION_SENSING_ACTIVE_MINUS_FIXED_ACTIONS": f"{action_sensing_summary['active_minus_fixed_actions']:.3f}",
        "ACTION_SENSING_COST_WEIGHT": f"{action_sensing_report['cost']['weight_in_expected_free_energy']:.3f}",
        "ACTION_SENSING_LEAKAGE_FREE": (
            "yes" if action_sensing_summary["leakage_free"] else "no"
        ),
        "PARAMETER_LEARNING_ARTIFACT": f"../{advanced_paths['parameter_learning']}",
        "PARAMETER_LEARNING_HOLDOUT_COUNT": str(
            parameter_learning_summary["fixture_count"]
        ),
        "PARAMETER_LEARNING_MAX_ITERATIONS": str(
            parameter_learning_report["max_iterations"]
        ),
        "PARAMETER_LEARNING_EM_MEAN_ACCURACY": f"{parameter_learning_summary['mean_weakly_supervised_accuracy']:.3f}",
        "PARAMETER_LEARNING_SUPERVISED_MEAN_ACCURACY": f"{parameter_learning_summary['mean_supervised_accuracy']:.3f}",
        "PARAMETER_LEARNING_REFERENCE_MEAN_ACCURACY": f"{parameter_learning_summary['mean_fixed_reference_accuracy']:.3f}",
        "PARAMETER_LEARNING_EM_MINUS_SUPERVISED": f"{parameter_learning_summary['weakly_supervised_minus_supervised']:.3f}",
        "PARAMETER_LEARNING_EM_MINUS_REFERENCE": f"{parameter_learning_summary['weakly_supervised_minus_reference']:.3f}",
        "PARAMETER_LEARNING_CONVERGED_HOLDOUT_COUNT": str(
            parameter_learning_summary["converged_holdout_count"]
        ),
        "PARAMETER_LEARNING_LEAKAGE_FREE": (
            "yes" if parameter_learning_summary["leakage_free"] else "no"
        ),
        "PARAMETER_LEARNING_MULTI_START_COUNT": str(
            parameter_learning_summary["multi_start_count"]
        ),
        "PARAMETER_LEARNING_MULTI_START_MEAN_ACCURACY": f"{parameter_learning_summary['multi_start_mean_accuracy']:.3f}",
        "PARAMETER_LEARNING_MULTI_START_MEAN_RANGE": f"{parameter_learning_summary['multi_start_mean_accuracy_range']:.3f}",
        "PARAMETER_LEARNING_MULTI_START_MEAN_AGREEMENT": f"{parameter_learning_summary['multi_start_mean_pairwise_label_agreement']:.3f}",
        "SCIENTIFIC_PROGRAM_ARTIFACT": f"../{advanced_paths['scientific_program']}",
        "SCIENTIFIC_PROGRAM_MILESTONE_COUNT": str(
            len(scientific_program_report["milestones"])
        ),
        "SCIENTIFIC_PROGRAM_IMPLEMENTED_COUNT": str(
            scientific_program_status_counts.get("implemented_local", 0)
        ),
        "SCIENTIFIC_PROGRAM_ACTIVE_COUNT": str(
            scientific_program_status_counts.get("active", 0)
        ),
        "SCIENTIFIC_PROGRAM_PLANNED_COUNT": str(
            scientific_program_status_counts.get("planned", 0)
        ),
        "QUOTIENT_PROTOCOL_ARTIFACT": f"../{advanced_paths['quotient_invariance_protocol']}",
        "REPRESENTATION_BENCHMARK_ARTIFACT": (
            f"../{advanced_paths['representation_invariance_benchmark']}"
        ),
        "REPRESENTATION_BENCHMARK_FAMILY_COUNT": str(
            representation_benchmark_report["family_count"]
        ),
        "REPRESENTATION_BENCHMARK_PAIR_COUNT": str(
            representation_benchmark_report["pair_count"]
        ),
        "REPRESENTATION_BENCHMARK_EQUIVALENCE_CONSISTENCY": f"{representation_benchmark_summary['equivalence_output_consistency']:.3f}",
        "REPRESENTATION_BENCHMARK_NEAR_MISS_SEPARATION": f"{representation_benchmark_summary['near_miss_separation_rate']:.3f}",
        "REPRESENTATION_BENCHMARK_SURFACE_BLIND_NEAR_MISS_COUNT": str(
            representation_benchmark_summary["surface_blind_near_miss_count"]
        ),
        "REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_SEPARATION": f"{representation_negative_control_summary['near_miss_separation_rate']:.3f}",
        "REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_UNSEPARATED_COUNT": str(
            representation_negative_control_unseparated_count
        ),
        "INTERVENTION_POWER_ARTIFACT": f"../{advanced_paths['intervention_power']}",
        "INTERVENTION_POWER_SUBSET_COUNT": str(
            intervention_power_report["subset_count"]
        ),
        "INTERVENTION_POWER_PAIR_COUNT": str(intervention_power_report["pair_count"]),
        "INTERVENTION_POWER_MINIMAL_CARDINALITY": str(
            intervention_power_report["minimal_separating_cardinality"]
            if intervention_power_report["minimal_separating_cardinality"] is not None
            else "not available"
        ),
        "INTERVENTION_POWER_MINIMAL_SET_COUNT": str(
            len(intervention_power_report["minimal_separating_sets"])
        ),
        "INTERVENTION_POWER_SINGLE_STATE_SEPARATION": f"{intervention_negative_control['near_miss_separation']:.3f}",
        "INTERVENTION_POWER_SINGLE_STATE_UNRESOLVED_COUNT": str(
            len(intervention_negative_control["unresolved_near_miss_families"])
        ),
        "CROSS_ARCHITECTURE_ARTIFACT": f"../{advanced_paths['cross_architecture_fixtures']}",
        "CROSS_ARCHITECTURE_COUNT": str(
            cross_architecture_summary["architecture_count"]
        ),
        "CROSS_ARCHITECTURE_ALL_PASSED": (
            "yes"
            if cross_architecture_summary["all_bounded_executions_passed"]
            else "no"
        ),
        "CROSS_ARCHITECTURE_RESULT": str(
            cross_architecture_report["rows"][0]["execution"]["final_accumulator"]
        ),
        "FORMAL_VERIFICATION_ARTIFACT": f"../{advanced_paths['formal_verification']}",
        "FORMAL_TOOLCHAIN": str(formal_verification_report["toolchain"]),
        "FORMAL_THEOREM_COUNT": str(len(formal_verification_report["theorems"])),
        "FORMAL_VERIFICATION_STATUS": str(
            formal_verification_report.get(
                "endpoint_status", formal_verification_report["status"]
            )
        ),
        "ACTIVE_INFORMATION_ARTIFACT": f"../{advanced_paths['active_information_frontier']}",
        "ACTIVE_INFORMATION_SENSOR_COUNT": str(
            len(active_information_report["sensors"])
        ),
        "ACTIVE_INFORMATION_FRONTIER_ROW_COUNT": str(
            len(active_information_report["frontier"])
        ),
        "ACTIVE_INFORMATION_EXACT": (
            "yes" if active_information_report["exact_enumeration"] else "no"
        ),
        "RESEARCH_PROTOCOL_ARTIFACT": f"../{advanced_paths['research_protocol']}",
        "RESEARCH_PROTOCOL_ID": str(research_protocol_report["protocol_id"]),
        "RESEARCH_PROTOCOL_HASH": str(research_protocol_report["protocol_sha256"]),
        "ARTIFACT_RECEIPT_SCHEMA_VERSION": "2.0",
        "ARTIFACT_RECEIPT_REQUIRED_FIELD_COUNT": str(len(COMMON_RECEIPT_FIELDS)),
        "ARTIFACT_RECEIPT_COUNT": str(artifact_receipt_count),
        "ARTIFACT_RECEIPT_ERROR_COUNT": str(len(artifact_receipt_errors)),
        "ARTIFACT_RECEIPT_AUDIT_COMMAND": (
            "uv run python scripts/run_scientific_program.py audit-artifacts"
        ),
        "QUOTIENT_CLASSIFIER_ARTIFACT": f"../{advanced_paths['quotient_classifier']}",
        "QUOTIENT_CLASSIFIER_STATUS": str(quotient_classifier_report["status"]),
        "QUOTIENT_CLASSIFIER_FAMILY_COUNT": str(
            quotient_classifier_report["family_count"]
        ),
        "QUOTIENT_CLASSIFIER_ROW_COUNT": str(quotient_classifier_report["row_count"]),
        "QUOTIENT_CLASSIFIER_TRAINING_FAMILY_COUNT": str(
            quotient_classifier_report["training_family_count"]
        ),
        "QUOTIENT_CLASSIFIER_DEVELOPMENT_FAMILY_COUNT": str(
            quotient_classifier_report["development_family_count"]
        ),
        "QUOTIENT_CLASSIFIER_HELD_OUT_FAMILY_COUNT": str(
            quotient_classifier_report["held_out_family_count"]
        ),
        "QUOTIENT_CLASSIFIER_INTERVENTION_COUNT": str(
            len(quotient_classifier_report["interventions"])
        ),
        "QUOTIENT_CLASSIFIER_INTERPRETABLE_BALANCED_ACCURACY": f"{quotient_classifier_report['models']['calibrated_interpretable']['balanced_accuracy']:.3f}",
        "QUOTIENT_CLASSIFIER_INTERPRETABLE_MACRO_F1": f"{quotient_classifier_report['models']['calibrated_interpretable']['macro_f1']:.3f}",
        "QUOTIENT_CLASSIFIER_TEMPORAL_BALANCED_ACCURACY": f"{quotient_classifier_report['models']['temporal_probabilistic']['balanced_accuracy']:.3f}",
        "QUOTIENT_CLASSIFIER_TEMPORAL_MACRO_F1": f"{quotient_classifier_report['models']['temporal_probabilistic']['macro_f1']:.3f}",
        "QUOTIENT_CLASSIFIER_NEURAL_BALANCED_ACCURACY": f"{quotient_classifier_report['models']['neural_sequence']['balanced_accuracy']:.3f}",
        "QUOTIENT_CLASSIFIER_NEURAL_MACRO_F1": f"{quotient_classifier_report['models']['neural_sequence']['macro_f1']:.3f}",
        "QUOTIENT_CLASSIFIER_BASELINE_BALANCED_ACCURACY": f"{quotient_classifier_report['models']['surface_baseline']['balanced_accuracy']:.3f}",
        "QUOTIENT_CLASSIFIER_BASELINE_MACRO_F1": f"{quotient_classifier_report['models']['surface_baseline']['macro_f1']:.3f}",
        "QUOTIENT_CLASSIFIER_RAW_BASELINE_COUNT": str(
            len(quotient_classifier_report["baseline_models"])
        ),
        "QUOTIENT_CLASSIFIER_RAW_COMPARISON_COUNT": str(
            sum(
                len(rows)
                for rows in quotient_classifier_report[
                    "surface_baseline_comparisons"
                ].values()
            )
        ),
        "FORMAL_CORRESPONDENCE_ARTIFACT": f"../{advanced_paths['formal_trace_correspondence']}",
        "FORMAL_CORRESPONDENCE_STATUS": str(formal_correspondence_report["status"]),
        "FORMAL_CORRESPONDENCE_FIXTURE_COUNT": str(
            formal_correspondence_report["fixture_count"]
        ),
        "FORMAL_CORRESPONDENCE_INTERVENTION_COUNT": str(
            formal_correspondence_report["intervention_count"]
        ),
        "FORMAL_CORRESPONDENCE_STEP_COUNT": str(
            sum(
                len(row["reference_steps"])
                for row in formal_correspondence_report["step_rows"]
            )
        ),
        "COMPILER_TRANSFER_ARTIFACT": f"../{advanced_paths['compiler_transfer']}",
        "COMPILER_TRANSFER_VALIDATED_COUNT": str(
            compiler_transfer_report["capability_summary"]["validated_static_objects"]
        ),
        "COMPILER_TRANSFER_UNAVAILABLE_COUNT": str(
            compiler_transfer_report["capability_summary"]["unavailable_rows"]
        ),
        "COMPILER_TRANSFER_ARCHITECTURES": ", ".join(
            compiler_transfer_report["capability_summary"]["architectures_validated"]
        ),
        "EXTERNAL_SOURCE_ARTIFACT": f"../{advanced_paths['external_source_validation']}",
        "EXTERNAL_SOURCE_STATUS": str(external_source_report["status"]),
        "EXTERNAL_SOURCE_PROJECT_COUNT": str(external_source_report["project_count"]),
        "EXTERNAL_SOURCE_LABELABLE_EXAMPLE_COUNT": str(
            external_source_report["labelable_example_count"]
        ),
        "EXTERNAL_SOURCE_GATE_COUNT": str(
            len(external_source_report["promotion_gate"])
        ),
        "EXTERNAL_STATIC_ARTIFACT": "../output/data/external_static_validation.json",
        "EXTERNAL_STATIC_STATUS": str(external_static_report["status"]),
        "EXTERNAL_STATIC_ROW_COUNT": str(len(external_static_report["rows"])),
        "EXTERNAL_STATIC_VALIDATED_COUNT": str(
            sum(
                row.get("status") == "validated"
                for row in external_static_report["rows"]
            )
        ),
        "EXTERNAL_CAPABILITIES_ARTIFACT": "../output/data/external_capabilities.json",
        "EXTERNAL_CAPABILITIES_STATUS": str(external_capabilities_report["status"]),
        "EXTERNAL_CAPABILITY_TARGET_COUNT": str(
            sum(
                bool(value)
                for row in external_capabilities_report["rows"]
                for value in row.get("targets", {}).values()
            )
        ),
        "INDEPENDENT_REPLAY_STATUS": str(
            independent_replay_report.get("endpoint_status", "not_yet_observed")
        ),
        "SENSOR_COST_ARTIFACT": f"../{advanced_paths['sensor_cost_receipt']}",
        "SENSOR_COST_SENSOR_COUNT": str(len(sensor_cost_report["sensors"])),
        "SENSOR_CERTIFICATE_COUNT": str(len(sensor_cost_report["policy_certificates"])),
        "SENSOR_CERTIFICATE_MAX_GAP": f"{max(float(row['optimality_gap']) for row in sensor_cost_report['policy_certificates']):.3f}",
        "REPLICATION_ARTIFACT": f"../{advanced_paths['replication_manifest']}",
        "REPLICATION_PUBLIC_COUNT": str(replication_report["public_row_count"]),
        "REPLICATION_GOLD_COUNT": str(replication_report["gold_row_count"]),
        "REPLICATION_LABEL_SEPARATED": (
            "yes" if replication_report["label_separated"] else "no"
        ),
        "REPLICATION_PUBLIC_BUNDLE_SHA256": str(
            replication_report["public_bundle_sha256"]
        ),
        "THEOREM_COUNTERTHEOREM_ARTIFACT": f"../{advanced_paths['theorem_or_countertheorem']}",
        "THEOREM_COUNTERTHEOREM_STATUS": str(theorem_report["promotion_status"]),
        "THEOREM_COUNTEREXAMPLE_FAMILY": str(
            theorem_report["counterexample"]["family_id"]
        ),
        "THEOREM_PROVED_COUNT": str(
            sum(
                item.get("status") == "proved"
                for item in theorem_report["candidate_theorems"]
            )
        ),
        "THEOREM_COMPUTATIONAL_ONLY_COUNT": str(
            sum(
                item.get("status") == "computationally_validated"
                for item in theorem_report["candidate_theorems"]
            )
        ),
        "FACTORIZED_COLLAPSED_MEAN_ACCURACY": f"{factorized_summary['mean_collapsed_accuracy']:.3f}",
        "FACTORIZED_STATIC_MEAN_ACCURACY": f"{factorized_summary['mean_static_factorized_accuracy']:.3f}",
        "FACTORIZED_MEAN_ACCURACY": f"{factorized_summary['mean_factorized_accuracy']:.3f}",
        "HIERARCHICAL_MEAN_ACCURACY": f"{factorized_summary['mean_hierarchical_accuracy']:.3f}",
        "FACTORIZED_MINUS_COLLAPSED": f"{factorized_summary['mean_factorized_minus_collapsed']:.3f}",
        "FACTORIZED_MINUS_STATIC": f"{factorized_summary['mean_factorized_minus_static']:.3f}",
        "MISSING_FIELD_MEAN_ACCURACY": f"{factorized_summary['mean_missing_memory_field_accuracy']:.3f}",
        "MISSING_FIELD_MEAN_DELTA": f"{factorized_summary['mean_missing_memory_field_delta']:.3f}",
        "FACTORIZED_ABLATION_GROUP_COUNT": str(
            len(factorized_report["field_group_ablation"])
        ),
        "TYPED_LEAKAGE_FREE": (
            "yes"
            if factorized_report["leakage_checks"]["held_out_excluded_from_training"]
            and factorized_report["leakage_checks"][
                "phase_label_keys_absent_from_typed_records"
            ]
            else "no"
        ),
        "AMBIGUITY_CROSSOVER_BLEND": (
            f"{ablation['crossover_blend']:.2f}"
            if ablation["crossover_blend"] is not None
            else "n/a"
        ),
        "ACTIVE_ADV_MAX_FRACTION": f"{ablation['active_max_advantage_fraction']:.3f}",
        "ACTIVE_ADV_BAND_MIN": f"{ablation['active_advantage_band'][0]:.2f}",
        "ACTIVE_ADV_BAND_MAX": f"{ablation['active_advantage_band'][1]:.2f}",
        "ACTIVE_NEVER_WORSE": (
            "yes"
            if evaluation_report["ambiguity_ablation"]["active_never_worse"]
            else "no"
        ),
        "AMBIGUITY_HIGH_BLEND": f"{max_blend_row['blend']:.2f}",
        "AMBIGUITY_HIGH_TEMPORAL_ACC": f"{max_blend_row['temporal_accuracy']:.3f}",
        "AMBIGUITY_HIGH_MEMORYLESS_ACC": (
            f"{max_blend_row['memoryless_accuracy']:.3f}"
        ),
        "SEGMENTATION_BOUNDARY_F1": f"{seg_pooled['boundary_f1']:.3f}",
        "SEGMENTATION_STEP_ACC": f"{seg_pooled['step_accuracy']:.3f}",
        "CORPUS_TEMPORAL_SPAN": f"{stability['temporal_pooled_range']['span']:.3f}",
        "CORPUS_TEMPORAL_MIN": f"{stability['temporal_pooled_range']['min']:.3f}",
        "CORPUS_TEMPORAL_MAX": f"{stability['temporal_pooled_range']['max']:.3f}",
        "PHASE_MACRO_F1": f"{phase_recovery['macro_f1']:.3f}",
        "HARDEST_PHASE": str(phase_recovery["hardest_phase"]),
        "SETUP_RECALL": f"{phase_metrics['setup']['recall']:.3f}",
        "ACCUMULATOR_RECALL": f"{phase_metrics['accumulator_update']['recall']:.3f}",
        "REPRESENTATION_F1": f"{phase_metrics['representation_transform']['f1']:.3f}",
        "EXIT_F1": f"{phase_metrics['exit']['f1']:.3f}",
        "GUARD_F1": f"{phase_metrics['guard_check']['f1']:.3f}",
        "DECODER_BEST_CALIBRATED": str(decoder_cal["best_calibrated"]),
        "TEMPORAL_ECE": f"{decoder_cal['temporal_ece']:.3f}",
        "IDENTITY_TRANSITION_ECE": f"{decoder_cal['identity_transition_ece']:.3f}",
        "MEMORYLESS_ECE": f"{decoder_cal['memoryless_ece']:.3f}",
        "MOST_INFORMATIVE_OBS": str(obs_info["most_informative"]),
        "LEAST_INFORMATIVE_OBS": str(obs_info["least_informative"]),
        "OBS_INFO_RANGE": f"{obs_info['info_gain_range']:.3f}",
        "TEMPORAL_LOG_EVIDENCE": f"{evidence['log_evidence']['temporal']:.3f}",
        "IDENTITY_TRANSITION_LOG_EVIDENCE": f"{evidence['log_evidence']['identity_transition']:.3f}",
        "MEMORYLESS_LOG_EVIDENCE": f"{evidence['log_evidence']['memoryless']:.3f}",
        "BEST_MODEL": str(evidence["best_model"]),
        "TEMPORAL_EVIDENCE_MARGIN": f"{evidence['temporal_margin_over_best_alternative']:.3f}",
        "TRANSITION_CONVERGENCE_BLEND": (
            f"{transition['convergence_blend']:.1f}"
            if transition["convergence_blend"] is not None
            else "n/a"
        ),
        "TRANSITION_PEAK_BLEND": f"{transition['peak_blend']:.1f}",
        "TRANSITION_PEAK_ACC": f"{transition['peak_temporal_accuracy']:.3f}",
        "GENERATIVE_ADEQUACY_REPLICATES": str(adequacy["replicates"]),
        "GENERATIVE_ADEQUACY_INSIDE_COUNT": str(adequacy["inside_interval_count"]),
        "GENERATIVE_ADEQUACY_METRIC_COUNT": str(adequacy["tested_metric_count"]),
        "GENERATIVE_ADEQUACY_INSIDE_FRACTION": f"{adequacy['inside_interval_fraction']:.3f}",
        "SEG_MEMORYLESS_F1": f"{ablation_rows[0]['memoryless_segmentation_f1']:.3f}",
        "SEG_HIGH_TEMPORAL_F1": f"{max_blend_row['temporal_segmentation_f1']:.3f}",
        "SEG_HIGH_MEMORYLESS_F1": f"{max_blend_row['memoryless_segmentation_f1']:.3f}",
        "TEMPORAL_INFERENCE_ARTIFACT": f"../{advanced_paths['temporal_inference']}",
        "EVALUATION_REPORT_ARTIFACT": f"../{advanced_paths['evaluation_report']}",
        "PHASE_SEGMENTS_ARTIFACT": f"../{advanced_paths['phase_segments']}",
        "ACTIVE_SELECTION_ARTIFACT": f"../{advanced_paths['active_selection']}",
        "HELD_OUT_MODEL_ARTIFACT": f"../{advanced_paths['held_out_model']}",
        "CFG_EVIDENCE_ARTIFACT": f"../{advanced_paths['cfg_evidence']}",
        "MEMORY_EVIDENCE_ARTIFACT": f"../{advanced_paths['memory_evidence']}",
        "OBSERVATION_FIGURE": "../output/figures/observation_pipeline.png",
        "POSTERIOR_FIGURE": "../output/figures/posterior_beliefs.png",
        "PACKAGE_PROVENANCE_FIGURE": "../output/figures/package_provenance.png",
        "TRACE_FEATURE_FIGURE": "../output/figures/trace_feature_matrix.png",
        "TRACE_REFINEMENT_FIGURE": "../output/figures/trace_refinement.png",
        "RXINFER_FREE_ENERGY_FIGURE": "../output/figures/rxinfer_free_energy.png",
        "RXINFER_CATEGORICAL_FREE_ENERGY_FIGURE": (
            "../output/figures/rxinfer_categorical_free_energy.png"
        ),
        "INFERENCE_DIAGNOSTIC_FIGURE": f"../{inference_diagnostic_path.relative_to(PROJECT_ROOT)}",
        "SCHOLARSHIP_FIGURE": "../output/figures/scholarship_source_roles.png",
        "SCHOLARSHIP_MATRIX_FIGURE": "../output/figures/scholarship_evidence_matrix.png",
        "CLAIM_FIGURE": "../output/figures/claim_evidence_map.png",
        "REPRODUCIBILITY_FIGURE": "../output/figures/artifact_hash_manifest.png",
        "TEMPORAL_BELIEFS_FIGURE": f"../{advanced_figure_paths['temporal_beliefs']}",
        "MODEL_COMPARISON_FIGURE": f"../{advanced_figure_paths['model_comparison']}",
        "ACTIVE_EFE_FIGURE": f"../{advanced_figure_paths['active_efe']}",
        "ASSEMBLY_CFG_FIGURE": f"../{advanced_figure_paths['assembly_cfg']}",
        "ASSEMBLY_MEMORY_FIGURE": (
            f"../{advanced_figure_paths['assembly_memory_timeline']}"
        ),
        "CALIBRATION_CURVE_FIGURE": f"../{advanced_figure_paths['calibration_curve']}",
        "CONFUSION_MATRIX_FIGURE": f"../{advanced_figure_paths['confusion_matrix']}",
        "AMBIGUITY_ABLATION_FIGURE": f"../{advanced_figure_paths['ambiguity_ablation']}",
        "SEGMENTATION_RIBBONS_FIGURE": (
            f"../{advanced_figure_paths['segmentation_ribbons']}"
        ),
        "CORPUS_STABILITY_FIGURE": f"../{advanced_figure_paths['corpus_stability']}",
        "PHASE_RECOVERY_FIGURE": f"../{advanced_figure_paths['phase_recovery']}",
        "DECODER_CALIBRATION_FIGURE": f"../{advanced_figure_paths['decoder_calibration']}",
        "GENERATIVE_MODEL_FIGURE": f"../{advanced_figure_paths['generative_model']}",
        "OBSERVATION_INFO_FIGURE": f"../{advanced_figure_paths['observation_informativeness']}",
        "MODEL_EVIDENCE_FIGURE": f"../{advanced_figure_paths['model_evidence']}",
        "TRANSITION_ABLATION_FIGURE": f"../{advanced_figure_paths['transition_ablation']}",
        "GENERATIVE_ADEQUACY_FIGURE": f"../{advanced_figure_paths['generative_adequacy']}",
        "GENERATIVE_ADEQUACY_ARTIFACT": f"../{advanced_paths['generative_adequacy']}",
        "TYPED_OBSERVATION_ARTIFACT": f"../{advanced_paths['typed_observations']}",
        "FACTORIZED_OBSERVATION_ARTIFACT": f"../{advanced_paths['factorized_observation_model']}",
        "TYPED_OBSERVATION_MATRIX_FIGURE": f"../{advanced_figure_paths['typed_observation_matrix']}",
        "FACTORIZED_OBSERVATION_COMPARISON_FIGURE": f"../{advanced_figure_paths['factorized_observation_comparison']}",
        "CORPUS_COVERAGE_FIGURE": f"../{advanced_figure_paths['corpus_coverage']}",
        "ACTION_SENSING_FIGURE": f"../{advanced_figure_paths['action_sensing']}",
        "PARAMETER_LEARNING_FIGURE": f"../{advanced_figure_paths['parameter_learning']}",
        "REPRESENTATION_BENCHMARK_FIGURE": (
            f"../{advanced_figure_paths['representation_invariance_benchmark']}"
        ),
        "INTERVENTION_POWER_FIGURE": f"../{advanced_figure_paths['intervention_power']}",
        "CROSS_ARCHITECTURE_FIGURE": (
            f"../{advanced_figure_paths['cross_architecture_benchmark']}"
        ),
        "FORMAL_VERIFICATION_FIGURE": (
            f"../{advanced_figure_paths['formal_verification_receipt']}"
        ),
        "ACTIVE_INFORMATION_FIGURE": (
            f"../{advanced_figure_paths['active_information_frontier']}"
        ),
        "QUOTIENT_CLASSIFIER_FIGURE": f"../{advanced_figure_paths['quotient_classifier']}",
        "COMPILER_TRANSFER_FIGURE": f"../{advanced_figure_paths['compiler_transfer']}",
        "FORMAL_CORRESPONDENCE_FIGURE": f"../{advanced_figure_paths['formal_correspondence']}",
        "SENSOR_COST_FIGURE": f"../{advanced_figure_paths['sensor_costs']}",
        "REPLICATION_FIGURE": f"../{advanced_figure_paths['replication_manifest']}",
        "EXTERNAL_SOURCE_PROVENANCE_FIGURE": (
            f"../{advanced_figure_paths['external_source_provenance']}"
        ),
        "ARTIFACT_RECEIPT_CLOSURE_FIGURE": (
            f"../{advanced_figure_paths['artifact_receipt_closure']}"
        ),
        "COVER_IMAGE_ARTIFACT": "manuscript/assets/active_assembly_cover.png",
        "DOCX_RENDER_ARTIFACT": (
            "../output/docx/active_assembly_theory_and_practice_combined.docx"
        ),
        "DOCX_RENDER_RECEIPT": "../output/data/docx_render_receipt.json",
        "DOCX_RENDER_STATUS": (
            "valid"
            if docx_payload.get("artifact_exists") and docx_payload.get("valid")
            else (
                "invalid" if docx_payload.get("artifact_exists") else "pending_render"
            )
        ),
        "DOCX_RENDER_FIGURE_COUNT": (
            str(docx_payload.get("embedded_media_count", 0))
            if docx_payload.get("artifact_exists")
            else "pending"
        ),
        "DOCX_RENDER_CAPTION_COUNT": (
            str(docx_payload.get("caption_count", 0))
            if docx_payload.get("artifact_exists")
            else "pending"
        ),
    }
    # These are populated after the first validation pass. Initial values
    # make the source-token coverage check valid even before hydration exists.
    variables.update(
        {
            "VALIDATION_PASS_COUNT": "pending",
            "VALIDATION_FAIL_COUNT": "pending",
            "VALIDATION_TOTAL_COUNT": "pending",
            "VALIDATION_ALL_PASSED": "pending",
            "VALIDATION_REPORT": "../output/data/validation_report.json",
            "VALIDATION_FIGURE": "../output/figures/validation_summary.png",
        }
    )
    output_path = OUTPUT_DIR / "data" / "manuscript_variables.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        output_path,
        variables,
        project_root=PROJECT_ROOT,
        producer_path=Path(__file__),
    )
    # The first validation pass is allowed to inspect the pre-hydration
    # interface; the final pass below receives the concrete validation values.
    write_build_provenance(PROJECT_ROOT)
    validation_report = write_validation_report()
    variables.update(
        {
            "VALIDATION_PASS_COUNT": str(validation_report.summary["passed"]),
            "VALIDATION_FAIL_COUNT": str(validation_report.summary["failed"]),
            "VALIDATION_TOTAL_COUNT": str(validation_report.summary["total"]),
            "VALIDATION_ALL_PASSED": str(
                validation_report.summary["all_passed"]
            ).lower(),
            "VALIDATION_REPORT": "../output/data/validation_report.json",
            "VALIDATION_FIGURE": "../output/figures/validation_summary.png",
        }
    )
    write_json_artifact(
        output_path,
        variables,
        project_root=PROJECT_ROOT,
        producer_path=Path(__file__),
    )
    write_variable_registry(variables, PROJECT_ROOT)
    write_build_provenance(PROJECT_ROOT)
    return variables
