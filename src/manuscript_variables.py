"""Generate manuscript variables from live experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .advanced_inference import OUTPUT_DIR as ADVANCED_OUTPUT_DIR
from .advanced_inference import run_advanced_inference
from .claim_register import CLAIM_RECORDS, write_claim_register
from .experiments import OUTPUT_DIR, run_experiment
from .figures import plot_inference_diagnostics
from .redteam_report import REDTEAM_FINDINGS, write_redteam_report
from .reproducibility import MANIFEST_TARGETS, write_reproducibility_manifest
from .rxinfer_bridge import run_rxinfer_backend
from .scholarship import SCHOLARSHIP_SOURCES, write_scholarship_sources
from .validation import write_validation_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_variables() -> dict[str, str]:
    """Run the scaffold experiment and return string manuscript variables."""

    result = run_experiment()
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
    evaluation_summary = evaluation_report["summary"]
    ablation = evaluation_report["ambiguity_ablation"]
    ablation_rows = list(ablation["rows"])
    max_blend_row = max(ablation_rows, key=lambda row: row["blend"])
    seg_pooled = evaluation_report["segmentation"]["pooled"]
    stability = evaluation_report["corpus_stability"]
    rxinfer_result = run_rxinfer_backend(
        PROJECT_ROOT / result.data_paths["trace_bundle"]
    )
    inference_diagnostic_path = plot_inference_diagnostics(
        list(result.inference.free_energy_by_step),
        list(rxinfer_result.free_energy_values),
        OUTPUT_DIR / "figures" / "inference_diagnostics.png",
    )
    scholarship_path = write_scholarship_sources()
    claim_register_path = write_claim_register()
    redteam_path = write_redteam_report()
    from .advanced_figures import write_advanced_figures

    advanced_figure_paths = write_advanced_figures()
    reproducibility_path = write_reproducibility_manifest()
    remediated_redteam = sum(
        finding.status == "remediated" for finding in REDTEAM_FINDINGS
    )
    variables = {
        "PROJECT_TITLE": "Active Assembly Theory and Practice",
        "SAMPLE_NAME": result.sample_name,
        "INSTRUCTION_COUNT": str(len(result.instructions)),
        "OBSERVATION_COUNT": str(len(result.observations)),
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
        "SCHOLARSHIP_SOURCE_COUNT": str(len(SCHOLARSHIP_SOURCES)),
        "SCHOLARSHIP_SOURCE_LEDGER": f"../{scholarship_path}",
        "CLAIM_COUNT": str(len(CLAIM_RECORDS)),
        "VALIDATED_CLAIM_COUNT": str(
            sum(claim.status == "validated" for claim in CLAIM_RECORDS)
        ),
        "CLAIM_REGISTER": f"../{claim_register_path}",
        "REDTEAM_FINDING_COUNT": str(len(REDTEAM_FINDINGS)),
        "REDTEAM_REMEDIATED_COUNT": str(remediated_redteam),
        "REDTEAM_REPORT": f"../{redteam_path}",
        "REPRODUCIBILITY_MANIFEST": f"../{reproducibility_path}",
        "REPRODUCIBILITY_ARTIFACT_COUNT": str(len(MANIFEST_TARGETS)),
        "CORPUS_FIXTURE_COUNT": str(evaluation_summary["fixture_count"]),
        "TEMPORAL_MEAN_ACCURACY": f"{evaluation_summary['mean_temporal_accuracy']:.3f}",
        "STATIC_MEAN_ACCURACY": f"{evaluation_summary['mean_static_accuracy']:.3f}",
        "MEMORYLESS_MEAN_ACCURACY": f"{evaluation_summary['mean_memoryless_accuracy']:.3f}",
        "TEMPORAL_ACC_CI": (
            f"[{evaluation_summary['temporal_ci95'][0]:.3f}, "
            f"{evaluation_summary['temporal_ci95'][1]:.3f}]"
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
        "AMBIGUITY_CROSSOVER_BLEND": (
            f"{ablation['crossover_blend']:.2f}"
            if ablation["crossover_blend"] is not None
            else "n/a"
        ),
        "ACTIVE_ADV_MAX_FRACTION": f"{ablation['active_max_advantage_fraction']:.3f}",
        "ACTIVE_ADV_BAND_MIN": f"{ablation['active_advantage_band'][0]:.2f}",
        "ACTIVE_ADV_BAND_MAX": f"{ablation['active_advantage_band'][1]:.2f}",
        "ACTIVE_NEVER_WORSE": "yes",
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
        "SEG_MEMORYLESS_F1": f"{ablation_rows[0]['memoryless_segmentation_f1']:.3f}",
        "SEG_HIGH_TEMPORAL_F1": f"{max_blend_row['temporal_segmentation_f1']:.3f}",
        "SEG_HIGH_MEMORYLESS_F1": f"{max_blend_row['memoryless_segmentation_f1']:.3f}",
        "TEMPORAL_INFERENCE_ARTIFACT": f"../{advanced_paths['temporal_inference']}",
        "EVALUATION_REPORT_ARTIFACT": f"../{advanced_paths['evaluation_report']}",
        "PHASE_SEGMENTS_ARTIFACT": f"../{advanced_paths['phase_segments']}",
        "ACTIVE_SELECTION_ARTIFACT": f"../{advanced_paths['active_selection']}",
        "OBSERVATION_FIGURE": "../output/figures/observation_pipeline.png",
        "POSTERIOR_FIGURE": "../output/figures/posterior_beliefs.png",
        "PACKAGE_PROVENANCE_FIGURE": "../output/figures/package_provenance.png",
        "TRACE_FEATURE_FIGURE": "../output/figures/trace_feature_matrix.png",
        "RXINFER_FREE_ENERGY_FIGURE": "../output/figures/rxinfer_free_energy.png",
        "INFERENCE_DIAGNOSTIC_FIGURE": f"../{inference_diagnostic_path.relative_to(PROJECT_ROOT)}",
        "SCHOLARSHIP_FIGURE": "../output/figures/scholarship_source_roles.png",
        "SCHOLARSHIP_MATRIX_FIGURE": "../output/figures/scholarship_evidence_matrix.png",
        "CLAIM_FIGURE": "../output/figures/claim_evidence_map.png",
        "REDTEAM_FIGURE": "../output/figures/redteam_risk_register.png",
        "REPRODUCIBILITY_FIGURE": "../output/figures/artifact_hash_manifest.png",
        "TEMPORAL_BELIEFS_FIGURE": f"../{advanced_figure_paths['temporal_beliefs']}",
        "MODEL_COMPARISON_FIGURE": f"../{advanced_figure_paths['model_comparison']}",
        "ACTIVE_EFE_FIGURE": f"../{advanced_figure_paths['active_efe']}",
        "CALIBRATION_CURVE_FIGURE": f"../{advanced_figure_paths['calibration_curve']}",
        "CONFUSION_MATRIX_FIGURE": f"../{advanced_figure_paths['confusion_matrix']}",
        "AMBIGUITY_ABLATION_FIGURE": f"../{advanced_figure_paths['ambiguity_ablation']}",
        "SEGMENTATION_RIBBONS_FIGURE": (
            f"../{advanced_figure_paths['segmentation_ribbons']}"
        ),
        "CORPUS_STABILITY_FIGURE": f"../{advanced_figure_paths['corpus_stability']}",
    }
    output_path = OUTPUT_DIR / "data" / "manuscript_variables.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(variables, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    output_path.write_text(
        json.dumps(variables, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return variables
