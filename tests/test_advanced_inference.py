"""Tests for the current advanced-inference orchestration over the corpus."""

from __future__ import annotations

import json

import pytest

from src.advanced_inference import (
    PROJECT_ROOT,
    active_selection_payload,
    action_sensing_payload_current,
    cfg_evidence_payload,
    corpus_coverage_payload_current,
    evaluation_report_payload,
    held_out_model_payload,
    memory_evidence_payload,
    parameter_learning_payload_current,
    phase_segments_payload,
    run_advanced_inference,
    temporal_inference_payload,
)
from src.advanced_figures import write_advanced_figures
from src.features import HIDDEN_STATE_LABELS
from src.corpus import build_corpus
from src.disassembly import disassemble_bytes
from src.emulation import emulate_x86_64
from src.formal_verification import formal_verification_payload
from src.information_policy import active_information_payload
from src.intervention_analysis import (
    intervention_power_payload,
    validate_intervention_power_payload,
)
from src.multi_arch import cross_architecture_payload
from src.figures import (
    plot_ambiguity_ablation,
    plot_action_sensing,
    plot_active_information_frontier,
    plot_artifact_receipt_closure,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_cross_architecture_benchmark,
    plot_corpus_coverage,
    plot_corpus_stability,
    plot_decoder_calibration,
    plot_generative_model,
    plot_generative_adequacy,
    plot_formal_verification_receipt,
    plot_assembly_cfg,
    plot_intervention_power,
    plot_model_evidence,
    plot_memory_access_timeline,
    plot_observation_informativeness,
    plot_parameter_learning,
    plot_phase_recovery,
    plot_segmentation_ribbons,
    plot_transition_ablation,
    plot_trace_refinement,
)
from src.symbolic_semantics import symbolic_execute


def test_temporal_inference_payload_covers_every_fixture() -> None:
    payload = temporal_inference_payload()
    per_fixture = payload["per_fixture"]
    assert set(per_fixture) >= {"x86_accumulator", "x86_long_accumulator"}
    for record in per_fixture.values():
        n = len(record["observation_ids"])
        assert len(record["truth_labels"]) == n
        assert len(record["filter"]["filtered_posteriors"]) == n
        assert len(record["smoothing"]["smoothed_posteriors"]) == n
        assert len(record["viterbi_path"]) == n


def test_phase_segments_partition_each_fixture() -> None:
    payload = phase_segments_payload()
    for record in payload["per_fixture"].values():
        segments = record["segments"]
        assert segments, "expected at least one segment"
        # Segments must exactly tile the step range with no gaps or overlaps.
        covered: list[int] = []
        for segment in segments:
            covered.extend(range(segment["start_index"], segment["end_index"] + 1))
        assert covered == list(range(len(covered)))
        assert record["segment_count"] == len(segments)


def test_active_selection_is_never_worse_than_fixed() -> None:
    payload = active_selection_payload()
    summary = payload["summary"]
    # Honest corpus outcome: active selection is never WORSE than the fixed scan;
    # on these short fixtures it ties because one observation saturates confidence.
    assert summary["active_never_worse"] is True
    assert (
        summary["ties"] + summary["active_strictly_fewer_steps"]
        == summary["fixture_count"]
    )
    assert all(
        row["predictive_selection"]["policy"] == "predictive"
        for row in payload["per_fixture"].values()
    )


def test_held_out_model_payload_is_fixture_level_leakage_safe() -> None:
    """The generated supervised fit records training complements explicitly."""

    payload = held_out_model_payload()
    assert payload["fixture_count"] == len(payload["per_fixture"])
    assert all(
        row["held_out_fixture"] not in row["training_fixtures"]
        for row in payload["per_fixture"]
    )


def test_cfg_evidence_preserves_external_branch_target() -> None:
    """The branch fixture produces multiple blocks and an explicit external edge."""

    payload = cfg_evidence_payload()
    assert payload["sample_id"] == "x86_guard"
    assert len(payload["basic_blocks"]) >= 2
    assert any(
        edge["target_block_id"].startswith("external_") for edge in payload["cfg_edges"]
    )
    assert len(payload["dynamic_addresses"]) == 4


def test_assembly_cfg_figure_written_nonempty(tmp_path) -> None:
    """The CFG evidence is rendered as a deterministic, nonblank figure."""

    path = plot_assembly_cfg(cfg_evidence_payload(), tmp_path / "assembly_cfg.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_trace_refinement_figure_uses_real_symbolic_and_unicorn_rows(tmp_path) -> None:
    fixture = build_corpus()[0]
    report = symbolic_execute(
        disassemble_bytes(fixture.code), emulate_x86_64(fixture.code)
    )
    path = plot_trace_refinement(report.to_dict(), tmp_path / "trace_refinement.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_cross_architecture_figure_written_from_real_adapter_payload(tmp_path) -> None:
    path = plot_cross_architecture_benchmark(
        cross_architecture_payload(), tmp_path / "cross_architecture_benchmark.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_formal_verification_figure_written_from_compiled_receipt(tmp_path) -> None:
    path = plot_formal_verification_receipt(
        formal_verification_payload(), tmp_path / "formal_verification_receipt.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_active_information_frontier_figure_written_from_exact_dp(tmp_path) -> None:
    path = plot_active_information_frontier(
        active_information_payload(), tmp_path / "active_information_frontier.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_intervention_power_receipt_and_figure_are_exact(tmp_path) -> None:
    payload = intervention_power_payload()
    valid, detail = validate_intervention_power_payload(payload)
    assert valid, detail
    assert payload["subset_count"] == 15
    assert payload["minimal_separating_cardinality"] == 1
    assert len(payload["minimal_separating_sets"]) == 3
    negative = payload["negative_control"]
    assert negative["near_miss_separation"] == 0.625
    assert len(negative["unresolved_near_miss_families"]) == 3
    path = plot_intervention_power(payload, tmp_path / "intervention_power.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_advanced_figure_registry_includes_open_roadmap_surfaces() -> None:
    paths = write_advanced_figures()
    required = {
        "quotient_classifier",
        "compiler_transfer",
        "formal_correspondence",
        "sensor_costs",
        "replication_manifest",
        "intervention_power",
    }
    assert required.issubset(paths)
    for key in required:
        assert (PROJECT_ROOT / paths[key]).is_file()
        assert (PROJECT_ROOT / paths[key]).stat().st_size > 0


def test_artifact_receipt_closure_figure_is_nonblank(tmp_path) -> None:
    path = plot_artifact_receipt_closure(
        {
            "protocol_id": "test-protocol",
            "protocol_sha256": "a" * 64,
            "status_counts": {
                "implemented_local": 2,
                "active": 1,
                "blocked_by_evidence": 1,
                "planned": 1,
            },
        },
        tmp_path / "artifact_receipt_closure.png",
        artifact_count=12,
        artifact_error_count=0,
    )
    assert path.is_file()
    assert path.stat().st_size > 0


def test_memory_evidence_joins_static_operands_to_dynamic_events() -> None:
    payload = memory_evidence_payload()

    assert payload["sample_id"] == "x86_memory"
    assert payload["static_memory_instruction_indexes"] == [0, 1]
    assert payload["summary"] == {
        "distinct_addresses": 1,
        "dynamic_event_count": 2,
        "read_count": 1,
        "static_memory_instruction_count": 2,
        "write_count": 1,
    }
    assert [event["value"] for event in payload["dynamic_memory_events"]] == [7, 7]


def test_memory_evidence_figure_written_nonempty(tmp_path) -> None:
    path = plot_memory_access_timeline(
        memory_evidence_payload(), tmp_path / "assembly_memory_timeline.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_corpus_coverage_records_memory_and_alternate_path_variation() -> None:
    payload = corpus_coverage_payload_current()
    summary = payload["summary"]
    assert payload["schema_version"] == "1.0.0"
    assert summary["fixture_count"] == 6
    assert summary["memory_fixture_count"] == 1
    assert summary["alternate_path_fixture_count"] == 1
    assert summary["repeated_phase_fixture_count"] == 4
    assert summary["unique_instruction_family_count"] == 5
    assert summary["unique_memory_mode_count"] == 1
    assert summary["unique_branch_target_class_count"] == 2


def test_action_sensing_payload_is_held_out_and_policy_complete() -> None:
    payload = action_sensing_payload_current()
    assert payload["schema_version"] == "1.0.0"
    assert payload["summary"]["fixture_count"] == 6
    assert payload["summary"]["leakage_free"] is True
    assert set(payload["policies"]) == {"active", "fixed", "random", "oracle"}
    assert payload["summary"]["action_pool_changes_evidence"] is True


def test_action_sensing_figure_written_nonempty(tmp_path) -> None:
    path = plot_action_sensing(
        action_sensing_payload_current(), tmp_path / "action_sensing.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_parameter_learning_payload_is_leakage_safe_and_figure_is_nonempty(
    tmp_path,
) -> None:
    payload = parameter_learning_payload_current()
    assert payload["schema_version"] == "2.0.0"
    assert payload["phase_labels_used_for_fit"] is False
    assert payload["summary"]["leakage_free"] is True
    path = plot_parameter_learning(payload, tmp_path / "parameter_learning.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_corpus_coverage_figure_written_nonempty(tmp_path) -> None:
    path = plot_corpus_coverage(
        corpus_coverage_payload_current(), tmp_path / "corpus_coverage.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_evaluation_report_temporal_beats_identity_transition_and_controls_are_near_chance() -> (
    None
):
    payload = evaluation_report_payload()
    summary = payload["summary"]
    mean_temporal = summary["mean_temporal_accuracy"]
    mean_identity_transition = summary["mean_identity_transition_accuracy"]
    # The proper temporal model must beat the implicit-identity comparator.
    assert mean_temporal > mean_identity_transition
    # Negative controls must sit well below the temporal model (discriminative metric).
    shuffle = payload["shuffled_label_control"]["mean_accuracy"]
    uniform = payload["uniform_belief_control"]["balanced_accuracy"]
    assert mean_temporal - shuffle >= 0.15
    assert mean_temporal > uniform


def test_run_advanced_inference_writes_deterministic_artifacts(tmp_path) -> None:
    paths_a = run_advanced_inference(tmp_path / "output")
    first = {
        name: (tmp_path / "output" / "data" / f"{name}.json").read_text(
            encoding="utf-8"
        )
        for name in (
            "temporal_inference",
            "phase_segments",
            "active_selection",
            "evaluation_report",
            "cfg_evidence",
            "memory_evidence",
            "held_out_model",
            "generative_adequacy",
            "typed_observations",
            "factorized_observation_model",
            "corpus_coverage",
            "action_sensing",
            "parameter_learning",
            "cross_architecture_fixtures",
            "active_information_frontier",
            "quotient_invariance_protocol",
            "representation_invariance_benchmark",
            "intervention_power",
            "scientific_program",
        )
    }
    run_advanced_inference(tmp_path / "output")
    second = {
        name: (tmp_path / "output" / "data" / f"{name}.json").read_text(
            encoding="utf-8"
        )
        for name in first
    }
    # Byte-identical across runs (determinism, no RNG).
    assert first == second
    # Every payload is valid JSON.
    for name in first:
        assert json.loads(first[name])
    assert set(paths_a) == {
        "temporal_inference",
        "phase_segments",
        "active_selection",
        "evaluation_report",
        "cfg_evidence",
        "memory_evidence",
        "held_out_model",
        "generative_adequacy",
        "typed_observations",
        "factorized_observation_model",
        "corpus_coverage",
        "action_sensing",
        "parameter_learning",
        "cross_architecture_fixtures",
        "active_information_frontier",
        "quotient_invariance_protocol",
        "representation_invariance_benchmark",
        "intervention_power",
        "scientific_program",
        "research_protocol",
        "quotient_classifier",
        "formal_trace_correspondence",
        "compiler_transfer",
        "sensor_cost_receipt",
        "replication_manifest",
        "theorem_or_countertheorem",
    }


def test_calibration_curve_written_nonempty(tmp_path) -> None:
    bins = [
        {
            "bin_index": i,
            "count": 4,
            "confidence": 0.3 + 0.1 * i,
            "accuracy": 0.3 + 0.1 * i,
        }
        for i in range(6)
    ]
    path = plot_calibration_curve(bins, tmp_path / "calibration_curve.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_calibration_curve_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_calibration_curve([], tmp_path / "calibration_curve.png")


def test_confusion_matrix_written_nonempty(tmp_path) -> None:
    labels = list(HIDDEN_STATE_LABELS)
    n = len(labels)
    matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    path = plot_confusion_matrix(
        {"labels": labels, "matrix": matrix}, tmp_path / "confusion_matrix.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_confusion_matrix_shape_mismatch_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_confusion_matrix(
            {"labels": list(HIDDEN_STATE_LABELS), "matrix": [[1, 2]]},
            tmp_path / "confusion_matrix.png",
        )


def test_ambiguity_ablation_figure_written_nonempty(tmp_path) -> None:
    rows = [
        {
            "blend": 0.0,
            "temporal_accuracy": 0.6,
            "memoryless_accuracy": 0.7,
            "identity_transition_accuracy": 0.3,
            "viterbi_accuracy": 0.5,
            "temporal_minus_memoryless": -0.1,
        },
        {
            "blend": 1.0,
            "temporal_accuracy": 0.6,
            "memoryless_accuracy": 0.3,
            "identity_transition_accuracy": 0.2,
            "viterbi_accuracy": 0.5,
            "temporal_minus_memoryless": 0.3,
        },
    ]
    path = plot_ambiguity_ablation(
        {"rows": rows, "crossover_blend": 0.8}, tmp_path / "ambiguity_ablation.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_ambiguity_ablation_figure_empty_rows_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_ambiguity_ablation({"rows": []}, tmp_path / "ambiguity_ablation.png")


def test_segmentation_ribbons_figure_written_nonempty(tmp_path) -> None:
    report: dict[str, object] = {
        "segmentation": {
            "per_fixture": {
                "x86_a": {
                    "truth_labels": ["setup", "accumulator_update", "exit"],
                    "predicted_labels": [
                        "setup",
                        "accumulator_update",
                        "accumulator_update",
                    ],
                    "boundary_f1": 0.5,
                },
                "x86_b": {
                    "truth_labels": ["setup", "guard_check", "exit"],
                    "predicted_labels": ["setup", "setup", "exit"],
                    "boundary_f1": 0.5,
                },
            },
            "pooled": {"boundary_f1": 0.5},
        }
    }
    path = plot_segmentation_ribbons(report, tmp_path / "segmentation_ribbons.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_segmentation_ribbons_empty_per_fixture_raises(tmp_path) -> None:
    report: dict[str, object] = {"segmentation": {"per_fixture": {}}}
    with pytest.raises(ValueError):
        plot_segmentation_ribbons(report, tmp_path / "segmentation_ribbons.png")


def test_corpus_stability_figure_written_nonempty(tmp_path) -> None:
    stability: dict[str, object] = {
        "reference": {
            "temporal": 0.667,
            "identity_transition": 0.389,
            "memoryless": 0.722,
        },
        "per_holdout": [
            {
                "held_out_fixture": "x86_accumulator",
                "temporal": 0.643,
                "identity_transition": 0.429,
                "memoryless": 0.714,
            },
            {
                "held_out_fixture": "x86_bitwise",
                "temporal": 0.714,
                "identity_transition": 0.357,
                "memoryless": 0.714,
            },
        ],
        "temporal_pooled_range": {"min": 0.643, "max": 0.714, "span": 0.071},
        "chance_floor": 0.2,
    }
    path = plot_corpus_stability(stability, tmp_path / "corpus_stability.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_corpus_stability_figure_empty_raises(tmp_path) -> None:
    stability: dict[str, object] = {"per_holdout": [], "reference": {}}
    with pytest.raises(ValueError):
        plot_corpus_stability(stability, tmp_path / "corpus_stability.png")


def test_phase_recovery_figure_written_nonempty(tmp_path) -> None:
    phase_recovery: dict[str, object] = {
        "per_phase": {
            "setup": {"precision": 0.67, "recall": 1.0, "f1": 0.8, "true_count": 4},
            "guard_check": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_count": 2,
            },
        },
        "macro_f1": 0.4,
        "hardest_phase": "guard_check",
    }
    path = plot_phase_recovery(phase_recovery, tmp_path / "phase_recovery.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_phase_recovery_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_phase_recovery({"per_phase": {}}, tmp_path / "phase_recovery.png")


def test_decoder_calibration_figure_written_nonempty(tmp_path) -> None:
    calibration_payload: dict[str, object] = {
        "decoders": {
            "memoryless": {"brier": 0.46, "ece": 0.09, "n_steps": 18},
            "temporal": {"brier": 0.5, "ece": 0.19, "n_steps": 18},
            "identity_transition": {
                "brier": 0.74,
                "ece": 0.36,
                "n_steps": 18,
            },
        },
        "best_calibrated": "memoryless",
        "worst_calibrated": "identity_transition",
    }
    path = plot_decoder_calibration(
        calibration_payload, tmp_path / "decoder_calibration.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_decoder_calibration_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_decoder_calibration({"decoders": {}}, tmp_path / "decoder_calibration.png")


def test_generative_model_figure_written_nonempty(tmp_path) -> None:
    path = plot_generative_model(tmp_path / "generative_model.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_observation_informativeness_figure_written_nonempty(tmp_path) -> None:
    payload: dict[str, object] = {
        "observations": [
            {
                "observation": "load_immediate",
                "info_gain": 0.966,
                "best_phase": "setup",
            },
            {
                "observation": "control_transfer",
                "info_gain": 0.374,
                "best_phase": "exit",
            },
        ],
        "most_informative": "load_immediate",
        "least_informative": "control_transfer",
    }
    path = plot_observation_informativeness(
        payload, tmp_path / "observation_informativeness.png"
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_observation_informativeness_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_observation_informativeness(
            {"observations": []}, tmp_path / "observation_informativeness.png"
        )


def test_model_evidence_figure_written_nonempty(tmp_path) -> None:
    payload: dict[str, object] = {
        "log_evidence": {
            "temporal": -26.72,
            "identity_transition": -29.34,
            "memoryless": -27.47,
        },
        "best_model": "temporal",
        "temporal_margin_over_best_alternative": 0.74,
    }
    path = plot_model_evidence(payload, tmp_path / "model_evidence.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_model_evidence_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_model_evidence({"log_evidence": {}}, tmp_path / "model_evidence.png")


def test_transition_ablation_figure_written_nonempty(tmp_path) -> None:
    payload: dict[str, object] = {
        "rows": [
            {
                "blend": 0.0,
                "temporal_mean_accuracy": 0.667,
                "temporal_minus_memoryless": -0.056,
                "temporal_minus_identity_transition": 0.278,
            },
            {
                "blend": 1.0,
                "temporal_mean_accuracy": 0.722,
                "temporal_minus_memoryless": 0.0,
                "temporal_minus_identity_transition": 0.333,
            },
        ],
        "memoryless_mean_accuracy": 0.722,
        "identity_transition_mean_accuracy": 0.389,
        "convergence_blend": 0.4,
        "peak_blend": 0.2,
        "peak_temporal_accuracy": 0.778,
    }
    path = plot_transition_ablation(payload, tmp_path / "transition_ablation.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_transition_ablation_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_transition_ablation({"rows": []}, tmp_path / "transition_ablation.png")


def test_generative_adequacy_figure_written_nonempty(tmp_path) -> None:
    payload: dict[str, object] = {
        "replicates": 32,
        "per_fixture": [
            {
                "fixture": "x86_accumulator",
                "observed": {
                    "transition_rate": 0.67,
                    "run_count": 3,
                    "temporal_recovery_accuracy": 0.75,
                },
                "predictive": {
                    "transition_rate": {"lower": 0.2, "median": 0.5, "upper": 0.8},
                    "temporal_recovery_accuracy": {
                        "lower": 0.4,
                        "median": 0.7,
                        "upper": 0.95,
                    },
                },
            },
            {
                "fixture": "x86_guard",
                "observed": {
                    "transition_rate": 0.33,
                    "run_count": 2,
                    "temporal_recovery_accuracy": 0.5,
                },
                "predictive": {
                    "transition_rate": {"lower": 0.1, "median": 0.4, "upper": 0.7},
                    "temporal_recovery_accuracy": {
                        "lower": 0.3,
                        "median": 0.6,
                        "upper": 0.9,
                    },
                },
            },
        ],
    }
    path = plot_generative_adequacy(payload, tmp_path / "generative_adequacy.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_generative_adequacy_figure_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_generative_adequacy(
            {"per_fixture": []}, tmp_path / "generative_adequacy.png"
        )
