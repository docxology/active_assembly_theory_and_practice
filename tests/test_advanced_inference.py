"""Tests for the v0.2 advanced-inference orchestration over the corpus."""

from __future__ import annotations

import json

import pytest

from src.advanced_inference import (
    active_selection_payload,
    evaluation_report_payload,
    phase_segments_payload,
    run_advanced_inference,
    temporal_inference_payload,
)
from src.features import HIDDEN_STATE_LABELS
from src.figures import (
    plot_ambiguity_ablation,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_corpus_stability,
    plot_decoder_calibration,
    plot_generative_model,
    plot_phase_recovery,
    plot_segmentation_ribbons,
)


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


def test_evaluation_report_temporal_beats_static_and_controls_are_near_chance() -> None:
    payload = evaluation_report_payload()
    summary = payload["summary"]
    mean_temporal = summary["mean_temporal_accuracy"]
    mean_static = summary["mean_static_accuracy"]
    # The proper temporal model must beat the implicit-identity static filter.
    assert mean_temporal > mean_static
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
            "static_accuracy": 0.3,
            "viterbi_accuracy": 0.5,
            "temporal_minus_memoryless": -0.1,
        },
        {
            "blend": 1.0,
            "temporal_accuracy": 0.6,
            "memoryless_accuracy": 0.3,
            "static_accuracy": 0.2,
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
        "reference": {"temporal": 0.667, "static": 0.389, "memoryless": 0.722},
        "per_holdout": [
            {
                "held_out_fixture": "x86_accumulator",
                "temporal": 0.643,
                "static": 0.429,
                "memoryless": 0.714,
            },
            {
                "held_out_fixture": "x86_bitwise",
                "temporal": 0.714,
                "static": 0.357,
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
            "static": {"brier": 0.74, "ece": 0.36, "n_steps": 18},
        },
        "best_calibrated": "memoryless",
        "worst_calibrated": "static",
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
