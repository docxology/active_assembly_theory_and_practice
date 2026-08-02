"""Tests for the quantitative evaluation and negative controls (no mocks)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.corpus import build_corpus, fixture_observation_ids
from src.evaluation import (
    ambiguity_ablation,
    calibration,
    calibration_bins,
    compare_models,
    corpus_stability_analysis,
    decoder_calibration,
    evaluate_decoding,
    evaluate_segmentation,
    evaluation_payload,
    free_energy_convergence,
    per_phase_recovery_metrics,
    pooled_decoding_accuracies,
    segmentation_evaluation,
    shuffled_label_control,
    uniform_belief_control,
    wilson_ci,
)
from src.features import HIDDEN_STATE_LABELS
from src.temporal_model import build_temporal_model, forward_filter


# --------------------------------------------------------------------------- #
# evaluate_decoding
# --------------------------------------------------------------------------- #
def test_isc26_evaluate_decoding_hand_computed_small_example() -> None:
    """ISC-26: exact match against a fully hand-computed example."""

    predicted = ["a", "b", "a"]
    truth = ["a", "a", "b"]
    labels = ["a", "b"]
    result = evaluate_decoding(predicted, truth, labels)

    # Only index 0 matches -> 1/3.
    assert result["accuracy"] == pytest.approx(1.0 / 3.0)
    # truth 'a' at idx0(pred a), idx1(pred b); truth 'b' at idx2(pred a).
    assert result["confusion"] == {
        "a": {"a": 1, "b": 1},
        "b": {"a": 1, "b": 0},
    }
    # recall(a)=1/2, recall(b)=0/1 -> mean 0.25.
    assert result["balanced_accuracy"] == pytest.approx(0.25)


def test_evaluate_decoding_perfect_prediction() -> None:
    labels = ["x", "y", "z"]
    seq = ["x", "y", "z", "x"]
    result = evaluate_decoding(seq, seq, labels)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_evaluate_decoding_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_decoding(["a"], ["a", "b"], ["a", "b"])


def test_evaluate_decoding_empty_truth_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_decoding([], [], ["a"])


def test_evaluate_decoding_confusion_covers_all_labels() -> None:
    labels = ["a", "b", "c"]
    result = evaluate_decoding(["a", "a"], ["a", "b"], labels)
    confusion = result["confusion"]
    assert set(confusion.keys()) == set(labels)
    for row in confusion.values():
        assert set(row.keys()) == set(labels)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_calibration_perfect_confident_predictor() -> None:
    """One-hot posteriors that are always correct -> brier 0, ece 0."""

    posteriors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    truth = [0, 1, 0]
    result = calibration(posteriors, truth)
    assert result["brier"] == pytest.approx(0.0)
    assert result["ece"] == pytest.approx(0.0)


def test_calibration_confident_but_wrong() -> None:
    """Confidence 1.0 but always wrong -> ece 1.0, brier 2.0."""

    posteriors = [[1.0, 0.0], [0.0, 1.0]]
    truth = [1, 0]
    result = calibration(posteriors, truth)
    # Each step: (1-0)^2 + (0-1)^2 = 2 -> mean 2.0.
    assert result["brier"] == pytest.approx(2.0)
    # Confidence 1.0, accuracy 0.0 in the top bin.
    assert result["ece"] == pytest.approx(1.0)


def test_calibration_brier_hand_computed_soft_posterior() -> None:
    posteriors = [[0.6, 0.4]]
    truth = [0]
    result = calibration(posteriors, truth)
    # (0.6-1)^2 + (0.4-0)^2 = 0.16 + 0.16 = 0.32.
    assert result["brier"] == pytest.approx(0.32)


def test_calibration_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        calibration([[1.0, 0.0]], [0, 1])


def test_calibration_empty_raises() -> None:
    with pytest.raises(ValueError):
        calibration([], [])


# --------------------------------------------------------------------------- #
# free_energy_convergence
# --------------------------------------------------------------------------- #
def test_free_energy_convergence_monotone_decrease() -> None:
    result = free_energy_convergence([3.0, 2.0, 1.0, 0.5])
    assert result["monotone_decrease_fraction"] == pytest.approx(1.0)
    assert result["delta"] == pytest.approx(-2.5)
    assert result["initial"] == pytest.approx(3.0)
    assert result["final"] == pytest.approx(0.5)


def test_free_energy_convergence_mixed() -> None:
    # Steps: 5->4 (decrease), 4->6 (increase), 6->6 (non-increase, equal).
    result = free_energy_convergence([5.0, 4.0, 6.0, 6.0])
    assert result["monotone_decrease_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["delta"] == pytest.approx(1.0)


def test_free_energy_convergence_single_step() -> None:
    result = free_energy_convergence([1.5])
    assert result["monotone_decrease_fraction"] == pytest.approx(1.0)
    assert result["delta"] == pytest.approx(0.0)
    assert result["initial"] == pytest.approx(1.5)
    assert result["final"] == pytest.approx(1.5)


def test_free_energy_convergence_empty_raises() -> None:
    with pytest.raises(ValueError):
        free_energy_convergence([])


# --------------------------------------------------------------------------- #
# compare_models
# --------------------------------------------------------------------------- #
def test_compare_models_keys_and_ranges() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    result = compare_models(corpus, model)
    assert set(result.keys()) == {fixture.name for fixture in corpus}
    for entry in result.values():
        assert set(entry.keys()) == {
            "static_accuracy",
            "memoryless_accuracy",
            "temporal_accuracy",
            "viterbi_accuracy",
        }
        for value in entry.values():
            assert 0.0 <= value <= 1.0


def test_memoryless_decoder_beats_temporal_on_this_corpus() -> None:
    """Honest comparator (Forge cross-vendor finding): on the short-program
    corpus a memoryless per-step decoder is the strongest non-temporal baseline
    and outperforms the temporal HMM, so temporal structure earns no net
    advantage here. This is recorded as a measured fact, not hidden."""

    corpus = build_corpus()
    model = build_temporal_model()
    result = compare_models(corpus, model)
    mean_memoryless = np.mean([e["memoryless_accuracy"] for e in result.values()])
    mean_temporal = np.mean([e["temporal_accuracy"] for e in result.values()])
    mean_static = np.mean([e["static_accuracy"] for e in result.values()])
    # The temporal HMM beats the sticky filter but not the memoryless decoder.
    assert mean_temporal > mean_static
    assert mean_memoryless > mean_temporal


def test_compare_models_temporal_beats_static_on_average() -> None:
    """The temporal model should not be worse than the sticky static filter."""

    corpus = build_corpus()
    model = build_temporal_model()
    result = compare_models(corpus, model)
    mean_temporal = np.mean([e["temporal_accuracy"] for e in result.values()])
    mean_static = np.mean([e["static_accuracy"] for e in result.values()])
    assert mean_temporal > mean_static


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def _concatenated_temporal_accuracy() -> float:
    corpus = build_corpus()
    model = build_temporal_model()
    labels = list(HIDDEN_STATE_LABELS)
    predicted: list[str] = []
    truth: list[str] = []
    for fixture in corpus:
        ids = fixture_observation_ids(fixture)
        predicted.extend(
            forward_filter([int(o) for o in ids], model).most_likely_states
        )
        truth.extend(fixture.phase_labels)
    return float(evaluate_decoding(predicted, truth, labels)["accuracy"])


def test_isc30_shuffled_control_far_below_best_model() -> None:
    """ISC-30: shuffled-label control is at least 0.15 below true-label accuracy."""

    corpus = build_corpus()
    model = build_temporal_model()
    shuffled = shuffled_label_control(corpus, model)
    best = _concatenated_temporal_accuracy()
    assert shuffled["mean_accuracy"] <= best - 0.15


def test_shuffled_control_is_deterministic() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    a = shuffled_label_control(corpus, model)
    b = shuffled_label_control(corpus, model)
    assert a == b


def test_shuffled_control_invalid_rounds_raises() -> None:
    with pytest.raises(ValueError):
        shuffled_label_control(build_corpus(), build_temporal_model(), rounds=0)


def test_isc31_uniform_predictor_worse_than_temporal() -> None:
    """ISC-31: constant-uniform predictor has strictly lower balanced accuracy."""

    corpus = build_corpus()
    model = build_temporal_model()
    labels = list(HIDDEN_STATE_LABELS)
    predicted: list[str] = []
    truth: list[str] = []
    for fixture in corpus:
        ids = fixture_observation_ids(fixture)
        predicted.extend(
            forward_filter([int(o) for o in ids], model).most_likely_states
        )
        truth.extend(fixture.phase_labels)
    temporal_balanced = float(
        evaluate_decoding(predicted, truth, labels)["balanced_accuracy"]
    )

    uniform = uniform_belief_control(corpus)
    assert uniform["balanced_accuracy"] < temporal_balanced


def test_uniform_belief_control_value() -> None:
    corpus = build_corpus()
    result = uniform_belief_control(corpus)
    # 5 hidden-state labels, only 'setup' recalled -> balanced 1/5.
    assert result["balanced_accuracy"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# evaluation_payload
# --------------------------------------------------------------------------- #
def test_isc32_evaluation_payload_byte_identical_across_runs() -> None:
    """ISC-32: payload is byte-identical across two json.dumps(sort_keys=True)."""

    corpus = build_corpus()
    model = build_temporal_model()
    first = json.dumps(evaluation_payload(corpus, model), sort_keys=True)
    second = json.dumps(evaluation_payload(corpus, model), sort_keys=True)
    assert first == second


# --------------------------------------------------------------------------- #
# wilson_ci / calibration bins / pooled statistics
# --------------------------------------------------------------------------- #
def test_wilson_ci_brackets_point_estimate() -> None:
    lo, hi = wilson_ci(10, 20)
    assert lo < 0.5 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_ci_narrows_with_sample_size() -> None:
    small = wilson_ci(5, 10)
    large = wilson_ci(50, 100)
    assert (small[1] - small[0]) > (large[1] - large[0])


def test_wilson_ci_perfect_accuracy_upper_near_one() -> None:
    lo, hi = wilson_ci(5, 5)
    assert hi > 0.9


def test_wilson_ci_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


def test_calibration_bins_reliability_shape() -> None:
    posteriors = [[0.7, 0.3], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]]
    truth = [0, 1, 0, 1]
    bins = calibration_bins(posteriors, truth)
    assert bins
    for record in bins:
        assert 0 <= record["confidence"] <= 1
        assert 0 <= record["accuracy"] <= 1
        assert record["count"] > 0


def test_calibration_returns_bins_and_n_steps() -> None:
    posteriors = [[0.8, 0.2], [0.6, 0.4]]
    truth = [0, 0]
    result = calibration(posteriors, truth)
    assert result["brier"] == pytest.approx(0.2)
    assert result["n_steps"] == 2
    assert isinstance(result["bins"], list)
    assert result["bins"] == calibration_bins(posteriors, truth)


def test_calibration_bins_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        calibration_bins([[1.0, 0.0]], [0, 1])


def test_evaluation_payload_pooled_stats_consistent() -> None:
    payload = evaluation_payload(build_corpus(), build_temporal_model())
    summary = payload["summary"]
    # The reported CI brackets the pooled (micro) accuracy of the same estimator.
    assert (
        summary["temporal_ci95"][0]
        <= summary["pooled_temporal_accuracy"]
        <= summary["temporal_ci95"][1]
    )
    assert (
        summary["memoryless_ci95"][0]
        <= summary["pooled_memoryless_accuracy"]
        <= summary["memoryless_ci95"][1]
    )
    # Pooled confusion sums to the number of labelled instructions in the corpus.
    pooled = payload["pooled_confusion"]
    assert pooled["labels"] == list(HIDDEN_STATE_LABELS)
    total_truth = sum(len(fixture.phase_labels) for fixture in build_corpus())
    assert sum(sum(row) for row in pooled["matrix"]) == total_truth
    assert isinstance(payload["pooled_calibration_bins"], list)
    # Shuffled control now reports dispersion around the mean.
    control = payload["shuffled_label_control"]
    assert control["std"] >= 0
    assert control["se"] >= 0
    # Determinism is preserved with the new pooled fields.
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        evaluation_payload(build_corpus(), build_temporal_model()), sort_keys=True
    )


def test_evaluation_payload_structure() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    payload = evaluation_payload(corpus, model)
    assert set(payload.keys()) == {
        "per_fixture",
        "compare_models",
        "shuffled_label_control",
        "uniform_belief_control",
        "summary",
        "pooled_confusion",
        "pooled_calibration_bins",
        "segmentation",
        "ambiguity_ablation",
        "corpus_stability",
        "phase_recovery",
        "decoder_calibration",
    }
    assert set(payload["per_fixture"].keys()) == {f.name for f in corpus}
    summary = payload["summary"]
    assert summary["fixture_count"] == len(corpus)
    assert summary["mean_temporal_accuracy"] >= summary["mean_static_accuracy"]


def test_evaluation_payload_per_fixture_fields() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    payload = evaluation_payload(corpus, model)
    for fixture in corpus:
        entry = payload["per_fixture"][fixture.name]
        assert entry["truth_labels"] == list(fixture.phase_labels)
        assert len(entry["observation_ids"]) == len(fixture.phase_labels)
        assert len(entry["temporal_labels"]) == len(fixture.phase_labels)
        assert "calibration" in entry
        assert "free_energy_convergence" in entry
        # calibration now also surfaces per-bin data used by the reliability diagram.
        assert set(entry["calibration"].keys()) == {"brier", "ece", "bins", "n_steps"}


def test_evaluation_payload_is_json_serializable() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    payload = evaluation_payload(corpus, model)
    # Must not raise.
    json.dumps(payload, sort_keys=True)


# --------------------------------------------------------------------------- #
# segmentation evaluation / ambiguity ablation
# --------------------------------------------------------------------------- #
def test_evaluate_segmentation_hand_computed() -> None:
    predicted = ["a", "a", "b", "b", "c"]
    truth = ["a", "a", "b", "b", "b"]
    result = evaluate_segmentation(predicted, truth)
    assert result["predicted_boundaries"] == [2, 4]
    assert result["true_boundaries"] == [2]
    assert result["boundary_precision"] == pytest.approx(0.5)
    assert result["boundary_recall"] == pytest.approx(1.0)
    assert result["boundary_f1"] == pytest.approx(2 / 3)
    assert result["step_accuracy"] == pytest.approx(0.8)


def test_evaluate_segmentation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_segmentation(["a"], ["a", "b"])


def test_segmentation_evaluation_pooled_is_measured() -> None:
    result = segmentation_evaluation(build_corpus(), build_temporal_model())
    pooled = result["pooled"]
    assert pooled["boundary_f1"] > 0
    assert pooled["step_accuracy"] > 0
    per_fixture = result["per_fixture"]
    assert set(per_fixture.keys()) == {f.name for f in build_corpus()}
    # Per-fixture entries carry the step sequences used to compute the metrics.
    for entry in per_fixture.values():
        assert len(entry["predicted_labels"]) == len(entry["truth_labels"]) > 0


def test_ambiguity_ablation_custom_blends() -> None:
    result = ambiguity_ablation(
        build_corpus(), build_temporal_model(), blends=(0.0, 1.0)
    )
    assert [round(row["blend"], 3) for row in result["rows"]] == [0.0, 1.0]


def test_pooled_decoding_accuracies_match_summary() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    pooled = pooled_decoding_accuracies(corpus, model)
    summary = evaluation_payload(corpus, model)["summary"]
    assert pooled["temporal"] == pytest.approx(summary["pooled_temporal_accuracy"])
    assert pooled["static"] == pytest.approx(summary["pooled_static_accuracy"])
    assert pooled["memoryless"] == pytest.approx(summary["pooled_memoryless_accuracy"])


def test_corpus_stability_honest_and_deterministic() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    result = corpus_stability_analysis(corpus, model)
    assert len(result["per_holdout"]) == len(corpus)
    # The headline result survives every single-fixture removal.
    assert result["all_holdouts_temporal_above_static"] is True
    assert result["all_holdouts_temporal_above_chance"] is True
    range_ = result["temporal_pooled_range"]
    assert 0.0 < range_["span"] < 0.5
    assert 0.0 < range_["min"] <= range_["max"] < 1.0
    # The full-corpus reference matches the pooled breakdown of the payload.
    assert result["reference"]["temporal"] == pytest.approx(
        evaluation_payload(corpus, model)["summary"]["pooled_temporal_accuracy"]
    )
    # Deterministic.
    assert json.dumps(result, sort_keys=True) == json.dumps(
        corpus_stability_analysis(corpus, model), sort_keys=True
    )


def test_corpus_stability_holdout_fixtures_named() -> None:
    corpus = build_corpus()
    result = corpus_stability_analysis(corpus, build_temporal_model())
    names = {row["held_out_fixture"] for row in result["per_holdout"]}
    assert names == {f.name for f in corpus}


def test_per_phase_recovery_metrics() -> None:
    payload = evaluation_payload(build_corpus(), build_temporal_model())
    phase_recovery = payload["phase_recovery"]
    assert set(phase_recovery["per_phase"].keys()) == set(HIDDEN_STATE_LABELS)
    assert 0 < phase_recovery["macro_f1"] <= 1.0
    assert phase_recovery["hardest_phase"] in HIDDEN_STATE_LABELS
    # Best-recall phase is fully recovered; the hardest phase is underdetermined.
    best = phase_recovery["best_recall_phase"]
    assert phase_recovery["per_phase"][best]["recall"] >= 1.0
    hardest = phase_recovery["hardest_phase"]
    assert phase_recovery["per_phase"][hardest]["recall"] < 0.5
    # Deterministic and consistent with the reported confusion matrix.
    assert json.dumps(phase_recovery, sort_keys=True) == json.dumps(
        per_phase_recovery_metrics(payload["pooled_confusion"]), sort_keys=True
    )


def test_per_phase_recovery_degenerate_all_zero() -> None:
    confusion: dict[str, object] = {
        "labels": list(HIDDEN_STATE_LABELS),
        "matrix": [
            [0] * len(HIDDEN_STATE_LABELS) for _ in range(len(HIDDEN_STATE_LABELS))
        ],
    }
    phase_recovery = per_phase_recovery_metrics(confusion)
    assert phase_recovery["macro_f1"] == 0.0
    for phase in HIDDEN_STATE_LABELS:
        assert phase_recovery["per_phase"][phase]["f1"] == 0.0


def test_decoder_calibration_ordering() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    payload = evaluation_payload(corpus, model)
    calibration_payload = payload["decoder_calibration"]
    decoders = calibration_payload["decoders"]
    # On this corpus calibration tracks accuracy: memoryless < temporal < static.
    assert decoders["memoryless"]["ece"] < decoders["temporal"]["ece"]
    assert decoders["temporal"]["ece"] < decoders["static"]["ece"]
    assert calibration_payload["best_calibrated"] == "memoryless"
    assert calibration_payload["worst_calibrated"] == "static"
    # Deterministic and consistent with the live computation.
    assert json.dumps(calibration_payload, sort_keys=True) == json.dumps(
        decoder_calibration(corpus, model), sort_keys=True
    )


def test_ambiguity_ablation_mechanism() -> None:
    result = ambiguity_ablation(build_corpus(), build_temporal_model())
    rows = result["rows"]
    assert len(rows) == 11
    first = rows[0]
    assert first["blend"] == pytest.approx(0.0)
    # No net advantage at the authored, informative likelihood.
    assert first["temporal_minus_memoryless"] <= 1e-6
    # Positive advantage once observations become ambiguous.
    assert result["crossover_blend"] is not None
    last = rows[-1]
    assert last["temporal_accuracy"] > last["memoryless_accuracy"]
    assert last["temporal_minus_memoryless"] > 0
    # Segmentation follows the same mechanism: memoryless boundary recovery is at
    # least as good at the authored likelihood but collapses under ambiguity while
    # the smoothed temporal segmentation retains/improves.
    assert (
        first["temporal_segmentation_f1"] <= first["memoryless_segmentation_f1"] + 1e-6
    )
    assert last["temporal_segmentation_f1"] > last["memoryless_segmentation_f1"]
    # Active layer: never worse at every blend, advantage appears somewhere in the
    # sweep, and is absent at the authored likelihood.
    assert result["active_never_worse"] is True
    assert result["active_max_advantage_fraction"] > 0
    assert result["authored_blend_active_advantage"] == 0
    band = result["active_advantage_band"]
    assert len(band) == 2 and band[0] < band[1]
    # Deterministic.
    assert json.dumps(result, sort_keys=True) == json.dumps(
        ambiguity_ablation(build_corpus(), build_temporal_model()), sort_keys=True
    )
