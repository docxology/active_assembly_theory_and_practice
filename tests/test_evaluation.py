"""Tests for the quantitative evaluation and negative controls (no mocks)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.corpus import build_corpus, fixture_observation_ids
from src.evaluation import (
    calibration,
    compare_models,
    evaluate_decoding,
    evaluation_payload,
    free_energy_convergence,
    shuffled_label_control,
    uniform_belief_control,
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
        predicted.extend(forward_filter([int(o) for o in ids], model).most_likely_states)
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
        predicted.extend(forward_filter([int(o) for o in ids], model).most_likely_states)
        truth.extend(fixture.phase_labels)
    temporal_balanced = float(evaluate_decoding(predicted, truth, labels)["balanced_accuracy"])

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
        assert set(entry["calibration"].keys()) == {"brier", "ece"}


def test_evaluation_payload_is_json_serializable() -> None:
    corpus = build_corpus()
    model = build_temporal_model()
    payload = evaluation_payload(corpus, model)
    # Must not raise.
    json.dumps(payload, sort_keys=True)
