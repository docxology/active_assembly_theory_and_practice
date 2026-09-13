"""Direct regression coverage for the co-primary quotient classifier portfolio."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import src.quotient_classifier as quotient_module

from src.quotient_classifier import (
    INTERVENTIONS,
    MODEL_FAMILIES,
    PairExample,
    RNN_HIDDEN_SIZE,
    SURFACE_BASELINE_NAMES,
    _adjusted_mutual_information,
    _baseline_matrix,
    _baseline_vocabulary,
    _binary_cross_entropy,
    _cluster_bootstrap,
    _code_descriptor,
    _dataset,
    _evaluate,
    _feature_names,
    _fit_logistic,
    _fit_markov,
    _fit_rnn,
    _holm_adjust,
    _interval,
    _metrics,
    _paired_cluster_bootstrap,
    _pad_nops,
    _pair_features,
    _predict_logistic,
    _predict_markov,
    _predict_rnn,
    _rnn_forward,
    _rnn_gradient_check,
    _seed_sensitivity,
    _sigmoid,
    _surface_indices,
    _token_vocabulary,
    _trace_summary,
    _wilson_interval,
    _family_specs,
    quotient_classifier_payload,
    validate_quotient_classifier_payload,
)


def _rows() -> tuple[PairExample, ...]:
    """Return a tiny two-class label-blind corpus for mathematical helpers."""

    return (
        PairExample(
            family_id="family_a",
            split="training",
            variant_role="equivalent",
            variant_index=0,
            label=1,
            reference_tokens=("ref", "add"),
            variant_tokens=("var", "add"),
            feature_vector=(1.0, 0.0, 1.0, 0.0),
            reference_code=bytes.fromhex("48 83 c0 01"),
            variant_code=bytes.fromhex("48 ff c0"),
        ),
        PairExample(
            family_id="family_b",
            split="training",
            variant_role="near_miss",
            variant_index=1,
            label=0,
            reference_tokens=("ref", "sub"),
            variant_tokens=("var", "sub"),
            feature_vector=(0.0, 1.0, 0.0, 1.0),
            reference_code=bytes.fromhex("48 83 c0 01"),
            variant_code=bytes.fromhex("48 83 e8 01"),
        ),
        PairExample(
            family_id="family_c",
            split="held_out",
            variant_role="equivalent",
            variant_index=2,
            label=1,
            reference_tokens=("ref", "nop"),
            variant_tokens=("var", "nop"),
            feature_vector=(1.0, 1.0, 0.0, 0.0),
            reference_code=b"\x90",
            variant_code=b"\x90\x90",
        ),
        PairExample(
            family_id="family_d",
            split="held_out",
            variant_role="near_miss",
            variant_index=3,
            label=0,
            reference_tokens=("ref", "cmp"),
            variant_tokens=("var", "cmp"),
            feature_vector=(0.0, 0.0, 1.0, 1.0),
            reference_code=bytes.fromhex("48 83 f8 00"),
            variant_code=bytes.fromhex("48 83 c0 01"),
        ),
    )


def test_family_catalog_and_trace_summary_are_explicit_and_bounded() -> None:
    families = _family_specs()
    assert len(families) == 64
    assert {family.split for family in families} == {
        "training",
        "development",
        "held_out",
    }
    assert len(INTERVENTIONS) == 8
    dataset = _dataset()
    assert len(dataset) == 4096
    assert {row.split for row in dataset} == {"training", "development", "held_out"}
    assert len(_feature_names()) > 20
    tokens, summary = _trace_summary(bytes.fromhex("48 83 c0 01"), 0)
    assert tokens
    assert summary["executed_instruction_count"] == 1.0
    assert _pad_nops(b"\x90", 2) == b"\x90\x90\x90"
    with pytest.raises(ValueError, match="NOP"):
        _pad_nops(b"\x90", -1)
    reference = {name: float(index) for index, name in enumerate(_feature_names())}
    variant = {name: value + 1.0 for name, value in reference.items()}
    assert len(_pair_features(reference, variant)) == 2 * len(_feature_names())


def test_all_model_families_and_statistics_run_on_small_real_rows() -> None:
    rows = _rows()
    vocabulary = _token_vocabulary(rows)
    x = np.asarray([row.feature_vector for row in rows], dtype=float)
    labels = np.asarray([row.label for row in rows], dtype=float)
    logistic = _fit_logistic(x, labels, seed=11)
    logistic_probabilities = _predict_logistic(x, logistic)
    markov = _fit_markov(rows, vocabulary)
    markov_probabilities = _predict_markov(rows, vocabulary, markov)
    rnn = _fit_rnn(rows, vocabulary, seed=11)
    rnn_probabilities = _predict_rnn(rows, vocabulary, rnn)
    assert all(
        probabilities.shape == (len(rows),)
        and np.all(np.isfinite(probabilities))
        and np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        for probabilities in (
            logistic_probabilities,
            markov_probabilities,
            rnn_probabilities,
        )
    )
    metrics = _metrics(labels, np.asarray([0.2, 0.8, 0.7, 0.1]))
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["selective_risk_coverage"]["selected_rows"] >= 0
    assert _metrics(np.ones(2), np.asarray([0.8, 0.9]))["negative_rows"] == 0
    assert _metrics(np.zeros(2), np.asarray([0.1, 0.2]))["positive_rows"] == 0
    empty_interval = _wilson_interval(0, 0)
    assert all(np.isnan(value) for value in empty_interval)
    assert _adjusted_mutual_information(np.asarray([]), np.asarray([])) == 0.0
    assert (
        _holm_adjust({"b": 0.02, "a": 0.01})["a"]
        <= _holm_adjust({"b": 0.02, "a": 0.01})["b"]
    )
    assert len(_interval([0.1, 0.2, 0.3])) == 2
    evaluated = _evaluate("tiny", rows, logistic_probabilities, seed=3)
    assert evaluated["family_count"] == 4
    cluster = _cluster_bootstrap(rows, logistic_probabilities, seed=3, replicates=3)
    paired = _paired_cluster_bootstrap(
        rows,
        logistic_probabilities,
        markov_probabilities,
        seed=3,
        replicates=3,
    )
    assert len(cluster["balanced_accuracy"]) == 3
    assert paired["family_count"] == 4


def test_surface_descriptors_gradient_and_seed_sensitivity_are_deterministic() -> None:
    rows = _rows()
    vocabulary = _baseline_vocabulary(rows)
    assert vocabulary
    descriptor = _code_descriptor(rows[0].reference_code, vocabulary)
    assert set(descriptor) == set(SURFACE_BASELINE_NAMES)
    for baseline in SURFACE_BASELINE_NAMES:
        matrix = _baseline_matrix(rows, baseline, vocabulary)
        assert matrix.shape[0] == len(rows)
        assert np.all(np.isfinite(matrix))
    surface = _surface_indices()
    assert surface
    token_vocabulary = _token_vocabulary(rows)
    x = np.asarray([row.feature_vector for row in rows], dtype=float)
    labels = np.asarray([row.label for row in rows], dtype=float)
    full_x = np.zeros((len(rows), 2 * len(_feature_names())), dtype=float)
    full_x[:, : x.shape[1]] = x
    sensitivity = _seed_sensitivity(
        rows[:2],
        rows[2:],
        full_x[:2],
        full_x[2:],
        labels[:2],
        token_vocabulary,
    )
    assert set(sensitivity) == {*MODEL_FAMILIES, "surface_baseline"}
    assert all(row["seed_count"] >= 2 for row in sensitivity.values())
    assert _binary_cross_entropy(0.5, 1) > 0.0
    ids = [
        token_vocabulary[token] for token in ("<REFERENCE>", *rows[0].reference_tokens)
    ]
    probability, states = _rnn_forward(
        ids,
        np.zeros((RNN_HIDDEN_SIZE, len(token_vocabulary))),
        np.zeros((RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE)),
        np.zeros(RNN_HIDDEN_SIZE),
        0.0,
    )
    assert probability == pytest.approx(0.5)
    assert len(states) == len(ids)
    assert _rnn_gradient_check(rows[0], token_vocabulary)["passed"] is True
    assert np.allclose(
        _sigmoid(np.asarray([-100.0, 0.0, 100.0])), [0.0, 0.5, 1.0], atol=1e-12
    )


def test_classifier_receipt_validation_rejects_core_drift() -> None:
    payload = quotient_classifier_payload()
    assert validate_quotient_classifier_payload(payload) == (True, "ok")
    cases = (
        ("schema_version", "1.0.0"),
        ("label_blind_features", False),
        ("split_overlap", True),
        ("models", {}),
        ("seed_sensitivity", {}),
        ("neural_gradient_check", {"passed": False}),
        ("negative_controls", {}),
        ("model_comparisons", {}),
        ("paired_comparisons", {}),
        ("baseline_models", {}),
        ("surface_baseline_comparisons", {}),
    )
    for key, value in cases:
        corrupted = copy.deepcopy(payload)
        corrupted[key] = value
        assert validate_quotient_classifier_payload(corrupted)[0] is False


def test_classifier_label_permutation_control_drift_fails_closed() -> None:
    """A permutation control at or above chance must fail the receipt gate."""

    payload = quotient_classifier_payload()
    corrupted = copy.deepcopy(payload)
    corrupted["negative_controls"]["label_permutation"]["balanced_accuracy"] = 0.9
    valid, detail = validate_quotient_classifier_payload(corrupted)
    assert valid is False
    assert "not below chance" in detail


def test_classifier_payload_recomputes_full_portfolio_when_cache_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the producer path, including raw baselines and Holm ledgers."""

    monkeypatch.setattr(
        quotient_module, "_cached_quotient_classifier_payload", lambda: None
    )
    payload = quotient_module.quotient_classifier_payload()
    valid, detail = validate_quotient_classifier_payload(payload)
    assert valid, detail
    assert payload["status"] == "validated_bounded"
    assert len(payload["surface_baseline_comparisons"]) == len(MODEL_FAMILIES)
    assert payload["neural_gradient_check"]["passed"] is True
    permutation = payload["negative_controls"]["label_permutation"]
    assert permutation["rows"] > 0
    # The label-permutation negative control must lose: at or below chance and
    # strictly below every co-primary model and the surface baseline.
    permutation_balanced_accuracy = permutation["balanced_accuracy"]
    assert permutation_balanced_accuracy < 0.5
    for name in (*MODEL_FAMILIES, "surface_baseline"):
        model_balanced_accuracy = payload["models"][name]["balanced_accuracy"]
        assert permutation_balanced_accuracy < model_balanced_accuracy, name
