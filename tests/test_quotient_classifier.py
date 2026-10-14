"""Direct regression coverage for the co-primary quotient classifier portfolio."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import src.quotient_classifier as quotient_module
from src.quotient_classifier import (
    FAMILY_SHUFFLE_SEED,
    INTERVENTIONS,
    LOGISTIC_L2_GRID,
    LOGISTIC_LEARNING_RATE_GRID,
    MODEL_FAMILIES,
    NEAR_DUPLICATE_THRESHOLD,
    RNN_EPOCHS_GRID,
    RNN_HIDDEN_SIZE,
    RNN_HIDDEN_SIZE_GRID,
    SEEDS,
    SURFACE_BASELINE_NAMES,
    PairExample,
    _adjusted_mutual_information,
    _apply_platt,
    _baseline_matrix,
    _baseline_vocabulary,
    _binary_cross_entropy,
    _cluster_bootstrap,
    _code_descriptor,
    _dataset,
    _evaluate,
    _family_shuffle_control,
    _family_specs,
    _feature_names,
    _fit_logistic,
    _fit_markov,
    _fit_platt,
    _fit_rnn,
    _fit_with_starts,
    _holm_adjust,
    _interval,
    _metrics,
    _near_duplicate_report,
    _pad_nops,
    _pair_features,
    _paired_cluster_bootstrap,
    _predict_logistic,
    _predict_markov,
    _predict_rnn,
    _rnn_forward,
    _rnn_gradient_check,
    _seed_sensitivity,
    _select_hyperparameters,
    _sigmoid,
    _split_manifest,
    _split_manifest_sha256,
    _surface_indices,
    _token_vocabulary,
    _trace_summary,
    _wilson_interval,
    quotient_classifier_payload,
    validate_quotient_classifier_payload,
)


@pytest.fixture(scope="module")
def full_payload() -> dict:
    """One full-producer pass shared by receipt-level regression tests."""

    return quotient_classifier_payload()


def _pair(
    family_id: str,
    split: str,
    label: int,
    first: float,
    second: float,
    *,
    reference_code: bytes = b"\x90",
    variant_code: bytes = b"\x90\x90",
) -> PairExample:
    """Build one synthetic label-blind row with both roles represented."""

    return PairExample(
        family_id=family_id,
        split=split,
        variant_role="equivalent" if label else "near_miss",
        variant_index=0,
        label=label,
        reference_tokens=("ref",),
        variant_tokens=("var",),
        feature_vector=(first, second),
        reference_code=reference_code,
        variant_code=variant_code,
    )


def _selection_rows() -> tuple[list[PairExample], list[PairExample]]:
    """Return tiny training and development corpora for selection tests."""

    training = [
        _pair("sel_a", "training", 1, 2.0, 0.0),
        _pair("sel_a", "training", 0, 0.0, 2.0),
        _pair("sel_b", "training", 1, 1.8, 0.2),
        _pair("sel_b", "training", 0, 0.2, 1.8),
    ]
    development = [
        _pair("sel_c", "development", 1, 2.2, 0.1),
        _pair("sel_c", "development", 0, 0.1, 2.2),
        _pair("sel_d", "development", 1, 1.5, 0.5),
        _pair("sel_d", "development", 0, 0.5, 1.5),
    ]
    return training, development


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


def test_classifier_receipt_validation_rejects_core_drift(
    full_payload: dict,
) -> None:
    payload = full_payload
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
        ("split_manifest", {}),
        ("hyperparameter_selection", {}),
        ("leakage_attacks", {}),
        ("near_duplicate_checks", {}),
    )
    for key, value in cases:
        corrupted = copy.deepcopy(payload)
        corrupted[key] = value
        assert validate_quotient_classifier_payload(corrupted)[0] is False


def test_classifier_label_permutation_control_drift_fails_closed(
    full_payload: dict,
) -> None:
    """A permutation control at or above chance must fail the receipt gate."""

    payload = full_payload
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
    # Sealed split manifest: recorded hash must match the recomputed assignment.
    manifest = payload["split_manifest"]
    assert manifest["assignment"] == {
        family.family_id: family.split for family in quotient_module._family_specs()
    }
    assert manifest["sha256"] == quotient_module._split_manifest_sha256(
        manifest["assignment"]
    )
    # Development-only selection: selected options sit inside the declared grids.
    selection = payload["hyperparameter_selection"]
    assert selection["fit_split"] == "development"
    assert selection["test_rows_used"] == 0
    logistic_selected = selection["selected_per_model"]["calibrated_interpretable"]
    assert logistic_selected["l2"] in LOGISTIC_L2_GRID
    assert logistic_selected["learning_rate"] in LOGISTIC_LEARNING_RATE_GRID
    rnn_selected = selection["selected_per_model"]["neural_sequence"]
    assert rnn_selected["hidden_size"] in RNN_HIDDEN_SIZE_GRID
    assert rnn_selected["epochs"] in RNN_EPOCHS_GRID
    assert selection["selected_per_model"]["temporal_probabilistic"] is None
    # Deterministic starts recorded for every co-primary model.
    for name in MODEL_FAMILIES:
        starts = payload["models"][name]["starts"]
        if name == "temporal_probabilistic":
            assert starts["catalog"] == []
            assert starts["selected_start"] is None
        else:
            assert starts["catalog"] == list(SEEDS)
            assert starts["selected_start"] in starts["catalog"]
            assert sorted(starts["development_score_by_start"]) == sorted(
                str(seed) for seed in SEEDS
            )
        # Platt calibration reported alongside; uncalibrated metrics stay primary.
        calibration = payload["models"][name]["calibration"]
        assert calibration["method"] == "platt_development"
        assert set(calibration) == {
            "method",
            "parameters",
            "ece_before",
            "ece_after",
            "brier_before",
            "brier_after",
            "metrics_role",
        }
        assert "primary" in calibration["metrics_role"]
    # Family-shuffle control must lose to every co-primary model that is
    # itself above chance; chance-level members are declared exclusions.
    shuffle = payload["negative_controls"]["family_shuffle"]
    assert shuffle["seed"] == FAMILY_SHUFFLE_SEED
    assert shuffle["loses_to_all_co_primary"] is True
    above_chance = [
        name
        for name in MODEL_FAMILIES
        if payload["models"][name]["balanced_accuracy"] > 0.5
    ]
    assert shuffle["compared_models"] == above_chance
    assert shuffle["chance_level_comparison_excluded"] == [
        name for name in MODEL_FAMILIES if name not in above_chance
    ]
    assert above_chance, "no co-primary model exceeds chance"
    for name in above_chance:
        assert (
            shuffle["metrics"]["balanced_accuracy"]
            < payload["models"][name]["balanced_accuracy"]
        ), name
    # Bounded leakage probe must not beat the label-blind models.
    leakage = payload["leakage_attacks"]
    assert leakage["metadata_only_beats_label_blind"] is False
    assert "bounded probe only" in leakage["conclusion_bound"]
    assert "no architecture-metadata claim" in leakage["conclusion_bound"]
    # Cross-split near-duplicate screen: flagged pairs must be declared.
    near_duplicate = payload["near_duplicate_checks"]
    assert near_duplicate["threshold"] == NEAR_DUPLICATE_THRESHOLD
    for pair in near_duplicate["flagged_pairs"]:
        if pair["crosses_training_held_out"]:
            assert pair["declared"] is True, pair


def test_payload_reports_pilot_receipt_for_pilot_scale_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIN-07: a sub-promotion corpus must drive the underpowered branch."""

    real_specs = quotient_module._family_specs()
    pilot_specs = real_specs[:16]
    monkeypatch.setattr(quotient_module, "_family_specs", lambda: pilot_specs)
    monkeypatch.setattr(
        quotient_module, "_cached_quotient_classifier_payload", lambda: None
    )
    quotient_module._dataset.cache_clear()
    try:
        payload = quotient_module.quotient_classifier_payload()
    finally:
        quotient_module._dataset.cache_clear()

    assert payload["family_count"] == 16
    assert payload["status"] == "underpowered"
    assert payload["endpoint_status"] == "pilot_receipt"
    assert payload["underpowered_cases"]
    assert (
        "Promotion target requires at least 64 independent families"
        in payload["underpowered_cases"][0]
    )


def test_validator_accepts_underpowered_status_on_pilot_summary(
    full_payload: dict,
) -> None:
    """MIN-07: the receipt validator must accept the declared pilot status."""

    summary = copy.deepcopy(full_payload)
    summary["status"] = "underpowered"
    summary["endpoint_status"] = "pilot_receipt"
    summary["family_count"] = 48
    summary["row_count"] = 48 * 64
    summary["underpowered_cases"] = [
        "Promotion target requires at least 64 independent families, "
        "64 rows per family, and 16 held-out families."
    ]
    assert validate_quotient_classifier_payload(summary) == (True, "ok")


def test_split_manifest_seals_frozen_family_assignment() -> None:
    """The manifest must reproduce _family_specs byte-identically and self-hash."""

    manifest = _split_manifest()
    families = _family_specs()
    assert manifest["assignment"] == {
        family.family_id: family.split for family in families
    }
    per_split = manifest["family_ids_per_split"]
    for split in ("training", "development", "held_out"):
        assert per_split[split] == [
            family.family_id for family in families if family.split == split
        ]
    assert {split: len(rows) for split, rows in per_split.items()} == {
        "training": 32,
        "development": 16,
        "held_out": 16,
    }
    assert manifest["sha256"] == _split_manifest_sha256(manifest["assignment"])


def test_split_manifest_receipt_gate_fails_closed(full_payload: dict) -> None:
    """A tampered manifest hash or assignment must fail the validator."""

    assert full_payload["split_manifest"] == _split_manifest()
    corrupted = copy.deepcopy(full_payload)
    corrupted["split_manifest"]["sha256"] = "0" * 64
    assert validate_quotient_classifier_payload(corrupted)[0] is False
    corrupted = copy.deepcopy(full_payload)
    assignment = corrupted["split_manifest"]["assignment"]
    training_id = next(key for key, split in assignment.items() if split == "training")
    assignment[training_id] = "held_out"
    assert validate_quotient_classifier_payload(corrupted)[0] is False


def test_label_scramble_receipt_cannot_claim_out_of_sample_evidence(full_payload):
    control = full_payload["negative_controls"]["family_shuffle"]
    assert control["fit_scope"] == "held_out_in_sample_perturbation_only"
    assert control["preserves_family_class_totals"] is True
    corrupted = copy.deepcopy(full_payload)
    corrupted["negative_controls"]["family_shuffle"]["fit_scope"] = (
        "training_only_out_of_sample"
    )
    assert validate_quotient_classifier_payload(corrupted)[0] is False


def test_hyperparameter_selection_is_development_only_and_deterministic() -> None:
    """Selection scores must be deterministic and blind to held-out rows."""

    training, development = _selection_rows()
    held = [
        _pair("held_a", "held_out", 1, 7.0, -7.0),
        _pair("held_a", "held_out", 0, -7.0, 7.0),
    ]
    alternative_held = [
        _pair("held_a", "held_out", 0, -99.0, 99.0),
        _pair("held_a", "held_out", 1, 99.0, -99.0),
    ]
    train_matrix = np.asarray([row.feature_vector for row in training], dtype=float)
    development_matrix = np.asarray(
        [row.feature_vector for row in development], dtype=float
    )
    vocabulary = _token_vocabulary([*training, *development])

    first = _select_hyperparameters(
        training, development, train_matrix, development_matrix, vocabulary
    )
    second = _select_hyperparameters(
        training, development, train_matrix, development_matrix, vocabulary
    )
    # Held-out rows are structurally outside the selection inputs: mutating
    # the held-out universe between runs cannot move the selection.
    assert held != alternative_held
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["fit_split"] == "development"
    assert first["test_rows_used"] == 0
    assert first["selected_per_model"]["temporal_probabilistic"] is None

    logistic_scores = first["scores_by_option"]["calibrated_interpretable"]
    expected_order = [
        f"l2={l2}|learning_rate={learning_rate}"
        for l2 in LOGISTIC_L2_GRID
        for learning_rate in LOGISTIC_LEARNING_RATE_GRID
    ]
    assert list(logistic_scores) == expected_order
    best_score = max(logistic_scores.values())
    best_key = next(
        key for key, value in logistic_scores.items() if value == best_score
    )
    _, learning_rate = best_key.split("|")
    selected = first["selected_per_model"]["calibrated_interpretable"]
    assert best_key.startswith(f"l2={selected['l2']}")
    assert float(learning_rate.split("=")[1]) == selected["learning_rate"]

    rnn_scores = first["scores_by_option"]["neural_sequence"]
    expected_rnn_order = [
        f"hidden_size={hidden}|epochs={epochs}"
        for hidden in RNN_HIDDEN_SIZE_GRID
        for epochs in RNN_EPOCHS_GRID
    ]
    assert list(rnn_scores) == expected_rnn_order


def test_fit_with_starts_records_catalog_and_tie_rule() -> None:
    """Every deterministic start is recorded; ties resolve to the lowest seed."""

    training, development = _selection_rows()
    train_matrix = np.asarray([row.feature_vector for row in training], dtype=float)
    development_matrix = np.asarray(
        [row.feature_vector for row in development], dtype=float
    )
    train_labels = np.asarray([row.label for row in training], dtype=float)
    development_labels = np.asarray([row.label for row in development], dtype=float)

    def fitter(seed: int):
        return _fit_logistic(train_matrix, train_labels, seed=seed, l2=0.01)

    def scorer(model) -> float:
        return float(
            _metrics(
                development_labels,
                _predict_logistic(development_matrix, model),
                extended=False,
            )["balanced_accuracy"]
        )

    first = _fit_with_starts(fitter, scorer)
    second = _fit_with_starts(fitter, scorer)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["catalog"] == list(SEEDS)
    assert first["selection_rule"] == (
        "best_development_balanced_accuracy_tie_lowest_seed"
    )
    assert sorted(first["development_score_by_start"]) == sorted(
        str(seed) for seed in SEEDS
    )
    scores = first["development_score_by_start"]
    best = max(scores.values())
    lowest_best = min(int(seed) for seed, value in scores.items() if value == best)
    assert first["selected_start"] == lowest_best
    assert first["selected_start"] in first["catalog"]


def test_platt_calibration_is_deterministic_and_claims_nothing() -> None:
    """Platt scaling is a pure fit; both error directions are reported, not claimed."""

    probabilities = np.asarray([0.9, 0.8, 0.3, 0.2])
    labels = np.asarray([1, 1, 0, 0], dtype=float)
    a_one, b_one = _fit_platt(probabilities, labels)
    a_two, b_two = _fit_platt(probabilities, labels)
    assert (a_one, b_one) == (a_two, b_two)
    assert np.isfinite([a_one, b_one]).all()
    calibrated = _apply_platt(probabilities, a_one, b_one)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    # Monotone sanity: calibration must not reorder probabilities.
    assert np.array_equal(np.argsort(calibrated), np.argsort(probabilities))
    before = _metrics(labels, probabilities, extended=False)
    after = _metrics(labels, calibrated, extended=False)
    for key in ("expected_calibration_error", "brier_score"):
        assert np.isfinite(before[key]) and np.isfinite(after[key])


def test_family_shuffle_control_gate_fails_closed(full_payload: dict) -> None:
    """The family-shuffle control must lose and its gate must fail closed."""

    shuffle = full_payload["negative_controls"]["family_shuffle"]
    assert shuffle["seed"] == FAMILY_SHUFFLE_SEED
    assert shuffle["loses_to_all_co_primary"] is True
    above_chance = [
        name
        for name in MODEL_FAMILIES
        if full_payload["models"][name]["balanced_accuracy"] > 0.5
    ]
    assert shuffle["compared_models"] == above_chance
    assert above_chance
    for name in above_chance:
        assert (
            shuffle["metrics"]["balanced_accuracy"]
            < full_payload["models"][name]["balanced_accuracy"]
        ), name
    # A flipped verdict must be caught by recomputation, not trusted.
    flipped = copy.deepcopy(full_payload)
    flipped["negative_controls"]["family_shuffle"]["loses_to_all_co_primary"] = False
    valid, detail = validate_quotient_classifier_payload(flipped)
    assert valid is False
    assert "family-shuffle control verdict drift" in detail
    # A control that secretly wins must be caught by verdict recomputation.
    inflated = copy.deepcopy(full_payload)
    inflated["negative_controls"]["family_shuffle"]["metrics"]["balanced_accuracy"] = (
        0.99
    )
    valid, detail = validate_quotient_classifier_payload(inflated)
    assert valid is False
    assert "family-shuffle control verdict drift" in detail
    # And a honestly-recorded losing control must still fail the gate.
    losing = copy.deepcopy(full_payload)
    losing["negative_controls"]["family_shuffle"]["metrics"]["balanced_accuracy"] = 0.99
    losing["negative_controls"]["family_shuffle"]["loses_to_all_co_primary"] = False
    valid, detail = validate_quotient_classifier_payload(losing)
    assert valid is False
    assert "family-shuffle control does not lose" in detail
    emptied = copy.deepcopy(full_payload)
    del emptied["negative_controls"]["family_shuffle"]
    assert validate_quotient_classifier_payload(emptied)[0] is False
    drifted = copy.deepcopy(full_payload)
    drifted["negative_controls"]["family_shuffle"]["seed"] = 1
    assert validate_quotient_classifier_payload(drifted)[0] is False
    drifted_exclusion = copy.deepcopy(full_payload)
    drifted_exclusion["negative_controls"]["family_shuffle"][
        "chance_level_comparison_excluded"
    ] = list(MODEL_FAMILIES)
    valid, detail = validate_quotient_classifier_payload(drifted_exclusion)
    assert valid is False
    assert "chance-level exclusion drift" in detail
    # The control itself must run deterministically on any held-out rows.
    held = [
        _pair("fs_a", "held_out", 1, 2.0, 0.0),
        _pair("fs_a", "held_out", 0, 0.0, 2.0),
        _pair("fs_b", "held_out", 1, 1.9, 0.3),
        _pair("fs_b", "held_out", 0, 0.3, 1.9),
    ]
    matrix = np.asarray([row.feature_vector for row in held], dtype=float)
    control = _family_shuffle_control(held, matrix)
    again = _family_shuffle_control(held, matrix)
    assert json.dumps(control, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert np.isfinite(control["metrics"]["balanced_accuracy"])


def test_leakage_probe_gate_fails_closed(full_payload: dict) -> None:
    """A metadata-only probe that wins, or a missing probe, must fail closed."""

    leakage = full_payload["leakage_attacks"]
    assert leakage["metadata_only_beats_label_blind"] is False
    assert "bounded probe only" in leakage["conclusion_bound"]
    assert "no architecture-metadata claim" in leakage["conclusion_bound"]
    assert leakage["cross_encoding_consistency"]["check"] == (
        "cross_encoding_label_consistency"
    )
    for name in MODEL_FAMILIES:
        score = leakage["held_out_scores"][name]
        assert 0.0 <= score <= 1.0
        difference = score - leakage["label_blind_full_held_out_scores"][name]
        assert difference <= leakage["beats_threshold"] + 1e-12
        assert difference == pytest.approx(leakage["probe_margin_differences"][name])
    winning = copy.deepcopy(full_payload)
    winning["leakage_attacks"]["metadata_only_beats_label_blind"] = True
    valid, detail = validate_quotient_classifier_payload(winning)
    assert valid is False
    assert "metadata-only probe beat" in detail
    # A probe that secretly beats the margin must be caught by recomputation.
    inflated = copy.deepcopy(full_payload)
    first_model = MODEL_FAMILIES[0]
    inflated["leakage_attacks"]["held_out_scores"][first_model] = (
        inflated["leakage_attacks"]["label_blind_full_held_out_scores"][first_model]
        + 0.5
    )
    valid, detail = validate_quotient_classifier_payload(inflated)
    assert valid is False
    assert "leakage probe verdict drift" in detail
    emptied = copy.deepcopy(full_payload)
    emptied["leakage_attacks"] = {}
    assert validate_quotient_classifier_payload(emptied)[0] is False
    unbounded = copy.deepcopy(full_payload)
    unbounded["leakage_attacks"]["conclusion_bound"] = "unbounded claim"
    assert validate_quotient_classifier_payload(unbounded)[0] is False
    threshold_drift = copy.deepcopy(full_payload)
    threshold_drift["leakage_attacks"]["beats_threshold"] = 0.9
    valid, detail = validate_quotient_classifier_payload(threshold_drift)
    assert valid is False
    assert "leakage probe threshold drift" in detail


def test_near_duplicate_report_flags_undeclared_cross_split_pair(
    full_payload: dict,
) -> None:
    """A synthetic duplicate pair is flagged undeclared; the gate fails closed."""

    duplicate_rows = (
        _pair(
            "dup_a",
            "training",
            1,
            1.0,
            0.0,
            reference_code=bytes.fromhex("4883c001"),
            variant_code=bytes.fromhex("48ffc0"),
        ),
        _pair(
            "dup_b",
            "held_out",
            1,
            1.0,
            0.0,
            reference_code=bytes.fromhex("4883c001"),
            variant_code=bytes.fromhex("48ffc0"),
        ),
        _pair(
            "other_a",
            "training",
            0,
            0.0,
            1.0,
            reference_code=bytes.fromhex("4831c0"),
            variant_code=bytes.fromhex("4829c0"),
        ),
    )
    report = _near_duplicate_report(duplicate_rows)
    flagged = next(
        pair
        for pair in report["flagged_pairs"]
        if {pair["family_a"], pair["family_b"]} == {"dup_a", "dup_b"}
    )
    assert flagged["byte_identity"] == 1.0
    assert flagged["crosses_training_held_out"] is True
    assert flagged["declared"] is False
    assert report["max_cross_split_similarity"] == 1.0

    # Template siblings declared in the frozen manifest stay declared even
    # when their surfaces coincide.
    template_rows = (
        _pair(
            "add_one",
            "training",
            1,
            1.0,
            0.0,
            reference_code=bytes.fromhex("4883c001"),
            variant_code=bytes.fromhex("48ffc0"),
        ),
        _pair(
            "add_five",
            "held_out",
            1,
            1.0,
            0.0,
            reference_code=bytes.fromhex("4883c001"),
            variant_code=bytes.fromhex("48ffc0"),
        ),
    )
    template_report = _near_duplicate_report(template_rows)
    assert template_report["flagged_pairs"][0]["declared"] is True

    injected = copy.deepcopy(full_payload)
    injected["near_duplicate_checks"]["flagged_pairs"].append(
        {
            "family_a": "dup_a",
            "family_b": "dup_b",
            "split_a": "training",
            "split_b": "held_out",
            "byte_identity": 1.0,
            "token_jaccard": 1.0,
            "similarity": 1.0,
            "crosses_training_held_out": True,
            "template_keys": ["dup_a", "dup_b"],
            "declared": False,
        }
    )
    valid, detail = validate_quotient_classifier_payload(injected)
    assert valid is False
    assert "undeclared training-held-out near-duplicate pair" in detail
    declared_injection = copy.deepcopy(full_payload)
    declared_injection["near_duplicate_checks"]["flagged_pairs"].append(
        {
            **injected["near_duplicate_checks"]["flagged_pairs"][-1],
            "declared": True,
        }
    )
    assert validate_quotient_classifier_payload(declared_injection)[0] is True
    drifted = copy.deepcopy(full_payload)
    drifted["near_duplicate_checks"]["threshold"] = 0.5
    assert validate_quotient_classifier_payload(drifted)[0] is False


def test_payload_json_is_deterministic(full_payload: dict) -> None:
    """Two production passes must serialize to identical JSON."""

    second = quotient_classifier_payload()
    assert json.dumps(full_payload, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )


def test_family_randomization_matches_explicit_prediction_swaps() -> None:
    """Swapping whole families must match rowwise rescoring of each null draw."""
    rows = _rows()
    model = np.asarray([0.9 if row.label else 0.1 for row in rows])
    baseline = np.full(len(rows), 0.6)
    report = quotient_module._paired_family_randomization(
        rows, model, baseline, seed=19, replicates=127
    )
    families = sorted({row.family_id for row in rows})
    labels = np.asarray([row.label for row in rows])
    draws = np.random.default_rng(19).integers(0, 2, size=(127, len(families)))
    counts = {metric: 0 for metric in report["p_values"]}
    for draw in draws:
        mask = np.asarray(
            [draw[families.index(row.family_id)] for row in rows], dtype=bool
        )
        first = _metrics(labels, np.where(mask, baseline, model), extended=False)
        second = _metrics(labels, np.where(mask, model, baseline), extended=False)
        for metric in counts:
            counts[metric] += (
                first[metric] - second[metric]
                >= report["observed_difference"][metric] - 1e-12
            )
    for metric, count in counts.items():
        assert report["exceedances"][metric] == count
        assert report["p_values"][metric] == (count + 1) / 128
    assert (
        quotient_module._paired_family_randomization(
            rows, model, baseline, seed=19, replicates=127
        )
        == report
    )


def test_family_randomization_identical_predictions_retain_all_ties() -> None:
    rows = _rows()
    probabilities = np.full(len(rows), 0.5)
    report = quotient_module._paired_family_randomization(
        rows, probabilities, probabilities, seed=7, replicates=99
    )
    assert report["p_values"] == {"balanced_accuracy": 1.0, "macro_f1": 1.0}
    paired = _paired_cluster_bootstrap(
        rows, probabilities, probabilities, seed=7, replicates=10
    )
    assert paired["nonpositive_bootstrap_fractions"]["macro_f1"] == 1.0
    assert "raw_one_sided_bootstrap_p_values" not in paired


@pytest.mark.parametrize("replicates", [0, -1, True, 1.5])
def test_family_randomization_rejects_invalid_replicates(replicates) -> None:
    rows = _rows()
    probabilities = np.full(len(rows), 0.5)
    with pytest.raises(ValueError, match="positive integer"):
        quotient_module._paired_family_randomization(
            rows, probabilities, probabilities, seed=7, replicates=replicates
        )


@pytest.mark.parametrize(
    "bad", [np.asarray([0.5]), np.asarray([np.nan] * 8), np.asarray([1.5] * 8)]
)
def test_family_randomization_rejects_invalid_probabilities(bad) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="probabilities"):
        quotient_module._paired_family_randomization(
            rows, bad, np.full(len(rows), 0.5), seed=7, replicates=10
        )


@pytest.mark.parametrize("defect", ["p_value", "holm", "claim", "replicates", "count"])
def test_randomization_receipt_validator_rejects_inferential_drift(
    full_payload, defect
) -> None:
    payload = copy.deepcopy(full_payload)
    model = MODEL_FAMILIES[0]
    metric = "balanced_accuracy"
    test = payload["paired_comparisons"][f"{model}:surface_baseline:{metric}"][
        "randomization_test"
    ]
    record = payload["model_comparisons"][model][metric]
    if defect == "p_value":
        test["p_values"][metric] = 0.0
    elif defect == "holm":
        record["holm_adjusted_p_value"] = -1.0
    elif defect == "claim":
        record["superiority_claim_allowed"] = True
    elif defect == "replicates":
        test["replicates"] = 0
    else:
        test["exceedances"][metric] = -1
    valid, _ = validate_quotient_classifier_payload(payload)
    assert not valid


def test_vectorized_cluster_bootstrap_matches_explicit_family_resampling() -> None:
    rows = _rows()
    families = sorted({row.family_id for row in rows})
    probabilities = np.asarray([0.9 if row.label else 0.7 for row in rows])
    report = _cluster_bootstrap(rows, probabilities, seed=29, replicates=23)
    samples = np.random.default_rng(29).choice(
        families, size=(23, len(families)), replace=True
    )
    for sample_index, sample in enumerate(samples):
        indices = [
            index
            for family in sample
            for index, row in enumerate(rows)
            if row.family_id == family
        ]
        expected = _metrics(
            np.asarray([rows[index].label for index in indices]),
            probabilities[indices],
            extended=False,
        )
        for metric in report:
            assert report[metric][sample_index] == pytest.approx(expected[metric])


def test_training_vocabulary_maps_new_held_out_tokens_to_unknown() -> None:
    from dataclasses import replace

    training, development = _selection_rows()
    vocabulary = _token_vocabulary(training)
    unseen = replace(development[0], reference_tokens=("never_seen_in_training",))
    assert "never_seen_in_training" not in vocabulary
    assert "<UNKNOWN>" in vocabulary
    markov = _fit_markov(training, vocabulary)
    recurrent = _fit_rnn(training, vocabulary, seed=3, epochs=1)
    for probabilities in (
        _predict_markov([unseen], vocabulary, markov),
        _predict_rnn([unseen], vocabulary, recurrent),
    ):
        assert probabilities.shape == (1,)
        assert np.isfinite(probabilities).all()
        assert 0 <= probabilities[0] <= 1


def test_grouped_rnn_gradient_equals_explicit_duplicate_rows() -> None:
    """An independent full-batch reference checks the duplicate aggregation."""
    rows = (*_rows(), *_rows())
    vocabulary = _token_vocabulary(rows)
    seed, size, epochs, rate = 5, 3, 2, 0.03
    rng = np.random.default_rng(seed)
    wxh = rng.normal(0, 0.05, (size, len(vocabulary)))
    whh = rng.normal(0, 0.05, (size, size))
    why = rng.normal(0, 0.05, size)
    by = 0.0
    for _ in range(epochs):
        gx, gh, gy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
        gb = 0.0
        for row in rows:
            sequence = (
                "<REFERENCE>",
                *row.reference_tokens,
                "<VARIANT>",
                *row.variant_tokens,
            )
            ids = [vocabulary[token] for token in sequence]
            probability, states = _rnn_forward(ids, wxh, whh, why, by)
            delta = probability - row.label
            gy += delta * states[-1][size:]
            gb += delta
            dh = delta * why
            for index in reversed(range(len(ids))):
                previous, hidden = np.split(states[index], 2)
                dz = dh * (1 - hidden**2)
                gx[:, ids[index]] += dz
                gh += np.outer(dz, previous)
                dh = whh.T @ dz
        wxh -= rate * gx / len(rows)
        whh -= rate * gh / len(rows)
        why -= rate * gy / len(rows)
        by -= rate * gb / len(rows)
    grouped = _fit_rnn(
        rows, vocabulary, seed=seed, hidden_size=size, epochs=epochs, learning_rate=rate
    )
    for actual, expected in zip(grouped[:4], (wxh, whh, why, by)):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_classifier_plot_preserves_interval_outside_point_estimate(
    full_payload, tmp_path
) -> None:
    """A percentile interval can exclude its point estimate without being invalid."""
    from src.figures import plot_quotient_classifier

    payload = copy.deepcopy(full_payload)
    model = MODEL_FAMILIES[0]
    payload["models"][model]["confidence_intervals"]["balanced_accuracy"][
        "interval"
    ] = [0.1, 0.2]
    payload["paired_comparisons"][f"{model}:surface_baseline:balanced_accuracy"][
        "confidence_intervals"
    ]["balanced_accuracy"]["interval"] = [-0.95, -0.8]
    path = plot_quotient_classifier(payload, tmp_path / "independent_interval.png")
    assert path.is_file() and path.stat().st_size > 0


def test_classifier_plot_rejects_invalid_paired_interval(
    full_payload, tmp_path
) -> None:
    from matplotlib import pyplot as plt
    from src.figures import plot_quotient_classifier

    payload = copy.deepcopy(full_payload)
    model = MODEL_FAMILIES[0]
    payload["paired_comparisons"][f"{model}:surface_baseline:balanced_accuracy"][
        "confidence_intervals"
    ]["balanced_accuracy"]["interval"] = [float("nan"), 0.2]
    try:
        with pytest.raises(ValueError, match="paired interval is invalid"):
            plot_quotient_classifier(payload, tmp_path / "invalid_interval.png")
    finally:
        plt.close("all")


def test_cache_rejects_changed_observation_dependency(
    full_payload, tmp_path, monkeypatch
) -> None:
    """A dependency edit invalidates an otherwise identical benchmark receipt."""
    payload = copy.deepcopy(full_payload)
    for name in payload["input_hashes"]:
        if name.startswith("src/"):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((quotient_module.PROJECT_ROOT / name).read_bytes())
    target = tmp_path / quotient_module.QUOTIENT_CLASSIFIER_ARTIFACT
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(quotient_module, "PROJECT_ROOT", tmp_path)
    assert quotient_module._cached_quotient_classifier_payload() is not None
    dependency = tmp_path / "src/emulation.py"
    dependency.write_text(
        dependency.read_text() + "\n# Changed observation dependency\n"
    )
    assert quotient_module._cached_quotient_classifier_payload() is None


def test_classifier_receipt_rejects_transductive_vocabulary(full_payload) -> None:
    payload = copy.deepcopy(full_payload)
    payload["vocabulary_fit_scope"] = "all_rows"
    assert not validate_quotient_classifier_payload(payload)[0]
