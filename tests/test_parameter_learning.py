"""Tests for observation-only ordered-HMM parameter learning."""

from __future__ import annotations

import copy
from typing import cast

import numpy as np
import pytest

from src.corpus import build_corpus, fixture_observation_ids
from src.parameter_learning import (
    fit_ordered_hmm_em,
    parameter_learning_payload,
    validate_parameter_learning_payload,
)
from src.temporal_model import build_temporal_model


def test_ordered_em_uses_real_probability_tensors_and_is_deterministic() -> None:
    fixtures = build_corpus()
    sequences = [list(fixture_observation_ids(fixture)) for fixture in fixtures[1:]]
    first_model, first_diagnostics = fit_ordered_hmm_em(
        sequences, initial_model=build_temporal_model()
    )
    second_model, second_diagnostics = fit_ordered_hmm_em(
        sequences, initial_model=build_temporal_model()
    )

    np.testing.assert_allclose(first_model.A, second_model.A)
    np.testing.assert_allclose(first_model.B, second_model.B)
    np.testing.assert_allclose(first_model.D, second_model.D)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["phase_labels_used_for_fit"] is False
    assert np.all(first_model.A > 0.0)
    assert np.all(first_model.B > 0.0)
    assert np.all(first_model.D > 0.0)
    np.testing.assert_allclose(first_model.A.sum(axis=0), 1.0)
    np.testing.assert_allclose(first_model.B.sum(axis=0), 1.0)
    np.testing.assert_allclose(first_model.D.sum(), 1.0)


def test_parameter_learning_payload_is_complete_and_leakage_safe() -> None:
    payload = parameter_learning_payload(build_corpus())

    valid, detail = validate_parameter_learning_payload(payload)
    assert valid, detail
    assert payload["phase_labels_used_for_fit"] is False
    summary = cast(dict[str, object], payload["summary"])
    rows = cast(list[dict[str, object]], payload["per_fixture"])
    assert summary["leakage_free"] is True
    assert summary["fixture_count"] == 6
    for row in rows:
        training_names = cast(list[object], row["training_fixtures"])
        observation_ids = cast(list[object], row["observation_ids"])
        truth_labels = cast(list[object], row["truth_labels_for_scoring_only"])
        assert row["held_out_fixture"] not in training_names
        assert row["fit_input_contract"] == {
            "phase_labels_used_for_fit": False,
            "labels_present_in_fit_input": False,
            "observation_only": True,
        }
        assert len(observation_ids) == len(truth_labels)
        diagnostics = cast(dict[str, object], row["fit_diagnostics"])
        assert diagnostics["phase_labels_used_for_fit"] is False


def test_parameter_learning_validator_rejects_label_leakage() -> None:
    payload = parameter_learning_payload(build_corpus())
    tampered = copy.deepcopy(payload)
    tampered_rows = cast(list[dict[str, object]], tampered["per_fixture"])
    tampered_contract = cast(dict[str, object], tampered_rows[0]["fit_input_contract"])
    tampered_contract["phase_labels_used_for_fit"] = True

    valid, detail = validate_parameter_learning_payload(tampered)
    assert valid is False
    assert "leakage" in detail


def test_parameter_learning_validator_fails_closed_on_malformed_score() -> None:
    payload = parameter_learning_payload(build_corpus())
    tampered = copy.deepcopy(payload)
    tampered_rows = cast(list[dict[str, object]], tampered["per_fixture"])
    score = cast(dict[str, object], tampered_rows[0]["fixed_reference"])
    score["accuracy"] = "not-a-number"

    valid, detail = validate_parameter_learning_payload(tampered)
    assert valid is False
    assert "fixed_reference" in detail


def test_ordered_em_rejects_empty_or_invalid_fit_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        fit_ordered_hmm_em([])
    with pytest.raises(ValueError, match="positive integer"):
        fit_ordered_hmm_em([[0]], max_iterations=0)
    with pytest.raises(ValueError, match="out-of-range"):
        fit_ordered_hmm_em([[999]])
