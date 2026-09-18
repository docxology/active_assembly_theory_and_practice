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
    # MIN-07: the failing-surface status must stay visible in the receipt; a
    # regression flipping it to a passing status must fail this suite.
    assert payload["status"] == "underpowered"
    assert payload["underpowered_cases"]
    summary = cast(dict[str, object], payload["summary"])
    rows = cast(list[dict[str, object]], payload["per_fixture"])
    assert summary["leakage_free"] is True
    assert summary["fixture_count"] == 10
    assert summary["multi_start_count"] == 50
    assert set(summary["multi_start_initializations"]) == {
        "reference",
        "uniform",
        "random_11",
        "random_23",
        "random_47",
    }
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
        starts = cast(list[dict[str, object]], row["multi_start"])
        assert len(starts) == 5
        assert all(
            cast(dict[str, object], start["fit_diagnostics"])[
                "phase_labels_used_for_fit"
            ]
            is False
            for start in starts
        )


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


def test_parameter_learning_validator_exercises_fail_closed_contract_matrix() -> None:
    """Cover the receipt validator's negative scientific-integrity branches."""

    payload = parameter_learning_payload(build_corpus())
    rows = cast(list[dict[str, object]], payload["per_fixture"])
    first = rows[0]

    def row_mutation(mutator):
        mutated = copy.deepcopy(payload)
        mutated_rows = cast(list[dict[str, object]], mutated["per_fixture"])
        mutator(mutated_rows[0], cast(dict[str, object], mutated["summary"]))
        return mutated

    mutations = [
        lambda row, summary: row.update(training_fixtures=[]),
        lambda row, summary: row.update(training_fixtures=[row["held_out_fixture"]]),
        lambda row, summary: row.update(fit_input_contract={"observation_only": True}),
        lambda row, summary: row.update(observation_ids=[]),
        lambda row, summary: row.update(truth_labels_for_scoring_only=["extra"]),
        lambda row, summary: row.update(fit_diagnostics={}),
        lambda row, summary: row.update(
            fit_diagnostics={"history": [], "phase_labels_used_for_fit": False}
        ),
        lambda row, summary: row.update(
            fit_diagnostics={
                "history": [{"log_evidence_before": float("nan")}],
                "iterations_used": 1,
                "phase_labels_used_for_fit": False,
            }
        ),
        lambda row, summary: row.update(learned_model={}),
        lambda row, summary: row.update(
            learned_model={"A": [[1.0]], "B": [[1.0]], "D": [1.0]}
        ),
        lambda row, summary: row.update(weakly_supervised_em={}),
        lambda row, summary: cast(dict[str, object], row["fixed_reference"]).update(
            accuracy=2.0
        ),
        lambda row, summary: row.update(posterior_predictive_check={}),
        lambda row, summary: row.update(
            posterior_predictive_check={"uses_phase_labels": True, "checks": {}}
        ),
        lambda row, summary: row.update(penalized_fit_diagnostics={}),
        lambda row, summary: row.update(multi_start=[]),
        lambda row, summary: row.update(multi_start=[{}]),
        lambda row, summary: row.update(
            multi_start=[
                {
                    "initialization": "reference",
                    "held_out_score": {"accuracy": 2.0},
                    "fit_diagnostics": {"phase_labels_used_for_fit": False},
                    "learned_model": {},
                }
            ]
        ),
        lambda row, summary: row.update(
            multi_start=[
                {
                    "initialization": "reference",
                    "held_out_score": {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "log_evidence": 0.0,
                        "temporal_labels": [],
                    },
                    "fit_diagnostics": {"phase_labels_used_for_fit": True},
                    "learned_model": {},
                },
                {
                    "initialization": "reference",
                    "held_out_score": {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "log_evidence": 0.0,
                        "temporal_labels": [],
                    },
                    "fit_diagnostics": {"phase_labels_used_for_fit": False},
                    "learned_model": {},
                },
                {
                    "initialization": "random",
                    "held_out_score": {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "log_evidence": 0.0,
                        "temporal_labels": [],
                    },
                    "fit_diagnostics": {"phase_labels_used_for_fit": False},
                    "learned_model": {},
                },
            ]
        ),
        lambda row, summary: row.update(training_observation_sequences=[]),
        lambda row, summary: row.update(training_observation_sequences=[[999]]),
    ]
    for mutation in mutations:
        assert validate_parameter_learning_payload(row_mutation(mutation))[0] is False

    top_level_mutations = [
        lambda candidate: candidate.update(schema_version="1.0.0"),
        lambda candidate: candidate.update(status="failed"),
        lambda candidate: candidate.update(phase_labels_used_for_fit=True),
        lambda candidate: candidate.update(per_fixture=[]),
        lambda candidate: candidate.update(fixture_count=99),
        lambda candidate: cast(dict[str, object], candidate["summary"]).update(
            leakage_free=False
        ),
        lambda candidate: candidate.update(
            model_selection_ledger={"held_out_labels_available_to_selection": True}
        ),
        lambda candidate: cast(dict[str, object], candidate["summary"]).update(
            multi_start_count=0
        ),
    ]
    for mutation in top_level_mutations:
        candidate = copy.deepcopy(payload)
        mutation(candidate)
        assert validate_parameter_learning_payload(candidate)[0] is False

    assert first["held_out_fixture"] not in cast(
        list[object], first["training_fixtures"]
    )
