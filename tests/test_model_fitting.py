"""Tests for leakage-safe supervised temporal-model fitting."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from src.corpus import build_corpus
from src.model_fitting import (
    _accuracy,
    _normalize_columns,
    fit_temporal_model,
    leave_one_fixture_out_evaluation,
)


def test_fit_temporal_model_returns_normalized_positive_tensors() -> None:
    """Dirichlet-smoothed count fitting produces valid temporal probabilities."""

    model = fit_temporal_model(build_corpus()[:-1])

    assert model.A.shape == (6, 5)
    assert model.B.shape == (5, 5)
    assert model.D.shape == (5,)
    assert np.all(model.A > 0.0)
    assert np.all(model.B > 0.0)
    assert np.all(model.D > 0.0)
    assert np.allclose(model.A.sum(axis=0), 1.0)
    assert np.allclose(model.B.sum(axis=0), 1.0)
    assert np.isclose(model.D.sum(), 1.0)


def test_leave_one_fixture_out_excludes_each_fixture_and_is_deterministic() -> None:
    """Every held-out row names an explicit disjoint training set."""

    corpus = build_corpus()
    first = leave_one_fixture_out_evaluation(corpus)
    second = leave_one_fixture_out_evaluation(corpus)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["fixture_count"] == len(corpus)
    assert len(first["per_fixture"]) == len(corpus)
    for row in first["per_fixture"]:
        assert row["held_out_fixture"] not in row["training_fixtures"]
        assert len(row["training_fixtures"]) == len(corpus) - 1
        assert 0.0 <= row["temporal_accuracy"] <= 1.0
        assert len(row["temporal_labels"]) == len(row["truth_labels"])
    summary = first["summary"]
    assert 0.0 <= summary["mean_held_out_temporal_accuracy"] <= 1.0
    assert 0.0 <= summary["mean_fixed_full_corpus_accuracy"] <= 1.0


@pytest.mark.parametrize("name", ("emission_alpha", "transition_alpha", "prior_alpha"))
def test_fit_temporal_model_rejects_nonpositive_smoothing(name: str) -> None:
    """Smoothing concentrations are explicit positive numeric parameters."""

    with pytest.raises(ValueError, match="finite and positive"):
        fit_temporal_model(build_corpus(), **{name: 0.0})


def test_leave_one_fixture_out_requires_multiple_fixtures() -> None:
    """A leave-one-out estimate needs a non-empty training complement."""

    with pytest.raises(ValueError, match="at least two"):
        leave_one_fixture_out_evaluation(build_corpus()[:1])


def test_temporal_fitting_rejects_invalid_inputs_and_private_contracts() -> None:
    corpus = build_corpus()
    with pytest.raises(ValueError, match="fixtures must not be empty"):
        fit_temporal_model([])
    with pytest.raises(ValueError, match="finite and positive"):
        fit_temporal_model(corpus, emission_alpha=float("nan"))
    with pytest.raises(ValueError, match="positive totals"):
        _normalize_columns(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-empty and equally sized"):
        _accuracy([], [])
    with pytest.raises(ValueError, match="non-empty and equally sized"):
        _accuracy(["setup"], ["setup", "exit"])

    fixture = corpus[0]
    with pytest.raises(ValueError, match="one observation per truth label"):
        fit_temporal_model([replace(fixture, phase_labels=fixture.phase_labels[:-1])])
    with pytest.raises(ValueError, match="unknown phase label"):
        fit_temporal_model(
            [replace(fixture, phase_labels=("not_a_phase", *fixture.phase_labels[1:]))]
        )
