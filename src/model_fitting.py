"""Leakage-safe categorical HMM fitting and leave-one-fixture-out evaluation.

The original manuscript model is deliberately hand-authored so that its
mechanism is transparent.  This module adds a complementary, reproducible
estimation path: smoothed maximum-a-posteriori counts are fit only on a stated
training subset of the labelled fixture corpus, then evaluated on fixtures that
were excluded from fitting.  It is still a tiny supervised demonstrator, not an
unsupervised discovery claim, but it removes the strongest methodological
weakness of evaluating a fixed model only on the same fixtures that motivated
its labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .evaluation import evaluate_decoding
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .temporal_model import TemporalGenerativeModel, forward_filter, viterbi_decode


def _validate_alpha(value: float, *, name: str) -> float:
    """Validate a strictly positive Dirichlet smoothing concentration."""

    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize a positive count matrix by source-state columns."""

    totals = matrix.sum(axis=0, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("count matrix columns must have positive totals")
    return matrix / totals


def fit_temporal_model(
    fixtures: Sequence[CorpusFixture],
    *,
    emission_alpha: float = 0.5,
    transition_alpha: float = 0.5,
    prior_alpha: float = 0.5,
) -> TemporalGenerativeModel:
    """Fit ``A``, ``B``, and ``D`` from labelled training fixtures.

    ``A[o, s]`` receives one count for each observed category ``o`` aligned to
    truth phase ``s``. ``B[next, current]`` receives one count for each adjacent
    truth-phase transition, and ``D`` receives one count for each fixture's
    initial truth phase. Positive Dirichlet pseudocounts keep every fitted
    probability strictly positive so it can be passed to the project's
    log-space temporal routines.
    """

    if not fixtures:
        raise ValueError("fixtures must not be empty")
    emission_alpha = _validate_alpha(emission_alpha, name="emission_alpha")
    transition_alpha = _validate_alpha(transition_alpha, name="transition_alpha")
    prior_alpha = _validate_alpha(prior_alpha, name="prior_alpha")

    state_index = {label: index for index, label in enumerate(HIDDEN_STATE_LABELS)}
    n_obs = len(OBSERVATION_LABELS)
    n_states = len(HIDDEN_STATE_LABELS)
    emission_counts = np.full((n_obs, n_states), emission_alpha, dtype=float)
    transition_counts = np.full((n_states, n_states), transition_alpha, dtype=float)
    prior_counts = np.full(n_states, prior_alpha, dtype=float)

    for fixture in fixtures:
        observation_ids = [int(value) for value in fixture_observation_ids(fixture)]
        truth = list(fixture.phase_labels)
        if len(observation_ids) != len(truth) or not truth:
            raise ValueError(
                f"fixture {fixture.name!r} must have one observation per truth label"
            )
        state_ids = [state_index.get(label, -1) for label in truth]
        if any(state_id < 0 for state_id in state_ids):
            raise ValueError(
                f"fixture {fixture.name!r} contains an unknown phase label"
            )
        prior_counts[state_ids[0]] += 1.0
        for observation_id, state_id in zip(observation_ids, state_ids, strict=True):
            if observation_id < 0 or observation_id >= n_obs:
                raise ValueError(
                    f"fixture {fixture.name!r} contains an invalid observation id"
                )
            emission_counts[observation_id, state_id] += 1.0
        for current, next_state in zip(state_ids[:-1], state_ids[1:], strict=True):
            transition_counts[next_state, current] += 1.0

    model = TemporalGenerativeModel(
        A=_normalize_columns(emission_counts),
        B=_normalize_columns(transition_counts),
        D=prior_counts / prior_counts.sum(),
        observation_labels=tuple(OBSERVATION_LABELS),
        hidden_state_labels=tuple(HIDDEN_STATE_LABELS),
    )
    model.validate()
    return model


def _accuracy(predicted: Sequence[str], truth: Sequence[str]) -> float:
    """Return exact per-step accuracy for one held-out fixture."""

    if len(predicted) != len(truth) or not truth:
        raise ValueError("predicted and truth must be non-empty and equally sized")
    return float(
        sum(p == t for p, t in zip(predicted, truth, strict=True)) / len(truth)
    )


def leave_one_fixture_out_evaluation(
    fixtures: Sequence[CorpusFixture],
    *,
    emission_alpha: float = 0.5,
    transition_alpha: float = 0.5,
    prior_alpha: float = 0.5,
) -> dict[str, Any]:
    """Fit on all-but-one fixtures and score each excluded fixture.

    The returned rows contain the explicit training and held-out fixture names,
    fitted model tensors, forward-filter and Viterbi predictions, and a fixed
    held-out baseline fitted on the full corpus only for reference.  The
    baseline is not used to tune the held-out model.
    """

    if len(fixtures) < 2:
        raise ValueError(
            "at least two fixtures are required for leave-one-out evaluation"
        )
    fixed_model = fit_temporal_model(
        fixtures,
        emission_alpha=emission_alpha,
        transition_alpha=transition_alpha,
        prior_alpha=prior_alpha,
    )
    rows: list[dict[str, object]] = []
    held_out_accuracies: list[float] = []
    fixed_accuracies: list[float] = []
    for held_out_index, held_out in enumerate(fixtures):
        training = [
            fixture for index, fixture in enumerate(fixtures) if index != held_out_index
        ]
        fitted = fit_temporal_model(
            training,
            emission_alpha=emission_alpha,
            transition_alpha=transition_alpha,
            prior_alpha=prior_alpha,
        )
        observation_ids = [int(value) for value in fixture_observation_ids(held_out)]
        truth = list(held_out.phase_labels)
        filtered = forward_filter(observation_ids, fitted)
        viterbi_labels, viterbi_log_probability = viterbi_decode(
            observation_ids, fitted
        )
        fixed_filtered = forward_filter(observation_ids, fixed_model)
        fitted_metrics = evaluate_decoding(
            filtered.most_likely_states, truth, list(HIDDEN_STATE_LABELS)
        )
        fixed_metrics = evaluate_decoding(
            fixed_filtered.most_likely_states, truth, list(HIDDEN_STATE_LABELS)
        )
        fitted_accuracy = _accuracy(filtered.most_likely_states, truth)
        fixed_accuracy = _accuracy(fixed_filtered.most_likely_states, truth)
        held_out_accuracies.append(fitted_accuracy)
        fixed_accuracies.append(fixed_accuracy)
        rows.append(
            {
                "held_out_fixture": held_out.name,
                "training_fixtures": [fixture.name for fixture in training],
                "observation_ids": observation_ids,
                "truth_labels": truth,
                "temporal_labels": list(filtered.most_likely_states),
                "viterbi_labels": list(viterbi_labels),
                "temporal_accuracy": fitted_accuracy,
                "temporal_balanced_accuracy": float(
                    fitted_metrics["balanced_accuracy"]
                ),
                "viterbi_accuracy": _accuracy(viterbi_labels, truth),
                "fixed_full_corpus_accuracy": fixed_accuracy,
                "fixed_full_corpus_balanced_accuracy": float(
                    fixed_metrics["balanced_accuracy"]
                ),
                "log_evidence": float(filtered.log_evidence),
                "viterbi_log_probability": float(viterbi_log_probability),
                "fitted_model": fitted.to_dict(),
            }
        )

    return {
        "method": "leave_one_fixture_out_supervised_map_hmm",
        "fit_scope": "training fixtures only; held-out fixture excluded from A, B, and D counts",
        "emission_alpha": float(emission_alpha),
        "transition_alpha": float(transition_alpha),
        "prior_alpha": float(prior_alpha),
        "fixture_count": int(len(fixtures)),
        "summary": {
            "mean_held_out_temporal_accuracy": float(np.mean(held_out_accuracies)),
            "mean_fixed_full_corpus_accuracy": float(np.mean(fixed_accuracies)),
            "held_out_minus_fixed_full_corpus": float(
                np.mean(held_out_accuracies) - np.mean(fixed_accuracies)
            ),
            "held_out_temporal_accuracy_min": float(np.min(held_out_accuracies)),
            "held_out_temporal_accuracy_max": float(np.max(held_out_accuracies)),
        },
        "per_fixture": rows,
        "full_corpus_fit": fixed_model.to_dict(),
    }
