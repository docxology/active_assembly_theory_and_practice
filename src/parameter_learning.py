"""Leakage-safe weakly supervised HMM parameter learning.

The canonical temporal model is intentionally hand-authored so that its
transition and emission assumptions are inspectable.  This module adds a
separate parameter-learning experiment: an ordered categorical HMM is fit by
Baum--Welch/expectation-maximization using only observation-category IDs from
the training fixtures.  Phase labels are reserved for the held-out scoring
layer and are never passed to the fitting routine.

The ordered state names remain an architectural scaffold rather than learned
semantics.  A small left-to-right transition prior anchors the state ordering,
and positive floors preserve the log-space contract of the project's temporal
inference routines.  The output is therefore a weakly supervised estimation
demonstrator, not unsupervised phase discovery or external validation.
"""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .evaluation import evaluate_decoding
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .model_fitting import fit_temporal_model
from .temporal_model import (
    TemporalGenerativeModel,
    _as_observation_ids,
    build_temporal_model,
    forward_filter,
)

PARAMETER_LEARNING_SCHEMA_VERSION = "1.0.0"
PARAMETER_LEARNING_MAX_ITERATIONS = 32
PARAMETER_LEARNING_TOLERANCE = 1e-8
PARAMETER_LEARNING_TRANSITION_FLOOR = 1e-6


def _validate_positive(value: float, *, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _validate_iterations(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_sequences(
    observation_sequences: Sequence[Sequence[int]],
    model: TemporalGenerativeModel,
) -> list[list[int]]:
    if not observation_sequences:
        raise ValueError("observation_sequences must not be empty")
    normalized: list[list[int]] = []
    for sequence_index, sequence in enumerate(observation_sequences):
        try:
            values = list(sequence)
        except TypeError as exc:
            raise ValueError(
                f"observation sequence {sequence_index} is not iterable"
            ) from exc
        if not values:
            raise ValueError(f"observation sequence {sequence_index} is empty")
        normalized.append(
            _as_observation_ids(values, observation_count=len(model.observation_labels))
        )
    return normalized


def _left_to_right_mask(state_count: int) -> np.ndarray:
    """Return the ordered self/next transition support mask."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    mask = np.zeros((state_count, state_count), dtype=bool)
    for current in range(state_count):
        mask[current, current] = True
        if current + 1 < state_count:
            mask[current + 1, current] = True
    return mask


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=0, keepdims=True)
    if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
        raise ValueError("count matrix columns must have positive finite totals")
    return matrix / totals


def _expected_statistics(
    observation_ids: Sequence[int], model: TemporalGenerativeModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return expected emission, transition, and initial counts for one sequence."""

    model.validate()
    ids = _as_observation_ids(
        list(observation_ids), observation_count=len(model.observation_labels)
    )
    n_steps = len(ids)
    n_states = len(model.hidden_state_labels)
    n_obs = len(model.observation_labels)
    matrix_a = np.asarray(model.A, dtype=float)
    matrix_b = np.asarray(model.B, dtype=float)
    prior_d = np.asarray(model.D, dtype=float)

    alpha = np.zeros((n_steps, n_states), dtype=float)
    scales = np.zeros(n_steps, dtype=float)
    for step, obs in enumerate(ids):
        prior = prior_d if step == 0 else matrix_b @ alpha[step - 1]
        unnormalized = prior * matrix_a[obs, :]
        scale = float(unnormalized.sum())
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("forward message has a non-positive marginal")
        scales[step] = scale
        alpha[step] = unnormalized / scale

    beta = np.ones((n_steps, n_states), dtype=float)
    for step in range(n_steps - 2, -1, -1):
        next_weight = matrix_a[ids[step + 1], :] * beta[step + 1]
        beta[step] = (matrix_b.T @ next_weight) / scales[step + 1]
        if not np.all(np.isfinite(beta[step])):
            raise ValueError("backward message contains non-finite values")

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    emission_counts = np.zeros((n_obs, n_states), dtype=float)
    for obs, posterior in zip(ids, gamma, strict=True):
        emission_counts[obs, :] += posterior

    transition_counts = np.zeros((n_states, n_states), dtype=float)
    for step in range(n_steps - 1):
        next_weight = matrix_a[ids[step + 1], :] * beta[step + 1]
        xi = alpha[step, :][np.newaxis, :] * matrix_b * next_weight[:, np.newaxis]
        xi /= xi.sum()
        transition_counts += xi

    log_evidence = float(np.log(scales).sum())
    return emission_counts, transition_counts, gamma[0, :], log_evidence


def fit_ordered_hmm_em(
    observation_sequences: Sequence[Sequence[int]],
    *,
    initial_model: TemporalGenerativeModel | None = None,
    emission_alpha: float = 0.5,
    transition_alpha: float = 0.5,
    prior_alpha: float = 0.5,
    max_iterations: int = PARAMETER_LEARNING_MAX_ITERATIONS,
    tolerance: float = PARAMETER_LEARNING_TOLERANCE,
) -> tuple[TemporalGenerativeModel, dict[str, object]]:
    """Fit an ordered categorical HMM using observation IDs only.

    The E step computes smoothed state and adjacent-state responsibilities.
    The M step updates ``A``, ``B``, and ``D`` with positive Dirichlet
    pseudocounts.  A soft left-to-right support prior keeps state order
    interpretable while leaving a tiny positive floor on disallowed entries so
    the shared temporal model can continue to operate in log space.
    """

    emission_alpha = _validate_positive(emission_alpha, name="emission_alpha")
    transition_alpha = _validate_positive(transition_alpha, name="transition_alpha")
    prior_alpha = _validate_positive(prior_alpha, name="prior_alpha")
    max_iterations = _validate_iterations(max_iterations, name="max_iterations")
    tolerance = _validate_positive(tolerance, name="tolerance")
    model = initial_model or build_temporal_model()
    model.validate()
    sequences = _validate_sequences(observation_sequences, model)
    n_obs = len(model.observation_labels)
    n_states = len(model.hidden_state_labels)
    support = _left_to_right_mask(n_states)
    current = TemporalGenerativeModel(
        A=np.asarray(model.A, dtype=float).copy(),
        B=np.asarray(model.B, dtype=float).copy(),
        D=np.asarray(model.D, dtype=float).copy(),
        observation_labels=tuple(model.observation_labels),
        hidden_state_labels=tuple(model.hidden_state_labels),
    )

    history: list[dict[str, float | int | bool]] = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        emission_counts = np.zeros((n_obs, n_states), dtype=float)
        transition_counts = np.zeros((n_states, n_states), dtype=float)
        prior_counts = np.zeros(n_states, dtype=float)
        log_evidence_before = 0.0
        for sequence in sequences:
            (
                sequence_emission,
                sequence_transition,
                sequence_prior,
                sequence_log_evidence,
            ) = _expected_statistics(sequence, current)
            emission_counts += sequence_emission
            transition_counts += sequence_transition
            prior_counts += sequence_prior
            log_evidence_before += sequence_log_evidence

        emission_parameters = _normalize_columns(emission_counts + emission_alpha)
        transition_parameters = np.full(
            (n_states, n_states), PARAMETER_LEARNING_TRANSITION_FLOOR, dtype=float
        )
        transition_parameters[support] += transition_counts[support] + transition_alpha
        transition_parameters = _normalize_columns(transition_parameters)
        prior_parameters = prior_counts + prior_alpha
        prior_parameters /= prior_parameters.sum()
        updated = TemporalGenerativeModel(
            A=emission_parameters,
            B=transition_parameters,
            D=prior_parameters,
            observation_labels=tuple(model.observation_labels),
            hidden_state_labels=tuple(model.hidden_state_labels),
        )
        updated.validate()
        delta = float(
            max(
                np.max(np.abs(updated.A - current.A)),
                np.max(np.abs(updated.B - current.B)),
                np.max(np.abs(updated.D - current.D)),
            )
        )
        log_evidence_after = 0.0
        for sequence in sequences:
            log_evidence_after += _expected_statistics(sequence, updated)[3]
        converged = delta <= tolerance
        history.append(
            {
                "iteration": iteration,
                "log_evidence_before": float(log_evidence_before),
                "log_evidence_after": float(log_evidence_after),
                "parameter_delta": delta,
                "converged": converged,
            }
        )
        current = updated
        if converged:
            break

    return current, {
        "algorithm": "ordered_baum_welch_em",
        "fit_input": "observation_category_ids_only",
        "phase_labels_used_for_fit": False,
        "transition_support": "soft_left_to_right_self_or_next_with_positive_floor",
        "emission_alpha": emission_alpha,
        "transition_alpha": transition_alpha,
        "prior_alpha": prior_alpha,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "transition_floor": PARAMETER_LEARNING_TRANSITION_FLOOR,
        "sequence_count": len(sequences),
        "observation_count": int(sum(len(sequence) for sequence in sequences)),
        "history": history,
        "iterations_used": len(history),
        "converged": converged,
        "final_log_evidence": float(history[-1]["log_evidence_after"]),
    }


def _score_model(
    observation_ids: Sequence[int],
    truth: Sequence[str],
    model: TemporalGenerativeModel,
) -> dict[str, object]:
    filtered = forward_filter(list(observation_ids), model)
    metrics = evaluate_decoding(
        filtered.most_likely_states, list(truth), list(HIDDEN_STATE_LABELS)
    )
    return {
        "temporal_labels": list(filtered.most_likely_states),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "log_evidence": float(filtered.log_evidence),
    }


def weakly_supervised_parameter_learning_payload(
    fixtures: Sequence[CorpusFixture],
    *,
    max_iterations: int = PARAMETER_LEARNING_MAX_ITERATIONS,
) -> dict[str, object]:
    """Evaluate observation-only EM under a complete-fixture holdout boundary."""

    if len(fixtures) < 2:
        raise ValueError("at least two fixtures are required for parameter learning")
    fixture_tuple = tuple(fixtures)
    reference_model = build_temporal_model()
    rows: list[dict[str, object]] = []
    weak_accuracies: list[float] = []
    supervised_accuracies: list[float] = []
    reference_accuracies: list[float] = []
    for held_out_index, held_out in enumerate(fixture_tuple):
        training = tuple(
            fixture
            for index, fixture in enumerate(fixture_tuple)
            if index != held_out_index
        )
        training_sequences = [
            list(fixture_observation_ids(fixture)) for fixture in training
        ]
        learned_model, fit_diagnostics = fit_ordered_hmm_em(
            training_sequences,
            initial_model=reference_model,
            max_iterations=max_iterations,
        )
        supervised_model = fit_temporal_model(training)
        observation_ids = list(fixture_observation_ids(held_out))
        truth = list(held_out.phase_labels)
        learned_score = _score_model(observation_ids, truth, learned_model)
        supervised_score = _score_model(observation_ids, truth, supervised_model)
        reference_score = _score_model(observation_ids, truth, reference_model)
        weak_accuracies.append(float(cast(float, learned_score["accuracy"])))
        supervised_accuracies.append(float(cast(float, supervised_score["accuracy"])))
        reference_accuracies.append(float(cast(float, reference_score["accuracy"])))
        rows.append(
            {
                "held_out_fixture": held_out.name,
                "training_fixtures": [fixture.name for fixture in training],
                "training_observation_sequences": training_sequences,
                "fit_input_contract": {
                    "phase_labels_used_for_fit": False,
                    "labels_present_in_fit_input": False,
                    "observation_only": True,
                },
                "observation_ids": observation_ids,
                "truth_labels_for_scoring_only": truth,
                "weakly_supervised_em": learned_score,
                "supervised_map": supervised_score,
                "fixed_reference": reference_score,
                "learned_model": learned_model.to_dict(),
                "fit_diagnostics": fit_diagnostics,
            }
        )

    return {
        "schema_version": PARAMETER_LEARNING_SCHEMA_VERSION,
        "method": "leave_one_fixture_out_ordered_weakly_supervised_em",
        "fit_scope": "training fixture observation IDs only; held-out fixture excluded",
        "phase_labels_used_for_fit": False,
        "fixture_count": len(fixture_tuple),
        "max_iterations": max_iterations,
        "policies": ["weakly_supervised_em", "supervised_map", "fixed_reference"],
        "summary": {
            "fixture_count": len(fixture_tuple),
            "mean_weakly_supervised_accuracy": float(np.mean(weak_accuracies)),
            "mean_supervised_accuracy": float(np.mean(supervised_accuracies)),
            "mean_fixed_reference_accuracy": float(np.mean(reference_accuracies)),
            "weakly_supervised_minus_supervised": float(
                np.mean(weak_accuracies) - np.mean(supervised_accuracies)
            ),
            "weakly_supervised_minus_reference": float(
                np.mean(weak_accuracies) - np.mean(reference_accuracies)
            ),
            "converged_holdout_count": int(
                sum(
                    bool(cast(dict[str, object], row["fit_diagnostics"])["converged"])
                    for row in rows
                )
            ),
            "leakage_free": True,
        },
        "reference_model": reference_model.to_dict(),
        "per_fixture": rows,
    }


def validate_parameter_learning_payload(payload: dict[str, object]) -> tuple[bool, str]:
    """Validate the structural and leakage contract of a learning payload."""

    if payload.get("schema_version") != PARAMETER_LEARNING_SCHEMA_VERSION:
        return False, "unexpected parameter-learning schema version"
    if payload.get("phase_labels_used_for_fit") is not False:
        return False, "phase labels must be excluded from fitting"
    rows = payload.get("per_fixture")
    if not isinstance(rows, list) or not rows:
        return False, "per_fixture rows are missing"
    if payload.get("fixture_count") != len(rows):
        return False, "fixture_count does not match rows"
    for row in rows:
        if not isinstance(row, dict):
            return False, "fixture row is not an object"
        training = row.get("training_fixtures")
        fit_contract = row.get("fit_input_contract")
        if (
            not isinstance(training, list)
            or not training
            or row.get("held_out_fixture") in training
        ):
            return False, "training and held-out fixtures overlap"
        if not isinstance(fit_contract, dict) or any(
            fit_contract.get(key) is not expected
            for key, expected in (
                ("phase_labels_used_for_fit", False),
                ("labels_present_in_fit_input", False),
                ("observation_only", True),
            )
        ):
            return False, "fit input leakage contract failed"
        observation_ids = row.get("observation_ids")
        truth_labels = row.get("truth_labels_for_scoring_only")
        if (
            not isinstance(observation_ids, list)
            or not isinstance(truth_labels, list)
            or not truth_labels
            or len(observation_ids) != len(truth_labels)
        ):
            return False, "held-out observations and scoring labels are misaligned"
        diagnostics = row.get("fit_diagnostics")
        if not isinstance(diagnostics, dict) or not isinstance(
            diagnostics.get("history"), list
        ):
            return False, "EM diagnostics are missing"
        if diagnostics.get("phase_labels_used_for_fit") is not False:
            return False, "EM diagnostics permit phase-label fitting"
        history = diagnostics["history"]
        if not history or diagnostics.get("iterations_used") != len(history):
            return False, "EM history length is inconsistent"
        for record in history:
            if not isinstance(record, dict):
                return False, "EM history contains a non-object record"
            for key in ("log_evidence_before", "log_evidence_after", "parameter_delta"):
                value = record.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    return False, f"EM history contains invalid {key}"
        model = row.get("learned_model")
        if not isinstance(model, dict):
            return False, "learned model is missing"
        try:
            emission = np.asarray(model["A"], dtype=float)
            transition = np.asarray(model["B"], dtype=float)
            prior = np.asarray(model["D"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"learned model tensors are invalid: {exc}"
        if (
            emission.shape != (len(OBSERVATION_LABELS), len(HIDDEN_STATE_LABELS))
            or transition.shape != (len(HIDDEN_STATE_LABELS), len(HIDDEN_STATE_LABELS))
            or prior.shape != (len(HIDDEN_STATE_LABELS),)
            or not np.all(np.isfinite(emission))
            or not np.all(np.isfinite(transition))
            or not np.all(np.isfinite(prior))
            or np.any(emission <= 0.0)
            or np.any(transition <= 0.0)
            or np.any(prior <= 0.0)
            or not np.allclose(emission.sum(axis=0), 1.0)
            or not np.allclose(transition.sum(axis=0), 1.0)
            or not np.isclose(prior.sum(), 1.0)
        ):
            return False, "learned model tensors violate probability contract"
        for score_name in ("weakly_supervised_em", "supervised_map", "fixed_reference"):
            score = row.get(score_name)
            if not isinstance(score, dict):
                return False, f"invalid {score_name} score"
            try:
                accuracy = float(score.get("accuracy", -1.0))
                balanced_accuracy = float(score.get("balanced_accuracy", -1.0))
                log_evidence = float(score.get("log_evidence", np.nan))
            except (TypeError, ValueError):
                return False, f"invalid {score_name} score"
            if (
                not 0.0 <= accuracy <= 1.0
                or not 0.0 <= balanced_accuracy <= 1.0
                or not np.isfinite(log_evidence)
            ):
                return False, f"invalid {score_name} score"
        sequences = row.get("training_observation_sequences")
        if not isinstance(sequences, list) or len(sequences) != len(training):
            return False, "training observation sequences are incomplete"
        try:
            for sequence in sequences:
                _as_observation_ids(
                    list(sequence), observation_count=len(OBSERVATION_LABELS)
                )
        except (TypeError, ValueError) as exc:
            return False, f"training observation sequence is invalid: {exc}"
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("leakage_free") is not True:
        return False, "summary leakage flag is false"
    return True, f"fixtures={len(rows)} observation_only=true"


def parameter_learning_payload(
    fixtures: Sequence[CorpusFixture],
) -> dict[str, object]:
    """Build the default deterministic parameter-learning artifact."""

    return weakly_supervised_parameter_learning_payload(fixtures)
