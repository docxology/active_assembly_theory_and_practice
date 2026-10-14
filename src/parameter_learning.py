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

from pathlib import Path
from typing import Sequence, cast

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .evaluation import evaluate_decoding
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .model_fitting import fit_temporal_model
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload
from .temporal_model import (
    TemporalGenerativeModel,
    _as_observation_ids,
    build_temporal_model,
    forward_filter,
)

PARAMETER_LEARNING_SCHEMA_VERSION = "2.0.0"
PARAMETER_LEARNING_MAX_ITERATIONS = 32
PARAMETER_LEARNING_TOLERANCE = 1e-8
PARAMETER_LEARNING_TRANSITION_FLOOR = 1e-6
PARAMETER_LEARNING_MULTI_START_SEEDS = (11, 23, 47)


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


def _sequence_summary(sequence: Sequence[int]) -> dict[str, float]:
    """Return label-free descriptive statistics for a categorical sequence."""

    values = np.asarray(list(sequence), dtype=int)
    counts = np.bincount(values, minlength=len(OBSERVATION_LABELS)).astype(float)
    probabilities = counts / max(float(counts.sum()), 1.0)
    positive = probabilities[probabilities > 0.0]
    return {
        "observation_entropy": float(-np.sum(positive * np.log(positive))),
        "observation_category_count": float(np.count_nonzero(counts)),
        "adjacent_change_rate": float(np.mean(values[1:] != values[:-1]))
        if len(values) > 1
        else 0.0,
    }


def _sample_model_sequence(
    model: TemporalGenerativeModel, length: int, rng: np.random.Generator
) -> list[int]:
    """Sample one categorical sequence from a fitted HMM without labels."""

    emission = np.asarray(model.A, dtype=float)
    transition = np.asarray(model.B, dtype=float)
    prior = np.asarray(model.D, dtype=float)
    state = int(rng.choice(len(prior), p=prior))
    observations: list[int] = []
    for _ in range(length):
        observations.append(int(rng.choice(emission.shape[0], p=emission[:, state])))
        state = int(rng.choice(transition.shape[0], p=transition[:, state]))
    return observations


def _posterior_predictive_check(
    model: TemporalGenerativeModel,
    observation_sequences: Sequence[Sequence[int]],
    *,
    seed: int,
    replicates: int = 64,
) -> dict[str, object]:
    """Compare observed label-free sequence summaries with model replicates."""

    observed = [_sequence_summary(sequence) for sequence in observation_sequences]
    observed_summary = {
        key: float(np.mean([row[key] for row in observed])) for key in observed[0]
    }
    rng = np.random.default_rng(seed)
    replicated: dict[str, list[float]] = {key: [] for key in observed_summary}
    for _ in range(replicates):
        summaries = [
            _sequence_summary(_sample_model_sequence(model, len(sequence), rng))
            for sequence in observation_sequences
        ]
        for key in replicated:
            replicated[key].append(float(np.mean([row[key] for row in summaries])))
    intervals = {
        key: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in replicated.items()
    }
    return {
        "replicates": replicates,
        "seed": seed,
        "observed_summary": observed_summary,
        "central_95_percent_intervals": intervals,
        "checks": {
            key: bool(interval[0] <= observed_summary[key] <= interval[1])
            for key, interval in intervals.items()
        },
        "all_checks_passed": all(
            interval[0] <= observed_summary[key] <= interval[1]
            for key, interval in intervals.items()
        ),
        "uses_phase_labels": False,
    }


def _initialization_models(
    reference_model: TemporalGenerativeModel,
) -> tuple[tuple[str, int | None, TemporalGenerativeModel], ...]:
    """Return deterministic reference, uniform, and random EM starts.

    The random starts are generated without labels and retain the same ordered
    state coordinate system.  They are robustness diagnostics, not a license
    to select the best start using the held-out fixture.
    """

    n_observations = len(reference_model.observation_labels)
    n_states = len(reference_model.hidden_state_labels)
    uniform = TemporalGenerativeModel(
        A=np.full((n_observations, n_states), 1.0 / n_observations, dtype=float),
        B=np.full((n_states, n_states), 1.0 / n_states, dtype=float),
        D=np.full(n_states, 1.0 / n_states, dtype=float),
        observation_labels=tuple(reference_model.observation_labels),
        hidden_state_labels=tuple(reference_model.hidden_state_labels),
    )
    starts: list[tuple[str, int | None, TemporalGenerativeModel]] = [
        ("reference", None, reference_model),
        ("uniform", None, uniform),
    ]
    for seed in PARAMETER_LEARNING_MULTI_START_SEEDS:
        rng = np.random.default_rng(seed)
        emission = rng.dirichlet(np.ones(n_observations), size=n_states).T
        transition = np.full((n_states, n_states), 1e-3, dtype=float)
        for current in range(n_states):
            allowed = [current]
            if current + 1 < n_states:
                allowed.append(current + 1)
            transition[allowed, current] += rng.dirichlet(np.ones(len(allowed)))
        transition = _normalize_columns(transition)
        prior = rng.dirichlet(np.ones(n_states))
        starts.append(
            (
                f"random_{seed}",
                seed,
                TemporalGenerativeModel(
                    A=emission,
                    B=transition,
                    D=prior,
                    observation_labels=tuple(reference_model.observation_labels),
                    hidden_state_labels=tuple(reference_model.hidden_state_labels),
                ),
            )
        )
    for _, _, model in starts:
        model.validate()
    return tuple(starts)


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
    penalized_accuracies: list[float] = []
    predictive_checks: list[dict[str, object]] = []
    for held_out_index, held_out in enumerate(fixture_tuple):
        training = tuple(
            fixture
            for index, fixture in enumerate(fixture_tuple)
            if index != held_out_index
        )
        training_sequences = [
            list(fixture_observation_ids(fixture)) for fixture in training
        ]
        multi_start_rows: list[dict[str, object]] = []
        multi_start_predictions: dict[str, list[str]] = {}
        for initialization, seed, initial_model in _initialization_models(
            reference_model
        ):
            fitted_model, diagnostics = fit_ordered_hmm_em(
                training_sequences,
                initial_model=initial_model,
                max_iterations=max_iterations,
            )
            score = _score_model(
                list(fixture_observation_ids(held_out)),
                list(held_out.phase_labels),
                fitted_model,
            )
            multi_start_predictions[initialization] = list(
                cast(list[str], score["temporal_labels"])
            )
            multi_start_rows.append(
                {
                    "initialization": initialization,
                    "seed": seed,
                    "held_out_score": score,
                    "fit_diagnostics": diagnostics,
                    "learned_model": fitted_model.to_dict(),
                }
            )
        canonical = next(
            row for row in multi_start_rows if row["initialization"] == "reference"
        )
        learned_score = cast(dict[str, object], canonical["held_out_score"])
        fit_diagnostics = cast(dict[str, object], canonical["fit_diagnostics"])
        supervised_model = fit_temporal_model(training)
        observation_ids = list(fixture_observation_ids(held_out))
        truth = list(held_out.phase_labels)
        supervised_score = _score_model(observation_ids, truth, supervised_model)
        reference_score = _score_model(observation_ids, truth, reference_model)
        penalized_model, penalized_diagnostics = fit_ordered_hmm_em(
            training_sequences,
            initial_model=reference_model,
            emission_alpha=1.0,
            transition_alpha=1.0,
            prior_alpha=1.0,
            max_iterations=max_iterations,
        )
        penalized_score = _score_model(observation_ids, truth, penalized_model)
        predictive_check = _posterior_predictive_check(
            penalized_model,
            training_sequences,
            seed=20260821 + held_out_index,
        )
        weak_accuracies.append(float(cast(float, learned_score["accuracy"])))
        supervised_accuracies.append(float(cast(float, supervised_score["accuracy"])))
        reference_accuracies.append(float(cast(float, reference_score["accuracy"])))
        penalized_accuracies.append(float(cast(float, penalized_score["accuracy"])))
        predictive_checks.append(predictive_check)
        multi_start_accuracy = {
            str(row["initialization"]): float(
                cast(
                    float,
                    cast(dict[str, object], row["held_out_score"])["accuracy"],
                )
            )
            for row in multi_start_rows
        }
        agreement_values: list[float] = []
        initialization_names = tuple(multi_start_predictions)
        for left_index, left_name in enumerate(initialization_names):
            for right_name in initialization_names[left_index + 1 :]:
                left = multi_start_predictions[left_name]
                right = multi_start_predictions[right_name]
                agreement_values.append(
                    float(sum(a == b for a, b in zip(left, right, strict=True)))
                    / len(left)
                )
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
                "penalized_dirichlet_map": penalized_score,
                "penalized_fit_diagnostics": penalized_diagnostics,
                "posterior_predictive_check": predictive_check,
                "learned_model": cast(dict[str, object], canonical["learned_model"]),
                "fit_diagnostics": fit_diagnostics,
                "multi_start": multi_start_rows,
                "multi_start_summary": {
                    "accuracy_by_initialization": multi_start_accuracy,
                    "accuracy_min": min(multi_start_accuracy.values()),
                    "accuracy_max": max(multi_start_accuracy.values()),
                    "accuracy_range": max(multi_start_accuracy.values())
                    - min(multi_start_accuracy.values()),
                    "mean_pairwise_label_agreement": float(np.mean(agreement_values)),
                    "pairwise_comparison_count": len(agreement_values),
                },
            }
        )

    multi_start_summary_rows = [
        cast(dict[str, object], row["multi_start_summary"]) for row in rows
    ]
    all_multi_start_scores = [
        float(value)
        for row in multi_start_summary_rows
        for value in cast(dict[str, float], row["accuracy_by_initialization"]).values()
    ]
    protocol = scientific_protocol_payload()
    limitations = [
        "The ordered state coordinates are an architectural scaffold rather than learned semantic identities.",
        "The multi-start analysis is a finite robustness result, not population-level inference.",
    ]
    underpowered_cases = [
        "The current authored-fixture surface is below the protocol minimum independent-family target; fixture identities are not independent source lineages.",
    ]
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={"src/parameter_learning.py": sha256_path(Path(__file__))},
        toolchain_versions=toolchain_versions(("numpy",)),
        status="underpowered",
        limitations=limitations,
        unsupported_cases=(
            "Held-out label fitting and post-hoc state relabeling are unsupported by the fitting contract.",
        ),
        underpowered_cases=underpowered_cases,
    )
    return {
        **envelope,
        "schema_version": PARAMETER_LEARNING_SCHEMA_VERSION,
        "artifact": "output/data/parameter_learning.json",
        "method": "leave_one_fixture_out_ordered_weakly_supervised_em",
        "fit_scope": "training fixture observation IDs only; held-out fixture excluded",
        "phase_labels_used_for_fit": False,
        "fixture_count": len(fixture_tuple),
        "max_iterations": max_iterations,
        "policies": [
            "weakly_supervised_em",
            "supervised_map",
            "fixed_reference",
            "penalized_dirichlet_map",
        ],
        "model_selection_ledger": {
            "candidate_models": [
                "weakly_supervised_em",
                "supervised_map",
                "fixed_reference",
                "penalized_dirichlet_map",
            ],
            "selected_fit": "reference_initialization_for_primary_endpoint",
            "selection_inputs": "training_observation_log_evidence_only",
            "held_out_labels_available_to_selection": False,
            "held_out_likelihood_reported_separately": True,
        },
        "summary": {
            "fixture_count": len(fixture_tuple),
            "mean_weakly_supervised_accuracy": float(np.mean(weak_accuracies)),
            "mean_supervised_accuracy": float(np.mean(supervised_accuracies)),
            "mean_fixed_reference_accuracy": float(np.mean(reference_accuracies)),
            "mean_penalized_dirichlet_map_accuracy": float(
                np.mean(penalized_accuracies)
            ),
            "weakly_supervised_minus_supervised": float(
                np.mean(weak_accuracies) - np.mean(supervised_accuracies)
            ),
            "weakly_supervised_minus_reference": float(
                np.mean(weak_accuracies) - np.mean(reference_accuracies)
            ),
            "penalized_minus_reference": float(
                np.mean(penalized_accuracies) - np.mean(reference_accuracies)
            ),
            "converged_holdout_count": int(
                sum(
                    bool(cast(dict[str, object], row["fit_diagnostics"])["converged"])
                    for row in rows
                )
            ),
            "leakage_free": True,
            "multi_start_initializations": [
                name for name, _, _ in _initialization_models(reference_model)
            ],
            "multi_start_count": len(all_multi_start_scores),
            "multi_start_mean_accuracy": float(np.mean(all_multi_start_scores)),
            "multi_start_min_accuracy": float(min(all_multi_start_scores)),
            "multi_start_max_accuracy": float(max(all_multi_start_scores)),
            "multi_start_mean_accuracy_range": float(
                np.mean(
                    [
                        float(cast(float, row["accuracy_range"]))
                        for row in multi_start_summary_rows
                    ]
                )
            ),
            "multi_start_mean_pairwise_label_agreement": float(
                np.mean(
                    [
                        float(cast(float, row["mean_pairwise_label_agreement"]))
                        for row in multi_start_summary_rows
                    ]
                )
            ),
            "posterior_predictive_check_pass_rate": float(
                np.mean([bool(row["all_checks_passed"]) for row in predictive_checks])
            ),
        },
        "reference_model": reference_model.to_dict(),
        "per_fixture": rows,
        "limitations": limitations,
        "unsupported_cases": list(envelope["unsupported_cases"]),
        "underpowered_cases": underpowered_cases,
    }


def validate_parameter_learning_payload(payload: dict[str, object]) -> tuple[bool, str]:
    """Validate the structural and leakage contract of a learning payload."""

    if payload.get("schema_version") != PARAMETER_LEARNING_SCHEMA_VERSION:
        return False, "unexpected parameter-learning schema version"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"parameter-learning receipt: {receipt_detail}"
    if payload.get("status") not in {"validated_bounded", "underpowered"}:
        return False, "parameter-learning evidence status is invalid"
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
        for score_name in (
            "weakly_supervised_em",
            "supervised_map",
            "fixed_reference",
            "penalized_dirichlet_map",
        ):
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
        predictive_check = row.get("posterior_predictive_check")
        if not isinstance(predictive_check, dict):
            return False, "posterior-predictive check is missing"
        if predictive_check.get("uses_phase_labels") is not False:
            return False, "posterior-predictive check used phase labels"
        if not isinstance(predictive_check.get("checks"), dict):
            return False, "posterior-predictive check diagnostics are missing"
        penalized_diagnostics = row.get("penalized_fit_diagnostics")
        if (
            not isinstance(penalized_diagnostics, dict)
            or penalized_diagnostics.get("phase_labels_used_for_fit") is not False
        ):
            return False, "penalized fit leakage contract failed"
        multi_start = row.get("multi_start")
        multi_summary = row.get("multi_start_summary")
        if (
            not isinstance(multi_start, list)
            or len(multi_start) < 3
            or not isinstance(multi_summary, dict)
        ):
            return False, "multi-start diagnostics are missing"
        observed_initializations: set[str] = set()
        for start in multi_start:
            if not isinstance(start, dict):
                return False, "multi-start row is not an object"
            initialization = start.get("initialization")
            score = start.get("held_out_score")
            diagnostics = start.get("fit_diagnostics")
            model = start.get("learned_model")
            if (
                not isinstance(initialization, str)
                or initialization in observed_initializations
                or not isinstance(score, dict)
                or not isinstance(diagnostics, dict)
                or not isinstance(model, dict)
            ):
                return False, "multi-start row is malformed"
            observed_initializations.add(initialization)
            try:
                score_accuracy = float(score.get("accuracy", -1.0))
                score_balanced = float(score.get("balanced_accuracy", -1.0))
                score_evidence = float(score.get("log_evidence", np.nan))
            except (TypeError, ValueError):
                return False, "multi-start score is invalid"
            if (
                not 0.0 <= score_accuracy <= 1.0
                or not 0.0 <= score_balanced <= 1.0
                or not np.isfinite(score_evidence)
            ):
                return False, "multi-start score is out of range"
            labels = score.get("temporal_labels")
            if (
                not isinstance(labels, list)
                or len(labels) != len(truth_labels)
                or any(not isinstance(label, str) for label in labels)
            ):
                return False, "multi-start labels are misaligned"
            if diagnostics.get("phase_labels_used_for_fit") is not False:
                return False, "multi-start fit used phase labels"
        expected_summary = multi_summary.get("accuracy_by_initialization")
        if (
            not isinstance(expected_summary, dict)
            or set(expected_summary) != observed_initializations
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
                for value in expected_summary.values()
            )
        ):
            return False, "multi-start accuracy summary is malformed"
        for key in (
            "accuracy_min",
            "accuracy_max",
            "accuracy_range",
            "mean_pairwise_label_agreement",
        ):
            value = multi_summary.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                return False, f"multi-start summary {key} is invalid"
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
    ledger = payload.get("model_selection_ledger")
    if (
        not isinstance(ledger, dict)
        or ledger.get("held_out_labels_available_to_selection") is not False
    ):
        return False, "model-selection ledger is not label blind"
    for key in (
        "mean_penalized_dirichlet_map_accuracy",
        "penalized_minus_reference",
        "posterior_predictive_check_pass_rate",
    ):
        value = summary.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
        ):
            return False, f"summary field {key} is invalid"
    for key in (
        "multi_start_count",
        "multi_start_mean_accuracy",
        "multi_start_min_accuracy",
        "multi_start_max_accuracy",
        "multi_start_mean_accuracy_range",
        "multi_start_mean_pairwise_label_agreement",
    ):
        value = summary.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
        ):
            return False, f"summary multi-start field {key} is invalid"
    if int(summary["multi_start_count"]) != sum(
        len(cast(list[object], row["multi_start"])) for row in rows
    ):
        return False, "summary multi-start count is inconsistent"
    return True, f"fixtures={len(rows)} observation_only=true"


def parameter_learning_payload(
    fixtures: Sequence[CorpusFixture],
) -> dict[str, object]:
    """Build the default deterministic parameter-learning artifact."""

    return weakly_supervised_parameter_learning_payload(fixtures)
