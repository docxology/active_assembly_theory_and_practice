"""Quantitative evaluation of phase decoding with negative controls.

This module turns the labelled :mod:`corpus` into the project's first *measured*
numbers. It scores how well three decoders recover the ground-truth latent
phase sequence of each program:

* ``static`` — the original sticky Bayesian filter with an implicit identity
  transition (``B = I``): ``posterior_t ∝ posterior_{t-1} * A[obs_t]``,
* ``temporal`` — the maximum-a-posteriori labels of the HMM forward filter from
  :func:`temporal_model.forward_filter` (genuine left-to-right dynamics),
* ``viterbi`` — the most-probable joint path from
  :func:`temporal_model.viterbi_decode`.

Two negative controls keep the metrics honest: a deterministically shuffled
truth (which a real decoder should beat by a wide margin) and a constant-uniform
predictor (which should be strictly worse than the temporal decoder). All
outputs are JSON-safe and deterministic across runs — no RNG is used anywhere.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .features import HIDDEN_STATE_LABELS
from .temporal_model import (
    TemporalGenerativeModel,
    forward_filter,
    viterbi_decode,
)

_EPS = 1e-12


def evaluate_decoding(
    predicted: Sequence[str],
    truth: Sequence[str],
    labels: Sequence[str],
) -> dict[str, object]:
    """Score a predicted label sequence against ground truth.

    Args:
        predicted: Predicted labels (one per step).
        truth: Ground-truth labels (one per step). Must match ``predicted``.
        labels: The full closed set of valid labels (rows/cols of the confusion
            matrix and the support set for balanced accuracy).

    Returns:
        A dict with ``accuracy`` (overall fraction correct),
        ``balanced_accuracy`` (mean of per-truth-label recall, averaged only
        over labels that actually appear in ``truth``), and ``confusion``
        (dict-of-dicts ``confusion[truth_label][predicted_label] -> count``).
    """

    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have equal length")
    if not truth:
        raise ValueError("truth must not be empty")

    label_list = list(labels)
    confusion: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in label_list}
        for true_label in label_list
    }
    correct = 0
    for pred, true in zip(predicted, truth):
        confusion[true][pred] += 1
        if pred == true:
            correct += 1
    accuracy = correct / len(truth)

    # Balanced accuracy: mean recall over the truth labels that appear.
    recalls: list[float] = []
    for true_label in label_list:
        row = confusion[true_label]
        total = sum(row.values())
        if total == 0:
            continue
        recalls.append(row[true_label] / total)
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion": {
            true_label: {pred_label: int(count) for pred_label, count in row.items()}
            for true_label, row in confusion.items()
        },
    }


def calibration(
    posteriors: Sequence[Sequence[float]],
    truth_state_ids: Sequence[int],
) -> dict[str, object]:
    """Compute Brier score and expected calibration error.

    Args:
        posteriors: Per-step posterior distributions over the hidden states.
        truth_state_ids: Ground-truth state index per step.

    Returns:
        A dict with ``brier`` (mean squared error of the full posterior against
        the one-hot truth) and ``ece`` (expected calibration error over 10 equal
        bins on the per-step maximum posterior probability).
    """

    if len(posteriors) != len(truth_state_ids):
        raise ValueError("posteriors and truth_state_ids must have equal length")
    if not posteriors:
        raise ValueError("posteriors must not be empty")

    matrix = np.asarray(posteriors, dtype=float)
    truth = np.asarray(truth_state_ids, dtype=int)
    n_steps, n_states = matrix.shape

    # Brier: mean over steps of sum_k (p_k - onehot_k)^2.
    onehot = np.zeros((n_steps, n_states), dtype=float)
    onehot[np.arange(n_steps), truth] = 1.0
    brier = float(np.mean(np.sum((matrix - onehot) ** 2, axis=1)))

    # ECE over 10 bins on the max-prob confidence.
    confidences = matrix.max(axis=1)
    predictions = matrix.argmax(axis=1)
    accuracies = (predictions == truth).astype(float)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # First bin is lower-inclusive so a confidence of exactly 0.0 is binned;
        # all later bins are lower-exclusive, upper-inclusive.
        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        bin_confidence = float(np.mean(confidences[in_bin]))
        bin_accuracy = float(np.mean(accuracies[in_bin]))
        ece += (count / n_steps) * abs(bin_accuracy - bin_confidence)

    return {"brier": float(brier), "ece": float(ece)}


def free_energy_convergence(
    free_energy_by_step: Sequence[float],
) -> dict[str, object]:
    """Summarize whether a free-energy trace decreases over steps.

    Args:
        free_energy_by_step: Per-step variational free-energy values.

    Returns:
        A dict with ``monotone_decrease_fraction`` (fraction of consecutive
        steps that did not increase), ``delta`` (final minus initial),
        ``final``, and ``initial``.
    """

    values = [float(value) for value in free_energy_by_step]
    if not values:
        raise ValueError("free_energy_by_step must not be empty")

    if len(values) == 1:
        monotone_fraction = 1.0
    else:
        non_increases = sum(
            1 for prev, curr in zip(values[:-1], values[1:]) if curr <= prev + _EPS
        )
        monotone_fraction = non_increases / (len(values) - 1)

    return {
        "monotone_decrease_fraction": float(monotone_fraction),
        "delta": float(values[-1] - values[0]),
        "final": float(values[-1]),
        "initial": float(values[0]),
    }


def _static_filter_labels(
    observation_ids: Sequence[int],
    model: TemporalGenerativeModel,
) -> tuple[str, ...]:
    """Decode labels with the original sticky filter (implicit ``B = I``).

    ``posterior_t ∝ posterior_{t-1} * A[obs_t]`` starting from the prior ``D``.
    Returns the per-step MAP labels.
    """

    matrix_a = np.asarray(model.A, dtype=float)
    posterior = np.asarray(model.D, dtype=float)
    posterior = posterior / posterior.sum()
    labels: list[str] = []
    for obs in observation_ids:
        unnormalized = posterior * matrix_a[int(obs), :]
        posterior = unnormalized / unnormalized.sum()
        labels.append(model.hidden_state_labels[int(np.argmax(posterior))])
    return tuple(labels)


def _temporal_filter_labels(
    observation_ids: Sequence[int],
    model: TemporalGenerativeModel,
) -> tuple[str, ...]:
    """Per-step MAP labels of the HMM forward filter."""

    result = forward_filter([int(o) for o in observation_ids], model)
    return result.most_likely_states


def _memoryless_decoder_labels(
    observation_ids: Sequence[int],
    model: TemporalGenerativeModel,
) -> tuple[str, ...]:
    """Decode labels with a memoryless per-step MAP decoder.

    ``argmax_s D[s] * A[obs_t, s]`` resets to the prior ``D`` at every step and
    uses no recursion. This is the *strongest honest non-temporal* baseline: it
    is what a decoder with the same observation model but no state memory would
    report. It is a fairer comparator than the sticky identity-transition filter,
    because the sticky filter is penalised by belief carried forward, not merely
    by the absence of dynamics.
    """

    matrix_a = np.asarray(model.A, dtype=float)
    prior = np.asarray(model.D, dtype=float)
    prior = prior / prior.sum()
    return tuple(
        model.hidden_state_labels[int(np.argmax(prior * matrix_a[int(obs), :]))]
        for obs in observation_ids
    )


def compare_models(
    corpus: Sequence[CorpusFixture],
    temporal_model: TemporalGenerativeModel,
) -> dict[str, dict[str, float]]:
    """Compare static, temporal, and Viterbi decoding accuracy per fixture.

    Args:
        corpus: The labelled fixtures.
        temporal_model: The temporal generative model (supplies ``A``/``B``/``D``
            for all three decoders).

    Returns:
        A dict mapping fixture name to a dict with ``static_accuracy``,
        ``temporal_accuracy``, and ``viterbi_accuracy`` — each the overall
        decoding accuracy against the fixture's ``phase_labels``.
    """

    labels = list(HIDDEN_STATE_LABELS)
    results: dict[str, dict[str, float]] = {}
    for fixture in corpus:
        observation_ids = fixture_observation_ids(fixture)
        truth = fixture.phase_labels

        static_labels = _static_filter_labels(observation_ids, temporal_model)
        memoryless_labels = _memoryless_decoder_labels(observation_ids, temporal_model)
        temporal_labels = _temporal_filter_labels(observation_ids, temporal_model)
        viterbi_labels, _ = viterbi_decode(
            [int(o) for o in observation_ids], temporal_model
        )

        results[fixture.name] = {
            "static_accuracy": float(
                evaluate_decoding(static_labels, truth, labels)["accuracy"]
            ),
            "memoryless_accuracy": float(
                evaluate_decoding(memoryless_labels, truth, labels)["accuracy"]
            ),
            "temporal_accuracy": float(
                evaluate_decoding(temporal_labels, truth, labels)["accuracy"]
            ),
            "viterbi_accuracy": float(
                evaluate_decoding(viterbi_labels, truth, labels)["accuracy"]
            ),
        }
    return results


def shuffled_label_control(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
    *,
    rounds: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Temporal decoding accuracy against seeded random label permutations.

    Each round draws a fresh permutation of the concatenated truth labels from a
    deterministic, seeded RNG and measures the temporal decoder's accuracy
    against that scrambled reference. A random permutation is a materially
    stronger null than a cyclic rotation: a rotation preserves local label
    adjacency and the run structure of the ground-truth sequence, whereas a full
    permutation destroys both. The mean over rounds should sit near chance — far
    below the accuracy against the true labels — proving the decoding metric is
    discriminative rather than vacuous.

    Only *distinct* permutations that also differ from the true label order are
    kept, so ``rounds`` in the returned payload is the actual number of unique
    scrambles averaged over. This avoids the wrap-around duplicates that a modular
    rotation schedule silently produces when ``rounds`` exceeds the label count
    (e.g. 20 requested rounds over an 18-label corpus). With a very small corpus
    the number of achievable distinct permutations can be below the requested
    ``rounds``; the returned ``rounds`` reflects what was actually used.

    Args:
        corpus: The labelled fixtures.
        model: The temporal generative model.
        rounds: Maximum number of distinct random permutations to average over.
        seed: Seed for the deterministic permutation RNG. Fixed so the control is
            byte-identical across repeated calls.

    Returns:
        A dict with ``mean_accuracy``, ``rounds`` (the actual distinct count
        used), and ``per_round`` accuracies.
    """

    if rounds <= 0:
        raise ValueError("rounds must be positive")

    labels = list(HIDDEN_STATE_LABELS)
    predicted: list[str] = []
    truth: list[str] = []
    for fixture in corpus:
        observation_ids = fixture_observation_ids(fixture)
        predicted.extend(_temporal_filter_labels(observation_ids, model))
        truth.extend(fixture.phase_labels)

    n = len(truth)
    rng = np.random.default_rng(seed)
    seen: set[tuple[str, ...]] = {tuple(truth)}
    per_round: list[float] = []
    # Bound the draws so a corpus admitting few distinct permutations cannot spin
    # forever; 64 attempts per requested round is ample headroom in practice.
    max_attempts = max(rounds * 64, 64)
    attempts = 0
    while len(per_round) < rounds and attempts < max_attempts:
        attempts += 1
        permutation = rng.permutation(n)
        scrambled = tuple(truth[int(index)] for index in permutation)
        if scrambled in seen:
            continue
        seen.add(scrambled)
        per_round.append(
            float(evaluate_decoding(predicted, list(scrambled), labels)["accuracy"])
        )

    if not per_round:
        raise ValueError(
            "corpus admits no label permutation distinct from the ground truth"
        )

    return {
        "mean_accuracy": float(np.mean(per_round)),
        "rounds": int(len(per_round)),
        "per_round": [float(value) for value in per_round],
    }


def uniform_belief_control(corpus: Sequence[CorpusFixture]) -> dict[str, object]:
    """Balanced accuracy of a constant-uniform (single-label) predictor.

    A maximally uninformative predictor always emits the first hidden-state
    label. Its balanced accuracy over the concatenated corpus is the floor that
    any informative decoder must clear.

    Args:
        corpus: The labelled fixtures.

    Returns:
        A dict with ``balanced_accuracy`` and ``accuracy`` of the constant
        predictor over the concatenated corpus labels.
    """

    labels = list(HIDDEN_STATE_LABELS)
    truth: list[str] = []
    for fixture in corpus:
        truth.extend(fixture.phase_labels)
    if not truth:
        raise ValueError("corpus must contain at least one labelled instruction")

    constant_label = labels[0]
    predicted = [constant_label] * len(truth)
    metrics = evaluate_decoding(predicted, truth, labels)
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
    }


def evaluation_payload(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
) -> dict[str, object]:
    """Bundle all evaluation metrics into a deterministic, JSON-safe payload.

    The payload aggregates per-fixture metrics (observation ids, truth labels,
    static/temporal/viterbi decoding, calibration of the forward filter, and the
    free-energy convergence summary), the cross-fixture ``compare_models``
    table, both negative controls, and a top-level summary of mean temporal and
    static accuracy.

    Args:
        corpus: The labelled fixtures.
        model: The temporal generative model.

    Returns:
        A JSON-safe dict that is byte-identical across repeated calls.
    """

    labels = list(HIDDEN_STATE_LABELS)
    label_index = {label: index for index, label in enumerate(labels)}

    per_fixture: dict[str, object] = {}
    temporal_accuracies: list[float] = []
    static_accuracies: list[float] = []
    memoryless_accuracies: list[float] = []

    for fixture in corpus:
        observation_ids = fixture_observation_ids(fixture)
        truth = fixture.phase_labels
        truth_state_ids = [label_index[label] for label in truth]

        filter_result = forward_filter([int(o) for o in observation_ids], model)
        temporal_labels = filter_result.most_likely_states
        static_labels = _static_filter_labels(observation_ids, model)
        memoryless_labels = _memoryless_decoder_labels(observation_ids, model)
        viterbi_labels, viterbi_log_prob = viterbi_decode(
            [int(o) for o in observation_ids], model
        )

        temporal_metrics = evaluate_decoding(temporal_labels, truth, labels)
        static_metrics = evaluate_decoding(static_labels, truth, labels)
        memoryless_metrics = evaluate_decoding(memoryless_labels, truth, labels)
        viterbi_metrics = evaluate_decoding(viterbi_labels, truth, labels)

        temporal_accuracies.append(float(temporal_metrics["accuracy"]))
        static_accuracies.append(float(static_metrics["accuracy"]))
        memoryless_accuracies.append(float(memoryless_metrics["accuracy"]))

        per_fixture[fixture.name] = {
            "observation_ids": [int(o) for o in observation_ids],
            "truth_labels": list(truth),
            "temporal_labels": list(temporal_labels),
            "static_labels": list(static_labels),
            "memoryless_labels": list(memoryless_labels),
            "viterbi_labels": list(viterbi_labels),
            "viterbi_log_prob": float(viterbi_log_prob),
            "temporal": temporal_metrics,
            "static": static_metrics,
            "memoryless": memoryless_metrics,
            "viterbi": viterbi_metrics,
            "calibration": calibration(
                [list(row) for row in filter_result.filtered_posteriors],
                truth_state_ids,
            ),
            "free_energy_convergence": free_energy_convergence(
                filter_result.free_energy_by_step
            ),
        }

    summary = {
        "mean_temporal_accuracy": float(np.mean(temporal_accuracies)),
        "mean_static_accuracy": float(np.mean(static_accuracies)),
        "mean_memoryless_accuracy": float(np.mean(memoryless_accuracies)),
        "best_non_temporal_accuracy": float(
            max(np.mean(static_accuracies), np.mean(memoryless_accuracies))
        ),
        "temporal_advantage_over_best_non_temporal": float(
            np.mean(temporal_accuracies)
            - max(np.mean(static_accuracies), np.mean(memoryless_accuracies))
        ),
        "fixture_count": int(len(corpus)),
    }

    return {
        "per_fixture": per_fixture,
        "compare_models": compare_models(corpus, model),
        "shuffled_label_control": shuffled_label_control(corpus, model),
        "uniform_belief_control": uniform_belief_control(corpus),
        "summary": summary,
    }
