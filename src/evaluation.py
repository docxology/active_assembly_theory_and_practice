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

import math
from typing import Any, Sequence, cast

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .features import HIDDEN_STATE_LABELS
from .active_selection import active_observation_order, fixed_order_steps_to_confidence
from .temporal_model import (
    TemporalGenerativeModel,
    forward_backward,
    forward_filter,
    viterbi_decode,
)

_EPS = 1e-12

# Upper bound on RNG draws per requested control round, so a corpus admitting
# very few distinct label permutations cannot spin forever in the loop below.
_PER_ROUND_DRAW_ATTEMPTS = 64


def evaluate_decoding(
    predicted: Sequence[str],
    truth: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the ``100*(1-alpha)`` Wilson score interval for the ratio ``k/n``.

    Unlike the normal approximation, the Wilson interval is well behaved for
    small ``n`` and for proportions near 0 or 1, so it is a robust way to report
    uncertainty around a decoding accuracy computed on a handful of steps.

    Args:
        k: Number of successes (e.g. correct labels).
        n: Number of trials (e.g. total labels).
        z: Standard normal quantile; 1.96 gives a 95% interval.

    Returns:
        The ``(lower, upper)`` bounds of the interval.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return (float(center - margin), float(center + margin))


def calibration_bins(
    posteriors: Sequence[Sequence[float]],
    truth_state_ids: Sequence[int],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Return per-bin calibration statistics over max-posterior confidence.

    Each bin records ``count``, ``confidence`` (mean max posterior) and
    ``accuracy`` (mean correctness) — the data behind a reliability diagram and
    the expected calibration error (ECE).

    Args:
        posteriors: Per-step posterior distributions over the hidden states.
        truth_state_ids: Ground-truth state index per step.
        n_bins: Number of equal-width confidence bins.

    Returns:
        A JSON-safe list of bin records, one per non-empty bin.
    """

    if len(posteriors) != len(truth_state_ids):
        raise ValueError("posteriors and truth_state_ids must have equal length")
    if not posteriors:
        raise ValueError("posteriors must not be empty")

    matrix = np.asarray(posteriors, dtype=float)
    truth = np.asarray(truth_state_ids, dtype=int)
    confidences = matrix.max(axis=1)
    predictions = matrix.argmax(axis=1)
    accuracies = (predictions == truth).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
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
        bins.append(
            {
                "bin_index": bin_index,
                "count": count,
                "confidence": float(np.mean(confidences[in_bin])),
                "accuracy": float(np.mean(accuracies[in_bin])),
            }
        )
    return bins


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
        the one-hot truth), ``ece`` (expected calibration error over 10 equal
        bins on the per-step maximum posterior probability), the per-bin
        ``bins`` (see :func:`calibration_bins`), and ``n_steps``.
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

    bins = calibration_bins(posteriors, truth_state_ids)
    ece = float(
        sum(
            (cast(int, bin_["count"]) / n_steps)
            * abs(cast(float, bin_["accuracy"]) - cast(float, bin_["confidence"]))
            for bin_ in bins
        )
    )

    return {
        "brier": brier,
        "ece": ece,
        "bins": bins,
        "n_steps": int(n_steps),
    }


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
    # forever (see _PER_ROUND_DRAW_ATTEMPTS).
    max_attempts = max(rounds * _PER_ROUND_DRAW_ATTEMPTS, _PER_ROUND_DRAW_ATTEMPTS)
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

    std = float(np.std(per_round))
    return {
        "mean_accuracy": float(np.mean(per_round)),
        "std": std,
        "se": std / math.sqrt(len(per_round)),
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


def evaluate_segmentation(
    predicted: Sequence[str],
    truth: Sequence[str],
) -> dict[str, Any]:
    """Score phase-segmentation recovery against ground truth.

    A phase boundary is a step index ``i`` (``1 <= i < n``) where the label
    changes from step ``i-1`` to step ``i``. We compare the predicted boundary
    set (from per-step decoded labels) against the ground-truth boundary set and
    report boundary precision/recall/F1 plus per-step label accuracy.

    Args:
        predicted: Per-step predicted labels.
        truth: Per-step ground-truth labels (same length).

    Returns:
        A JSON-safe dict with ``boundary_precision``, ``boundary_recall``,
        ``boundary_f1``, ``step_accuracy``, and the two boundary-index sets.
    """

    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have equal length")
    if not truth:
        raise ValueError("truth must not be empty")

    n = len(truth)
    predicted_boundaries = {i for i in range(1, n) if predicted[i] != predicted[i - 1]}
    true_boundaries = {i for i in range(1, n) if truth[i] != truth[i - 1]}
    true_positives = len(predicted_boundaries & true_boundaries)
    precision = (
        true_positives / len(predicted_boundaries) if predicted_boundaries else 0.0
    )
    recall = true_positives / len(true_boundaries) if true_boundaries else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    step_accuracy = sum(1 for p, t in zip(predicted, truth) if p == t) / n

    return {
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
        "step_accuracy": float(step_accuracy),
        "predicted_boundaries": sorted(predicted_boundaries),
        "true_boundaries": sorted(true_boundaries),
    }


def segmentation_evaluation(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
) -> dict[str, Any]:
    """Measure phase-segmentation recovery from the smoothed forward-backward beliefs.

    Per-fixture and pooled boundary precision/recall/F1 plus step accuracy are
    reported, so the F3 segmentation module is now *measured* against the
    hand-authored ground truth rather than only self-consistent.

    Args:
        corpus: The labelled fixtures.
        model: The temporal generative model.

    Returns:
        A JSON-safe dict with ``per_fixture`` metrics and a ``pooled`` summary.
    """

    per_fixture: dict[str, dict[str, Any]] = {}
    pooled_predicted: list[str] = []
    pooled_truth: list[str] = []
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        smoothing = forward_backward(observation_ids, model)
        step_labels = [
            model.hidden_state_labels[int(np.argmax(row))]
            for row in smoothing.smoothed_posteriors
        ]
        per_fixture[fixture.name] = evaluate_segmentation(
            step_labels, list(fixture.phase_labels)
        )
        per_fixture[fixture.name]["predicted_labels"] = list(step_labels)
        per_fixture[fixture.name]["truth_labels"] = list(fixture.phase_labels)
        pooled_predicted.extend(step_labels)
        pooled_truth.extend(fixture.phase_labels)
    return {
        "per_fixture": per_fixture,
        "pooled": evaluate_segmentation(pooled_predicted, pooled_truth),
    }


def _ambiguous_likelihood_temporal_model(
    model: TemporalGenerativeModel,
    blend: float,
) -> TemporalGenerativeModel:
    """Return a temporal model whose likelihood is blended toward uniform.

    ``blend = 0`` keeps the authored likelihood; ``blend = 1`` makes the
    likelihood uniform over observations (maximal per-step ambiguity). Each
    column is renormalised so the result stays a valid categorical likelihood.
    This is the controlled knob for the ambiguity ablation: it changes only
    the observation model (``A``), leaving ``B`` and ``D`` identical.
    """

    authored = np.clip(np.asarray(model.A, dtype=float), 1e-6, None)
    uniform = np.ones_like(authored) / authored.shape[1]
    mixed = (1.0 - blend) * authored + blend * uniform
    mixed = mixed / mixed.sum(axis=0, keepdims=True)
    ambiguous = TemporalGenerativeModel(
        A=mixed.astype(float),
        B=np.asarray(model.B, dtype=float),
        D=np.asarray(model.D, dtype=float),
        observation_labels=model.observation_labels,
        hidden_state_labels=model.hidden_state_labels,
    )
    ambiguous.validate()
    return ambiguous


def _segmentation_boundary_f1_for_decoder(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
    *,
    memoryless: bool = False,
) -> float:
    """Pooled boundary F1 of a decoder's per-step labels against ground truth.

    ``memoryless=False`` decodes each step from the smoothed forward-backward
    posteriors (the temporal segmentation); ``memoryless=True`` decodes each step
    independently as ``argmax(D * A[obs, :])``. Both use the same generative
    model, so comparing the two at a given ambiguity blend isolates how much the
    transition structure improves boundary recovery when per-step decoding is
    ambiguous.
    """

    predicted: list[str] = []
    truth: list[str] = []
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        if memoryless:
            prior = np.asarray(model.D, dtype=float)
            likelihood = np.asarray(model.A, dtype=float)
            step_labels = [
                model.hidden_state_labels[int(np.argmax(prior * likelihood[obs, :]))]
                for obs in observation_ids
            ]
        else:
            smoothing = forward_backward(observation_ids, model)
            step_labels = [
                model.hidden_state_labels[int(np.argmax(row))]
                for row in smoothing.smoothed_posteriors
            ]
        predicted.extend(step_labels)
        truth.extend(fixture.phase_labels)
    return float(evaluate_segmentation(predicted, truth)["boundary_f1"])


def ambiguity_ablation(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
    blends: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Sweep observation ambiguity and report decoder accuracy at each blend.

    Because the authored likelihood is near-diagonal, per-step decoding already
    resolves most phases and the temporal transition matrix adds little. Blending
    the likelihood toward uniform makes individual observations ambiguous; here
    the left-to-right transition structure carries real information. This
    ablation therefore *demonstrates the mechanism* behind the head-to-head
    result: the temporal advantage over the memoryless decoder emerges exactly
    when observations become ambiguous (and the corpus is unchanged, only the
    likelihood conditioning is swept).

    For each blend the rows also record pooled phase-boundary F1 for the temporal
    (smoothed) and memoryless segmentations, so the same mechanism is shown for
    *segmentation*, not only step decoding.

    Args:
        corpus: The labelled fixtures.
        model: The temporal generative model (supplies ``A``/``B``/``D``).
        blends: Likelihood-ambiguity values to sweep (defaults to 0.0..1.0 in
            steps of 0.1). Deterministic.

    Returns:
        A JSON-safe dict with ``blends``, per-blend ``rows`` of decoder mean
        accuracies and segmentation boundary F1, and ``crossover_blend`` (the
        first blend at which the temporal decoder beats the memoryless decoder).
    """

    if blends is None:
        blends = tuple(round(float(v), 3) for v in np.linspace(0.0, 1.0, 11))
    rows: list[dict[str, object]] = []
    crossover: float | None = None
    for blend in blends:
        ambiguous = _ambiguous_likelihood_temporal_model(model, float(blend))
        result = compare_models(corpus, ambiguous)
        names = list(result.keys())

        def _mean(key: str) -> float:
            return float(sum(result[n][key] for n in names) / len(names))

        temporal = _mean("temporal_accuracy")
        memoryless = _mean("memoryless_accuracy")
        sticky = _mean("static_accuracy")
        viterbi = _mean("viterbi_accuracy")
        segmentation_temporal = _segmentation_boundary_f1_for_decoder(
            corpus, ambiguous, memoryless=False
        )
        segmentation_memoryless = _segmentation_boundary_f1_for_decoder(
            corpus, ambiguous, memoryless=True
        )
        # Active layer: does the EFE policy reach confidence in fewer steps than a
        # fixed scan? Under intermediate ambiguity observations become
        # differentially informative, so the greedy EFE ordering matters (and is
        # never worse than the fixed order at any blend).
        active_advantage = 0
        active_worse = 0
        prior = np.asarray(model.D, dtype=float)
        preference = np.ones(ambiguous.A.shape[0], dtype=float)
        ambiguous_a = np.asarray(ambiguous.A, dtype=float)
        for fixture in corpus:
            ids = [int(o) for o in fixture_observation_ids(fixture)]
            active_steps = active_observation_order(
                ids, ambiguous_a, prior, preference, 0.7
            ).steps_to_confidence
            fixed_steps = fixed_order_steps_to_confidence(ids, ambiguous_a, prior, 0.7)
            if active_steps is None and fixed_steps is None:
                continue
            if active_steps is None or (
                fixed_steps is not None and active_steps > fixed_steps
            ):
                active_worse += 1
            elif fixed_steps is None or active_steps < fixed_steps:
                active_advantage += 1
        gap = temporal - memoryless
        if crossover is None and gap > 1e-9:
            crossover = float(blend)
        rows.append(
            {
                "blend": float(blend),
                "temporal_accuracy": temporal,
                "memoryless_accuracy": memoryless,
                "static_accuracy": sticky,
                "viterbi_accuracy": viterbi,
                "temporal_minus_memoryless": float(gap),
                "temporal_segmentation_f1": segmentation_temporal,
                "memoryless_segmentation_f1": segmentation_memoryless,
                "active_advantage_count": int(active_advantage),
                "active_worse_count": int(active_worse),
            }
        )

    authored_blend_active_advantage = int(cast(int, rows[0]["active_advantage_count"]))
    active_never_worse = all(
        int(cast(int, row["active_worse_count"])) == 0 for row in rows
    )
    advantage_fractions = [
        int(cast(int, row["active_advantage_count"])) / max(1, len(corpus))
        for row in rows
    ]
    advantage_blends = [
        float(cast(float, rows[i]["blend"]))
        for i, fraction in enumerate(advantage_fractions)
        if fraction > 0
    ]
    max_advantage_fraction = (
        float(max(advantage_fractions)) if advantage_fractions else 0.0
    )

    return {
        "blends": [float(v) for v in blends],
        "rows": rows,
        "crossover_blend": crossover,
        "active_max_advantage_fraction": max_advantage_fraction,
        "active_advantage_band": (
            [min(advantage_blends), max(advantage_blends)] if advantage_blends else []
        ),
        "active_never_worse": bool(active_never_worse),
        "authored_blend_active_advantage": authored_blend_active_advantage,
    }


def pooled_decoding_accuracies(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
) -> dict[str, float]:
    """Pooled (micro) decoding accuracy of each decoder over the given corpus.

    Hits and totals are accumulated across all fixture steps, so the pools are
    comparable to the Wilson CIs reported in the summary (which bracket pooled,
    not macro, accuracy).
    """

    hits = {"temporal": 0, "static": 0, "memoryless": 0}
    total = 0
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        truth = list(fixture.phase_labels)
        temporal = forward_filter(observation_ids, model).most_likely_states
        static = _static_filter_labels(observation_ids, model)
        memoryless = _memoryless_decoder_labels(observation_ids, model)
        hits["temporal"] += sum(1 for p, t in zip(temporal, truth) if p == t)
        hits["static"] += sum(1 for p, t in zip(static, truth) if p == t)
        hits["memoryless"] += sum(1 for p, t in zip(memoryless, truth) if p == t)
        total += len(truth)
    if not total:
        return {"temporal": 0.0, "static": 0.0, "memoryless": 0.0}
    return {key: float(count / total) for key, count in hits.items()}


def corpus_stability_analysis(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
) -> dict[str, Any]:
    """Leave-one-out sensitivity of the pooled decoding accuracies.

    Because the corpus is small (see the manuscript's honest limits), the
    headline pooled accuracies could in principle be driven by a single fixture.
    This analysis holds out one fixture at a time and recomputes the pooled
    accuracies over the remaining corpus, reporting the resulting range and
    whether the headline pattern (temporal beats the static filter, and stays
    well above chance) survives every single-fixture removal.

    Args:
        corpus: The labelled fixtures.
        model: The temporal generative model.

    Returns:
        A JSON-safe dict with ``reference`` (full-corpus pools), ``per_holdout``
        results, per-decoder pooled ``*_range``, and two booleans binding whether
        the pattern holds under every holdout.
    """

    reference = pooled_decoding_accuracies(corpus, model)
    per_holdout: list[dict[str, object]] = []
    temporal_above_static = True
    temporal_above_chance = True
    chance_floor = 0.2
    clearance = 0.25  # temporal pooled must stay >= chance+clearance under holdout

    for fixture in corpus:
        reduced = [f for f in corpus if f is not fixture]
        accuracies = pooled_decoding_accuracies(reduced, model)
        per_holdout.append(
            {
                "held_out_fixture": fixture.name,
                "temporal": accuracies["temporal"],
                "static": accuracies["static"],
                "memoryless": accuracies["memoryless"],
            }
        )
        if not (accuracies["temporal"] > accuracies["static"]):
            temporal_above_static = False
        if not (accuracies["temporal"] > chance_floor + clearance):
            temporal_above_chance = False

    def _range(key: str) -> dict[str, float]:
        values = [float(cast(float, row[key])) for row in per_holdout]
        return {
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
        }

    return {
        "reference": reference,
        "per_holdout": per_holdout,
        "temporal_pooled_range": _range("temporal"),
        "static_pooled_range": _range("static"),
        "memoryless_pooled_range": _range("memoryless"),
        "chance_floor": float(chance_floor),
        "chance_clearance": float(clearance),
        "all_holdouts_temporal_above_static": temporal_above_static,
        "all_holdouts_temporal_above_chance": temporal_above_chance,
    }


def per_phase_recovery_metrics(pooled_confusion: dict[str, object]) -> dict[str, Any]:
    """Per-phase precision/recall/F1 from a pooled confusion matrix.

    The pooled confusion matrix (rows = truth, columns = predicted, counts) from
    the temporal forward filter is collapsed into per-class recovery metrics.
    This quantifies *which* phases are recoverable — e.g. that the ``exit`` (and
    especially ``guard_check``) phase is observation-underdetermined — instead of
    citing the limitation qualitatively. A phase never predicted anywhere (its
    column all zero) or never present (its row) is reported with 0.0 metrics.

    Args:
        pooled_confusion: Dict with ``labels`` and a square ``matrix`` of counts.

    Returns:
        A JSON-safe dict with ``labels``, per-phase metrics, ``macro_f1``, the
        hardest phase, and the best-recall phase.
    """

    labels = list(cast(list[str], pooled_confusion["labels"]))
    matrix: list[list[int]] = [
        [int(value) for value in cast(list, row)]
        for row in cast(list, pooled_confusion["matrix"])
    ]
    per_phase: dict[str, dict[str, object]] = {}
    for i, phase in enumerate(labels):
        true_positives = matrix[i][i]
        false_positives = sum(matrix[j][i] for j in range(len(labels)) if j != i)
        false_negatives = sum(matrix[i][j] for j in range(len(labels)) if j != i)
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives)
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives)
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_phase[phase] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_count": int(true_positives + false_negatives),
        }
    macro_f1 = float(sum(cast(float, per_phase[p]["f1"]) for p in labels) / len(labels))
    hardest = min(labels, key=lambda p: float(cast(float, per_phase[p]["f1"])))
    best_recall = max(labels, key=lambda p: float(cast(float, per_phase[p]["recall"])))
    return {
        "labels": labels,
        "per_phase": per_phase,
        "macro_f1": macro_f1,
        "hardest_phase": hardest,
        "best_recall_phase": best_recall,
    }


def evaluation_payload(
    corpus: Sequence[CorpusFixture],
    model: TemporalGenerativeModel,
) -> dict[str, Any]:
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

    # Pooled (concatenated-corpus) accumulators for confidence intervals and
    # the confusion-matrix / calibration visualizations.
    temporal_hits = 0
    temporal_total = 0
    static_hits = 0
    static_total = 0
    memoryless_hits = 0
    memoryless_total = 0
    pooled_confusion: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels} for true_label in labels
    }
    pooled_posteriors: list[list[float]] = []
    pooled_truth_ids: list[int] = []

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

        temporal_accuracies.append(cast(float, temporal_metrics["accuracy"]))
        static_accuracies.append(cast(float, static_metrics["accuracy"]))
        memoryless_accuracies.append(cast(float, memoryless_metrics["accuracy"]))

        temporal_hits += sum(1 for p, t in zip(temporal_labels, truth) if p == t)
        temporal_total += len(truth)
        static_hits += sum(1 for p, t in zip(static_labels, truth) if p == t)
        static_total += len(truth)
        memoryless_hits += sum(1 for p, t in zip(memoryless_labels, truth) if p == t)
        memoryless_total += len(truth)

        for true_label, row in temporal_metrics["confusion"].items():
            for pred_label, count in row.items():
                pooled_confusion[true_label][pred_label] += int(count)

        pooled_posteriors.extend(
            [list(row) for row in filter_result.filtered_posteriors]
        )
        pooled_truth_ids.extend(truth_state_ids)

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

    mean_temporal = float(np.mean(temporal_accuracies))
    mean_static = float(np.mean(static_accuracies))
    mean_memoryless = float(np.mean(memoryless_accuracies))
    best_non_temporal = max(mean_static, mean_memoryless)
    summary = {
        "mean_temporal_accuracy": mean_temporal,
        "mean_static_accuracy": mean_static,
        "mean_memoryless_accuracy": mean_memoryless,
        "best_non_temporal_accuracy": best_non_temporal,
        "temporal_advantage_over_best_non_temporal": mean_temporal - best_non_temporal,
        "temporal_memoryless_diff": mean_temporal - mean_memoryless,
        "pooled_temporal_accuracy": temporal_hits / temporal_total,
        "pooled_static_accuracy": static_hits / static_total,
        "pooled_memoryless_accuracy": memoryless_hits / memoryless_total,
        "temporal_ci95": list(wilson_ci(temporal_hits, temporal_total)),
        "static_ci95": list(wilson_ci(static_hits, static_total)),
        "memoryless_ci95": list(wilson_ci(memoryless_hits, memoryless_total)),
        "fixture_count": int(len(corpus)),
    }

    formatted_confusion: dict[str, object] = {
        "labels": list(labels),
        "matrix": [
            [pooled_confusion[true_label][pred_label] for pred_label in labels]
            for true_label in labels
        ],
    }

    return {
        "per_fixture": per_fixture,
        "compare_models": compare_models(corpus, model),
        "shuffled_label_control": shuffled_label_control(corpus, model),
        "uniform_belief_control": uniform_belief_control(corpus),
        "summary": summary,
        "pooled_confusion": formatted_confusion,
        "pooled_calibration_bins": calibration_bins(
            pooled_posteriors, pooled_truth_ids
        ),
        "segmentation": segmentation_evaluation(corpus, model),
        "ambiguity_ablation": ambiguity_ablation(corpus, model),
        "corpus_stability": corpus_stability_analysis(corpus, model),
        "phase_recovery": per_phase_recovery_metrics(formatted_confusion),
    }
