"""Tests for the temporal HMM over latent assembly phases.

No mocks: every expected value is either an exact rational hand-computed
constant or an independent numpy reference computation that does not call the
function under test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.temporal_model import (
    FilterResult,
    SmoothingResult,
    TemporalGenerativeModel,
    build_temporal_model,
    forward_backward,
    forward_filter,
    viterbi_decode,
)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #


def test_build_temporal_model_shapes_and_normalization() -> None:
    model = build_temporal_model()
    assert model.A.shape == (6, 5)
    assert model.B.shape == (5, 5)
    assert model.D.shape == (5,)
    assert model.observation_labels == (
        "load_immediate",
        "arithmetic_update",
        "bitwise_transform",
        "comparison",
        "control_transfer",
        "other",
    )
    assert model.hidden_state_labels == (
        "setup",
        "accumulator_update",
        "representation_transform",
        "guard_check",
        "exit",
    )
    assert np.allclose(model.A.sum(axis=0), 1.0)
    assert np.allclose(model.B.sum(axis=0), 1.0)
    assert math.isclose(float(model.D.sum()), 1.0, abs_tol=TOL)
    assert np.all(model.A > 0)
    assert np.all(model.B > 0)
    assert np.all(model.D > 0)
    # validate() must not raise on the canonical model.
    model.validate()


def test_build_temporal_model_is_left_to_right() -> None:
    """B carries mass only on the diagonal and the first sub-diagonal."""
    model = build_temporal_model()
    n = model.B.shape[0]
    for source in range(n):
        for target in range(n):
            mass = model.B[target, source]
            if target == source or target == source + 1:
                assert mass > 1e-3
            else:
                # Off the left-to-right band only the positivity clip remains.
                assert mass < 1e-4
    # exit (last column) is (almost) absorbing.
    assert model.B[n - 1, n - 1] > 0.99


def _good_model() -> TemporalGenerativeModel:
    return build_temporal_model()


def test_validate_rejects_bad_A_shape() -> None:
    good = _good_model()
    bad = TemporalGenerativeModel(
        A=good.A[:5, :],
        B=good.B,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="A shape"):
        bad.validate()


def test_validate_rejects_bad_B_shape() -> None:
    good = _good_model()
    bad = TemporalGenerativeModel(
        A=good.A,
        B=good.B[:, :4],
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="B shape"):
        bad.validate()


def test_validate_rejects_bad_D_shape() -> None:
    good = _good_model()
    bad = TemporalGenerativeModel(
        A=good.A,
        B=good.B,
        D=good.D[:4],
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="D shape"):
        bad.validate()


def test_validate_rejects_non_positive_entries() -> None:
    good = _good_model()
    bad_a = good.A.copy()
    bad_a[0, 0] = 0.0
    # Renormalize that column so the only defect is the non-positive entry.
    bad_a[:, 0] = bad_a[:, 0] / bad_a[:, 0].sum()
    bad = TemporalGenerativeModel(
        A=bad_a,
        B=good.B,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        bad.validate()


def test_validate_rejects_unnormalized_A_columns() -> None:
    good = _good_model()
    bad_a = good.A.copy()
    bad_a[:, 0] = bad_a[:, 0] * 2.0  # column no longer sums to one
    bad = TemporalGenerativeModel(
        A=bad_a,
        B=good.B,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="A column"):
        bad.validate()


def test_validate_rejects_unnormalized_B_columns() -> None:
    good = _good_model()
    bad_b = good.B.copy()
    bad_b[:, 0] = bad_b[:, 0] * 3.0
    bad = TemporalGenerativeModel(
        A=good.A,
        B=bad_b,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="B column"):
        bad.validate()


def test_validate_rejects_unnormalized_D() -> None:
    good = _good_model()
    bad_d = good.D.copy()
    bad_d = bad_d * 1.5
    bad = TemporalGenerativeModel(
        A=good.A,
        B=good.B,
        D=bad_d,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="D must normalize"):
        bad.validate()


# --------------------------------------------------------------------------- #
# Exact hand-computed 2-step forward filter
# --------------------------------------------------------------------------- #


def test_forward_filter_two_step_hand_computed() -> None:
    """Hand-derived forward-filter values for obs = [0, 1], checked to 1e-9.

    Independent reference arithmetic below uses only the model tensors and the
    HMM predict/update equations, never the function under test.
    """
    model = build_temporal_model()
    ids = [0, 1]

    a = np.asarray(model.A, dtype=float)
    b = np.asarray(model.B, dtype=float)
    d = np.asarray(model.D, dtype=float)

    # Step 0: prior = D, no prediction.
    prior0 = d
    u0 = prior0 * a[0, :]
    marg0 = u0.sum()
    post0 = u0 / marg0

    # Step 1: predict with B, update with A[1].
    prior1 = b @ post0
    prior1 = prior1 / prior1.sum()
    u1 = prior1 * a[1, :]
    marg1 = u1.sum()
    post1 = u1 / marg1

    expected_log_evidence = math.log(marg0) + math.log(marg1)

    result = forward_filter(ids, model)
    assert isinstance(result, FilterResult)

    np.testing.assert_allclose(result.predicted_priors[0], prior0, atol=TOL)
    np.testing.assert_allclose(result.predicted_priors[1], prior1, atol=TOL)
    np.testing.assert_allclose(result.filtered_posteriors[0], post0, atol=TOL)
    np.testing.assert_allclose(result.filtered_posteriors[1], post1, atol=TOL)
    np.testing.assert_allclose(result.final_posterior, post1, atol=TOL)
    assert math.isclose(result.log_evidence, expected_log_evidence, abs_tol=TOL)

    # Hand spot-check of the concrete step-0 marginal (D . A[0]).
    manual_marg0 = (
        0.34 * a[0, 0]
        + 0.24 * a[0, 1]
        + 0.18 * a[0, 2]
        + 0.14 * a[0, 3]
        + 0.10 * a[0, 4]
    )
    assert math.isclose(marg0, manual_marg0, abs_tol=TOL)

    # MAP labels.
    assert (
        result.most_likely_states[0] == model.hidden_state_labels[int(np.argmax(post0))]
    )
    assert result.final_state == model.hidden_state_labels[int(np.argmax(post1))]

    # Free energy matches the categorical VFE definition independently.
    logits0 = np.log(prior0) + np.log(a[0, :])
    fe0 = float(np.sum(post0 * (np.log(post0) - logits0)))
    assert math.isclose(result.free_energy_by_step[0], fe0, abs_tol=TOL)


def test_forward_filter_posteriors_sum_to_one() -> None:
    model = build_temporal_model()
    result = forward_filter([0, 1, 4, 4, 3], model)
    for row in result.filtered_posteriors:
        assert math.isclose(sum(row), 1.0, abs_tol=TOL)
    for row in result.predicted_priors:
        assert math.isclose(sum(row), 1.0, abs_tol=TOL)


def test_forward_filter_empty_raises() -> None:
    model = build_temporal_model()
    with pytest.raises(ValueError, match="must not be empty"):
        forward_filter([], model)


# --------------------------------------------------------------------------- #
# The temporal correction is real: B != identity changes the answer
# --------------------------------------------------------------------------- #


def test_temporal_filter_differs_from_static_identity_filter() -> None:
    """A left-to-right B yields a different posterior than an identity B.

    The observation sequence implies a phase change (arithmetic updates then a
    clear control-transfer/exit signature). A static (identity-B) filter cannot
    move probability mass forward across phases the way the temporal model can.
    """
    model = build_temporal_model()
    ids = [1, 1, 4, 4]  # accumulator updates, then control transfers

    temporal = forward_filter(ids, model)

    # Reference static filter with B = identity, recomputed independently.
    a = np.asarray(model.A, dtype=float)
    d = np.asarray(model.D, dtype=float)
    posterior = d
    static_posteriors = []
    for step, obs in enumerate(ids):
        prior = d if step == 0 else posterior  # identity transition
        prior = prior / prior.sum()
        u = prior * a[obs, :]
        posterior = u / u.sum()
        static_posteriors.append(posterior.copy())

    temporal_final = np.asarray(temporal.final_posterior, dtype=float)
    static_final = static_posteriors[-1]

    # The two filters must genuinely disagree (correction is real).
    assert not np.allclose(temporal_final, static_final, atol=1e-3)

    # Concretely: the temporal model should place more mass on the terminal
    # "exit" phase than the memoryless static filter after the exit-like tail.
    exit_idx = model.hidden_state_labels.index("exit")
    assert temporal_final[exit_idx] > static_final[exit_idx]


# --------------------------------------------------------------------------- #
# Viterbi
# --------------------------------------------------------------------------- #


def test_viterbi_two_step_hand_computed() -> None:
    """Exhaustive best-path reference over all 5x5 paths for obs = [0, 1]."""
    model = build_temporal_model()
    ids = [0, 1]

    log_a = np.log(np.asarray(model.A, dtype=float))
    log_b = np.log(np.asarray(model.B, dtype=float))
    log_d = np.log(np.asarray(model.D, dtype=float))

    n = len(model.hidden_state_labels)
    best_logp = -np.inf
    best_path = (-1, -1)
    for s0 in range(n):
        for s1 in range(n):
            logp = log_d[s0] + log_a[ids[0], s0] + log_b[s1, s0] + log_a[ids[1], s1]
            if logp > best_logp:
                best_logp = logp
                best_path = (s0, s1)

    expected_labels = (
        model.hidden_state_labels[best_path[0]],
        model.hidden_state_labels[best_path[1]],
    )

    labels, log_prob = viterbi_decode(ids, model)
    assert labels == expected_labels
    assert math.isclose(log_prob, float(best_logp), abs_tol=TOL)


def test_viterbi_single_step() -> None:
    model = build_temporal_model()
    log_a = np.log(np.asarray(model.A, dtype=float))
    log_d = np.log(np.asarray(model.D, dtype=float))
    obs = 4
    delta = log_d + log_a[obs, :]
    expected = model.hidden_state_labels[int(np.argmax(delta))]
    labels, log_prob = viterbi_decode([obs], model)
    assert labels == (expected,)
    assert math.isclose(log_prob, float(np.max(delta)), abs_tol=TOL)


def test_viterbi_empty_raises() -> None:
    model = build_temporal_model()
    with pytest.raises(ValueError, match="must not be empty"):
        viterbi_decode([], model)


# --------------------------------------------------------------------------- #
# Forward-backward smoothing
# --------------------------------------------------------------------------- #


def test_forward_backward_last_step_equals_filter() -> None:
    model = build_temporal_model()
    ids = [0, 1, 2, 4]

    smoothing = forward_backward(ids, model)
    assert isinstance(smoothing, SmoothingResult)

    filtering = forward_filter(ids, model)

    np.testing.assert_allclose(
        smoothing.final_posterior,
        filtering.final_posterior,
        atol=TOL,
    )
    np.testing.assert_allclose(
        smoothing.smoothed_posteriors[-1],
        filtering.filtered_posteriors[-1],
        atol=TOL,
    )


def test_forward_backward_rows_sum_to_one() -> None:
    model = build_temporal_model()
    smoothing = forward_backward([0, 1, 2, 3, 4], model)
    for row in smoothing.smoothed_posteriors:
        assert math.isclose(sum(row), 1.0, abs_tol=TOL)
    assert len(smoothing.most_likely_states) == 5


def test_forward_backward_empty_raises() -> None:
    model = build_temporal_model()
    with pytest.raises(ValueError, match="must not be empty"):
        forward_backward([], model)


# --------------------------------------------------------------------------- #
# JSON-safety of to_dict outputs
# --------------------------------------------------------------------------- #


def test_model_to_dict_json_safe() -> None:
    model = build_temporal_model()
    payload = model.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert isinstance(restored["A"], list)
    assert isinstance(restored["A"][0], list)
    assert isinstance(restored["B"], list)
    assert isinstance(restored["D"], list)
    assert restored["observation_labels"] == list(model.observation_labels)
    np.testing.assert_allclose(np.array(restored["B"]), model.B, atol=TOL)


def test_filter_result_to_dict_json_safe() -> None:
    model = build_temporal_model()
    result = forward_filter([0, 1, 4], model)
    payload = result.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert isinstance(restored["filtered_posteriors"], list)
    assert isinstance(restored["filtered_posteriors"][0], list)
    assert isinstance(restored["predicted_priors"], list)
    assert isinstance(restored["free_energy_by_step"], list)
    assert isinstance(restored["log_evidence"], float)
    assert restored["final_state"] == result.final_state


def test_smoothing_result_to_dict_json_safe() -> None:
    model = build_temporal_model()
    smoothing = forward_backward([0, 1, 4], model)
    payload = smoothing.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert isinstance(restored["smoothed_posteriors"], list)
    assert isinstance(restored["smoothed_posteriors"][0], list)
    assert isinstance(restored["most_likely_states"], list)
    assert restored["final_posterior"] == list(smoothing.final_posterior)


def test_determinism_repeated_runs_identical() -> None:
    model = build_temporal_model()
    ids = [0, 1, 2, 3, 4]
    first = forward_filter(ids, model).to_dict()
    second = forward_filter(ids, model).to_dict()
    assert first == second
