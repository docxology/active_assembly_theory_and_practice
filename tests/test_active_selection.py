"""Tests for Expected Free Energy and active observation selection.

No mocks. All expected values are hand-computed from real numpy arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.active_selection import (
    ExpectedFreeEnergy,
    SelectionResult,
    active_observation_order,
    expected_free_energy,
    fixed_order_steps_to_confidence,
    observation_info_gain,
    select_next_observation,
    state_belief_update,
)
from src.generative_model import build_generative_model


# Synthetic fixtures with hand-checkable arithmetic.
# 3 hidden states; obs row 0 is one-hot onto state 0, obs row 1 is uniform.
ONE_HOT_VS_UNIFORM_A = np.array(
    [
        [1.0, 0.0, 0.0],  # observation 0: one-hot onto state 0
        [1.0, 1.0, 1.0],  # observation 1: uniform across states
    ]
)
UNIFORM_BELIEF_3 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_state_belief_update_hand_computed():
    """Posterior proportional to belief * A row, normalized."""

    belief = np.array([0.5, 0.3, 0.2])
    A = np.array([[0.6, 0.2, 0.1]])  # single observation row
    posterior = state_belief_update(belief, A, 0)

    # unnormalized: [0.30, 0.06, 0.02] -> sum 0.38
    expected = np.array([0.30, 0.06, 0.02]) / 0.38
    assert np.allclose(posterior, expected)
    assert math.isclose(float(posterior.sum()), 1.0, rel_tol=1e-12)


def test_state_belief_update_one_hot_collapses_belief():
    """A one-hot observation column collapses the belief onto that state."""

    posterior = state_belief_update(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 0)
    assert np.allclose(posterior, np.array([1.0, 0.0, 0.0]))


def test_state_belief_update_uniform_observation_is_inert():
    """A uniform observation row leaves the belief unchanged."""

    posterior = state_belief_update(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 1)
    assert np.allclose(posterior, UNIFORM_BELIEF_3)


def test_observation_info_gain_one_hot_value():
    """Info gain for a one-hot observation equals the full prior entropy.

    H(uniform over 3) = ln 3; posterior is a point mass with entropy 0.
    """

    gain = observation_info_gain(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 0)
    assert math.isclose(gain, math.log(3.0), rel_tol=1e-9)


def test_observation_info_gain_uniform_is_zero():
    """A uniform (uninformative) observation yields ~zero info gain."""

    gain = observation_info_gain(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 1)
    assert math.isclose(gain, 0.0, abs_tol=1e-9)


def test_observation_info_gain_non_negative_on_real_model():
    """Info gain is approximately non-negative for every category."""

    spec = build_generative_model()
    belief = np.asarray(spec.D, dtype=float)
    for observation_id in range(spec.A.shape[0]):
        gain = observation_info_gain(belief, spec.A, observation_id)
        assert gain >= -1e-9


def test_isc12_one_hot_strictly_more_informative_than_uniform():
    """ISC-12: a one-hot column is STRICTLY more informative than a uniform one."""

    one_hot_gain = observation_info_gain(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 0)
    uniform_gain = observation_info_gain(UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, 1)
    assert one_hot_gain > uniform_gain
    # And the magnitude is large, not a rounding artifact.
    assert one_hot_gain - uniform_gain > 1.0


def test_expected_free_energy_decomposition_hand_computed():
    """EFE = -(epistemic + pragmatic); pragmatic = log(normalized C)."""

    belief = UNIFORM_BELIEF_3
    A = ONE_HOT_VS_UNIFORM_A
    C = np.array([1.0, 3.0])  # normalizes to [0.25, 0.75]
    efe = expected_free_energy(belief, A, C, candidate_index=4, observation_id=0)

    expected_epistemic = math.log(3.0)
    expected_pragmatic = math.log(0.25)
    expected_efe = -(expected_epistemic + expected_pragmatic)

    assert efe.candidate_index == 4
    assert efe.observation_id == 0
    assert math.isclose(efe.epistemic_value, expected_epistemic, rel_tol=1e-9)
    assert math.isclose(efe.pragmatic_value, expected_pragmatic, rel_tol=1e-9)
    assert math.isclose(efe.expected_free_energy, expected_efe, rel_tol=1e-9)


def test_isc16_sign_control_pragmatic_value():
    """ISC-16: with C peaked on k, observation_id=k has strictly HIGHER
    pragmatic value than observation_id!=k.

    Fails if the pragmatic-value sign is flipped: log of the larger
    normalized preference must be the larger (less negative) value.
    """

    belief = UNIFORM_BELIEF_3
    A = ONE_HOT_VS_UNIFORM_A
    k = 0
    # Preference peaked on category k=0.
    C = np.array([0.9, 0.1])

    efe_on_peak = expected_free_energy(belief, A, C, candidate_index=0, observation_id=k)
    efe_off_peak = expected_free_energy(belief, A, C, candidate_index=1, observation_id=1)

    assert efe_on_peak.pragmatic_value > efe_off_peak.pragmatic_value
    # Concrete hand values: log(0.9) > log(0.1).
    assert math.isclose(efe_on_peak.pragmatic_value, math.log(0.9), rel_tol=1e-9)
    assert math.isclose(efe_off_peak.pragmatic_value, math.log(0.1), rel_tol=1e-9)


def test_select_next_observation_minimizes_efe():
    """Selection returns the candidate with minimum EFE."""

    belief = UNIFORM_BELIEF_3
    A = ONE_HOT_VS_UNIFORM_A
    C = np.array([1.0, 1.0])  # flat preference -> epistemic dominates
    candidates = [(7, 1), (3, 0)]  # uniform vs one-hot
    chosen = select_next_observation(belief, candidates, A, C)

    # The one-hot observation (id 0) has higher info gain -> lower EFE.
    assert chosen.candidate_index == 3
    assert chosen.observation_id == 0
    # Confirm it truly is the minimum.
    other = expected_free_energy(belief, A, C, 7, 1)
    assert chosen.expected_free_energy < other.expected_free_energy


def test_select_next_observation_empty_raises():
    """Selecting from no candidates is an error."""

    with pytest.raises(ValueError):
        select_next_observation(UNIFORM_BELIEF_3, [], ONE_HOT_VS_UNIFORM_A, np.array([1.0, 1.0]))


def test_isc15_active_beats_fixed_when_informative_is_last():
    """ISC-15 mechanism: a single informative observation placed LAST in natural
    order is found first by active selection, reaching confidence in fewer steps.
    """

    A = np.array(
        [
            [0.9, 0.05, 0.05],  # observation 0: strongly diagnostic of state 0
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],  # observation 1: uninformative
        ]
    )
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = np.array([1.0, 1.0])  # flat preference

    # Informative observation (id 0) sits LAST; two uninformative ones first.
    observation_ids = [1, 1, 0]

    fixed_steps = fixed_order_steps_to_confidence(observation_ids, A, D, threshold=0.7)
    result = active_observation_order(observation_ids, A, D, C, confidence_threshold=0.7)

    assert fixed_steps == 3
    assert result.steps_to_confidence == 1
    assert result.steps_to_confidence < fixed_steps
    # The active policy reveals the informative observation first.
    assert result.observation_id_order[0] == 0


def test_active_observation_order_records_full_trajectory():
    """The greedy result records one entry per observation in order."""

    A = np.array(
        [
            [0.9, 0.05, 0.05],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = np.array([1.0, 1.0])
    observation_ids = [1, 0, 1]

    result = active_observation_order(observation_ids, A, D, C)

    assert len(result.order) == 3
    assert len(result.observation_id_order) == 3
    assert len(result.efe_by_step) == 3
    assert len(result.belief_entropy_by_step) == 3
    # Every candidate index is used exactly once.
    assert sorted(result.order) == [0, 1, 2]
    # Entropy is non-increasing once the informative observation is consumed.
    assert result.belief_entropy_by_step[0] <= math.log(3.0) + 1e-9


def test_active_observation_order_on_real_model():
    """Greedy ordering runs against the realistic generative model A."""

    spec = build_generative_model()
    C = np.ones(spec.A.shape[0])
    observation_ids = [5, 0, 1, 2, 3, 4]

    result = active_observation_order(
        observation_ids, spec.A, spec.D, C, confidence_threshold=0.7
    )

    assert sorted(result.order) == list(range(len(observation_ids)))
    assert all(math.isfinite(value) for value in result.efe_by_step)
    assert all(value >= 0.0 for value in result.belief_entropy_by_step)


def test_active_observation_order_no_confidence_returns_none():
    """If confidence is never reached, steps_to_confidence is None."""

    A = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],  # uniform, never resolves
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = np.array([1.0, 1.0])

    result = active_observation_order([0, 1], A, D, C, confidence_threshold=0.7)
    assert result.steps_to_confidence is None


def test_fixed_order_steps_to_confidence_immediate():
    """Confidence reached on the first observation."""

    A = np.array([[0.95, 0.025, 0.025]])
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    steps = fixed_order_steps_to_confidence([0], A, D, threshold=0.7)
    assert steps == 1


def test_fixed_order_steps_to_confidence_none_when_flat():
    """A flat observation never produces confidence."""

    A = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert fixed_order_steps_to_confidence([0, 0, 0], A, D, threshold=0.7) is None


def test_empty_inputs_raise():
    """Empty observation pools are rejected."""

    A = np.array([[0.9, 0.1]])
    D = np.array([0.5, 0.5])
    C = np.array([1.0])
    with pytest.raises(ValueError):
        active_observation_order([], A, D, C)
    with pytest.raises(ValueError):
        fixed_order_steps_to_confidence([], A, D)


def test_expected_free_energy_to_dict_json_safe():
    """ExpectedFreeEnergy.to_dict returns JSON-native scalar types."""

    import json

    efe = expected_free_energy(
        UNIFORM_BELIEF_3, ONE_HOT_VS_UNIFORM_A, np.array([1.0, 1.0]), 0, 0
    )
    data = efe.to_dict()
    assert isinstance(data["candidate_index"], int)
    assert isinstance(data["observation_id"], int)
    assert isinstance(data["epistemic_value"], float)
    assert isinstance(data["pragmatic_value"], float)
    assert isinstance(data["expected_free_energy"], float)
    # Round-trips through JSON without error.
    assert json.loads(json.dumps(data)) == data


def test_selection_result_to_dict_json_safe():
    """SelectionResult.to_dict returns JSON-native lists and scalars."""

    import json

    A = np.array(
        [
            [0.9, 0.05, 0.05],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = np.array([1.0, 1.0])
    result = active_observation_order([1, 0], A, D, C)
    data = result.to_dict()

    assert isinstance(data["order"], list)
    assert all(isinstance(value, int) for value in data["order"])
    assert isinstance(data["observation_id_order"], list)
    assert all(isinstance(value, int) for value in data["observation_id_order"])
    assert all(isinstance(value, float) for value in data["efe_by_step"])
    assert all(isinstance(value, float) for value in data["belief_entropy_by_step"])
    assert isinstance(data["confidence_threshold"], float)
    assert json.loads(json.dumps(data)) == data


def test_selection_result_to_dict_none_confidence():
    """to_dict serializes a None steps_to_confidence as JSON null."""

    A = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = np.array([1.0, 1.0])
    result = active_observation_order([0, 1], A, D, C, confidence_threshold=0.7)
    data = result.to_dict()
    assert data["steps_to_confidence"] is None


def test_state_belief_update_zero_overlap_raises():
    """An observation with zero overlap with the belief is unnormalizable."""

    belief = np.array([0.0, 0.0, 1.0])
    A = np.array([[1.0, 1.0, 0.0]])  # zero on the only supported state
    with pytest.raises(ValueError):
        state_belief_update(belief, A, 0)


def test_dataclasses_are_frozen():
    """Result dataclasses are immutable."""

    efe = ExpectedFreeEnergy(0, 0, 1.0, -1.0, 0.0)
    with pytest.raises(Exception):
        efe.candidate_index = 5  # type: ignore[misc]

    result = SelectionResult((0,), (0,), (0.0,), (0.0,), 1, 0.7)
    with pytest.raises(Exception):
        result.confidence_threshold = 0.9  # type: ignore[misc]
