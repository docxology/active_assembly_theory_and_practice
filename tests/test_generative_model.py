"""Tests for the discrete GenerativeModelSpec and its validate() raise paths.

These mirror the temporal model's validate() coverage so the central observation
model the whole pipeline depends on has every error branch exercised directly
(no mocks: each bad spec is built from the real canonical model and mutated).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generative_model import GenerativeModelSpec, build_generative_model


def _good_spec() -> GenerativeModelSpec:
    return build_generative_model()


def test_build_generative_model_validate_does_not_raise() -> None:
    """The canonical model must pass validate() without error."""
    spec = _good_spec()
    # Must not raise; explicit call for parity with the temporal model test.
    spec.validate()
    assert spec.A.shape == (
        len(spec.observation_labels),
        len(spec.hidden_state_labels),
    )
    assert spec.D.shape == (len(spec.hidden_state_labels),)


def test_validate_rejects_bad_A_shape() -> None:
    good = _good_spec()
    bad = GenerativeModelSpec(
        A=good.A[:-1, :],  # drop one observation row -> wrong shape
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="A shape"):
        bad.validate()


def test_validate_rejects_bad_D_shape() -> None:
    good = _good_spec()
    bad = GenerativeModelSpec(
        A=good.A,
        D=good.D[:-1],  # too few hidden-state priors
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="D shape"):
        bad.validate()


def test_validate_rejects_unnormalized_A_columns() -> None:
    good = _good_spec()
    bad_a = good.A.copy()
    bad_a[:, 0] = bad_a[:, 0] * 2.0  # first column no longer sums to one
    bad = GenerativeModelSpec(
        A=bad_a,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="A column"):
        bad.validate()


def test_validate_rejects_unnormalized_D() -> None:
    good = _good_spec()
    bad_d = good.D.copy() * 1.5  # no longer sums to one
    bad = GenerativeModelSpec(
        A=good.A,
        D=bad_d,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="D must normalize"):
        bad.validate()


def test_validate_rejects_non_positive_entries() -> None:
    good = _good_spec()
    bad_a = good.A.copy()
    bad_a[0, 0] = 0.0
    # Renormalize that column so the ONLY remaining defect is the zero entry,
    # forcing the strictly-positive branch (not the column-normalization one).
    bad_a[:, 0] = bad_a[:, 0] / bad_a[:, 0].sum()
    bad = GenerativeModelSpec(
        A=bad_a,
        D=good.D,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        bad.validate()


def test_validate_rejects_non_positive_prior() -> None:
    good = _good_spec()
    bad_d = good.D.copy()
    # Move all of one state's mass onto another so D still sums to one but has a
    # zero entry -> strictly-positive branch on D.
    bad_d[1] += bad_d[0]
    bad_d[0] = 0.0
    bad = GenerativeModelSpec(
        A=good.A,
        D=bad_d,
        observation_labels=good.observation_labels,
        hidden_state_labels=good.hidden_state_labels,
    )
    assert np.isclose(bad_d.sum(), 1.0)  # normalization is intact by construction
    with pytest.raises(ValueError, match="strictly positive"):
        bad.validate()
