"""Expected Free Energy and active observation selection.

This module implements the "Active" in Active Inference for the assembly-state
inference setting. The latent program phase (one of the five
``HIDDEN_STATE_LABELS``) is never directly observed. Each candidate instruction
has a *known* observation category (cheap to decode) but its contribution to
resolving the latent phase is not known a priori. Active disassembly chooses
which observation to reveal next so as to most reduce expected free energy
(EFE), trading off epistemic value (information gain about the phase) against
pragmatic value (a prior preference over observation categories).

Selection MINIMIZES expected free energy. EFE is defined here as the negated
sum of epistemic and pragmatic value, so larger information gain and larger
preference both pull EFE lower (better).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np


CONFIDENCE_THRESHOLD = 0.7


@dataclass(frozen=True)
class ExpectedFreeEnergy:
    """Expected-free-energy decomposition for a single candidate observation."""

    candidate_index: int
    observation_id: int
    epistemic_value: float
    pragmatic_value: float
    expected_free_energy: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["candidate_index"] = int(self.candidate_index)
        data["observation_id"] = int(self.observation_id)
        data["epistemic_value"] = float(self.epistemic_value)
        data["pragmatic_value"] = float(self.pragmatic_value)
        data["expected_free_energy"] = float(self.expected_free_energy)
        return data


@dataclass(frozen=True)
class SelectionResult:
    """Greedy active-selection trajectory over a set of observations."""

    order: tuple[int, ...]
    observation_id_order: tuple[int, ...]
    efe_by_step: tuple[float, ...]
    belief_entropy_by_step: tuple[float, ...]
    steps_to_confidence: int | None
    confidence_threshold: float
    policy: str = "realized"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["order"] = [int(value) for value in self.order]
        data["observation_id_order"] = [
            int(value) for value in self.observation_id_order
        ]
        data["efe_by_step"] = [float(value) for value in self.efe_by_step]
        data["belief_entropy_by_step"] = [
            float(value) for value in self.belief_entropy_by_step
        ]
        data["steps_to_confidence"] = (
            None if self.steps_to_confidence is None else int(self.steps_to_confidence)
        )
        data["confidence_threshold"] = float(self.confidence_threshold)
        data["policy"] = str(self.policy)
        return data


def _normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Return a finite, non-negative, strictly-normalized probability vector."""

    arr = np.asarray(vector, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must not contain negative values")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum")
    return arr / total


def _as_integral_id(value: object, *, name: str, upper_bound: int) -> int:
    """Validate an observation/candidate id without truncating floats."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    integer = int(numeric)
    if integer < 0 or integer >= upper_bound:
        raise ValueError(
            f"{name} is out-of-range; expected an integer in [0, {upper_bound})"
        )
    return integer


def _as_probability_threshold(value: object, *, name: str) -> float:
    """Validate a confidence threshold without relying on NumPy coercion."""

    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


def _validate_inputs(
    belief: np.ndarray,
    A: np.ndarray,
    C: np.ndarray | None = None,
    *,
    require_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate conditional weights; predictive objectives require a full channel."""

    matrix_a = np.asarray(A, dtype=float)
    if matrix_a.ndim != 2 or matrix_a.shape[0] == 0 or matrix_a.shape[1] == 0:
        raise ValueError("A must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix_a)) or np.any(matrix_a < 0.0):
        raise ValueError("A must contain only finite, non-negative values")
    if np.any(matrix_a.sum(axis=0) <= 0.0):
        raise ValueError("each A hidden-state column must have positive support")
    if require_channel and not np.allclose(
        matrix_a.sum(axis=0), 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("each A hidden-state column must normalize to one")
    prior = _normalize(belief, name="belief")
    if prior.size != matrix_a.shape[1]:
        raise ValueError("belief length must match A hidden-state dimension")
    preferences: np.ndarray | None = None
    if C is not None:
        preferences = _normalize(C, name="C")
        if preferences.size != matrix_a.shape[0]:
            raise ValueError("C length must match A observation dimension")
    return prior, matrix_a, preferences


def _shannon_entropy(belief: np.ndarray) -> float:
    """Return the Shannon entropy (in nats) of a belief distribution."""

    normalized = _normalize(belief, name="belief")
    safe = np.clip(normalized, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def state_belief_update(
    belief: np.ndarray,
    A: np.ndarray,
    observation_id: int,
) -> np.ndarray:
    """Static Bayesian belief update.

    Computes the posterior over hidden phases given a single observation
    category: ``posterior ∝ belief * A[observation_id, :]``, normalized.

    Args:
        belief: Prior belief over the hidden phases (shape ``(n_states,)``).
        A: Likelihood matrix with shape ``(n_obs, n_states)``, columns normalized.
        observation_id: Row index of the observed category.

    Returns:
        The normalized posterior belief over hidden phases.
    """

    prior, matrix_a, _ = _validate_inputs(belief, A)
    index = _as_integral_id(
        observation_id, name="observation_id", upper_bound=matrix_a.shape[0]
    )
    likelihood = matrix_a[index, :]
    unnormalized = prior * likelihood
    return _normalize(unnormalized, name="posterior")


def observation_info_gain(
    belief: np.ndarray,
    A: np.ndarray,
    observation_id: int,
) -> float:
    """Realized reduction in Shannon entropy of the belief from this observation.

    Returns ``H(belief) - H(update(belief, A, observation_id))`` for the *known*
    observation category — the realized (conditional) information gain, not an
    expectation over possible observations. For a single realized observation
    this value can be negative (an observation may move the posterior toward a
    flatter distribution); it is the expectation over observations that is
    guaranteed non-negative. The active policy uses the realized gain because the
    candidate's category is already decoded before selection.
    """

    prior_entropy = _shannon_entropy(belief)
    posterior = state_belief_update(belief, A, observation_id)
    posterior_entropy = _shannon_entropy(posterior)
    return float(prior_entropy - posterior_entropy)


def predictive_information_contribution(
    belief: np.ndarray,
    A: np.ndarray,
    observation_id: int,
) -> float:
    """Return a category's predictive contribution to mutual information.

    The realized entropy reduction used by the original policy conditions on a
    category that has already been decoded.  Before revealing an observation,
    the category itself has predictive probability ``P(o) = A[o] @ belief``.
    Weighting its realized entropy reduction by ``P(o)`` gives that category's
    contribution to the one-step expected information gain.  The contributions
    sum to the mutual information of the categorical channel and therefore
    provide a principled predictive diagnostic without inventing a hidden
    distribution over candidate instructions.
    """

    prior, matrix_a, _ = _validate_inputs(belief, A, require_channel=True)
    index = _as_integral_id(
        observation_id, name="observation_id", upper_bound=matrix_a.shape[0]
    )
    predictive_probability = float(matrix_a[index, :] @ prior)
    if predictive_probability <= 0.0:
        return 0.0
    return float(predictive_probability * observation_info_gain(prior, matrix_a, index))


def expected_information_gain(belief: np.ndarray, A: np.ndarray) -> float:
    """Return the one-step expected information gain of the observation channel."""

    prior, matrix_a, _ = _validate_inputs(belief, A, require_channel=True)
    contributions = [
        predictive_information_contribution(prior, matrix_a, observation_id)
        for observation_id in range(matrix_a.shape[0])
    ]
    return float(sum(contributions))


def expected_free_energy(
    belief: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    candidate_index: int,
    observation_id: int,
) -> ExpectedFreeEnergy:
    """Compute the expected-free-energy decomposition for one candidate.

    Args:
        belief: Current belief over the hidden phases.
        A: Likelihood matrix with shape ``(n_obs, n_states)``.
        C: Prior preference over the ``n_obs`` observation categories.
        candidate_index: Index identifying the candidate instruction.
        observation_id: Observation category that this candidate would reveal.

    Returns:
        An :class:`ExpectedFreeEnergy` with epistemic, pragmatic, and total EFE.
        EFE is ``-(epistemic_value + pragmatic_value)``; lower is better.
    """

    prior, matrix_a, preferences = _validate_inputs(belief, A, C)
    del prior
    assert preferences is not None  # C is required by this public function.
    index = _as_integral_id(
        observation_id, name="observation_id", upper_bound=matrix_a.shape[0]
    )
    try:
        candidate_numeric = float(candidate_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_index must be a non-negative integer") from exc
    if (
        isinstance(candidate_index, bool)
        or not np.isfinite(candidate_numeric)
        or not candidate_numeric.is_integer()
        or candidate_numeric < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    epistemic_value = observation_info_gain(belief, matrix_a, index)
    pragmatic_value = float(np.log(np.clip(preferences[index], 1e-12, 1.0)))
    efe = -(epistemic_value + pragmatic_value)
    return ExpectedFreeEnergy(
        candidate_index=int(candidate_numeric),
        observation_id=index,
        epistemic_value=float(epistemic_value),
        pragmatic_value=float(pragmatic_value),
        expected_free_energy=float(efe),
    )


def predictive_expected_free_energy(
    belief: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    candidate_index: int,
    observation_id: int,
) -> ExpectedFreeEnergy:
    """Compute EFE using a predictive information contribution.

    This is the calibrated companion to :func:`expected_free_energy`.  The
    pragmatic term is unchanged; only the epistemic term is weighted by the
    prior predictive probability of the candidate category.  The result is
    still candidate-conditioned on the decoded category, so it is reported as
    a bounded predictive diagnostic rather than as a claim about an unmodelled
    instruction-level sensing process.
    """

    prior, matrix_a, preferences = _validate_inputs(belief, A, C, require_channel=True)
    assert preferences is not None
    index = _as_integral_id(
        observation_id, name="observation_id", upper_bound=matrix_a.shape[0]
    )
    try:
        candidate_numeric = float(candidate_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_index must be a non-negative integer") from exc
    if (
        isinstance(candidate_index, bool)
        or not np.isfinite(candidate_numeric)
        or not candidate_numeric.is_integer()
        or candidate_numeric < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    epistemic_value = predictive_information_contribution(prior, matrix_a, index)
    pragmatic_value = float(np.log(np.clip(preferences[index], 1e-12, 1.0)))
    return ExpectedFreeEnergy(
        candidate_index=int(candidate_numeric),
        observation_id=index,
        epistemic_value=float(epistemic_value),
        pragmatic_value=pragmatic_value,
        expected_free_energy=float(-(epistemic_value + pragmatic_value)),
    )


def select_next_observation(
    belief: np.ndarray,
    candidates: list[tuple[int, int]],
    A: np.ndarray,
    C: np.ndarray,
    *,
    policy: str = "realized",
) -> ExpectedFreeEnergy:
    """Select the candidate with MINIMUM expected free energy.

    Args:
        belief: Current belief over the hidden phases.
        candidates: List of ``(candidate_index, observation_id)`` tuples.
        A: Likelihood matrix with shape ``(n_obs, n_states)``.
        C: Prior preference over the ``n_obs`` observation categories.

    Returns:
        The :class:`ExpectedFreeEnergy` of the minimizing candidate.
    """

    if not candidates:
        raise ValueError("candidates must not be empty")
    _validate_inputs(belief, A, C)
    candidate_indexes = [candidate_index for candidate_index, _ in candidates]
    if len(set(candidate_indexes)) != len(candidate_indexes):
        raise ValueError("candidate indexes must be unique")
    if policy not in {"realized", "predictive"}:
        raise ValueError("policy must be 'realized' or 'predictive'")
    scorer = (
        expected_free_energy
        if policy == "realized"
        else predictive_expected_free_energy
    )
    evaluations = [
        scorer(belief, A, C, candidate_index, observation_id)
        for candidate_index, observation_id in candidates
    ]
    return min(evaluations, key=lambda efe: efe.expected_free_energy)


def active_observation_order(
    observation_ids: list[int],
    A: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    confidence_threshold: float = 0.7,
    *,
    policy: str = "realized",
) -> SelectionResult:
    """Greedily order observations by minimum expected free energy.

    Starts from the prior ``D`` (normalized), repeatedly selects the remaining
    observation with minimum EFE, applies the Bayesian update, and records the
    selected EFE and resulting belief entropy at each step.

    Args:
        observation_ids: The pool of observation category ids available to reveal.
        A: Likelihood matrix with shape ``(n_obs, n_states)``.
        D: Prior over the hidden phases.
        C: Prior preference over the ``n_obs`` observation categories.
        confidence_threshold: ``max(belief)`` level deemed confident.
        policy: ``"realized"`` reproduces the original decoded-category EFE;
            ``"predictive"`` uses the category's predictive information
            contribution.

    Returns:
        A :class:`SelectionResult` describing the greedy trajectory.
    """

    if not observation_ids:
        raise ValueError("observation_ids must not be empty")

    prior, matrix_a, _ = _validate_inputs(D, A, C)
    if policy not in {"realized", "predictive"}:
        raise ValueError("policy must be 'realized' or 'predictive'")
    confidence_threshold = _as_probability_threshold(
        confidence_threshold, name="confidence_threshold"
    )
    normalized_ids = [
        _as_integral_id(value, name="observation_id", upper_bound=matrix_a.shape[0])
        for value in observation_ids
    ]

    belief = prior
    remaining: list[tuple[int, int]] = list(enumerate(normalized_ids))

    order: list[int] = []
    observation_id_order: list[int] = []
    efe_by_step: list[float] = []
    belief_entropy_by_step: list[float] = []
    steps_to_confidence: int | None = None

    step = 0
    while remaining:
        chosen = select_next_observation(belief, remaining, A, C, policy=policy)
        order.append(chosen.candidate_index)
        observation_id_order.append(chosen.observation_id)
        efe_by_step.append(chosen.expected_free_energy)

        belief = state_belief_update(belief, A, chosen.observation_id)
        belief_entropy_by_step.append(_shannon_entropy(belief))

        step += 1
        if (
            steps_to_confidence is None
            and float(np.max(belief)) >= confidence_threshold
        ):
            steps_to_confidence = step

        remaining = [
            (candidate_index, observation_id)
            for candidate_index, observation_id in remaining
            if candidate_index != chosen.candidate_index
        ]

    return SelectionResult(
        order=tuple(order),
        observation_id_order=tuple(observation_id_order),
        efe_by_step=tuple(efe_by_step),
        belief_entropy_by_step=tuple(belief_entropy_by_step),
        steps_to_confidence=steps_to_confidence,
        confidence_threshold=float(confidence_threshold),
        policy=policy,
    )


def fixed_order_steps_to_confidence(
    observation_ids: list[int],
    A: np.ndarray,
    D: np.ndarray,
    threshold: float = 0.7,
) -> int | None:
    """Steps to reach confidence applying observations left-to-right.

    Uses the same Bayesian belief update as :func:`active_observation_order` but
    in the given (natural) order. Provided for comparison against the active
    selection policy.

    Returns:
        The 1-based step at which ``max(belief) >= threshold``, or ``None``.
    """

    if not observation_ids:
        raise ValueError("observation_ids must not be empty")

    prior, matrix_a, _ = _validate_inputs(D, A)
    threshold = _as_probability_threshold(threshold, name="threshold")
    normalized_ids = [
        _as_integral_id(value, name="observation_id", upper_bound=matrix_a.shape[0])
        for value in observation_ids
    ]

    belief = prior
    for step, observation_id in enumerate(normalized_ids, start=1):
        belief = state_belief_update(belief, matrix_a, observation_id)
        if float(np.max(belief)) >= threshold:
            return step
    return None
