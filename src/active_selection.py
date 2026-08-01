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

import numpy as np


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
        return data


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a strictly-normalized probability vector."""

    arr = np.asarray(vector, dtype=float)
    total = arr.sum()
    if total <= 0:
        raise ValueError("cannot normalize a non-positive vector")
    return arr / total


def _shannon_entropy(belief: np.ndarray) -> float:
    """Return the Shannon entropy (in nats) of a belief distribution."""

    safe = np.clip(np.asarray(belief, dtype=float), 1e-12, 1.0)
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

    prior = np.asarray(belief, dtype=float)
    likelihood = np.asarray(A, dtype=float)[observation_id, :]
    unnormalized = prior * likelihood
    return _normalize(unnormalized)


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

    prior_entropy = _shannon_entropy(_normalize(belief))
    posterior = state_belief_update(belief, A, observation_id)
    posterior_entropy = _shannon_entropy(posterior)
    return float(prior_entropy - posterior_entropy)


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

    epistemic_value = observation_info_gain(belief, A, observation_id)
    preferences = _normalize(C)
    pragmatic_value = float(np.log(np.clip(preferences[observation_id], 1e-12, 1.0)))
    efe = -(epistemic_value + pragmatic_value)
    return ExpectedFreeEnergy(
        candidate_index=int(candidate_index),
        observation_id=int(observation_id),
        epistemic_value=float(epistemic_value),
        pragmatic_value=float(pragmatic_value),
        expected_free_energy=float(efe),
    )


def select_next_observation(
    belief: np.ndarray,
    candidates: list[tuple[int, int]],
    A: np.ndarray,
    C: np.ndarray,
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
    evaluations = [
        expected_free_energy(belief, A, C, candidate_index, observation_id)
        for candidate_index, observation_id in candidates
    ]
    return min(evaluations, key=lambda efe: efe.expected_free_energy)


def active_observation_order(
    observation_ids: list[int],
    A: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    confidence_threshold: float = 0.7,
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

    Returns:
        A :class:`SelectionResult` describing the greedy trajectory.
    """

    if not observation_ids:
        raise ValueError("observation_ids must not be empty")

    belief = _normalize(D)
    remaining: list[tuple[int, int]] = list(enumerate(observation_ids))

    order: list[int] = []
    observation_id_order: list[int] = []
    efe_by_step: list[float] = []
    belief_entropy_by_step: list[float] = []
    steps_to_confidence: int | None = None

    step = 0
    while remaining:
        chosen = select_next_observation(belief, remaining, A, C)
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

    belief = _normalize(D)
    for step, observation_id in enumerate(observation_ids, start=1):
        belief = state_belief_update(belief, A, observation_id)
        if float(np.max(belief)) >= threshold:
            return step
    return None
