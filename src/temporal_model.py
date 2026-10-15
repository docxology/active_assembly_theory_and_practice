"""Temporal HMM over latent assembly-program phases.

The static bridge in :mod:`active_inference_bridge` performs per-step Bayesian
updates with an implicit identity transition (``B = I``), so latent program
state can never evolve across steps. This module supplies a genuine temporal
generative model: a column-stochastic transition matrix ``B`` describing a
left-to-right progression through the five program phases, together with
standard HMM inference routines (forward filtering, Viterbi decoding, and
forward-backward smoothing).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .generative_model import build_generative_model


@dataclass(frozen=True)
class TemporalGenerativeModel:
    """A discrete temporal generative model with explicit state dynamics.

    Attributes:
        A: Likelihood matrix ``(n_obs, n_states)`` with normalized columns,
            where ``A[o, s] = P(observation=o | state=s)``.
        B: Transition matrix ``(n_states, n_states)``, column-stochastic, where
            ``B[:, s] = P(next_state | current_state=s)``.
        D: Prior over the initial state ``(n_states,)``.
        observation_labels: Human-readable observation category names.
        hidden_state_labels: Human-readable latent-phase names.
    """

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    observation_labels: tuple[str, ...]
    hidden_state_labels: tuple[str, ...]

    def validate(self) -> None:
        """Validate categorical probability tensors.

        Raises:
            ValueError: On any shape mismatch, non-normalized column of ``A`` or
                ``B``, non-normalized ``D``, or any non-positive entry.
        """

        n_obs = len(self.observation_labels)
        n_states = len(self.hidden_state_labels)
        if self.A.shape != (n_obs, n_states):
            raise ValueError("A shape must be observations x hidden states")
        if self.B.shape != (n_states, n_states):
            raise ValueError("B shape must be hidden states x hidden states")
        if self.D.shape != (n_states,):
            raise ValueError("D shape must match hidden state count")
        if not (
            np.all(np.isfinite(self.A))
            and np.all(np.isfinite(self.B))
            and np.all(np.isfinite(self.D))
        ):
            raise ValueError("A, B and D must contain only finite values")
        if np.any(self.A <= 0) or np.any(self.B <= 0) or np.any(self.D <= 0):
            raise ValueError("A, B and D must be strictly positive for log updates")
        if not np.allclose(self.A.sum(axis=0), 1.0):
            raise ValueError("each A column must normalize to one")
        if not np.allclose(self.B.sum(axis=0), 1.0):
            raise ValueError("each B column must normalize to one")
        if not np.isclose(self.D.sum(), 1.0):
            raise ValueError("D must normalize to one")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return {
            "A": [list(row) for row in np.asarray(self.A, dtype=float)],
            "B": [list(row) for row in np.asarray(self.B, dtype=float)],
            "D": list(np.asarray(self.D, dtype=float)),
            "observation_labels": list(self.observation_labels),
            "hidden_state_labels": list(self.hidden_state_labels),
        }


def build_temporal_model(model_path: Path | None = None) -> TemporalGenerativeModel:
    from .gnn_parser import compile_gnn

    compiled = compile_gnn(
        model_path
        if model_path is not None
        else Path(__file__).resolve().parents[1] / "configs/gnn/passive_phase.md"
    )
    base = build_generative_model(model_path)
    if compiled.B[0].shape[2] != 1:
        raise ValueError("GNN_PASSIVE_CONTRACT_MISMATCH")
    model = TemporalGenerativeModel(
        A=base.A,
        B=compiled.B[0][:, :, 0].copy(),
        D=base.D,
        observation_labels=base.observation_labels,
        hidden_state_labels=base.hidden_state_labels,
    )
    model.validate()
    return model


def _variational_free_energy(logits: np.ndarray, posterior: np.ndarray) -> float:
    """Compute a finite categorical variational free-energy diagnostic.

    Mirrors ``active_inference_bridge._variational_free_energy``: the posterior
    is scored against unnormalized ``logits = log_prior + log_likelihood``.
    """

    positive = posterior > 0.0
    return float(
        np.sum(posterior[positive] * (np.log(posterior[positive]) - logits[positive]))
    )


@dataclass(frozen=True)
class FilterResult:
    """Forward-filtering posteriors and free-energy diagnostics."""

    filtered_posteriors: tuple[tuple[float, ...], ...]
    predicted_priors: tuple[tuple[float, ...], ...]
    free_energy_by_step: tuple[float, ...]
    most_likely_states: tuple[str, ...]
    final_posterior: tuple[float, ...]
    final_state: str
    log_evidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["filtered_posteriors"] = [list(row) for row in self.filtered_posteriors]
        data["predicted_priors"] = [list(row) for row in self.predicted_priors]
        data["free_energy_by_step"] = list(self.free_energy_by_step)
        data["most_likely_states"] = list(self.most_likely_states)
        data["final_posterior"] = list(self.final_posterior)
        data["log_evidence"] = float(self.log_evidence)
        return data


def _as_observation_ids(
    observation_ids: list[int], *, observation_count: int | None = None
) -> list[int]:
    if not observation_ids:
        raise ValueError("observation_ids must not be empty")
    normalized: list[int] = []
    for value in observation_ids:
        if isinstance(value, bool):
            raise ValueError("observation_ids must contain integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_ids must contain integers") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("observation_ids must contain finite integers")
        integer = int(numeric)
        if integer < 0 or (
            observation_count is not None and integer >= observation_count
        ):
            raise ValueError("observation_ids contain an out-of-range category")
        normalized.append(integer)
    return normalized


def forward_filter(
    observation_ids: list[int],
    model: TemporalGenerativeModel,
) -> FilterResult:
    """Standard HMM forward filtering with explicit dynamics.

    Step 0 uses the prior ``D`` directly; step ``t`` predicts the prior as
    ``B @ posterior_{t-1}`` and updates it with the likelihood column
    ``A[obs_t, :]``. ``log_evidence`` accumulates the incremental marginal
    log-likelihood ``sum_t log( sum_s prior_t[s] * A[obs_t, s] )``.
    """

    model.validate()
    ids = _as_observation_ids(
        observation_ids, observation_count=len(model.observation_labels)
    )

    log_posterior = np.log(model.D)
    log_a = np.log(model.A)
    log_b = np.log(model.B)
    filtered_posteriors: list[tuple[float, ...]] = []
    predicted_priors: list[tuple[float, ...]] = []
    free_energy_by_step: list[float] = []
    most_likely_states: list[str] = []
    log_evidence = 0.0

    for step, obs in enumerate(ids):
        log_prior = (
            np.log(model.D)
            if step == 0
            else np.logaddexp.reduce(log_b + log_posterior[np.newaxis, :], axis=1)
        )
        log_prior -= np.logaddexp.reduce(log_prior)
        predicted_priors.append(tuple(float(value) for value in np.exp(log_prior)))
        logits = log_prior + log_a[obs, :]
        log_marginal = float(np.logaddexp.reduce(logits))
        log_evidence += log_marginal
        log_posterior = logits - log_marginal
        posterior = np.exp(log_posterior)

        filtered_posteriors.append(tuple(float(value) for value in posterior))
        free_energy_by_step.append(_variational_free_energy(logits, posterior))
        most_likely_states.append(model.hidden_state_labels[int(np.argmax(posterior))])

    return FilterResult(
        filtered_posteriors=tuple(filtered_posteriors),
        predicted_priors=tuple(predicted_priors),
        free_energy_by_step=tuple(free_energy_by_step),
        most_likely_states=tuple(most_likely_states),
        final_posterior=tuple(float(value) for value in posterior),
        final_state=most_likely_states[-1],
        log_evidence=float(log_evidence),
    )


def viterbi_decode(
    observation_ids: list[int],
    model: TemporalGenerativeModel,
) -> tuple[tuple[str, ...], float]:
    """Most-probable state path via log-space Viterbi recursion.

    Returns the MAP path as labels and its joint log-probability under the
    model ``log P(states, observations)``.
    """

    model.validate()
    ids = _as_observation_ids(
        observation_ids, observation_count=len(model.observation_labels)
    )

    n_states = len(model.hidden_state_labels)
    log_a = np.log(np.asarray(model.A, dtype=float))
    log_b = np.log(np.asarray(model.B, dtype=float))
    log_d = np.log(np.asarray(model.D, dtype=float))

    delta = log_d + log_a[ids[0], :]
    backpointers: list[np.ndarray] = []

    for obs in ids[1:]:
        # scores[next, prev] = delta[prev] + log B[next, prev]
        scores = delta[np.newaxis, :] + log_b
        best_prev = np.argmax(scores, axis=1)
        delta = scores[np.arange(n_states), best_prev] + log_a[obs, :]
        backpointers.append(best_prev)

    best_last = int(np.argmax(delta))
    log_prob = float(delta[best_last])

    path = [best_last]
    for pointers in reversed(backpointers):
        path.append(int(pointers[path[-1]]))
    path.reverse()

    labels = tuple(model.hidden_state_labels[state] for state in path)
    return labels, log_prob


@dataclass(frozen=True)
class SmoothingResult:
    """Forward-backward smoothed posteriors."""

    smoothed_posteriors: tuple[tuple[float, ...], ...]
    most_likely_states: tuple[str, ...]
    final_posterior: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["smoothed_posteriors"] = [list(row) for row in self.smoothed_posteriors]
        data["most_likely_states"] = list(self.most_likely_states)
        data["final_posterior"] = list(self.final_posterior)
        return data


def forward_backward(
    observation_ids: list[int],
    model: TemporalGenerativeModel,
) -> SmoothingResult:
    """Standard forward-backward smoothing.

    Computes log-space forward (alpha) and backward (beta) messages and returns
    the per-step smoothed posteriors ``gamma_t ∝ alpha_t * beta_t``. Each step
    sums to one; at the last step gamma equals the forward filter posterior.
    """

    model.validate()
    ids = _as_observation_ids(
        observation_ids, observation_count=len(model.observation_labels)
    )

    n_steps = len(ids)
    n_states = len(model.hidden_state_labels)
    log_a = np.log(model.A)
    log_b = np.log(model.B)
    log_alpha = np.zeros((n_steps, n_states), dtype=float)
    log_prior = np.log(model.D)
    for step, obs in enumerate(ids):
        if step > 0:
            log_prior = np.logaddexp.reduce(
                log_b + log_alpha[step - 1][np.newaxis, :], axis=1
            )
        logits = log_prior + log_a[obs, :]
        log_alpha[step] = logits - np.logaddexp.reduce(logits)

    log_beta = np.zeros((n_steps, n_states), dtype=float)
    for step in range(n_steps - 2, -1, -1):
        weighted = log_a[ids[step + 1], :] + log_beta[step + 1]
        message = np.logaddexp.reduce(log_b + weighted[:, np.newaxis], axis=0)
        log_beta[step] = message - np.logaddexp.reduce(message)
    log_smoothed = log_alpha + log_beta
    log_smoothed -= np.logaddexp.reduce(log_smoothed, axis=1, keepdims=True)
    smoothed = np.exp(log_smoothed)

    smoothed_posteriors = tuple(
        tuple(float(value) for value in row) for row in smoothed
    )
    most_likely_states = tuple(
        model.hidden_state_labels[int(np.argmax(row))] for row in smoothed
    )
    final_posterior = tuple(float(value) for value in smoothed[-1])

    return SmoothingResult(
        smoothed_posteriors=smoothed_posteriors,
        most_likely_states=most_likely_states,
        final_posterior=final_posterior,
    )
