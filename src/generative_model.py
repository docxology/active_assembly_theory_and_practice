"""Discrete generative model for assembly observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS


@dataclass(frozen=True)
class GenerativeModelSpec:
    """A discrete likelihood and prior over program-state hypotheses."""

    A: np.ndarray
    D: np.ndarray
    observation_labels: tuple[str, ...]
    hidden_state_labels: tuple[str, ...]

    def validate(self) -> None:
        """Validate categorical probability tensors."""

        if not (np.all(np.isfinite(self.A)) and np.all(np.isfinite(self.D))):
            raise ValueError("A and D must contain only finite values")
        if self.A.shape != (
            len(self.observation_labels),
            len(self.hidden_state_labels),
        ):
            raise ValueError("A shape must be observations x hidden states")
        if self.D.shape != (len(self.hidden_state_labels),):
            raise ValueError("D shape must match hidden state count")
        if not np.allclose(self.A.sum(axis=0), 1.0):
            raise ValueError("each A column must normalize to one")
        if not np.isclose(self.D.sum(), 1.0):
            raise ValueError("D must normalize to one")
        if np.any(self.A <= 0) or np.any(self.D <= 0):
            raise ValueError("A and D must be strictly positive for log updates")


def _normalized_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=0, keepdims=True)


def build_generative_model() -> GenerativeModelSpec:
    """Build a small observation model for assembly-state inference."""

    # Columns are latent states. Rows are observation categories.
    likelihood = np.array(
        [
            [0.72, 0.06, 0.05, 0.05, 0.06],  # load_immediate
            [0.08, 0.74, 0.08, 0.06, 0.08],  # arithmetic_update
            [0.06, 0.10, 0.76, 0.05, 0.08],  # bitwise_transform
            [0.05, 0.04, 0.04, 0.76, 0.10],  # comparison
            [0.04, 0.03, 0.04, 0.08, 0.68],  # control_transfer
            [0.05, 0.03, 0.03, 0.00, 0.00],  # other
        ],
        dtype=float,
    )
    likelihood = np.clip(likelihood, 1e-6, None)
    prior = np.array([0.34, 0.24, 0.18, 0.14, 0.10], dtype=float)
    spec = GenerativeModelSpec(
        A=_normalized_columns(likelihood),
        D=prior / prior.sum(),
        observation_labels=OBSERVATION_LABELS,
        hidden_state_labels=HIDDEN_STATE_LABELS,
    )
    spec.validate()
    return spec
