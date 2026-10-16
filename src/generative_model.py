"""Discrete generative model for assembly observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def build_generative_model(model_path: Path | None = None) -> GenerativeModelSpec:
    """Adapt the canonical GNN likelihood/prior to the historical result shape."""
    from .gnn_numeric import canonical_model

    model = canonical_model(
        Path(model_path)
        if model_path is not None
        else Path(__file__).resolve().parents[1] / "configs/gnn/passive_phase.md"
    )
    if (
        len(model.factors) != 1
        or model.modalities != ("category",)
        or model.outcomes[0] != OBSERVATION_LABELS
        or model.state_labels[0] != HIDDEN_STATE_LABELS
    ):
        raise ValueError("GNN_PASSIVE_CONTRACT_MISMATCH")
    spec = GenerativeModelSpec(
        A=model.A[0].copy(),
        D=model.D[0].copy(),
        observation_labels=model.outcomes[0],
        hidden_state_labels=model.state_labels[0],
    )
    spec.validate()
    return spec
