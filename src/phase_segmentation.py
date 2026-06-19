"""Segment belief-posterior traces into contiguous program phases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A contiguous run of steps sharing one maximum-a-posteriori label."""

    start_index: int
    end_index: int
    label: str
    mean_confidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["start_index"] = int(self.start_index)
        data["end_index"] = int(self.end_index)
        data["label"] = str(self.label)
        data["mean_confidence"] = float(self.mean_confidence)
        return data


def _as_matrix(posteriors: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a posterior sequence to a 2D float array."""

    if len(posteriors) == 0:
        raise ValueError("posteriors must not be empty")
    matrix = np.asarray(posteriors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("posteriors must be a 2D sequence of distributions")
    return matrix


def map_state_sequence(
    posteriors: Sequence[Sequence[float]],
    hidden_state_labels: Sequence[str],
) -> tuple[str, ...]:
    """Return the maximum-a-posteriori label per step."""

    matrix = _as_matrix(posteriors)
    if matrix.shape[1] != len(hidden_state_labels):
        raise ValueError("posterior width must match hidden state label count")
    indices = np.argmax(matrix, axis=1)
    return tuple(hidden_state_labels[int(index)] for index in indices)


def detect_changepoints(posteriors: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return step indices t>=1 where the MAP state differs from step t-1."""

    matrix = _as_matrix(posteriors)
    indices = np.argmax(matrix, axis=1)
    changepoints = [
        int(step)
        for step in range(1, len(indices))
        if int(indices[step]) != int(indices[step - 1])
    ]
    return tuple(changepoints)


def segment_phases(
    posteriors: Sequence[Sequence[float]],
    hidden_state_labels: Sequence[str],
) -> tuple[Segment, ...]:
    """Partition a trace into contiguous segments of equal MAP label.

    The returned segments exactly partition ``range(len(posteriors))`` with no
    gaps or overlaps. Each segment's ``mean_confidence`` is the mean of the
    maximum posterior probability over the segment's steps.
    """

    matrix = _as_matrix(posteriors)
    if matrix.shape[1] != len(hidden_state_labels):
        raise ValueError("posterior width must match hidden state label count")

    map_indices = np.argmax(matrix, axis=1)
    max_posterior = np.max(matrix, axis=1)

    segments: list[Segment] = []
    start = 0
    n_steps = len(map_indices)
    for step in range(1, n_steps + 1):
        is_boundary = step == n_steps or int(map_indices[step]) != int(map_indices[start])
        if is_boundary:
            mean_confidence = float(np.mean(max_posterior[start:step]))
            segments.append(
                Segment(
                    start_index=int(start),
                    end_index=int(step - 1),
                    label=hidden_state_labels[int(map_indices[start])],
                    mean_confidence=mean_confidence,
                )
            )
            start = step
    return tuple(segments)
