"""Tests for phase segmentation of belief-posterior traces."""

from __future__ import annotations

import math

import pytest

from src.features import HIDDEN_STATE_LABELS
from src.phase_segmentation import (
    Segment,
    detect_changepoints,
    map_state_sequence,
    segment_phases,
)


def test_map_state_sequence_argmax_per_step():
    """MAP label per step is the argmax over each distribution."""

    posteriors = [
        [0.7, 0.1, 0.1, 0.05, 0.05],  # setup
        [0.1, 0.6, 0.1, 0.1, 0.1],  # accumulator_update
        [0.05, 0.05, 0.8, 0.05, 0.05],  # representation_transform
    ]
    result = map_state_sequence(posteriors, HIDDEN_STATE_LABELS)
    assert result == ("setup", "accumulator_update", "representation_transform")


def test_detect_changepoints_monotone():
    """Changepoints occur exactly where the MAP state changes."""

    # MAP indices: 0, 0, 1, 3, 4 -> changes at steps 2, 3, 4.
    posteriors = [
        [0.7, 0.1, 0.1, 0.05, 0.05],  # setup (0)
        [0.6, 0.2, 0.1, 0.05, 0.05],  # setup (0)
        [0.1, 0.7, 0.1, 0.05, 0.05],  # accumulator_update (1)
        [0.1, 0.1, 0.1, 0.6, 0.1],  # guard_check (3)
        [0.05, 0.05, 0.1, 0.1, 0.7],  # exit (4)
    ]
    assert detect_changepoints(posteriors) == (2, 3, 4)


def test_isc19_partition_coverage():
    """ISC-19: concatenated [start..end] exactly equals 0..n-1, no overlap."""

    posteriors = [
        [0.7, 0.1, 0.1, 0.05, 0.05],  # 0
        [0.6, 0.2, 0.1, 0.05, 0.05],  # 0
        [0.1, 0.7, 0.1, 0.05, 0.05],  # 1
        [0.1, 0.7, 0.1, 0.05, 0.05],  # 1
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 3
        [0.05, 0.05, 0.1, 0.1, 0.7],  # 4
        [0.05, 0.05, 0.1, 0.1, 0.7],  # 4
    ]
    segments = segment_phases(posteriors, HIDDEN_STATE_LABELS)

    covered: list[int] = []
    for seg in segments:
        assert seg.start_index <= seg.end_index
        covered.extend(range(seg.start_index, seg.end_index + 1))
    assert covered == list(range(len(posteriors)))

    # Segments are contiguous: each starts right after the prior ends.
    for prev, nxt in zip(segments, segments[1:]):
        assert nxt.start_index == prev.end_index + 1


def test_isc20_monotone_phase_sequence_segments():
    """ISC-20: hand-built monotone argmax sequence yields expected segments.

    Argmax labels: setup, setup, accumulator_update, guard_check, exit.
    Expected: 4 segments with correct labels and bounds.
    """

    posteriors = [
        [0.7, 0.1, 0.1, 0.05, 0.05],  # setup (0)
        [0.6, 0.2, 0.1, 0.05, 0.05],  # setup (0)
        [0.1, 0.7, 0.1, 0.05, 0.05],  # accumulator_update (1)
        [0.1, 0.1, 0.1, 0.6, 0.1],  # guard_check (3)
        [0.05, 0.05, 0.1, 0.1, 0.7],  # exit (4)
    ]
    segments = segment_phases(posteriors, HIDDEN_STATE_LABELS)

    assert len(segments) == 4
    labels = tuple(seg.label for seg in segments)
    assert labels == ("setup", "accumulator_update", "guard_check", "exit")

    bounds = tuple((seg.start_index, seg.end_index) for seg in segments)
    assert bounds == ((0, 1), (2, 2), (3, 3), (4, 4))

    # mean_confidence of the first segment is mean of max-posterior over steps 0,1.
    first = segments[0]
    assert math.isclose(first.mean_confidence, (0.7 + 0.6) / 2.0, rel_tol=0, abs_tol=1e-12)
    # Single-step segment confidence equals that step's max posterior.
    assert math.isclose(segments[1].mean_confidence, 0.7, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(segments[3].mean_confidence, 0.7, rel_tol=0, abs_tol=1e-12)


def test_isc21_constant_belief_single_segment():
    """ISC-21: a constant-belief sequence yields exactly one segment."""

    constant = [0.5, 0.2, 0.1, 0.1, 0.1]  # argmax index 0 -> setup
    posteriors = [list(constant) for _ in range(6)]

    assert detect_changepoints(posteriors) == ()

    segments = segment_phases(posteriors, HIDDEN_STATE_LABELS)
    assert len(segments) == 1
    only = segments[0]
    assert only.start_index == 0
    assert only.end_index == 5
    assert only.label == "setup"
    assert math.isclose(only.mean_confidence, 0.5, rel_tol=0, abs_tol=1e-12)


def test_segment_to_dict_json_safe():
    """Segment.to_dict returns JSON-safe native types."""

    import json

    seg = Segment(start_index=0, end_index=2, label="setup", mean_confidence=0.65)
    data = seg.to_dict()
    assert data == {
        "start_index": 0,
        "end_index": 2,
        "label": "setup",
        "mean_confidence": 0.65,
    }
    assert isinstance(data["start_index"], int)
    assert isinstance(data["mean_confidence"], float)
    # Round-trips through JSON without error.
    assert json.loads(json.dumps(data)) == data


def test_single_step_trace():
    """A one-step trace yields a single segment covering index 0."""

    posteriors = [[0.4, 0.3, 0.1, 0.1, 0.1]]
    assert detect_changepoints(posteriors) == ()
    segments = segment_phases(posteriors, HIDDEN_STATE_LABELS)
    assert len(segments) == 1
    assert segments[0].start_index == 0
    assert segments[0].end_index == 0
    assert segments[0].label == "setup"
    assert math.isclose(segments[0].mean_confidence, 0.4, rel_tol=0, abs_tol=1e-12)


def test_empty_posteriors_raises():
    """Empty posteriors are rejected for all functions."""

    with pytest.raises(ValueError):
        map_state_sequence([], HIDDEN_STATE_LABELS)
    with pytest.raises(ValueError):
        detect_changepoints([])
    with pytest.raises(ValueError):
        segment_phases([], HIDDEN_STATE_LABELS)


def test_mismatched_width_raises():
    """A posterior width not matching the label count is rejected."""

    bad = [[0.5, 0.5]]  # width 2, labels has 5
    with pytest.raises(ValueError):
        map_state_sequence(bad, HIDDEN_STATE_LABELS)
    with pytest.raises(ValueError):
        segment_phases(bad, HIDDEN_STATE_LABELS)


def test_non_2d_raises():
    """A 1D posterior sequence is rejected."""

    with pytest.raises(ValueError):
        map_state_sequence([0.5, 0.5], HIDDEN_STATE_LABELS)
