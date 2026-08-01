"""Compose the v0.2 temporal/active/evaluation pipeline over the corpus.

This thin orchestrator wires the four new capability modules together and writes
their deterministic artifacts under ``output/data``:

* ``temporal_inference.json`` — per-fixture HMM forward filter, smoothing, and
  Viterbi decoding from :mod:`temporal_model` (the proper ``B``-transition model
  that replaces the implicit ``B = identity`` of the original sticky filter),
* ``phase_segments.json`` — per-fixture phase segmentation of the smoothed
  beliefs from :mod:`phase_segmentation`,
* ``active_selection.json`` — per-fixture active-disassembly ordering and the
  honest active-vs-fixed step comparison from :mod:`active_selection`,
* ``evaluation_report.json`` — corpus decoding accuracy, calibration,
  free-energy convergence, model comparison, and two negative controls from
  :mod:`evaluation`.

Every value written here is computed from the live modules; nothing is
hand-stamped. The orchestrator owns no domain logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .active_selection import active_observation_order, fixed_order_steps_to_confidence
from .corpus import build_corpus, fixture_observation_ids
from .evaluation import evaluation_payload
from .generative_model import build_generative_model
from .phase_segmentation import segment_phases
from .temporal_model import (
    build_temporal_model,
    forward_backward,
    forward_filter,
    viterbi_decode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

CONFIDENCE_THRESHOLD = 0.7


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def temporal_inference_payload() -> dict[str, object]:
    """Per-fixture forward filter, smoothing, and Viterbi decoding."""

    model = build_temporal_model()
    corpus = build_corpus()
    per_fixture: dict[str, object] = {}
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        filter_result = forward_filter(observation_ids, model)
        smoothing = forward_backward(observation_ids, model)
        viterbi_path, viterbi_log_prob = viterbi_decode(observation_ids, model)
        per_fixture[fixture.name] = {
            "observation_ids": observation_ids,
            "truth_labels": list(fixture.phase_labels),
            "filter": filter_result.to_dict(),
            "smoothing": smoothing.to_dict(),
            "viterbi_path": list(viterbi_path),
            "viterbi_log_prob": float(viterbi_log_prob),
        }
    return {
        "model": model.to_dict(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "per_fixture": per_fixture,
    }


def phase_segments_payload() -> dict[str, object]:
    """Per-fixture phase segmentation of the smoothed beliefs."""

    model = build_temporal_model()
    corpus = build_corpus()
    per_fixture: dict[str, object] = {}
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        smoothing = forward_backward(observation_ids, model)
        segments = segment_phases(
            smoothing.smoothed_posteriors, model.hidden_state_labels
        )
        per_fixture[fixture.name] = {
            "segment_count": len(segments),
            "segments": [segment.to_dict() for segment in segments],
        }
    return {"per_fixture": per_fixture}


def active_selection_payload() -> dict[str, object]:
    """Per-fixture active-disassembly ordering and the active-vs-fixed comparison.

    Honest reporting: with a flat preference and the peaked likelihood ``A`` of
    short fixtures, the first observation already saturates confidence, so the
    active ordering and the fixed left-to-right scan reach the threshold in the
    same number of steps. The active *mechanism* is unit-verified in
    ``tests/test_active_selection.py``; this artifact records the corpus outcome
    as measured, including ties.
    """

    model = build_generative_model()
    preference = np.ones(len(model.observation_labels), dtype=float)
    corpus = build_corpus()
    per_fixture: dict[str, dict[str, object]] = {}
    active_wins = 0
    ties = 0
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        selection = active_observation_order(
            observation_ids,
            model.A,
            model.D,
            preference,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        fixed_steps = fixed_order_steps_to_confidence(
            observation_ids, model.A, model.D, CONFIDENCE_THRESHOLD
        )
        active_steps = selection.steps_to_confidence
        if (
            active_steps is not None
            and fixed_steps is not None
            and active_steps < fixed_steps
        ):
            active_wins += 1
        elif active_steps == fixed_steps:
            ties += 1
        per_fixture[fixture.name] = {
            "selection": selection.to_dict(),
            "active_steps_to_confidence": active_steps,
            "fixed_steps_to_confidence": fixed_steps,
            "active_no_worse_than_fixed": (
                active_steps is not None
                and fixed_steps is not None
                and active_steps <= fixed_steps
            ),
        }
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "preference": [float(value) for value in preference],
        "summary": {
            "fixture_count": len(corpus),
            "active_strictly_fewer_steps": int(active_wins),
            "ties": int(ties),
            "active_never_worse": all(
                row["active_no_worse_than_fixed"] for row in per_fixture.values()
            ),
        },
        "per_fixture": per_fixture,
    }


def evaluation_report_payload() -> dict[str, Any]:
    """Corpus evaluation metrics with negative controls."""

    model = build_temporal_model()
    corpus = build_corpus()
    return evaluation_payload(corpus, model)


def run_advanced_inference(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the four v0.2 artifacts and return their relative paths."""

    data_dir = output_dir / "data"
    return {
        "temporal_inference": _write_json(
            data_dir / "temporal_inference.json", temporal_inference_payload()
        ),
        "phase_segments": _write_json(
            data_dir / "phase_segments.json", phase_segments_payload()
        ),
        "active_selection": _write_json(
            data_dir / "active_selection.json", active_selection_payload()
        ),
        "evaluation_report": _write_json(
            data_dir / "evaluation_report.json", evaluation_report_payload()
        ),
    }
