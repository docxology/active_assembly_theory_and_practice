"""Exact finite-horizon information-per-cost policy analysis.

This module is intentionally separate from the held-out action-sensing
experiment. The latter evaluates a leakage-safe policy on fixture records; this
module solves a finite sensor decision problem exactly by enumerating every
sensor/outcome branch for a bounded horizon. It therefore supplies a
mechanism-level frontier and a route to a future theorem or counterexample,
not a deployment claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .corpus import build_corpus
from .factorized_model import (
    FactorizedObservationModel,
    build_typed_observation_corpus,
    fit_factorized_observation_model,
)
from .features import HIDDEN_STATE_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFORMATION_POLICY_ARTIFACT = "output/data/active_information_frontier.json"
INFORMATION_POLICY_SCHEMA_VERSION = "1.0.0"
INFORMATION_POLICY_FIELDS = (
    "instruction_family",
    "operand_class",
    "register_read_set",
    "register_write_set",
)
INFORMATION_POLICY_HORIZONS = (0, 1, 2, 3)
INFORMATION_POLICY_COST_WEIGHTS = (0.0, 0.05, 0.2, 1.0)


@dataclass(frozen=True)
class FiniteSensor:
    """A categorical sensor with a declared cost and state-conditioned channel."""

    sensor_id: str
    outcomes: tuple[str, ...]
    cost: float
    likelihood: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["outcomes"] = list(self.outcomes)
        data["likelihood"] = [list(row) for row in self.likelihood]
        return data


@dataclass(frozen=True)
class ExactPolicyResult:
    """Optimal finite-horizon policy summary for one cost weight."""

    horizon: int
    cost_weight: float
    first_sensor: str | None
    expected_terminal_entropy: float
    expected_information_gain: float
    expected_cost: float
    objective: float
    evaluated_policy_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("belief must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("belief must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("belief must have positive mass")
    return values / total


def entropy(belief: np.ndarray) -> float:
    """Return Shannon entropy in nats."""

    normalized = _normalize(belief)
    positive = normalized[normalized > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def sensors_from_factorized_model(
    model: FactorizedObservationModel,
    fields: Sequence[str] = INFORMATION_POLICY_FIELDS,
) -> tuple[FiniteSensor, ...]:
    """Project a fitted label-blind field model into finite sensor channels."""

    selected = tuple(fields)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("sensor fields must be non-empty and unique")
    if any(field not in model.fields for field in selected):
        raise ValueError("all sensor fields must be present in the fitted model")
    sensors: list[FiniteSensor] = []
    for field in selected:
        outcomes = tuple(model.field_support[field])
        likelihood = tuple(
            tuple(
                float(model.probabilities[field][state][outcome])
                for state in model.hidden_state_labels
            )
            for outcome in outcomes
        )
        sensors.append(
            FiniteSensor(
                sensor_id=field,
                outcomes=outcomes,
                cost=1.0,
                likelihood=likelihood,
            )
        )
    return tuple(sensors)


def _validate_sensor(sensor: FiniteSensor, state_count: int) -> np.ndarray:
    if not sensor.outcomes or sensor.cost <= 0.0:
        raise ValueError("sensor outcomes and positive cost are required")
    matrix = np.asarray(sensor.likelihood, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(sensor.outcomes), state_count):
        raise ValueError(f"sensor {sensor.sensor_id!r} has an invalid likelihood shape")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"sensor {sensor.sensor_id!r} likelihood is invalid")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError(f"sensor {sensor.sensor_id!r} columns must normalize")
    return matrix


def _posterior_for_outcome(
    belief: np.ndarray, matrix: np.ndarray, outcome_index: int
) -> tuple[float, np.ndarray]:
    predictive = float(matrix[outcome_index] @ belief)
    if predictive <= 0.0:
        return 0.0, _normalize(belief)
    return predictive, _normalize(belief * matrix[outcome_index])


def _optimal(
    belief: np.ndarray,
    sensors: tuple[FiniteSensor, ...],
    remaining: tuple[int, ...],
    horizon: int,
    cost_weight: float,
    memo: dict[
        tuple[tuple[float, ...], tuple[int, ...], int],
        tuple[float, float, float, int, str | None],
    ],
) -> tuple[float, float, float, int, str | None]:
    """Return objective, entropy, cost, policy count, and first sensor."""

    normalized = _normalize(belief)
    key = (tuple(np.round(normalized, 12)), remaining, horizon)
    if key in memo:
        return memo[key]
    if horizon <= 0 or not remaining:
        result: tuple[float, float, float, int, str | None] = (
            entropy(normalized),
            entropy(normalized),
            0.0,
            1,
            None,
        )
        memo[key] = result
        return result

    candidates: list[tuple[float, float, float, int, str, int]] = []
    total_policy_count = 0
    for sensor_index in remaining:
        sensor = sensors[sensor_index]
        matrix = np.asarray(sensor.likelihood, dtype=float)
        expected_objective = cost_weight * sensor.cost
        expected_entropy = 0.0
        expected_cost = sensor.cost
        policy_count = 1
        for outcome_index in range(matrix.shape[0]):
            probability, posterior = _posterior_for_outcome(
                normalized, matrix, outcome_index
            )
            if probability <= 0.0:
                continue
            child = _optimal(
                posterior,
                sensors,
                tuple(index for index in remaining if index != sensor_index),
                horizon - 1,
                cost_weight,
                memo,
            )
            expected_objective += probability * child[0]
            expected_entropy += probability * child[1]
            expected_cost += probability * child[2]
            policy_count += child[3]
        candidates.append(
            (
                expected_objective,
                expected_entropy,
                expected_cost,
                policy_count,
                sensor.sensor_id,
                sensor_index,
            )
        )
        total_policy_count += policy_count
    chosen = min(candidates, key=lambda row: (row[0], row[5]))
    result = (
        float(chosen[0]),
        float(chosen[1]),
        float(chosen[2]),
        int(total_policy_count),
        str(chosen[4]),
    )
    memo[key] = result
    return result


def solve_exact_policy(
    belief: Sequence[float] | np.ndarray,
    sensors: Sequence[FiniteSensor],
    *,
    horizon: int,
    cost_weight: float,
) -> ExactPolicyResult:
    """Solve the finite sensor problem by exhaustive outcome-branch DP."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if cost_weight < 0.0 or not np.isfinite(cost_weight):
        raise ValueError("cost_weight must be finite and non-negative")
    sensor_tuple = tuple(sensors)
    if not sensor_tuple:
        raise ValueError("at least one sensor is required")
    normalized = _normalize(np.asarray(belief, dtype=float))
    for sensor in sensor_tuple:
        _validate_sensor(sensor, normalized.size)
    prior_entropy = entropy(normalized)
    objective, terminal_entropy, expected_cost, policy_count, first_sensor = _optimal(
        normalized,
        sensor_tuple,
        tuple(range(len(sensor_tuple))),
        horizon,
        float(cost_weight),
        {},
    )
    return ExactPolicyResult(
        horizon=int(horizon),
        cost_weight=float(cost_weight),
        first_sensor=first_sensor,
        expected_terminal_entropy=float(terminal_entropy),
        expected_information_gain=float(prior_entropy - terminal_entropy),
        expected_cost=float(expected_cost),
        objective=float(objective),
        evaluated_policy_count=int(policy_count),
    )


def active_information_payload() -> dict[str, object]:
    """Generate the exact information-per-cost frontier from label-blind fields."""

    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    model = fit_factorized_observation_model(
        corpus,
        {fixture.name: typed[fixture.name] for fixture in corpus},
        fields=INFORMATION_POLICY_FIELDS,
    )
    sensors = sensors_from_factorized_model(model)
    belief = _normalize(np.asarray(model.D, dtype=float))
    frontier = [
        solve_exact_policy(
            belief,
            sensors,
            horizon=horizon,
            cost_weight=cost_weight,
        ).to_dict()
        for horizon in INFORMATION_POLICY_HORIZONS
        for cost_weight in INFORMATION_POLICY_COST_WEIGHTS
    ]
    return {
        "schema_version": INFORMATION_POLICY_SCHEMA_VERSION,
        "method": "exact_finite_horizon_sensor_dynamic_programming",
        "fields": list(INFORMATION_POLICY_FIELDS),
        "hidden_state_labels": list(HIDDEN_STATE_LABELS),
        "initial_belief": [float(value) for value in belief],
        "initial_entropy": entropy(belief),
        "sensors": [sensor.to_dict() for sensor in sensors],
        "frontier": frontier,
        "exact_enumeration": True,
        "scope": (
            "finite label-blind sensor-channel mechanism analysis with measured "
            "model-derived proxy costs; not a deployment cost law, adaptive "
            "binary-analysis system, or theorem about all sensor families"
        ),
    }


def validate_active_information_payload(
    payload: dict[str, Any],
) -> tuple[bool, str]:
    """Validate the exact-DP receipt and its finite frontier rows."""

    if payload.get("schema_version") != INFORMATION_POLICY_SCHEMA_VERSION:
        return False, "schema version drift"
    if payload.get("exact_enumeration") is not True:
        return False, "exact enumeration flag is missing"
    sensors = payload.get("sensors")
    frontier = payload.get("frontier")
    if not isinstance(sensors, list) or not sensors:
        return False, "sensors are missing"
    if not isinstance(frontier, list) or not frontier:
        return False, "frontier is missing"
    state_count = len(payload.get("hidden_state_labels", []))
    for raw_sensor in sensors:
        if not isinstance(raw_sensor, dict):
            return False, "sensor row is not an object"
        sensor = FiniteSensor(
            sensor_id=str(raw_sensor.get("sensor_id")),
            outcomes=tuple(str(value) for value in raw_sensor.get("outcomes", [])),
            cost=float(raw_sensor.get("cost", 0.0)),
            likelihood=tuple(
                tuple(float(value) for value in row)
                for row in raw_sensor.get("likelihood", [])
            ),
        )
        try:
            _validate_sensor(sensor, state_count)
        except (TypeError, ValueError) as exc:
            return False, str(exc)
    for row in frontier:
        if not isinstance(row, dict):
            return False, "frontier row is not an object"
        for key in (
            "horizon",
            "cost_weight",
            "expected_terminal_entropy",
            "expected_information_gain",
            "expected_cost",
            "objective",
            "evaluated_policy_count",
        ):
            if key not in row:
                return False, f"frontier key missing: {key}"
        if float(row["expected_information_gain"]) < -1e-8:
            return False, "information gain cannot be negative"
        if int(row["evaluated_policy_count"]) <= 0:
            return False, "policy count must be positive"
    return True, f"sensors={len(sensors)} frontier_rows={len(frontier)}"


def write_active_information_payload(project_root: Path = PROJECT_ROOT) -> str:
    """Write the exact finite-horizon information frontier artifact."""

    payload = active_information_payload()
    valid, detail = validate_active_information_payload(payload)
    if not valid:
        raise ValueError(f"active-information payload is invalid: {detail}")
    path = project_root / INFORMATION_POLICY_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return INFORMATION_POLICY_ARTIFACT


__all__ = [
    "ExactPolicyResult",
    "FiniteSensor",
    "INFORMATION_POLICY_ARTIFACT",
    "INFORMATION_POLICY_FIELDS",
    "INFORMATION_POLICY_SCHEMA_VERSION",
    "active_information_payload",
    "entropy",
    "sensors_from_factorized_model",
    "solve_exact_policy",
    "validate_active_information_payload",
    "write_active_information_payload",
]
