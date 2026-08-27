"""Deterministic sensor-cost vectors and finite policy certificates.

The finite information frontier and the measured-cost layer are deliberately
separate: the former uses model-channel units, while this module records an
auditable action-cost vector and optional local timing calibration. The exact
dynamic-programming certificate is scoped to the finite declared sensor/action
space; it is not a universal cost law.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, cast

import numpy as np

from .artifact_contract import write_json_artifact
from .information_policy import (
    FiniteSensor,
    active_information_payload,
    entropy,
    solve_exact_policy,
)
from .provenance import (
    generator_revision,
    sha256_bytes,
    sha256_path,
    toolchain_versions,
)
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSOR_COST_SCHEMA_VERSION = "2.0.0"
SENSOR_COST_ARTIFACT = "output/data/sensor_cost_receipt.json"
CALIBRATION_TRIALS = 64
CALIBRATION_REPETITIONS = 5
COST_COMPONENTS = (
    "sensor_invocations",
    "decoded_instructions",
    "bounded_execution_steps",
    "register_flag_observations",
    "branch_observations",
    "memory_hook_events",
    "serialized_bytes",
    "unsupported_attempts",
)
POLICY_MODES = (
    "exact_oracle",
    "fixed_first",
    "greedy_information",
    "seeded_random",
    "abstain",
)


def _deterministic_vector(sensor: dict[str, Any]) -> dict[str, int]:
    sensor_id = str(sensor["sensor_id"])
    outcomes = [str(value) for value in sensor["outcomes"]]
    return {
        "sensor_invocations": 1,
        "decoded_instructions": 1,
        "bounded_execution_steps": 1,
        "register_flag_observations": int(
            "register" in sensor_id or "flag" in sensor_id
        ),
        "branch_observations": int("branch" in sensor_id),
        "memory_hook_events": int("memory" in sensor_id),
        "serialized_bytes": len(sensor_id.encode("utf-8"))
        + sum(len(outcome.encode("utf-8")) for outcome in outcomes),
        "unsupported_attempts": 0,
    }


def _timed_sensor(sensor: dict[str, Any], belief: np.ndarray) -> list[float]:
    matrix = np.asarray(sensor["likelihood"], dtype=float)
    observations: list[float] = []
    for _ in range(CALIBRATION_REPETITIONS):
        start = time.perf_counter_ns()
        accumulator = 0.0
        for _ in range(CALIBRATION_TRIALS):
            accumulator += float(np.sum(matrix @ belief))
        elapsed = time.perf_counter_ns() - start
        if accumulator <= 0.0:
            raise RuntimeError("sensor calibration accumulator is non-positive")
        observations.append(elapsed / CALIBRATION_TRIALS)
    return observations


def _timed_overhead() -> float:
    """Measure the fixed loop/timer overhead using the same trial count."""

    samples: list[float] = []
    for _ in range(CALIBRATION_REPETITIONS):
        start = time.perf_counter_ns()
        accumulator = 0.0
        for _ in range(CALIBRATION_TRIALS):
            accumulator += 1.0
        elapsed = time.perf_counter_ns() - start
        if accumulator <= 0.0:
            raise RuntimeError(
                "sensor calibration overhead accumulator is non-positive"
            )
        samples.append(elapsed / CALIBRATION_TRIALS)
    return float(statistics.median(samples))


def _zero_cost_vector() -> dict[str, float]:
    return {component: 0.0 for component in COST_COMPONENTS}


def _add_cost_vectors(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {
        component: float(left[component] + right[component])
        for component in COST_COMPONENTS
    }


def _policy_value(
    belief: np.ndarray,
    sensors: tuple[FiniteSensor, ...],
    remaining: tuple[int, ...],
    horizon: int,
    cost_weight: float,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
    memo: dict[tuple[tuple[float, ...], tuple[int, ...], int], dict[str, Any]]
    | None = None,
    calibrated_costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Enumerate outcome branches for one finite policy family.

    The oracle mode recursively chooses the minimum objective over all finite
    sensor actions. Other modes choose an action without consulting labels or
    held-out outcomes. Since the state and action spaces are finite, the
    returned expectations are exact conditional on the fixed random-policy
    seed.
    """

    normalized = np.asarray(belief, dtype=float)
    normalized = normalized / normalized.sum()
    key = (tuple(np.round(normalized, 12)), remaining, horizon)
    if mode == "exact_oracle" and memo is not None and key in memo:
        return dict(memo[key])

    terminal = {
        "objective": float(entropy(normalized)),
        "terminal_entropy": float(entropy(normalized)),
        "action_count": 0.0,
        "deterministic_cost_vector": _zero_cost_vector(),
        "calibrated_cost_nanoseconds": 0.0,
        "first_sensor": None,
        "evaluated_policy_count": 1,
    }
    if horizon <= 0 or not remaining or mode == "abstain":
        if mode == "exact_oracle" and memo is not None:
            memo[key] = dict(terminal)
        return terminal

    matrix_by_index = {
        index: np.asarray(sensors[index].likelihood, dtype=float) for index in remaining
    }
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    if mode == "exact_oracle":
        selected_indices = remaining
    elif mode == "fixed_first":
        selected_indices = (min(remaining),)
    elif mode == "greedy_information":
        information_scores: list[tuple[float, int]] = []
        prior_entropy = entropy(normalized)
        for index in remaining:
            matrix = matrix_by_index[index]
            expected_entropy = 0.0
            for outcome_index in range(matrix.shape[0]):
                probability, posterior = _posterior_for_outcome(
                    normalized, matrix, outcome_index
                )
                expected_entropy += probability * entropy(posterior)
            information_scores.append((prior_entropy - expected_entropy, index))
        selected_indices = (
            max(information_scores, key=lambda item: (item[0], -item[1]))[1],
        )
    elif mode == "seeded_random":
        if rng is None:
            raise ValueError("seeded_random policy requires a generator")
        selected_indices = (int(rng.choice(np.asarray(remaining, dtype=int))),)
    else:
        raise ValueError(f"unknown finite policy mode {mode!r}")

    for sensor_index in selected_indices:
        sensor = sensors[sensor_index]
        matrix = matrix_by_index[sensor_index]
        expected_entropy = 0.0
        expected_actions = 1.0
        expected_vector = {
            component: float(_deterministic_vector(sensor.to_dict())[component])
            for component in COST_COMPONENTS
        }
        expected_calibrated = (
            float(calibrated_costs.get(sensor.sensor_id, 0.0))
            if calibrated_costs is not None
            else 0.0
        )
        child_count = 1
        for outcome_index in range(matrix.shape[0]):
            probability, posterior = _posterior_for_outcome(
                normalized, matrix, outcome_index
            )
            if probability <= 0.0:
                continue
            child = _policy_value(
                posterior,
                sensors,
                tuple(index for index in remaining if index != sensor_index),
                horizon - 1,
                cost_weight,
                mode,
                rng=rng,
                memo=memo,
                calibrated_costs=calibrated_costs,
            )
            expected_entropy += probability * float(child["terminal_entropy"])
            expected_actions += probability * float(child["action_count"])
            expected_vector = _add_cost_vectors(
                expected_vector,
                {
                    component: probability
                    * float(child["deterministic_cost_vector"][component])
                    for component in COST_COMPONENTS
                },
            )
            expected_calibrated += probability * float(
                child["calibrated_cost_nanoseconds"]
            )
            child_count += int(child["evaluated_policy_count"])
        candidate = {
            "objective": float(expected_entropy + cost_weight * expected_actions),
            "terminal_entropy": float(expected_entropy),
            "action_count": float(expected_actions),
            "deterministic_cost_vector": expected_vector,
            "calibrated_cost_nanoseconds": float(expected_calibrated),
            "first_sensor": sensor.sensor_id,
            "evaluated_policy_count": child_count,
        }
        candidates.append(
            (
                float(expected_entropy + cost_weight * expected_actions),
                sensor_index,
                candidate,
            )
        )

    chosen = min(candidates, key=lambda item: (item[0], item[1]))
    result = dict(chosen[2])
    if mode == "exact_oracle" and memo is not None:
        memo[key] = dict(result)
    return result


def _posterior_for_outcome(
    belief: np.ndarray, matrix: np.ndarray, outcome_index: int
) -> tuple[float, np.ndarray]:
    probability = float(matrix[outcome_index] @ belief)
    if probability <= 0.0:
        return 0.0, belief
    posterior = belief * matrix[outcome_index]
    return probability, posterior / posterior.sum()


def _cached_sensor_cost_payload(project_root: Path) -> dict[str, Any] | None:
    """Reuse a matching calibration receipt so repeated producers are stable."""

    path = project_root / SENSOR_COST_ARTIFACT
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    protocol = scientific_protocol_payload()
    if payload.get("protocol_sha256") != protocol["protocol_sha256"] or payload.get(
        "generator_revision"
    ) != generator_revision(Path(__file__)):
        return None
    input_hashes = payload.get("input_hashes")
    expected_frontier_hash = sha256_bytes(
        json.dumps(
            active_information_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    if (
        not isinstance(input_hashes, dict)
        or input_hashes.get("active_information_payload") != expected_frontier_hash
        or input_hashes.get("src/sensor_costs.py") != sha256_path(Path(__file__))
    ):
        return None
    valid, _ = validate_sensor_cost_payload(payload)
    return payload if valid else None


def sensor_cost_payload(
    project_root: Path = PROJECT_ROOT, *, reuse_existing: bool = False
) -> dict[str, Any]:
    """Return deterministic vectors, local calibration, and exact certificates.

    ``reuse_existing`` is used by the normal producer path after a matching
    receipt has been established.  This keeps a measured wall-clock figure
    source-bound across ordinary manuscript hydration runs while allowing an
    explicit recalibration through :func:`write_sensor_cost_payload`.
    """

    if reuse_existing:
        cached = _cached_sensor_cost_payload(project_root)
        if cached is not None:
            return cached

    frontier_payload = active_information_payload()
    belief = np.asarray(frontier_payload["initial_belief"], dtype=float)
    sensors = cast(list[dict[str, Any]], frontier_payload["sensors"])
    sensor_rows: list[dict[str, Any]] = []
    overhead = _timed_overhead()
    for sensor in sensors:
        samples = _timed_sensor(sensor, belief)
        deterministic = _deterministic_vector(sensor)
        corrected = [max(0.0, value - overhead) for value in samples]
        sensor_rows.append(
            {
                "sensor_id": sensor["sensor_id"],
                "deterministic_cost_vector": deterministic,
                "wall_clock_nanoseconds_per_query": {
                    "median": float(statistics.median(samples)),
                    "min": float(min(samples)),
                    "max": float(max(samples)),
                    "samples": [float(value) for value in samples],
                    "overhead_corrected_median": float(statistics.median(corrected)),
                    "iqr": float(
                        np.percentile(samples, 75) - np.percentile(samples, 25)
                    ),
                    "p95": float(np.percentile(samples, 95)),
                },
            }
        )

    design = np.asarray(
        [
            [1.0]
            + [
                float(row["deterministic_cost_vector"][component])
                for component in COST_COMPONENTS
            ]
            for row in sensor_rows
        ],
        dtype=float,
    )
    timing = np.asarray(
        [
            float(row["wall_clock_nanoseconds_per_query"]["median"])
            for row in sensor_rows
        ],
        dtype=float,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, timing, rcond=None)
    fitted = design @ coefficients
    residuals = timing - fitted
    total_sum_squares = float(np.sum((timing - timing.mean()) ** 2))
    regression = {
        "design_columns": ["intercept", *COST_COMPONENTS],
        "coefficients": [float(value) for value in coefficients],
        "residuals": [float(value) for value in residuals],
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
        "residual_max_absolute": float(np.max(np.abs(residuals))),
        "r_squared": float(
            1.0 - float(np.sum(residuals**2)) / total_sum_squares
            if total_sum_squares > 0.0
            else 1.0
        ),
    }

    finite_sensors = tuple(
        FiniteSensor(
            sensor_id=str(sensor["sensor_id"]),
            outcomes=tuple(str(value) for value in sensor["outcomes"]),
            cost=1.0,
            likelihood=tuple(
                tuple(float(value) for value in values)
                for values in sensor["likelihood"]
            ),
        )
        for sensor in sensors
    )
    calibrated_costs = {
        str(row["sensor_id"]): float(
            row["wall_clock_nanoseconds_per_query"]["overhead_corrected_median"]
        )
        for row in sensor_rows
    }
    certificates: list[dict[str, Any]] = []
    policy_evaluations: list[dict[str, Any]] = []
    frontier_rows = cast(list[dict[str, Any]], frontier_payload["frontier"])
    for row in frontier_rows:
        horizon = int(row["horizon"])
        cost_weight = float(row["cost_weight"])
        oracle_reference = solve_exact_policy(
            belief,
            finite_sensors,
            horizon=horizon,
            cost_weight=cost_weight,
        )
        oracle = _policy_value(
            belief,
            finite_sensors,
            tuple(range(len(finite_sensors))),
            horizon,
            cost_weight,
            "exact_oracle",
            memo={},
            calibrated_costs=calibrated_costs,
        )
        if not np.isclose(
            float(oracle["objective"]), float(oracle_reference.objective), atol=1e-10
        ):
            raise RuntimeError("local finite oracle disagrees with information policy")
        evaluated: dict[str, dict[str, Any]] = {}
        for mode in POLICY_MODES:
            evaluated[mode] = _policy_value(
                belief,
                finite_sensors,
                tuple(range(len(finite_sensors))),
                horizon,
                cost_weight,
                mode,
                rng=np.random.default_rng(20260821)
                if mode == "seeded_random"
                else None,
                memo={} if mode == "exact_oracle" else None,
                calibrated_costs=calibrated_costs,
            )
            result = evaluated[mode]
            oracle_objective = float(oracle["objective"])
            regret = max(0.0, float(result["objective"]) - oracle_objective)
            denominator = max(abs(oracle_objective), 1e-12)
            policy_evaluations.append(
                {
                    "horizon": horizon,
                    "cost_weight": cost_weight,
                    "policy": mode,
                    "first_sensor": result["first_sensor"],
                    "objective": float(result["objective"]),
                    "terminal_entropy": float(result["terminal_entropy"]),
                    "information_gain": float(
                        entropy(belief) - float(result["terminal_entropy"])
                    ),
                    "expected_sensor_invocations": float(result["action_count"]),
                    "deterministic_cost_vector": result["deterministic_cost_vector"],
                    "calibrated_cost_nanoseconds": float(
                        result["calibrated_cost_nanoseconds"]
                    ),
                    "regret_vs_exact_oracle": regret,
                    "optimality_gap": float(regret / denominator),
                    "confidence_interval": None,
                    "uncertainty": "exact_outcome_enumeration_fixed_policy_seed",
                    "evaluated_policy_count": int(result["evaluated_policy_count"]),
                }
            )
        exact = evaluated["exact_oracle"]
        certificates.append(
            {
                "horizon": horizon,
                "cost_weight": cost_weight,
                "policy": "exact_dynamic_program",
                "oracle_objective": float(oracle_reference.objective),
                "achieved_objective": float(exact["objective"]),
                "lower_bound": float(oracle_reference.objective),
                "deterministic_cost_vector": exact["deterministic_cost_vector"],
                "calibrated_cost_nanoseconds": float(
                    exact["calibrated_cost_nanoseconds"]
                ),
                "optimality_gap": 0.0,
                "within_preregistered_tolerance": True,
                "scope": "finite declared sensor/action space only",
            }
        )
    protocol = scientific_protocol_payload()
    limitations = [
        "wall-clock values are machine-specific calibration observations",
        "the exact certificate is finite and model-channel specific",
        "the exact oracle is compared with fixed, greedy, seeded-random, and abstaining policies",
    ]
    underpowered_cases = [
        "No cross-machine timing comparison is promoted without a matching calibration receipt.",
        "The finite certificate does not establish universal active-sensing optimality.",
    ]
    input_hashes = {
        "active_information_payload": sha256_bytes(
            json.dumps(frontier_payload, sort_keys=True, separators=(",", ":")).encode()
        ),
        "src/sensor_costs.py": sha256_path(Path(__file__)),
    }
    toolchains = toolchain_versions(("numpy",))
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=input_hashes,
        toolchain_versions=toolchains,
        status="validated_bounded",
        limitations=limitations,
        unsupported_cases=(
            "Rejected or unsupported sensor attempts remain explicit in the deterministic vector.",
        ),
        underpowered_cases=underpowered_cases,
    )
    return {
        **envelope,
        "schema_version": SENSOR_COST_SCHEMA_VERSION,
        "artifact": SENSOR_COST_ARTIFACT,
        "endpoint_status": "calibrated_local_finite_certificate",
        "primary_cost": "deterministic_vector",
        "calibration": {
            "trials_per_repetition": CALIBRATION_TRIALS,
            "repetitions": CALIBRATION_REPETITIONS,
            "timing_overhead_nanoseconds_per_query": overhead,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "entropy_nats": entropy(belief),
            "regression": regression,
        },
        "sensors": sensor_rows,
        "policy_certificates": certificates,
        "policy_evaluations": policy_evaluations,
        "limitations": limitations,
        "unsupported_cases": list(envelope["unsupported_cases"]),
        "underpowered_cases": underpowered_cases,
    }


def validate_sensor_cost_payload(payload: object) -> tuple[bool, str]:
    """Validate cost-vector shape, finite timing, and zero-gap certificates."""

    if not isinstance(payload, dict):
        return False, "sensor cost payload must be an object"
    if payload.get("schema_version") != SENSOR_COST_SCHEMA_VERSION:
        return False, "sensor cost schema mismatch"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"sensor cost receipt: {receipt_detail}"
    if payload.get("status") != "validated_bounded":
        return False, "sensor cost evidence status is invalid"
    if (
        not isinstance(payload.get("generator_revision"), str)
        or not isinstance(payload.get("input_hashes"), dict)
        or not isinstance(payload.get("toolchain_versions"), dict)
    ):
        return False, "sensor cost provenance is incomplete"
    sensors = payload.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        return False, "sensor cost rows are missing"
    required_components = {
        "sensor_invocations",
        "decoded_instructions",
        "bounded_execution_steps",
        "register_flag_observations",
        "branch_observations",
        "memory_hook_events",
        "serialized_bytes",
        "unsupported_attempts",
    }
    for row in sensors:
        if not isinstance(row, dict):
            return False, "sensor cost row is not an object"
        vector = row.get("deterministic_cost_vector")
        if not isinstance(vector, dict) or set(vector) != required_components:
            return False, "sensor cost vector components drift"
        if any(not isinstance(value, int) or value < 0 for value in vector.values()):
            return False, "sensor cost vector must contain non-negative integers"
        timing = row.get("wall_clock_nanoseconds_per_query")
        try:
            timing_valid = (
                isinstance(timing, dict)
                and float(timing.get("median", 0.0)) > 0.0
                and float(timing.get("iqr", -1.0)) >= 0.0
                and float(timing.get("p95", 0.0)) > 0.0
                and float(timing.get("overhead_corrected_median", -1.0)) >= 0.0
                and isinstance(timing.get("samples"), list)
                and len(timing["samples"]) == CALIBRATION_REPETITIONS
                and all(float(value) > 0.0 for value in timing["samples"])
            )
        except (TypeError, ValueError):
            timing_valid = False
        if not timing_valid:
            return False, "sensor cost timing is invalid"
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("regression"), dict
    ):
        return False, "sensor calibration regression diagnostics are missing"
    regression = calibration["regression"]
    if regression.get("design_columns") != ["intercept", *COST_COMPONENTS]:
        return False, "sensor calibration design columns drift"
    for key in (
        "coefficients",
        "residuals",
    ):
        values = regression.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, (int, float)) or not np.isfinite(float(value))
            for value in values
        ):
            return False, "sensor calibration regression values are invalid"
    for key in ("residual_rms", "residual_max_absolute", "r_squared"):
        value = regression.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return False, "sensor calibration residual diagnostics are invalid"
    certificates = payload.get("policy_certificates")
    if not isinstance(certificates, list) or not certificates:
        return False, "sensor cost certificates are missing"
    for certificate in certificates:
        if not isinstance(certificate, dict):
            return False, "finite oracle certificate is not an object"
        if certificate.get("optimality_gap") != 0.0:
            return False, "finite oracle certificate has a non-zero gap"
        if certificate.get("within_preregistered_tolerance") is not True:
            return False, "finite oracle certificate is not within tolerance"
        vector = certificate.get("deterministic_cost_vector")
        if not isinstance(vector, dict) or set(vector) != set(COST_COMPONENTS):
            return False, "finite oracle deterministic cost vector is missing"
    evaluations = payload.get("policy_evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        return False, "finite policy comparison rows are missing"
    observed_modes = {row.get("policy") for row in evaluations if isinstance(row, dict)}
    if observed_modes != set(POLICY_MODES):
        return False, "finite policy comparison modes are incomplete"
    for row in evaluations:
        if not isinstance(row, dict):
            return False, "finite policy comparison row is not an object"
        for key in (
            "objective",
            "terminal_entropy",
            "information_gain",
            "expected_sensor_invocations",
            "calibrated_cost_nanoseconds",
            "regret_vs_exact_oracle",
            "optimality_gap",
        ):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                return False, f"finite policy comparison value {key} is invalid"
        vector = row.get("deterministic_cost_vector")
        if not isinstance(vector, dict) or set(vector) != set(COST_COMPONENTS):
            return False, "finite policy comparison vector is invalid"
        if any(float(value) < -1e-12 for value in vector.values()):
            return False, "finite policy comparison vector is negative"
    return True, "ok"


def write_sensor_cost_payload(
    project_root: Path = PROJECT_ROOT, *, recalibrate: bool = False
) -> Path:
    """Write the deterministic sensor-cost and policy-certificate receipt."""

    payload = sensor_cost_payload(project_root, reuse_existing=not recalibrate)
    valid, detail = validate_sensor_cost_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / SENSOR_COST_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path
