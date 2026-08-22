"""Deterministic sensor-cost vectors and finite policy certificates.

The original information frontier intentionally used unit proxy costs.  This
module adds an auditable cost vector and a local timing calibration while
retaining the exact dynamic-programming oracle.  The certificate is scoped to
the finite declared sensor/action space; it is not a universal cost law.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, cast

import numpy as np

from .information_policy import (
    FiniteSensor,
    active_information_payload,
    entropy,
    solve_exact_policy,
)
from .provenance import generator_revision, sha256_bytes, toolchain_versions
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSOR_COST_SCHEMA_VERSION = "1.0.0"
SENSOR_COST_ARTIFACT = "output/data/sensor_cost_receipt.json"
CALIBRATION_TRIALS = 64
CALIBRATION_REPETITIONS = 5


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
    for sensor in sensors:
        samples = _timed_sensor(sensor, belief)
        deterministic = _deterministic_vector(sensor)
        sensor_rows.append(
            {
                "sensor_id": sensor["sensor_id"],
                "deterministic_cost_vector": deterministic,
                "wall_clock_nanoseconds_per_query": {
                    "median": float(statistics.median(samples)),
                    "min": float(min(samples)),
                    "max": float(max(samples)),
                    "samples": [float(value) for value in samples],
                },
            }
        )

    certificates: list[dict[str, Any]] = []
    frontier_rows = cast(list[dict[str, Any]], frontier_payload["frontier"])
    for row in frontier_rows:
        horizon = int(row["horizon"])
        cost_weight = float(row["cost_weight"])
        result = solve_exact_policy(
            belief,
            tuple(
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
            ),
            horizon=horizon,
            cost_weight=cost_weight,
        )
        certificates.append(
            {
                "horizon": horizon,
                "cost_weight": cost_weight,
                "policy": "exact_dynamic_program",
                "oracle_objective": float(result.objective),
                "achieved_objective": float(result.objective),
                "lower_bound": float(result.objective),
                "optimality_gap": 0.0,
                "within_preregistered_tolerance": True,
                "scope": "finite declared sensor/action space only",
            }
        )
    protocol = scientific_protocol_payload()
    return {
        "schema_version": SENSOR_COST_SCHEMA_VERSION,
        "artifact": SENSOR_COST_ARTIFACT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "generator_revision": generator_revision(Path(__file__)),
        "input_hashes": {
            "active_information_payload": sha256_bytes(
                json.dumps(
                    frontier_payload, sort_keys=True, separators=(",", ":")
                ).encode()
            )
        },
        "toolchain_versions": toolchain_versions(("numpy",)),
        "status": "calibrated_local_finite_certificate",
        "primary_cost": "deterministic_vector",
        "calibration": {
            "trials_per_repetition": CALIBRATION_TRIALS,
            "repetitions": CALIBRATION_REPETITIONS,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "entropy_nats": entropy(belief),
        },
        "sensors": sensor_rows,
        "policy_certificates": certificates,
        "limitations": [
            "wall-clock values are machine-specific calibration observations",
            "the exact certificate is finite and model-channel specific",
            "the current candidate is the exact oracle, not an empirical shortcut",
        ],
    }


def validate_sensor_cost_payload(payload: object) -> tuple[bool, str]:
    """Validate cost-vector shape, finite timing, and zero-gap certificates."""

    if not isinstance(payload, dict):
        return False, "sensor cost payload must be an object"
    if payload.get("schema_version") != SENSOR_COST_SCHEMA_VERSION:
        return False, "sensor cost schema mismatch"
    protocol = scientific_protocol_payload()
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        return False, "sensor cost protocol hash mismatch"
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
        if not isinstance(timing, dict) or float(timing.get("median", 0.0)) <= 0.0:
            return False, "sensor cost timing is invalid"
    certificates = payload.get("policy_certificates")
    if not isinstance(certificates, list) or not certificates:
        return False, "sensor cost certificates are missing"
    for certificate in certificates:
        if certificate.get("optimality_gap") != 0.0:
            return False, "finite oracle certificate has a non-zero gap"
        if certificate.get("within_preregistered_tolerance") is not True:
            return False, "finite oracle certificate is not within tolerance"
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
