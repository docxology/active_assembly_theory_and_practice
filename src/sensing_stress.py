"""Active-sensing stress study for the exact finite-horizon sensor DP (MED-04).

Objective: stress the exact finite-horizon sensor dynamic program
(:mod:`src.information_policy`) by comparing fixed, greedy, seeded-random, and
abstaining policies against the exact reference on a horizon x budget x
cost-weight grid, by quantifying the cross-repetition stability of the local
timing calibration regression, and by tracing how much information value a
bounded action budget forgoes relative to the unconstrained exact bound.

Primary endpoint: ``output/data/sensing_stress.json`` — a fail-closed,
provenance-sealed receipt produced by :func:`sensing_stress_payload`,
validated by :func:`validate_sensing_stress_payload`, and written by
:func:`write_sensing_stress`.

Grid semantics: for every ``(horizon, budget, cost_weight)`` cell the exact
reference is :func:`solve_exact_policy` at the cell horizon (the reference is
never budget constrained); the fixed, greedy, seeded-random, and abstaining
policies roll out over the same horizon but may take at most ``budget``
sensing actions, after which they abstain for the remaining horizon. Regret
is clamped at zero exactly as in :mod:`src.sensor_costs`, because a
budget-constrained rollout may strictly beat the forced-action exact program
when the cost weight dominates.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from . import information_policy as _information_policy
from . import sensor_costs as _sensor_costs
from .artifact_contract import write_json_artifact
from .corpus import build_corpus
from .factorized_model import (
    build_typed_observation_corpus,
    fit_factorized_observation_model,
)
from .information_policy import (
    FiniteSensor,
    INFORMATION_POLICY_COST_WEIGHTS,
    INFORMATION_POLICY_FIELDS,
    INFORMATION_POLICY_HORIZONS,
    entropy,
    sensors_from_factorized_model,
    solve_exact_policy,
)
from .provenance import sha256_path
from .receipt import EVIDENCE_STATUSES
from .scientific_protocol import scientific_protocol_payload
from .sensor_costs import (
    CALIBRATION_REPETITIONS,
    CALIBRATION_TRIALS,
    COST_COMPONENTS,
    POLICY_MODES,
    _deterministic_vector,
    _timed_overhead,
    _timed_sensor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSING_STRESS_SCHEMA_VERSION = "1.0.0"
SENSING_STRESS_ARTIFACT = "output/data/sensing_stress.json"
SENSING_STRESS_METHOD = "budget_constrained_active_sensing_stress_study"
ACTION_BUDGETS = (1, 2, 3)
SENSING_STRESS_RANDOM_SEED = 20260821
CALIBRATION_STABILITY_REPETITIONS = 3
ORACLE_CROSS_CHECK_ATOL = 1e-10
RUNTIME_BUDGET_SECONDS = 60.0
UPSTREAM_FRONTIER_ARTIFACT = "output/data/active_information_frontier.json"
UPSTREAM_ABSENT = "upstream_absent"

UNCERTAINTY_EXACT = "exact_outcome_enumeration"
UNCERTAINTY_ROLLOUT = "exact_outcome_enumeration_fixed_policy_seed"
UNCERTAINTY_TIMING = "machine_specific_local_timing_not_comparable_across_machines"

STOPPING_RULE = (
    "fixed predetermined enumeration: the complete horizon x budget x "
    "cost-weight x policy grid is enumerated to exhaustion with a memoized "
    "exact dynamic program, calibration stability runs exactly R=3 timing "
    "repetitions, and the value curve is evaluated at every declared budget; "
    "there is no adaptive early stopping, no timeout-driven truncation, and "
    "no convergence criterion; the producer is sized to finish the entire "
    "study in under 60 seconds of wall-clock on the reference machine"
)

_GRID_ROW_KEYS = (
    "policy",
    "horizon",
    "budget",
    "cost_weight",
    "expected_information_gain",
    "expected_cost",
    "objective",
    "regret_vs_exact",
    "optimality_gap",
    "first_sensor",
    "evaluated_policy_count",
    "uncertainty",
)

_REQUIRED_INPUT_HASHES = (
    "research_protocol.json",
    "src/sensing_stress.py",
    "src/information_policy.py",
    "src/sensor_costs.py",
)

_RolloutMemo = dict[tuple[tuple[float, ...], tuple[int, ...], int, int], dict[str, Any]]


@lru_cache(maxsize=1)
def _sensing_state() -> tuple[tuple[float, ...], tuple[FiniteSensor, ...]]:
    """Build the deterministic label-blind sensors once per process."""

    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    model = fit_factorized_observation_model(
        corpus,
        {fixture.name: typed[fixture.name] for fixture in corpus},
        fields=INFORMATION_POLICY_FIELDS,
    )
    sensors = sensors_from_factorized_model(model)
    belief = np.asarray(model.D, dtype=float)
    belief = belief / belief.sum()
    return tuple(float(value) for value in belief), sensors


def _posterior_for_outcome(
    belief: np.ndarray, matrix: np.ndarray, outcome_index: int
) -> tuple[float, np.ndarray]:
    probability = float(matrix[outcome_index] @ belief)
    if probability <= 0.0:
        return 0.0, belief / belief.sum()
    posterior = belief * matrix[outcome_index]
    return probability, posterior / posterior.sum()


def _rollout_value(
    belief: np.ndarray,
    sensors: tuple[FiniteSensor, ...],
    remaining: tuple[int, ...],
    horizon: int,
    budget: int,
    cost_weight: float,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
    memo: _RolloutMemo | None = None,
) -> dict[str, Any]:
    """Enumerate outcome branches for one policy family under an action budget.

    The exact-oracle mode recursively minimizes the objective over all finite
    sensor actions and is cross-checked against :func:`solve_exact_policy`.
    The other modes commit to an action without consulting labels or held-out
    outcomes. Expectations are exact conditional on the fixed random-policy
    seed because the state, action, and outcome spaces are finite.
    """

    normalized = np.asarray(belief, dtype=float)
    normalized = normalized / normalized.sum()
    key = (tuple(np.round(normalized, 12)), remaining, horizon, budget)
    if mode == "exact_oracle" and memo is not None and key in memo:
        return dict(memo[key])
    prior = entropy(normalized)
    terminal: dict[str, Any] = {
        "objective": float(prior),
        "terminal_entropy": float(prior),
        "expected_cost": 0.0,
        "expected_action_count": 0.0,
        "first_sensor": None,
        "evaluated_policy_count": 1,
    }
    if horizon <= 0 or budget <= 0 or not remaining or mode == "abstain":
        if mode == "exact_oracle" and memo is not None:
            memo[key] = dict(terminal)
        return terminal

    if mode == "exact_oracle":
        selected_indices = remaining
    elif mode == "fixed_first":
        selected_indices = (min(remaining),)
    elif mode == "greedy_information":
        information_scores: list[tuple[float, int]] = []
        for index in remaining:
            matrix = np.asarray(sensors[index].likelihood, dtype=float)
            expected_entropy = 0.0
            for outcome_index in range(matrix.shape[0]):
                probability, posterior = _posterior_for_outcome(
                    normalized, matrix, outcome_index
                )
                expected_entropy += probability * entropy(posterior)
            information_scores.append((prior - expected_entropy, index))
        selected_indices = (
            max(information_scores, key=lambda item: (item[0], -item[1]))[1],
        )
    elif mode == "seeded_random":
        if rng is None:
            raise ValueError("seeded_random policy requires a generator")
        selected_indices = (int(rng.choice(np.asarray(remaining, dtype=int))),)
    else:
        raise ValueError(f"unknown finite policy mode {mode!r}")

    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for sensor_index in selected_indices:
        sensor = sensors[sensor_index]
        matrix = np.asarray(sensor.likelihood, dtype=float)
        expected_objective = cost_weight * sensor.cost
        expected_entropy = 0.0
        expected_cost = sensor.cost
        expected_actions = 1.0
        policy_count = 1
        for outcome_index in range(matrix.shape[0]):
            probability, posterior = _posterior_for_outcome(
                normalized, matrix, outcome_index
            )
            if probability <= 0.0:
                continue
            child = _rollout_value(
                posterior,
                sensors,
                tuple(index for index in remaining if index != sensor_index),
                horizon - 1,
                budget - 1,
                cost_weight,
                mode,
                rng=rng,
                memo=memo,
            )
            expected_objective += probability * float(child["objective"])
            expected_entropy += probability * float(child["terminal_entropy"])
            expected_cost += probability * float(child["expected_cost"])
            expected_actions += probability * float(child["expected_action_count"])
            policy_count += int(child["evaluated_policy_count"])
        candidates.append(
            (
                float(expected_objective),
                sensor_index,
                {
                    "objective": float(expected_objective),
                    "terminal_entropy": float(expected_entropy),
                    "expected_cost": float(expected_cost),
                    "expected_action_count": float(expected_actions),
                    "first_sensor": sensor.sensor_id,
                    "evaluated_policy_count": policy_count,
                },
            )
        )

    chosen = min(candidates, key=lambda item: (item[0], item[1]))
    result = dict(chosen[2])
    if mode == "exact_oracle" and memo is not None:
        memo[key] = dict(result)
    return result


def _grid_and_cross_check(
    belief: np.ndarray, sensors: tuple[FiniteSensor, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the horizon x budget x cost-weight x policy grid."""

    prior = entropy(belief)
    rows: list[dict[str, Any]] = []
    cross_checks: list[dict[str, Any]] = []
    for horizon in INFORMATION_POLICY_HORIZONS:
        for cost_weight in INFORMATION_POLICY_COST_WEIGHTS:
            reference = solve_exact_policy(
                belief, sensors, horizon=horizon, cost_weight=cost_weight
            )
            local = _rollout_value(
                belief,
                sensors,
                tuple(range(len(sensors))),
                horizon,
                horizon,
                cost_weight,
                "exact_oracle",
                memo={},
            )
            if not bool(
                np.isclose(
                    float(local["objective"]),
                    float(reference.objective),
                    atol=ORACLE_CROSS_CHECK_ATOL,
                )
            ):
                raise RuntimeError(
                    "local finite oracle disagrees with solve_exact_policy"
                )
            cross_checks.append(
                {
                    "horizon": int(horizon),
                    "cost_weight": float(cost_weight),
                    "reference_objective": float(reference.objective),
                    "local_oracle_objective": float(local["objective"]),
                    "atol": ORACLE_CROSS_CHECK_ATOL,
                    "match": True,
                }
            )
            reference_objective = float(reference.objective)
            for budget in ACTION_BUDGETS:
                for mode in POLICY_MODES:
                    if mode == "exact_oracle":
                        row = {
                            "policy": mode,
                            "horizon": int(horizon),
                            "budget": int(budget),
                            "cost_weight": float(cost_weight),
                            "expected_information_gain": float(
                                reference.expected_information_gain
                            ),
                            "expected_cost": float(reference.expected_cost),
                            "objective": reference_objective,
                            "regret_vs_exact": 0.0,
                            "optimality_gap": 0.0,
                            "first_sensor": reference.first_sensor,
                            "evaluated_policy_count": int(
                                reference.evaluated_policy_count
                            ),
                            "uncertainty": UNCERTAINTY_EXACT,
                        }
                    else:
                        value = _rollout_value(
                            belief,
                            sensors,
                            tuple(range(len(sensors))),
                            horizon,
                            budget,
                            cost_weight,
                            mode,
                            rng=(
                                np.random.default_rng(SENSING_STRESS_RANDOM_SEED)
                                if mode == "seeded_random"
                                else None
                            ),
                        )
                        objective = float(value["objective"])
                        regret = max(0.0, objective - reference_objective)
                        row = {
                            "policy": mode,
                            "horizon": int(horizon),
                            "budget": int(budget),
                            "cost_weight": float(cost_weight),
                            "expected_information_gain": float(
                                prior - float(value["terminal_entropy"])
                            ),
                            "expected_cost": float(value["expected_cost"]),
                            "objective": objective,
                            "regret_vs_exact": float(regret),
                            "optimality_gap": float(
                                regret / max(abs(reference_objective), 1e-12)
                            ),
                            "first_sensor": value["first_sensor"],
                            "evaluated_policy_count": int(
                                value["evaluated_policy_count"]
                            ),
                            "uncertainty": UNCERTAINTY_ROLLOUT,
                        }
                    rows.append(row)
    return rows, cross_checks


def _calibration_stability(
    belief: np.ndarray, sensors: tuple[FiniteSensor, ...]
) -> dict[str, Any]:
    """Repeat the local timing regression and report coefficient drift.

    Every number in this block is a machine-specific wall-clock observation of
    the local harness; it is never comparable across machines.
    """

    per_repetition: list[dict[str, Any]] = []
    for repetition in range(CALIBRATION_STABILITY_REPETITIONS):
        overhead = _timed_overhead()
        medians: list[float] = []
        design_rows: list[list[float]] = []
        for sensor in sensors:
            sensor_dict = sensor.to_dict()
            samples = _timed_sensor(sensor_dict, belief)
            vector = _deterministic_vector(sensor_dict)
            medians.append(float(statistics.median(samples)))
            design_rows.append(
                [1.0] + [float(vector[component]) for component in COST_COMPONENTS]
            )
        design = np.asarray(design_rows, dtype=float)
        timing = np.asarray(medians, dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(design, timing, rcond=None)
        fitted = design @ coefficients
        residuals = timing - fitted
        total_sum_squares = float(np.sum((timing - timing.mean()) ** 2))
        per_repetition.append(
            {
                "repetition": int(repetition),
                "timing_overhead_nanoseconds_per_query": float(overhead),
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
        )
    coefficient_matrix = np.asarray(
        [row["coefficients"] for row in per_repetition], dtype=float
    )
    coefficient_medians = np.median(coefficient_matrix, axis=0)
    coefficient_ranges = coefficient_matrix.max(axis=0) - coefficient_matrix.min(axis=0)
    relative_drift = coefficient_ranges / np.maximum(np.abs(coefficient_medians), 1e-9)
    return {
        "repetitions": CALIBRATION_STABILITY_REPETITIONS,
        "trials_per_repetition": CALIBRATION_TRIALS,
        "samples_per_sensor_repetition": CALIBRATION_REPETITIONS,
        "design_columns": ["intercept", *COST_COMPONENTS],
        "per_repetition": per_repetition,
        "coefficient_relative_drift": [float(value) for value in relative_drift],
        "max_relative_coefficient_drift": float(np.max(relative_drift)),
        "uncertainty": UNCERTAINTY_TIMING,
    }


def _budget_value_curve(
    belief: np.ndarray, sensors: tuple[FiniteSensor, ...]
) -> list[dict[str, Any]]:
    """Trace the budget-constrained information value against the exact bound."""

    bound = solve_exact_policy(belief, sensors, horizon=len(sensors), cost_weight=0.0)
    rows: list[dict[str, Any]] = []
    for budget in ACTION_BUDGETS:
        achieved = solve_exact_policy(belief, sensors, horizon=budget, cost_weight=0.0)
        rows.append(
            {
                "budget": int(budget),
                "cost_weight": 0.0,
                "achieved_information_gain": float(achieved.expected_information_gain),
                "upper_bound_information_gain": float(bound.expected_information_gain),
                "unattained_information_gain_gap": float(
                    bound.expected_information_gain - achieved.expected_information_gain
                ),
                "upper_bound_horizon": len(sensors),
                "uncertainty": UNCERTAINTY_EXACT,
            }
        )
    return rows


def _input_hashes(project_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Hash the protocol, producers, and the upstream frontier when present."""

    protocol = scientific_protocol_payload()
    hashes = {
        "research_protocol.json": str(protocol["protocol_sha256"]),
        "src/sensing_stress.py": sha256_path(Path(__file__)),
        "src/information_policy.py": sha256_path(Path(_information_policy.__file__)),
        "src/sensor_costs.py": sha256_path(Path(_sensor_costs.__file__)),
    }
    upstream: dict[str, str] = {}
    frontier_path = project_root / UPSTREAM_FRONTIER_ARTIFACT
    if frontier_path.is_file():
        hashes[UPSTREAM_FRONTIER_ARTIFACT] = sha256_path(frontier_path)
    else:
        upstream[UPSTREAM_FRONTIER_ARTIFACT] = UPSTREAM_ABSENT
    return hashes, upstream


def sensing_stress_payload(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate the full active-sensing stress study payload."""

    belief_values, sensors = _sensing_state()
    belief = np.asarray(belief_values, dtype=float)
    grid, cross_checks = _grid_and_cross_check(belief, sensors)
    stability = _calibration_stability(belief, sensors)
    curve = _budget_value_curve(belief, sensors)
    input_hashes, upstream = _input_hashes(project_root)
    limitations = [
        "the exact reference and grid rollouts are bounded to the declared "
        "label-blind sensor channels and finite horizons",
        "calibration-stability timings are machine-specific local wall-clock "
        "observations and are never comparable across machines",
        "the seeded-random policy expectations are exact only conditional on "
        "the fixed policy seed",
    ]
    unsupported_cases = [
        "No claim is made about continuous action spaces, adaptive sensing "
        "machines, or sensor families outside the declared model.",
    ]
    underpowered_cases = [
        "The cross-repetition coefficient drift does not establish a "
        "cross-machine timing law.",
    ]
    return {
        "schema_version": SENSING_STRESS_SCHEMA_VERSION,
        "artifact": SENSING_STRESS_ARTIFACT,
        "method": SENSING_STRESS_METHOD,
        "fields": list(INFORMATION_POLICY_FIELDS),
        "initial_belief": [float(value) for value in belief],
        "initial_entropy": entropy(belief),
        "sensors": [sensor.to_dict() for sensor in sensors],
        "grid_dimensions": {
            "horizons": list(INFORMATION_POLICY_HORIZONS),
            "budgets": list(ACTION_BUDGETS),
            "cost_weights": list(INFORMATION_POLICY_COST_WEIGHTS),
            "policies": list(POLICY_MODES),
            "rows": len(grid),
        },
        "stopping_rule": STOPPING_RULE,
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "random_seed": SENSING_STRESS_RANDOM_SEED,
        "grid": grid,
        "exact_reference_cross_check": cross_checks,
        "calibration_stability": stability,
        "budget_value_curve": curve,
        "upstream": upstream,
        "input_hashes": input_hashes,
        "status": "validated_bounded",
        "limitations": limitations,
        "unsupported_cases": unsupported_cases,
        "underpowered_cases": underpowered_cases,
    }


def _finite(value: object) -> bool:
    try:
        return bool(math.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def validate_sensing_stress_payload(payload: object) -> tuple[bool, str]:
    """Fail-closed validation of the active-sensing stress study receipt."""

    if not isinstance(payload, dict):
        return False, "sensing stress payload must be an object"
    if payload.get("schema_version") != SENSING_STRESS_SCHEMA_VERSION:
        return False, "sensing stress schema version drift"
    if payload.get("method") != SENSING_STRESS_METHOD:
        return False, "sensing stress method mismatch"
    status = payload.get("status")
    if status not in EVIDENCE_STATUSES or status != "validated_bounded":
        return False, f"unexpected evidence status={status!r}"
    if payload.get("fields") != list(INFORMATION_POLICY_FIELDS):
        return False, "sensing stress field list drift"
    stopping_rule = payload.get("stopping_rule")
    if not isinstance(stopping_rule, str) or not stopping_rule.strip():
        return False, "stopping rule is missing"
    if float(payload.get("runtime_budget_seconds", 0.0)) <= 0.0:
        return False, "runtime budget must be positive"

    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        return False, "input hashes are missing"
    for key in _REQUIRED_INPUT_HASHES:
        value = input_hashes.get(key)
        if not isinstance(value, str) or len(value) != 64:
            return False, f"input hash missing or malformed: {key}"
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict):
        return False, "upstream marker block is missing"
    if (
        UPSTREAM_FRONTIER_ARTIFACT not in input_hashes
        and upstream.get(UPSTREAM_FRONTIER_ARTIFACT) != UPSTREAM_ABSENT
    ):
        return False, "absent upstream frontier is not explicitly marked"

    belief = payload.get("initial_belief")
    if (
        not isinstance(belief, list)
        or not belief
        or not all(_finite(value) for value in belief)
    ):
        return False, "initial belief is missing or malformed"
    if (
        not _finite(payload.get("initial_entropy"))
        or float(payload["initial_entropy"]) < 0.0
    ):
        return False, "initial entropy is missing or negative"
    sensors = payload.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        return False, "sensors are missing"
    for raw_sensor in sensors:
        if not isinstance(raw_sensor, dict) or not raw_sensor.get("sensor_id"):
            return False, "sensor row is malformed"

    expected_cells = {
        (horizon, budget, cost_weight, mode)
        for horizon in INFORMATION_POLICY_HORIZONS
        for budget in ACTION_BUDGETS
        for cost_weight in INFORMATION_POLICY_COST_WEIGHTS
        for mode in POLICY_MODES
    }
    grid = payload.get("grid")
    if not isinstance(grid, list) or not grid:
        return False, "grid is missing"
    seen_cells: set[tuple[int, int, float, str]] = set()
    for row in grid:
        if not isinstance(row, dict):
            return False, "grid row is not an object"
        for key in _GRID_ROW_KEYS:
            if key not in row:
                return False, f"grid key missing: {key}"
        policy = row["policy"]
        if policy not in POLICY_MODES:
            return False, f"unknown grid policy {policy!r}"
        try:
            horizon = int(row["horizon"])
            budget = int(row["budget"])
            cost_weight = float(row["cost_weight"])
        except (TypeError, ValueError):
            return False, "grid row coordinates are malformed"
        if horizon not in INFORMATION_POLICY_HORIZONS or budget not in ACTION_BUDGETS:
            return False, "grid row horizon or budget is out of range"
        if not any(
            abs(cost_weight - declared) < 1e-12
            for declared in INFORMATION_POLICY_COST_WEIGHTS
        ):
            return False, "grid row cost weight is out of range"
        cell = (horizon, budget, cost_weight, str(policy))
        if cell in seen_cells:
            return False, "duplicate grid cell"
        seen_cells.add(cell)
        for key in (
            "expected_information_gain",
            "expected_cost",
            "objective",
            "regret_vs_exact",
            "optimality_gap",
        ):
            if not _finite(row[key]):
                return False, f"grid value is not finite: {key}"
        if float(row["expected_cost"]) < 0.0:
            return False, "expected cost cannot be negative"
        if float(row["regret_vs_exact"]) < 0.0:
            return False, "regret cannot be negative"
        if float(row["optimality_gap"]) < 0.0:
            return False, "optimality gap cannot be negative"
        if policy == "exact_oracle":
            if (
                float(row["regret_vs_exact"]) != 0.0
                or float(row["optimality_gap"]) != 0.0
            ):
                return False, "exact reference regret must be exactly zero"
        else:
            if float(row["expected_cost"]) > budget + 1e-9:
                return False, "budget cap violated by rollout policy"
        uncertainty = row["uncertainty"]
        if not isinstance(uncertainty, str) or not uncertainty.strip():
            return False, "grid row uncertainty label is missing"
        expected_uncertainty = (
            UNCERTAINTY_EXACT if policy == "exact_oracle" else UNCERTAINTY_ROLLOUT
        )
        if uncertainty != expected_uncertainty:
            return False, "grid row uncertainty does not match the policy class"
    if seen_cells != expected_cells:
        return False, "grid does not cover the declared cell product exactly"

    cross_checks = payload.get("exact_reference_cross_check")
    if not isinstance(cross_checks, list) or not cross_checks:
        return False, "exact reference cross-check is missing"
    for check in cross_checks:
        if not isinstance(check, dict) or check.get("match") is not True:
            return False, "exact reference cross-check did not match"
        if float(check.get("atol", 0.0)) != ORACLE_CROSS_CHECK_ATOL:
            return False, "cross-check tolerance drifted"
        if not _finite(check.get("reference_objective")) or not _finite(
            check.get("local_oracle_objective")
        ):
            return False, "cross-check objectives are not finite"
        if not bool(
            np.isclose(
                float(check["reference_objective"]),
                float(check["local_oracle_objective"]),
                atol=ORACLE_CROSS_CHECK_ATOL,
            )
        ):
            return False, "recorded cross-check objectives disagree"

    stability = payload.get("calibration_stability")
    if not isinstance(stability, dict):
        return False, "calibration stability block is missing"
    if stability.get("repetitions") != CALIBRATION_STABILITY_REPETITIONS:
        return False, "calibration stability repetition count drifted"
    per_repetition = stability.get("per_repetition")
    if not isinstance(per_repetition, list) or len(per_repetition) != (
        CALIBRATION_STABILITY_REPETITIONS
    ):
        return False, "calibration stability repetitions are incomplete"
    for repetition in per_repetition:
        if not isinstance(repetition, dict):
            return False, "calibration repetition row is malformed"
        for key in ("residual_rms", "r_squared"):
            if not _finite(repetition.get(key)):
                return False, f"calibration {key} is not finite"
        if float(repetition["residual_rms"]) < 0.0:
            return False, "residual rms cannot be negative"
    drift = stability.get("max_relative_coefficient_drift")
    if drift is None or not _finite(drift) or float(drift) < 0.0:
        return False, "coefficient drift is not finite and non-negative"
    if stability.get("uncertainty") != UNCERTAINTY_TIMING:
        return False, "calibration timing is not labeled machine-specific"

    curve = payload.get("budget_value_curve")
    if not isinstance(curve, list) or not curve:
        return False, "budget value curve is missing"
    curve_budgets = set()
    for row in curve:
        if not isinstance(row, dict):
            return False, "value curve row is not an object"
        curve_budgets.add(int(row.get("budget", -1)))
        if not _finite(row.get("achieved_information_gain")) or not _finite(
            row.get("upper_bound_information_gain")
        ):
            return False, "value curve gains are not finite"
        if (
            float(row["upper_bound_information_gain"])
            < float(row["achieved_information_gain"]) - 1e-12
        ):
            return False, "value curve upper bound is below the achieved gain"
        gap = row.get("unattained_information_gain_gap")
        if gap is None or not _finite(gap) or float(gap) < -1e-12:
            return False, "unattained value gap is not finite and non-negative"
        if row.get("uncertainty") != UNCERTAINTY_EXACT:
            return False, "value curve uncertainty must declare exact enumeration"
    if curve_budgets != set(ACTION_BUDGETS):
        return False, "value curve does not cover every declared budget"

    dimensions = payload.get("grid_dimensions")
    if not isinstance(dimensions, dict) or dimensions.get("rows") != len(grid):
        return False, "grid dimensions do not match the recorded grid"
    return True, f"grid_rows={len(grid)} repetitions={len(per_repetition)}"


def write_sensing_stress(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the sealed active-sensing stress study artifact."""

    payload = sensing_stress_payload(project_root)
    valid, detail = validate_sensing_stress_payload(payload)
    if not valid:
        raise ValueError(f"sensing-stress payload is invalid: {detail}")
    path = project_root / SENSING_STRESS_ARTIFACT
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        status="validated_bounded",
    )
    return path


__all__ = [
    "ACTION_BUDGETS",
    "CALIBRATION_STABILITY_REPETITIONS",
    "ORACLE_CROSS_CHECK_ATOL",
    "POLICY_MODES",
    "PROJECT_ROOT",
    "RUNTIME_BUDGET_SECONDS",
    "SENSING_STRESS_ARTIFACT",
    "SENSING_STRESS_METHOD",
    "SENSING_STRESS_RANDOM_SEED",
    "SENSING_STRESS_SCHEMA_VERSION",
    "STOPPING_RULE",
    "sensing_stress_payload",
    "validate_sensing_stress_payload",
    "write_sensing_stress",
]
