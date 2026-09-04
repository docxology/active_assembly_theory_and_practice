"""Tests for the active-sensing stress study (roadmap MED-04)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.artifact_contract import validate_artifact_envelope
from src.information_policy import (
    FiniteSensor,
    INFORMATION_POLICY_COST_WEIGHTS,
    INFORMATION_POLICY_HORIZONS,
    solve_exact_policy,
)
from src.sensing_stress import (
    ACTION_BUDGETS,
    CALIBRATION_STABILITY_REPETITIONS,
    SENSING_STRESS_ARTIFACT,
    SENSING_STRESS_RANDOM_SEED,
    UNCERTAINTY_TIMING,
    UPSTREAM_ABSENT,
    UPSTREAM_FRONTIER_ARTIFACT,
    sensing_stress_payload,
    validate_sensing_stress_payload,
    write_sensing_stress,
)
from src.sensing_stress import _input_hashes
from src.sensor_costs import CALIBRATION_REPETITIONS, CALIBRATION_TRIALS


@pytest.fixture(scope="session")
def payload() -> dict[str, Any]:
    return sensing_stress_payload()


def _sensors_from_payload(data: dict[str, Any]) -> tuple[FiniteSensor, ...]:
    return tuple(
        FiniteSensor(
            sensor_id=str(sensor["sensor_id"]),
            outcomes=tuple(str(value) for value in sensor["outcomes"]),
            cost=1.0,
            likelihood=tuple(
                tuple(float(value) for value in row) for row in sensor["likelihood"]
            ),
        )
        for sensor in data["sensors"]
    )


def test_grid_exact_rows_match_solve_exact_policy_references(
    payload: dict[str, Any],
) -> None:
    belief = [float(value) for value in payload["initial_belief"]]
    sensors = _sensors_from_payload(payload)
    exact_rows = [row for row in payload["grid"] if row["policy"] == "exact_oracle"]
    assert len(exact_rows) == len(INFORMATION_POLICY_HORIZONS) * len(
        ACTION_BUDGETS
    ) * len(INFORMATION_POLICY_COST_WEIGHTS)
    for row in exact_rows:
        reference = solve_exact_policy(
            belief,
            sensors,
            horizon=int(row["horizon"]),
            cost_weight=float(row["cost_weight"]),
        )
        assert np.isclose(row["objective"], reference.objective, atol=1e-10)
        assert np.isclose(
            row["expected_information_gain"],
            reference.expected_information_gain,
            atol=1e-10,
        )
        assert row["first_sensor"] == reference.first_sensor
    for check in payload["exact_reference_cross_check"]:
        assert check["match"] is True
        assert np.isclose(
            check["reference_objective"],
            check["local_oracle_objective"],
            atol=check["atol"],
        )


def test_exact_policy_regret_is_zero_and_rollout_regret_is_nonnegative(
    payload: dict[str, Any],
) -> None:
    grid = payload["grid"]
    for row in grid:
        assert row["regret_vs_exact"] >= 0.0
        assert row["optimality_gap"] >= 0.0
        if row["policy"] == "exact_oracle":
            assert row["regret_vs_exact"] == 0.0
            assert row["optimality_gap"] == 0.0
    for policy in ("seeded_random", "abstain"):
        rows = [row for row in grid if row["policy"] == policy]
        assert rows, f"policy {policy} is absent from the grid"
        assert any(row["regret_vs_exact"] >= 0.0 for row in rows)


def test_rollout_budget_cap_is_respected(payload: dict[str, Any]) -> None:
    grid = payload["grid"]
    for row in grid:
        if row["policy"] == "exact_oracle":
            continue
        # Every sensor costs 1.0, so expected cost is the expected number of
        # sensing actions: it must never exceed the declared budget.
        assert row["expected_cost"] <= row["budget"] + 1e-9
    trapped = [
        row
        for row in grid
        if row["policy"] != "exact_oracle"
        and row["horizon"] == 3
        and row["budget"] == 1
    ]
    assert trapped
    assert all(row["expected_cost"] <= 1.0 + 1e-9 for row in trapped)


def test_calibration_stability_has_three_repetitions_and_finite_residuals(
    payload: dict[str, Any],
) -> None:
    stability = payload["calibration_stability"]
    assert stability["repetitions"] == CALIBRATION_STABILITY_REPETITIONS == 3
    assert stability["trials_per_repetition"] == CALIBRATION_TRIALS
    assert stability["samples_per_sensor_repetition"] == CALIBRATION_REPETITIONS
    assert len(stability["per_repetition"]) == CALIBRATION_STABILITY_REPETITIONS
    for repetition in stability["per_repetition"]:
        assert math.isfinite(repetition["residual_rms"])
        assert repetition["residual_rms"] >= 0.0
        assert math.isfinite(repetition["r_squared"])
        assert math.isfinite(repetition["timing_overhead_nanoseconds_per_query"])
    drift = stability["max_relative_coefficient_drift"]
    assert math.isfinite(drift)
    assert drift >= 0.0
    assert len(stability["coefficient_relative_drift"]) == len(
        stability["design_columns"]
    )
    assert stability["uncertainty"] == UNCERTAINTY_TIMING


def test_budget_value_curve_reports_unattained_gap(payload: dict[str, Any]) -> None:
    curve = payload["budget_value_curve"]
    assert {int(row["budget"]) for row in curve} == set(ACTION_BUDGETS)
    for row in curve:
        assert row["upper_bound_information_gain"] >= row["achieved_information_gain"]
        assert row["unattained_information_gain_gap"] >= 0.0
    # The largest budget still forgoes information relative to the
    # unconstrained four-sensor exact bound; the gap is never zero-filled.
    assert any(row["unattained_information_gain_gap"] > 0.0 for row in curve)


def test_upstream_frontier_absence_is_marked(payload: dict[str, Any]) -> None:
    """Present upstream frontier is hashed; absent frontier is explicitly marked."""

    assert payload["upstream"] == {}
    frontier_hash = payload["input_hashes"][UPSTREAM_FRONTIER_ARTIFACT]
    assert len(frontier_hash) == 64 and frontier_hash == frontier_hash.lower()
    absent_hashes, absent_upstream = _input_hashes(Path("/nonexistent-project-root"))
    assert absent_upstream == {UPSTREAM_FRONTIER_ARTIFACT: UPSTREAM_ABSENT}
    assert UPSTREAM_FRONTIER_ARTIFACT not in absent_hashes


def test_validator_accepts_full_payload(payload: dict[str, Any]) -> None:
    valid, detail = validate_sensing_stress_payload(payload)
    assert valid, detail
    assert "grid_rows=" in detail


def test_validator_rejects_wrong_schema_version(
    payload: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(payload)
    mutated["schema_version"] = "0.9.0"
    valid, detail = validate_sensing_stress_payload(mutated)
    assert not valid
    assert "schema version drift" in detail


def test_validator_rejects_negative_optimality_gap(
    payload: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(payload)
    rollout_row = next(
        row for row in mutated["grid"] if row["policy"] != "exact_oracle"
    )
    rollout_row["optimality_gap"] = -0.1
    valid, detail = validate_sensing_stress_payload(mutated)
    assert not valid
    assert "optimality gap" in detail


def test_validator_rejects_missing_uncertainty_field(
    payload: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(payload)
    del mutated["grid"][0]["uncertainty"]
    valid, detail = validate_sensing_stress_payload(mutated)
    assert not valid
    assert "uncertainty" in detail


def test_writer_round_trips_sealed_envelope(tmp_path) -> None:
    path = write_sensing_stress(tmp_path)
    assert path == tmp_path / SENSING_STRESS_ARTIFACT
    data = json.loads(path.read_text(encoding="utf-8"))
    valid, detail = validate_sensing_stress_payload(data)
    assert valid, detail
    envelope_valid, envelope_detail = validate_artifact_envelope(data)
    assert envelope_valid, envelope_detail
    assert data["receipt_schema_version"] == "2.0.0"
    assert data["artifact_id"] == SENSING_STRESS_ARTIFACT
    assert data["status"] == "validated_bounded"
    assert len(data["protocol_sha256"]) == 64
    assert data["input_hashes"]["src/sensing_stress.py"]


def test_seeded_random_grid_rows_are_seed_deterministic() -> None:
    first = sensing_stress_payload()
    second = sensing_stress_payload()
    assert first["random_seed"] == SENSING_STRESS_RANDOM_SEED
    seeded_first = [
        row
        for row in first["grid"]
        if row["policy"] == "seeded_random" and row["horizon"] == 2
    ]
    seeded_second = [
        row
        for row in second["grid"]
        if row["policy"] == "seeded_random" and row["horizon"] == 2
    ]
    assert seeded_first == seeded_second
