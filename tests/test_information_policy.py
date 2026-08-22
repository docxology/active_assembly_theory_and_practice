from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from src.corpus import build_corpus
from src.factorized_model import build_typed_observation_corpus
from src.information_policy import (
    FiniteSensor,
    active_information_payload,
    solve_exact_policy,
    validate_active_information_payload,
    write_active_information_payload,
)
import src.information_policy as information_module


def _binary_sensor() -> FiniteSensor:
    return FiniteSensor(
        sensor_id="exact",
        outcomes=("zero", "one"),
        cost=2.0,
        likelihood=((1.0, 0.0), (0.0, 1.0)),
    )


def test_exact_policy_dynamic_program_has_hand_checkable_information_gain() -> None:
    result = solve_exact_policy(
        [0.5, 0.5],
        [_binary_sensor()],
        horizon=1,
        cost_weight=0.0,
    )
    assert result.first_sensor == "exact"
    assert result.expected_information_gain == pytest.approx(np.log(2.0))
    assert result.expected_terminal_entropy == pytest.approx(0.0)
    assert result.expected_cost == pytest.approx(2.0)
    assert result.evaluated_policy_count > 0


def test_exact_policy_respects_cost_weight_and_zero_horizon() -> None:
    result = solve_exact_policy(
        [0.5, 0.5],
        [_binary_sensor()],
        horizon=0,
        cost_weight=1.0,
    )
    assert result.first_sensor is None
    assert result.expected_information_gain == pytest.approx(0.0)
    assert result.expected_cost == pytest.approx(0.0)

    with pytest.raises(ValueError):
        solve_exact_policy([0.5, 0.5], [_binary_sensor()], horizon=1, cost_weight=-1.0)


def test_active_information_payload_is_exact_and_source_bound() -> None:
    payload = active_information_payload()
    valid, detail = validate_active_information_payload(payload)
    assert valid, detail
    assert payload["exact_enumeration"] is True
    frontier = cast(list[dict[str, object]], payload["frontier"])
    assert len(frontier) == 16
    assert all(cast(int, row["evaluated_policy_count"]) > 0 for row in frontier)


def test_active_information_validator_rejects_channel_drift() -> None:
    payload = active_information_payload()
    sensors = cast(list[dict[str, object]], payload["sensors"])
    likelihood = cast(list[list[object]], sensors[0]["likelihood"])
    likelihood[0][0] = 2.0
    valid, detail = validate_active_information_payload(payload)
    assert not valid
    assert "normalize" in detail


def test_active_information_writer_round_trips(tmp_path) -> None:
    path = tmp_path / write_active_information_payload(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert validate_active_information_payload(payload)[0]


def test_information_policy_rejects_invalid_beliefs_fields_and_sensors() -> None:
    with pytest.raises(ValueError, match="non-empty vector"):
        information_module.entropy(np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        information_module.entropy(np.asarray([float("nan")]))
    with pytest.raises(ValueError, match="unique"):
        information_module.sensors_from_factorized_model(
            information_module.fit_factorized_observation_model(
                build_corpus()[:2],
                build_typed_observation_corpus(build_corpus()[:2]),
            ),
            fields=("instruction_family", "instruction_family"),
        )

    sensor = FiniteSensor(
        sensor_id="bad",
        outcomes=("a",),
        cost=0.0,
        likelihood=((1.0, 0.0),),
    )
    with pytest.raises(ValueError, match="outcomes and positive cost"):
        solve_exact_policy([1.0, 0.0], [sensor], horizon=1, cost_weight=0.0)


def test_information_policy_validator_rejects_missing_and_malformed_rows() -> None:
    payload = active_information_payload()
    payload["exact_enumeration"] = False
    assert "exact enumeration" in validate_active_information_payload(payload)[1]

    payload = active_information_payload()
    sensors = cast(list[dict[str, object]], payload["sensors"])
    sensors[0]["likelihood"] = []
    valid, detail = validate_active_information_payload(payload)
    assert not valid
    assert "likelihood shape" in detail

    payload = active_information_payload()
    frontier = cast(list[dict[str, object]], payload["frontier"])
    frontier[0].pop("objective")
    valid, detail = validate_active_information_payload(payload)
    assert not valid
    assert "frontier key missing" in detail
