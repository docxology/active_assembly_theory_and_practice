"""Tests for the exact finite intervention-power diagnostic."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.intervention_analysis import (
    _outcome_matrix,
    intervention_power_payload,
    validate_intervention_power_payload,
    write_intervention_power_payload,
)


def _custom_benchmark(
    *,
    catalog: list[int] | None = None,
    equivalent: bool = True,
    near_miss_equalities: tuple[bool, ...] = (True, True),
) -> dict[str, object]:
    """Build a tiny label-boundary benchmark for branch-level contracts."""

    values = catalog or [0, 1]
    pairs = [
        {
            "family_id": "equivalent",
            "expected_equivalent": equivalent,
            "interventions": [
                {"initial_rax": value, "final_rax_equal": True} for value in values
            ],
        },
        {
            "family_id": "near_miss",
            "expected_equivalent": not equivalent,
            "interventions": [
                {"initial_rax": value, "final_rax_equal": equality}
                for value, equality in zip(values, near_miss_equalities)
            ],
        },
    ]
    return {"initial_state_interventions": values, "pairs": pairs}


def test_intervention_power_enumerates_the_frozen_catalog() -> None:
    payload = intervention_power_payload()

    valid, detail = validate_intervention_power_payload(payload)

    assert valid, detail
    assert payload["subset_count"] == 15
    assert len(payload["subset_evaluations"]) == 15
    assert payload["minimal_separating_cardinality"] == 1
    assert len(payload["minimal_separating_sets"]) == 3
    negative = payload["negative_control"]
    assert negative["values"] == [0]
    assert negative["near_miss_separation"] == 0.625
    assert len(negative["unresolved_near_miss_families"]) == 3
    assert payload["promotion_status"] == "fail_closed_no_identifiability_promotion"


def test_intervention_power_rejects_duplicate_subsets_and_promotion_drift() -> None:
    payload = intervention_power_payload()

    duplicate = deepcopy(payload)
    duplicate["subset_evaluations"][1] = deepcopy(duplicate["subset_evaluations"][0])
    valid, detail = validate_intervention_power_payload(duplicate)
    assert not valid
    assert "duplicates" in detail

    promoted = deepcopy(payload)
    promoted["promotion_status"] = "semantic_identifiability"
    valid, detail = validate_intervention_power_payload(promoted)
    assert not valid
    assert "promote identifiability" in detail


def test_intervention_power_writer_is_source_bound_and_deterministic(tmp_path) -> None:
    first_path = write_intervention_power_payload(tmp_path)
    first = first_path.read_text(encoding="utf-8")
    second_path = write_intervention_power_payload(tmp_path)
    second = second_path.read_text(encoding="utf-8")

    assert first_path == second_path
    assert first == second
    assert json.loads(first)["subset_count"] == 15


def test_intervention_power_reports_underpowered_custom_matrix() -> None:
    payload = intervention_power_payload(
        _custom_benchmark(near_miss_equalities=(True, True))
    )

    assert payload["status"] == "underpowered"
    assert payload["minimal_separating_cardinality"] is None
    assert payload["minimal_separating_sets"] == []
    assert payload["per_near_miss_minimal_interventions"] == [
        {
            "family_id": "near_miss",
            "separating_interventions": [],
            "minimum_cardinality": None,
        }
    ]
    assert validate_intervention_power_payload(payload)[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda benchmark: benchmark.pop("initial_state_interventions"), "catalog"),
        (
            lambda benchmark: benchmark.update(initial_state_interventions=[]),
            "catalog",
        ),
        (
            lambda benchmark: benchmark.update(initial_state_interventions=[0, 0]),
            "duplicate",
        ),
        (lambda benchmark: benchmark.pop("pairs"), "pair rows"),
        (lambda benchmark: benchmark.update(pairs=[]), "pair rows"),
        (lambda benchmark: benchmark.update(pairs=["not an object"]), "not an object"),
        (
            lambda benchmark: benchmark["pairs"][0].update(family_id=""),
            "family identifier",
        ),
        (
            lambda benchmark: benchmark["pairs"][0].update(expected_equivalent="yes"),
            "boolean",
        ),
        (
            lambda benchmark: benchmark["pairs"][0].update(interventions=[]),
            "incomplete",
        ),
        (
            lambda benchmark: benchmark["pairs"][0].update(
                interventions=["bad", "bad"]
            ),
            "malformed",
        ),
        (
            lambda benchmark: benchmark["pairs"][0]["interventions"][0].update(
                initial_rax=True
            ),
            "non-integer",
        ),
        (
            lambda benchmark: benchmark["pairs"][0]["interventions"][0].update(
                initial_rax=99
            ),
            "undeclared",
        ),
        (
            lambda benchmark: benchmark["pairs"][0]["interventions"][1].update(
                initial_rax=0
            ),
            "repeats",
        ),
        (
            lambda benchmark: benchmark["pairs"][0]["interventions"][0].update(
                final_rax_equal=None
            ),
            "boolean",
        ),
    ],
)
def test_intervention_power_rejects_malformed_source_matrix(mutation, message) -> None:
    benchmark = _custom_benchmark()
    mutation(benchmark)

    with pytest.raises(ValueError, match=message):
        _outcome_matrix(benchmark)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="0.0.0"), "schema"),
        (lambda payload: payload.update(protocol_sha256="bad"), "protocol"),
        (lambda payload: payload.update(intervention_catalog=[True]), "catalog"),
        (lambda payload: payload.update(intervention_catalog=[]), "catalog"),
        (lambda payload: payload.update(intervention_catalog=[0, 0]), "catalog"),
        (
            lambda payload: payload.update(
                subset_evaluations=payload["subset_evaluations"][:-1]
            ),
            "incomplete",
        ),
        (
            lambda payload: payload["subset_evaluations"].__setitem__(0, "bad"),
            "subset row",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(values=[]),
            "subset values",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(values=[999]),
            "subset values",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(values=[0, 0]),
            "subset values",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(size=99),
            "subset values",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(
                equivalent_consistency=2
            ),
            "metric",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(
                near_miss_separation=True
            ),
            "metric",
        ),
        (
            lambda payload: payload["subset_evaluations"][0].update(
                near_miss_unresolved_families=[1]
            ),
            "unresolved",
        ),
        (lambda payload: payload.update(outcome_matrix=[]), "matrix"),
        (lambda payload: payload.update(pair_count=0), "pair count"),
        (
            lambda payload: payload["outcome_matrix"].__setitem__(0, "bad"),
            "outcome row",
        ),
        (
            lambda payload: payload["outcome_matrix"][0].update(outcomes=None),
            "outcome row",
        ),
        (
            lambda payload: payload["outcome_matrix"][0].update(outcomes={"0": True}),
            "outcome row",
        ),
        (
            lambda payload: payload["outcome_matrix"][0].update(
                outcomes={"0": True, "1": 1}
            ),
            "outcome row",
        ),
        (lambda payload: payload.pop("negative_control"), "negative control"),
        (
            lambda payload: payload["negative_control"].update(values=[1]),
            "first declared",
        ),
        (
            lambda payload: payload["negative_control"].update(
                near_miss_separation=True
            ),
            "separation",
        ),
        (
            lambda payload: payload["negative_control"].update(
                unresolved_near_miss_families=[1]
            ),
            "unresolved",
        ),
    ],
)
def test_intervention_power_validator_fails_closed(mutation, message) -> None:
    payload = intervention_power_payload()
    mutation(payload)

    valid, detail = validate_intervention_power_payload(payload)

    assert not valid
    assert message in detail
