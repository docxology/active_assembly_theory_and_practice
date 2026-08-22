"""Tests for the source-family-disjoint P3 benchmark protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

import src.representation_invariance as representation_module
from src.representation_invariance import (
    DEVELOPMENT_SPLIT,
    HELD_OUT_SPLIT,
    NON_SEPARATING_INTERVENTION_INITIAL_RAX,
    _family_specs,
    _minimum_output_difference,
    _summary,
    _wilson_interval,
    representation_invariance_payload,
    validate_representation_invariance_payload,
    write_representation_invariance_benchmark,
)


def test_representation_benchmark_is_family_disjoint_and_has_endpoints() -> None:
    payload = representation_invariance_payload()
    valid, detail = validate_representation_invariance_payload(payload)

    assert valid, detail
    assert payload["sampling_unit"] == "source_program_family"
    assert payload["source_family_disjoint"] is True
    assert payload["family_count"] == 8
    assert payload["pair_count"] == 16
    assert set(payload["families_by_split"]) == {DEVELOPMENT_SPLIT, HELD_OUT_SPLIT}
    endpoint_names = {row["name"] for row in payload["primary_endpoints"]}
    assert {
        "equivalence_output_consistency",
        "near_miss_separation_rate",
        "invariance_gap",
        "separation_gap",
    } <= endpoint_names


def test_representation_benchmark_preserves_surface_blind_near_misses() -> None:
    payload = representation_invariance_payload()
    summary = payload["summary"]

    assert summary["equivalence_output_consistency"] == 1.0
    assert summary["near_miss_separation_rate"] == 1.0
    assert summary["surface_blind_near_miss_count"] >= 1
    assert summary["surface_blind_near_miss_rate"] > 0.0


def test_representation_benchmark_reports_family_level_descriptive_intervals() -> None:
    payload = representation_invariance_payload()
    uncertainty = payload["summary"]["uncertainty"]
    negative_uncertainty = payload["negative_control"]["summary"]["uncertainty"]

    assert uncertainty["unit"] == "source_program_family"
    assert uncertainty["descriptive_only"] is True
    assert uncertainty["source_family_count"] == payload["family_count"]
    assert uncertainty["confidence_level"] == 0.95
    for endpoint in ("equivalence_output_consistency", "near_miss_separation_rate"):
        record = uncertainty[endpoint]
        assert record["trials"] == payload["family_count"]
        assert record["interval"][0] <= record["successes"] / record["trials"]
        assert record["successes"] / record["trials"] <= record["interval"][1]
        assert negative_uncertainty[endpoint]["trials"] == payload["family_count"]

    lower, upper = _wilson_interval(0, 8)
    assert 0.0 <= lower < upper < 0.5


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (
            lambda payload: payload["families"][0].update(split=HELD_OUT_SPLIT),
            "family split does not match split manifest",
        ),
        (
            lambda payload: payload["pairs"][0].update(split=HELD_OUT_SPLIT),
            "pair split does not match family split",
        ),
        (
            lambda payload: payload.update(family_manifest_sha256="0" * 64),
            "family_manifest_sha256 mismatch",
        ),
        (
            lambda payload: payload["pairs"][0].update(reference_code_hex="b801000000"),
            "reference code drift",
        ),
        (
            lambda payload: payload["pairs"][0].update(expected_equivalent=False),
            "expected_equivalent-role mismatch",
        ),
        (
            lambda payload: payload["pairs"][0].update(relation="unrelated"),
            "relation-role mismatch",
        ),
        (
            lambda payload: payload["pairs"][0]["interventions"][0].update(
                final_rax_equal=False
            ),
            "final-RAX equality drift",
        ),
        (
            lambda payload: payload["summary"]["uncertainty"].update(
                descriptive_only=False
            ),
            "summary mismatch",
        ),
    ],
)
def test_representation_validator_rejects_provenance_and_uncertainty_drift(
    mutation, expected_detail
) -> None:
    payload = deepcopy(representation_invariance_payload())
    mutation(payload)

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert expected_detail in detail


def test_representation_validator_rejects_negative_control_split_drift() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["negative_control"]["pairs"][0]["split"] = HELD_OUT_SPLIT

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert "pair split does not match family split" in detail


def test_representation_benchmark_records_non_separating_negative_control() -> None:
    payload = representation_invariance_payload()
    negative_control = payload["negative_control"]
    summary = negative_control["summary"]

    assert negative_control["initial_state_interventions"] == list(
        NON_SEPARATING_INTERVENTION_INITIAL_RAX
    )
    assert summary["equivalence_output_consistency"] == 1.0
    assert summary["near_miss_separation_rate"] == 0.625
    assert any(
        not pair["near_miss_separated"]
        for pair in negative_control["pairs"]
        if pair["variant_role"] == "near_miss"
    )


def test_representation_benchmark_rejects_split_overlap() -> None:
    payload = representation_invariance_payload()
    split_manifest = dict(payload["families_by_split"])
    split_manifest[HELD_OUT_SPLIT] = [
        split_manifest[HELD_OUT_SPLIT][0],
        split_manifest[DEVELOPMENT_SPLIT][0],
        *split_manifest[HELD_OUT_SPLIT][1:],
    ]
    payload["families_by_split"] = split_manifest

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert "split overlap" in detail


def test_representation_benchmark_rejects_summary_drift() -> None:
    payload = representation_invariance_payload()
    summary = dict(payload["summary"])
    summary["near_miss_separation_rate"] = 0.0
    payload["summary"] = summary

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert "summary mismatch" in detail


def test_representation_benchmark_writes_under_requested_project_root(tmp_path) -> None:
    relative_path = write_representation_invariance_benchmark(tmp_path)

    assert (tmp_path / relative_path).is_file()


def test_representation_payload_rejects_empty_duplicate_and_unknown_families() -> None:
    with pytest.raises(ValueError, match="at least one transformation family"):
        representation_invariance_payload([])

    canonical_family = _family_specs()[0]
    with pytest.raises(ValueError, match="IDs must be unique"):
        representation_invariance_payload([canonical_family, canonical_family])

    invalid_split = replace(canonical_family, split="unregistered")
    with pytest.raises(ValueError, match="unsupported benchmark splits"):
        representation_invariance_payload([invalid_split])


def test_representation_internal_summary_guards_empty_inputs() -> None:
    with pytest.raises(ValueError, match="equivalent and near-miss"):
        _summary([])
    with pytest.raises(ValueError, match="at least one intervention"):
        _minimum_output_difference({"interventions": []})


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (lambda payload: payload.update(schema_version="0.0.0"), "schema="),
        (lambda payload: payload.update(families=None), "families must be"),
        (lambda payload: payload.update(pairs=None), "pairs must be"),
        (
            lambda payload: payload["families"].__setitem__(
                1, dict(payload["families"][0])
            ),
            "family IDs must be unique",
        ),
        (lambda payload: payload.update(families_by_split=None), "families_by_split"),
        (
            lambda payload: payload["families_by_split"][DEVELOPMENT_SPLIT].pop(),
            "does not cover",
        ),
        (lambda payload: payload.update(source_family_disjoint=False), "must be true"),
    ],
)
def test_representation_validator_rejects_manifest_contract_errors(
    mutation, expected_detail
) -> None:
    payload = deepcopy(representation_invariance_payload())
    mutation(payload)

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert expected_detail in detail


def test_representation_validator_rejects_pair_witness_and_identity_errors() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][0] = "not-a-pair"
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "non-object pair" in detail

    payload = deepcopy(representation_invariance_payload())
    first_pair = payload["pairs"][0]
    first_pair.update(
        {
            "family_id": "unknown-family",
            "variant_role": "unknown-role",
            "split": "unknown-split",
            "observed_equivalent": False,
            "interventions": [],
        }
    )
    second_pair = payload["pairs"][1]
    second_pair["near_miss_separated"] = False
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "unknown-family" in detail
    assert "unknown-role" in detail
    assert "unknown-split" in detail
    assert "intervention-row-count" in detail

    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][0]["observed_equivalent"] = False
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "equivalent-witness-failed" in detail

    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][1]["near_miss_separated"] = False
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "near-miss-not-separated" in detail


def test_representation_validator_rejects_pair_set_and_count_drift() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][1]["pair_id"] = payload["pairs"][0]["pair_id"]
    payload["pair_count"] += 1
    payload["family_count"] += 1
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "duplicate pair IDs" in detail
    assert "pair_count mismatch" in detail
    assert "family_count mismatch" in detail

    payload = deepcopy(representation_invariance_payload())
    payload["pairs"] = payload["pairs"][:-1]
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "one equivalent and one near-miss" in detail
    assert detail


def test_representation_validator_reports_summary_recomputation_failure() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][0].pop("interventions")

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert detail


def test_representation_validator_rejects_negative_control_drift() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["negative_control"]["summary"] = dict(
        payload["negative_control"]["summary"]
    )
    payload["negative_control"]["summary"]["near_miss_separation_rate"] = 1.0

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert "negative_control summary mismatch" in detail


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (
            lambda payload: payload.update(initial_state_interventions=[0]),
            "primary intervention catalog differs",
        ),
        (
            lambda payload: payload.update(intervention_catalogs=None),
            "must be an object",
        ),
        (
            lambda payload: payload["intervention_catalogs"].update(primary=None),
            "primary intervention catalog metadata is invalid",
        ),
        (
            lambda payload: payload["intervention_catalogs"]["primary"].update(
                values=[0]
            ),
            "metadata differs from canonical",
        ),
        (
            lambda payload: payload["intervention_catalogs"].update(
                non_separating_negative_control=None
            ),
            "negative-control intervention catalog metadata is invalid",
        ),
        (
            lambda payload: payload["intervention_catalogs"][
                "non_separating_negative_control"
            ].update(values=[1]),
            "negative-control intervention catalog metadata differs",
        ),
    ],
)
def test_representation_validator_rejects_intervention_metadata_drift(
    mutation, expected_detail
) -> None:
    payload = deepcopy(representation_invariance_payload())
    mutation(payload)

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert expected_detail in detail


def test_representation_validator_rejects_primary_pair_catalog_drift() -> None:
    payload = deepcopy(representation_invariance_payload())
    payload["pairs"][0]["initial_state_interventions"] = [0]

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert "intervention-catalog" in detail


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (lambda payload: payload.update(negative_control=None), "must be an object"),
        (
            lambda payload: payload["negative_control"].update(catalog_id="wrong"),
            "catalog ID mismatch",
        ),
        (
            lambda payload: payload["negative_control"].update(
                initial_state_interventions=[1]
            ),
            "intervention catalog mismatch",
        ),
        (
            lambda payload: payload["negative_control"].update(pairs=None),
            "pairs must be a non-empty list",
        ),
        (
            lambda payload: payload["negative_control"].update(pairs=["bad"]),
            "contains a non-object pair",
        ),
        (
            lambda payload: payload["negative_control"]["pairs"][0].update(
                initial_state_interventions=[1]
            ),
            "pair catalog mismatch",
        ),
    ],
)
def test_representation_validator_rejects_negative_control_structure_drift(
    mutation, expected_detail
) -> None:
    payload = deepcopy(representation_invariance_payload())
    mutation(payload)

    valid, detail = validate_representation_invariance_payload(payload)

    assert not valid
    assert expected_detail in detail


def test_representation_validator_rejects_negative_control_family_and_role_drift() -> (
    None
):
    payload = deepcopy(representation_invariance_payload())
    payload["negative_control"]["pairs"] = payload["negative_control"]["pairs"][:-2]
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "negative_control family coverage mismatch" in detail

    payload = deepcopy(representation_invariance_payload())
    payload["negative_control"]["pairs"][0]["variant_role"] = "other"
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "negative_control role coverage mismatch" in detail

    payload = deepcopy(representation_invariance_payload())
    next(
        pair
        for pair in payload["negative_control"]["pairs"]
        if pair["variant_role"] == "near_miss"
    ).pop("interventions")
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "negative_control summary recomputation failed" in detail

    payload = deepcopy(representation_invariance_payload())
    for pair in payload["negative_control"]["pairs"]:
        if pair["variant_role"] == "near_miss":
            pair["near_miss_separated"] = True
    valid, detail = validate_representation_invariance_payload(payload)
    assert not valid
    assert "negative_control did not retain an unresolved near miss" in detail


def test_representation_writer_refuses_invalid_canonical_payload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        representation_module,
        "validate_representation_invariance_payload",
        lambda payload: (False, "forced-invalid-test"),
    )

    with pytest.raises(ValueError, match="forced-invalid-test"):
        representation_module.write_representation_invariance_benchmark(tmp_path)
