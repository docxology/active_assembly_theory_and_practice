"""Tests for the bounded action-conditioned typed-field sensing experiment."""

from __future__ import annotations

import json

from src.action_sensing import (
    ACTION_POLICY_NAMES,
    ACTION_SENSING_SCHEMA_VERSION,
    action_conditioned_sensing_payload,
    validate_action_conditioned_sensing_payload,
)
from src.corpus import build_corpus


def test_action_sensing_is_deterministic_and_leakage_safe() -> None:
    first = action_conditioned_sensing_payload(build_corpus())
    second = action_conditioned_sensing_payload(build_corpus())

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["schema_version"] == ACTION_SENSING_SCHEMA_VERSION
    assert first["summary"]["leakage_free"] is True
    assert first["summary"]["action_pool_changes_evidence"] is False
    assert set(first["policies"]) == set(ACTION_POLICY_NAMES)


def test_action_sensing_uses_common_budget_and_explicit_oracle() -> None:
    payload = action_conditioned_sensing_payload(build_corpus())
    for row in payload["per_fixture"]:
        assert row["held_out_fixture"] not in row["training_fixtures"]
        assert row["action_candidates"]
        candidate_ids = {
            candidate["action_id"] for candidate in row["action_candidates"]
        }
        for policy in ACTION_POLICY_NAMES:
            result = row["policies"][policy]
            assert result["policy"] == policy
            assert result["actions_used"] <= payload["max_steps"]
            assert set(result["selected_action_ids"]) <= candidate_ids
            assert (
                result["actions_to_confidence"] is not None
                or not result["reached_confidence"]
            )
    assert payload["selection_contract"]["oracle"].startswith("inadmissible")


def test_action_sensing_validation_rejects_policy_leakage() -> None:
    payload = action_conditioned_sensing_payload(build_corpus())
    valid, detail = validate_action_conditioned_sensing_payload(payload)
    assert valid, detail

    broken = json.loads(json.dumps(payload))
    broken["per_fixture"][0]["training_fixtures"].append(
        broken["per_fixture"][0]["held_out_fixture"]
    )
    valid, detail = validate_action_conditioned_sensing_payload(broken)
    assert not valid
    assert "leakage" in detail


def test_action_sensing_validation_requires_all_policies() -> None:
    payload = action_conditioned_sensing_payload(build_corpus())
    broken = json.loads(json.dumps(payload))
    del broken["per_fixture"][0]["policies"]["oracle"]
    valid, detail = validate_action_conditioned_sensing_payload(broken)
    assert not valid
    assert "policy" in detail
