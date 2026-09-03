"""Additional negative-path coverage for the frozen replication interface."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

import src.replication as replication_module

from src.replication import (
    INDEPENDENT_REPLAY_ARTIFACT,
    GOLD_LABELS,
    REPLICATION_ARTIFACT,
    _public_bundle_digest,
    ingest_independent_replay,
    _replay_timestamp,
    _timestamps_ordered,
    replay_receipt_payload,
    run_label_blind_model,
    score_replication_predictions,
    validate_public_replay_bundle,
    validate_independent_replay_artifact,
    validate_replay_receipt,
    write_replay_receipt,
    write_replication_bundle,
)


def _public_path(tmp_path: Path) -> Path:
    path = tmp_path / "public.json"
    path.write_text(
        json.dumps(
            [
                {"record_id": "a", "sequence": ["x"], "features": [0.0, 1.0]},
                {"record_id": "b", "sequence": ["y", "z"], "features": [1.0, 2.0]},
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "model", ["calibrated_interpretable", "temporal_probabilistic", "neural_sequence"]
)
def test_all_label_blind_reference_models_use_only_public_rows(
    tmp_path: Path, model: str
) -> None:
    output = run_label_blind_model(
        _public_path(tmp_path), tmp_path / f"{model}.json", model=model
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["label_blind"] is True
    assert len(payload["predictions"]) == 2
    assert all(0.0 <= row["probability"] <= 1.0 for row in payload["predictions"])


def test_public_toolchain_receipt_is_frozen_across_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host drift cannot silently change the replay input hash."""

    write_replication_bundle(tmp_path)
    receipt_path = tmp_path / "output/replication/public/toolchain_receipt.json"
    original = receipt_path.read_bytes()

    monkeypatch.setattr(replication_module.platform, "platform", lambda: "drifted")
    monkeypatch.setattr(replication_module.platform, "python_version", lambda: "99.0.0")
    monkeypatch.setattr(replication_module.platform, "machine", lambda: "drifted")
    write_replication_bundle(tmp_path)

    assert receipt_path.read_bytes() == original


def test_label_blind_model_rejects_unreadable_empty_and_leaking_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        run_label_blind_model(
            _public_path(tmp_path), tmp_path / "out.json", model="bad"
        )
    with pytest.raises(ValueError, match="unreadable"):
        run_label_blind_model(tmp_path / "missing.json", tmp_path / "out.json")
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        run_label_blind_model(empty, tmp_path / "out.json")
    leaking = tmp_path / "leaking.json"
    leaking.write_text(
        json.dumps([{"record_id": "a", "sequence": [], "features": [], "label": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label blind"):
        run_label_blind_model(leaking, tmp_path / "out.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps([{"record_id": 1, "sequence": [], "features": []}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid fields"):
        run_label_blind_model(malformed, tmp_path / "out.json")


def test_replication_scoring_rejects_prediction_contract_drift(tmp_path: Path) -> None:
    write_replication_bundle(tmp_path)
    gold_path = tmp_path / "output/replication/private/gold_labels.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    valid_rows = [
        {"record_id": row["record_id"], "probability": float(row["label"])}
        for row in gold
    ]

    def score(candidate: object, name: str) -> None:
        path = tmp_path / name
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ValueError):
            score_replication_predictions(path, tmp_path)

    score({"wrong": []}, "missing_predictions.json")
    score([{"record_id": "x", "probability": 0.5, "extra": 1}], "extra.json")
    score(
        [{"record_id": valid_rows[0]["record_id"], "probability": 0.5}] * 2,
        "duplicate.json",
    )
    score(
        [{"record_id": valid_rows[0]["record_id"], "probability": 2.0}],
        "out_of_range.json",
    )

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(valid_rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="identifiers"):
        score_replication_predictions(incomplete, tmp_path)

    wrong_gold = tmp_path / "wrong_gold.json"
    wrong_gold.write_text(
        json.dumps([{**gold[0], "label": 1 - gold[0]["label"]}]), encoding="utf-8"
    )
    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps(valid_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed manifest hash"):
        score_replication_predictions(complete, tmp_path, gold_path=wrong_gold)


def test_replay_receipt_is_pending_and_binds_public_hashes(tmp_path: Path) -> None:
    write_replication_bundle(tmp_path)
    public = tmp_path / "output/replication/public/public_manifest.json"
    predictions = tmp_path / "predictions.json"
    run_label_blind_model(public, predictions)
    receipt = replay_receipt_payload(
        public,
        predictions,
        operator_id="operator-test",
        machine_id="machine-test",
        replay_bundle_sha256=_public_bundle_digest(tmp_path),
        model="calibrated_interpretable",
        seed=17,
        command=["replay", "--offline"],
        network_disabled=True,
        project_root=tmp_path,
    )
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=predictions
    )
    assert valid, detail
    assert receipt["endpoint_status"] == "submitted_pending_verification"
    assert receipt["independent_replay"] == "not_yet_observed"
    assert receipt["attestation"]["separate_controlled_run"] is False

    receipt["public_bundle_sha256"] = "0" * 64
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=predictions
    )
    assert not valid
    assert "bundle hash" in detail


def test_replay_receipt_can_bind_a_committed_runtime_source_archive(
    tmp_path: Path,
) -> None:
    receipt, predictions, _ = _pending_receipt_fixture(tmp_path)
    receipt["source_revision"] = "a" * 40
    receipt["source_archive_sha256"] = "b" * 64
    receipt["input_hashes"] = {
        **cast(dict[str, str], receipt["input_hashes"]),
        "runtime_source_archive": "b" * 64,
    }
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=predictions
    )
    assert valid, detail

    unbound = deepcopy(receipt)
    cast(dict[str, str], unbound["input_hashes"]).pop("runtime_source_archive")
    valid, detail = validate_replay_receipt(
        unbound, project_root=tmp_path, predictions_path=predictions
    )
    assert not valid
    assert "bound" in detail

    half_bound = deepcopy(receipt)
    half_bound.pop("source_archive_sha256")
    valid, detail = validate_replay_receipt(
        half_bound, project_root=tmp_path, predictions_path=predictions
    )
    assert not valid
    assert "must appear together" in detail


def test_public_replay_requires_no_private_gold_or_owner_manifest(
    tmp_path: Path,
) -> None:
    write_replication_bundle(tmp_path)
    public = tmp_path / "output/replication/public/public_manifest.json"
    predictions = tmp_path / "predictions.json"
    (tmp_path / GOLD_LABELS).unlink()
    (tmp_path / REPLICATION_ARTIFACT).unlink()

    valid, detail = validate_public_replay_bundle(tmp_path)
    assert valid, detail
    run_label_blind_model(public, predictions)
    receipt = replay_receipt_payload(
        public,
        predictions,
        operator_id="vm-operator",
        machine_id="vm-machine",
        replay_bundle_sha256=_public_bundle_digest(tmp_path),
        model="calibrated_interpretable",
        seed=0,
        command=["replay", "--offline"],
        network_disabled=True,
        project_root=tmp_path,
    )
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=predictions
    )
    assert valid, detail


def test_local_receipt_cannot_promote_independent_replay(tmp_path: Path) -> None:
    write_replication_bundle(tmp_path)
    public = tmp_path / "output/replication/public/public_manifest.json"
    predictions = tmp_path / "predictions.json"
    run_label_blind_model(public, predictions)
    receipt_path = tmp_path / "external-receipt.json"
    receipt = replay_receipt_payload(
        public,
        predictions,
        operator_id="local-process",
        machine_id="local-machine",
        replay_bundle_sha256=_public_bundle_digest(tmp_path),
        model="calibrated_interpretable",
        seed=1,
        command=["replay", "--offline"],
        network_disabled=True,
        project_root=tmp_path,
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="separately controlled"):
        ingest_independent_replay(receipt_path, predictions, tmp_path)
    assert not (tmp_path / INDEPENDENT_REPLAY_ARTIFACT).exists()


def _pending_receipt_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    write_replication_bundle(tmp_path)
    public = tmp_path / "output/replication/public/public_manifest.json"
    predictions = tmp_path / "predictions.json"
    run_label_blind_model(public, predictions)
    receipt = replay_receipt_payload(
        public,
        predictions,
        operator_id="operator-test",
        machine_id="machine-test",
        replay_bundle_sha256=_public_bundle_digest(tmp_path),
        model="calibrated_interpretable",
        seed=11,
        command=["replay", "--offline"],
        network_disabled=True,
        project_root=tmp_path,
    )
    return receipt, predictions, public


def test_pending_replay_validator_rejects_core_contract_drift(tmp_path: Path) -> None:
    receipt, predictions, _ = _pending_receipt_fixture(tmp_path)
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=predictions
    )
    assert valid, detail

    mutations = []

    def remove_required(candidate: dict[str, object]) -> None:
        candidate.pop("operator_id")

    def change_kind(candidate: dict[str, object]) -> None:
        candidate["receipt_kind"] = "local_rerun"

    def change_status(candidate: dict[str, object]) -> None:
        candidate["status"] = "validated"

    def empty_operator(candidate: dict[str, object]) -> None:
        candidate["operator_id"] = ""

    def disable_network(candidate: dict[str, object]) -> None:
        candidate["network_disabled"] = False

    def promote_endpoint(candidate: dict[str, object]) -> None:
        candidate["endpoint_status"] = "independent_replay_observed"

    def claim_independent(candidate: dict[str, object]) -> None:
        candidate["independent_replay"] = "observed_external_replay"

    def bad_outcome(candidate: dict[str, object]) -> None:
        candidate["outcome"] = "validated"

    def bad_attestation(candidate: dict[str, object]) -> None:
        candidate["attestation"] = None

    def unsafe_flag(candidate: dict[str, object]) -> None:
        cast(dict[str, object], candidate["attestation"])["gold_labels_accessed"] = True

    def bad_separate_flag(candidate: dict[str, object]) -> None:
        cast(dict[str, object], candidate["attestation"])["separate_controlled_run"] = (
            "yes"
        )

    def bad_confirmation(candidate: dict[str, object]) -> None:
        cast(dict[str, object], candidate["attestation"])[
            "external_operator_confirmation"
        ] = None

    def empty_command(candidate: dict[str, object]) -> None:
        candidate["command"] = []

    def bool_seed(candidate: dict[str, object]) -> None:
        candidate["seed"] = True

    def negative_count(candidate: dict[str, object]) -> None:
        candidate["prediction_row_count"] = -1

    def bad_tolerance(candidate: dict[str, object]) -> None:
        candidate["numerical_tolerance"] = "not-a-number"

    def bad_digest(candidate: dict[str, object]) -> None:
        candidate["prediction_file_sha256"] = "not-a-digest"

    def bad_timestamp(candidate: dict[str, object]) -> None:
        candidate["started_at"] = "operator_supplied"

    def naive_timestamp(candidate: dict[str, object]) -> None:
        candidate["completed_at"] = "2026-08-27T12:00:00"

    def reversed_timestamps(candidate: dict[str, object]) -> None:
        candidate["started_at"] = "2026-08-27T12:00:01+00:00"
        candidate["completed_at"] = "2026-08-27T12:00:00+00:00"

    mutations.extend(
        (
            remove_required,
            change_kind,
            change_status,
            empty_operator,
            disable_network,
            promote_endpoint,
            claim_independent,
            bad_outcome,
            bad_attestation,
            unsafe_flag,
            bad_separate_flag,
            bad_confirmation,
            empty_command,
            bool_seed,
            negative_count,
            bad_tolerance,
            bad_digest,
            bad_timestamp,
            naive_timestamp,
            reversed_timestamps,
        )
    )
    for mutation in mutations:
        candidate = deepcopy(receipt)
        mutation(candidate)
        assert not validate_replay_receipt(
            candidate, project_root=tmp_path, predictions_path=predictions
        )[0]


def test_replay_timestamp_contract_requires_timezone_and_order() -> None:
    assert _replay_timestamp("2026-08-27T12:00:00Z") == "2026-08-27T12:00:00Z"
    assert _timestamps_ordered("2026-08-27T12:00:00+00:00", "2026-08-27T12:00:01+00:00")
    assert not _timestamps_ordered(
        "2026-08-27T12:00:01+00:00", "2026-08-27T12:00:00+00:00"
    )
    with pytest.raises(ValueError, match="valid ISO-8601"):
        _replay_timestamp("operator_supplied")
    with pytest.raises(ValueError, match="timezone"):
        _replay_timestamp("2026-08-27T12:00:00")


def test_attested_replay_ingestion_is_protected_and_not_reimplementation(
    tmp_path: Path,
) -> None:
    receipt, predictions, public = _pending_receipt_fixture(tmp_path)
    attestation = cast(dict[str, object], receipt["attestation"])
    attestation["separate_controlled_run"] = True
    attestation["external_operator_confirmation"] = "operator-confirmed-run-1"
    receipt_path = tmp_path / "external-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    output = ingest_independent_replay(receipt_path, predictions, tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    valid, detail = validate_independent_replay_artifact(payload, tmp_path)
    assert valid, detail
    assert payload["endpoint_status"] == "independent_replay_observed"
    assert payload["independent_reimplementation"] is False
    assert payload["gold_labels_exposed_to_operator"] is False
    assert payload["protected_score"]["balanced_accuracy"] >= 0.0

    written = tmp_path / "pending-receipt.json"
    result = write_replay_receipt(
        public,
        predictions,
        written,
        operator_id="operator-test",
        machine_id="machine-test",
        replay_bundle_sha256=_public_bundle_digest(tmp_path),
        model="calibrated_interpretable",
        seed=11,
        command=["replay", "--offline"],
        project_root=tmp_path,
    )
    assert result == written
    assert json.loads(written.read_text(encoding="utf-8"))["independent_replay"] == (
        "not_yet_observed"
    )


def test_independent_replay_artifact_validator_rejects_promotion_drift(
    tmp_path: Path,
) -> None:
    receipt, predictions, _ = _pending_receipt_fixture(tmp_path)
    attestation = cast(dict[str, object], receipt["attestation"])
    attestation["separate_controlled_run"] = True
    attestation["external_operator_confirmation"] = "operator-confirmed-run-2"
    receipt_path = tmp_path / "external-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = ingest_independent_replay(receipt_path, predictions, tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    candidates = [
        None,
        {**payload, "receipt_schema_version": "0.0.0"},
        {**payload, "schema_version": "0.0.0"},
        {**payload, "endpoint_status": "pending"},
        {**payload, "independent_replay": "not_yet_observed"},
        {**payload, "external_receipt": None},
        {**payload, "gold_labels_exposed_to_operator": True},
        {**payload, "independent_reimplementation": True},
    ]
    for candidate in candidates:
        valid, _ = validate_independent_replay_artifact(candidate, tmp_path)
        assert not valid


def test_replication_private_boundary_and_prediction_parser_fail_closed(
    tmp_path: Path,
) -> None:
    receipt, predictions, _ = _pending_receipt_fixture(tmp_path)
    missing = tmp_path / "missing.json"
    valid, detail = validate_replay_receipt(
        receipt, project_root=tmp_path, predictions_path=missing
    )
    assert not valid
    assert "unreadable" in detail

    bad_predictions = tmp_path / "bad-predictions.json"
    bad_cases = [
        {"predictions": [], "label_blind": False},
        {"predictions": "not-a-list", "label_blind": True},
        [{"record_id": "x", "probability": "not-a-number"}],
        [{"record_id": "x", "probability": float("nan")}],
        [
            {"record_id": "x", "probability": 0.5},
            {"record_id": "x", "probability": 0.5},
        ],
    ]
    for candidate in bad_cases:
        bad_predictions.write_text(json.dumps(candidate), encoding="utf-8")
        valid, _ = validate_replay_receipt(
            receipt, project_root=tmp_path, predictions_path=bad_predictions
        )
        assert not valid

    missing_bundle_file = tmp_path / "output/replication/public/toolchain_receipt.json"
    original = missing_bundle_file.read_text(encoding="utf-8")
    missing_bundle_file.unlink()
    with pytest.raises(ValueError, match="bundle file is missing"):
        _public_bundle_digest(tmp_path)
    missing_bundle_file.write_text(original, encoding="utf-8")
