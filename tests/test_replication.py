"""Additional negative-path coverage for the frozen replication interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.replication import (
    run_label_blind_model,
    score_replication_predictions,
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
