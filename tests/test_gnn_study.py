import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.gnn_validation import ARTIFACTS, validate_gnn_study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def study(gnn_study):
    return gnn_study


def hashes(output):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("*.json")
    }


def test_real_study_audit_preserves_all_bytes(study):
    output, upstream, payloads = study
    before = hashes(output)
    assert set(payloads) == set(ARTIFACTS)
    assert validate_gnn_study(output, upstream) == []
    assert hashes(output) == before


@pytest.mark.parametrize(
    "control",
    [
        "policy",
        "action",
        "acquisition",
        "observation",
        "posterior",
        "source",
        "upstream",
        "empty",
        "control",
        "horizon",
        "baseline",
        "figures",
    ],
)
def test_audit_rejects_real_corrupt_receipts(study, tmp_path, control):
    _, upstream, original = study
    payloads = copy.deepcopy(original)
    if control == "upstream":
        payloads["upstream"]["sensor"]["revision"] = "0" * 40
    elif control == "empty":
        payloads["sensor_run"]["trajectory"] = []
    elif control == "source":
        payloads["controls"]["input_hashes"] = {}
    elif control == "control":
        payloads["controls"]["controls"].pop("B")
    elif control == "horizon":
        payloads["controls"]["horizon_sensitivity"]["horizon"] = 1
    elif control == "baseline":
        payloads["controls"]["baselines"]["fixed_static"]["run"]["trajectory"][0][
            "action"
        ] = [0, 2]
    elif control == "figures":
        payloads["figure_inputs"]["negative_efe"][0] += 1
    else:
        row = payloads["sensor_run"]["trajectory"][0]
        if control == "policy":
            row["q_pi"] = [1.0, 0.0, 0.0]
        elif control == "action":
            row["action"] = [0, 0]
        elif control == "acquisition":
            row["decision_before_acquisition"] = False
        elif control == "observation":
            row["revealed_observation"] = [0, 0]
        else:
            row["posterior"][0] = [0.9, 0.1]
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    before = hashes(tmp_path)
    assert validate_gnn_study(tmp_path, upstream)
    assert hashes(tmp_path) == before


@pytest.mark.parametrize("mode", ["missing", "invalid"])
def test_audit_rejects_absent_or_invalid_json(tmp_path, mode):
    if mode == "invalid":
        (tmp_path / "upstream.json").write_text("{")
    assert validate_gnn_study(tmp_path, tmp_path)


def test_canonical_assembly_cli_runs_upstream_study(study, tmp_path):
    _, upstream, _ = study
    stale_native = tmp_path / "native_comparison.json"
    stale_native.write_text('{"prior_native_receipt": "preserved"}')
    before = stale_native.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_assembly_inference.py"),
            "--gnn-root",
            str(upstream),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "models: passive_phase fixture_sensor" in result.stdout
    assert validate_gnn_study(tmp_path, upstream) == []
    assert stale_native.read_bytes() == before
    assert validate_gnn_study(tmp_path, upstream, require_native=True)
