import json
import shutil

import pytest

from src.gnn_validation import validate_gnn_study


@pytest.mark.parametrize("corruption", ["figures", "empty-native", "oracles"])
def test_required_native_audit_rejects_corrupted_derivatives(
    native_gnn_study, tmp_path, corruption
):
    source, upstream, _ = native_gnn_study
    assert validate_gnn_study(source, upstream, require_native=True) == []
    output = tmp_path / "copy"
    shutil.copytree(source, output)
    if corruption == "figures":
        path = output / "figure_inputs.json"
        payload = json.loads(path.read_text())
        payload["sensor_actions"] = [[999, 999]]
        payload["sensor_posteriors"] = [[99, -98]]
    elif corruption == "empty-native":
        path = output / "native_backend/result.json"
        payload = {}
    else:
        path = output / "controls.json"
        payload = json.loads(path.read_text())
        payload["oracle_max_error"] = 999
        payload["tolerance"] = -1
        payload["model_digests"] = {"passive": "false", "sensor": "false"}
        for row in (*payload["controls"].values(), *payload["baselines"].values()):
            row["oracle"]["q_pi"] = [123]
            row["oracle_max_error"] = 999
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    assert validate_gnn_study(output, upstream, require_native=True)
    assert path.read_bytes() == before
