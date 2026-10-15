from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gnn_parser import compile_gnn
from src.gnn_reference import forward_backward
from src.rxinfer_bridge import run_shared_gnn_native

ROOT = Path(__file__).resolve().parents[1]


def test_native_first_observation_and_smoothing(tmp_path):
    m = compile_gnn(ROOT / "configs/gnn/passive_phase.md")
    first = run_shared_gnn_native(m, [0], tmp_path / "one")
    initial = m.A[0][0] * m.D[0]
    initial /= initial.sum()
    wrong = m.A[0][0] * (m.B[0][:, :, 0] @ m.D[0])
    wrong /= wrong.sum()
    assert np.max(np.abs(initial - wrong)) > 0.01
    np.testing.assert_allclose(first["native_smoothed"][0], initial, atol=1e-5)

    np.testing.assert_allclose(
        first["native_smoothed"], forward_backward(m, [0])["smoothed"], atol=1e-5
    )
    result = run_shared_gnn_native(m, [0, 1, 2, 1], tmp_path / "sequence")
    assert result["max_abs_error"] < 1e-5
    changed = replace(m, D=(np.array([0.1, 0.1, 0.1, 0.1, 0.6]),))
    other = run_shared_gnn_native(changed, [0], tmp_path / "one")
    assert other["model_digest"] != first["model_digest"]
    assert other["native_smoothed"] != first["native_smoothed"]


def test_successful_process_without_fresh_native_output_is_rejected(
    tmp_path, monkeypatch
):
    model = compile_gnn(ROOT / "configs/gnn/passive_phase.md")
    executable = tmp_path / "no-output-julia"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    output = tmp_path / "native"
    output.mkdir()
    result = output / "result.json"
    result.write_text('{"previous": "preserved"}')
    before = result.read_bytes()
    monkeypatch.setenv("JULIA", str(executable))
    with pytest.raises(RuntimeError, match="GNN_NATIVE_OUTPUT_NOT_FRESH"):
        run_shared_gnn_native(model, [0], output)
    assert result.read_bytes() == before
    assert (output / "stdout.log").is_file()
