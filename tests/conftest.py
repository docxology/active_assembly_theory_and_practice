"""Pytest configuration for Active Assembly tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def canonical_artifacts():
    """Generate the real canonical package once for artifact integration checks.

    Consumers validate current source/input hashes. This avoids repeatedly
    refitting identical models and rendering the entire monograph in one suite.
    """
    from src.template_hydration import run_hydration_gate

    # Exercise the same source-bound gate used before rendering. Read only the
    # variables produced by this invocation, never a receipt from a prior run.
    assert run_hydration_gate() == 0
    values = json.loads((ROOT / "output/data/manuscript_variables.json").read_text())
    return {key: value for key, value in values.items() if key.isupper()}


@pytest.fixture(scope="session")
def gnn_study(tmp_path_factory):
    configured = os.environ.get("GNN_ROOT")
    if not configured:
        pytest.skip("real GNN study requires GNN_ROOT")
    from src.gnn_experiments import run_gnn_study

    upstream = Path(configured)
    output = tmp_path_factory.mktemp("gnn-study")
    return output, upstream, run_gnn_study(output, upstream)


@pytest.fixture(scope="session")
def native_gnn_study(gnn_study):
    from src.gnn_experiments import write_payload
    from src.gnn_model import CompiledModel
    from src.rxinfer_bridge import julia_available, run_shared_gnn_native

    if not julia_available():
        pytest.skip("native GNN parity requires pinned Julia")
    output, upstream, payloads = gnn_study
    model = CompiledModel.from_dict(payloads["passive_model"])
    observations = [r["observation"][0] for r in payloads["passive_run"]["trajectory"]]
    native = run_shared_gnn_native(model, observations, output / "native_backend")
    write_payload(output / "native_comparison.json", native)
    return output, upstream, payloads
