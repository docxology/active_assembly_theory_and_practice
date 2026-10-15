from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gnn_model import CompiledModel, GNNError
from src.gnn_parser import compile_gnn

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "before,after",
    [
        ("GNN v1.1", ""),
        ("ActInfPOMDP", "ignored"),
        (
            'factors: ["hypothesis", "sensor_mode"]',
            'factors: ["hypothesis", "hypothesis"]',
        ),
        ("horizon: 1", "horizon: true"),
        ("seed: 0", "seed: -1"),
        ("A_dependencies: [[0, 1], [1]]", "A_dependencies: [[false, 1], [1]]"),
    ],
)
def test_malformed_contract_fails_with_stable_error(tmp_path, before, after):
    p = tmp_path / "model.md"
    p.write_text(
        (ROOT / "configs/gnn/fixture_sensor.md").read_text().replace(before, after)
    )
    with pytest.raises(GNNError, match="GNN-E"):
        compile_gnn(p)


def test_export_roundtrip_and_immutable_tensors():
    m = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    restored = CompiledModel.from_dict(m.to_dict())
    assert restored.tensor_digest == m.tensor_digest
    assert restored.C[1][-1] == -0.1
    with pytest.raises(ValueError):
        restored.A[0][0, 0, 0] = 0
    with pytest.raises(GNNError):
        replace(m, E=np.array([0.2, 0.2, 0.2]))


@pytest.mark.parametrize(
    "before,after",
    [
        ("D0>A0", "D0>A1"),
        ("D0>A0", "D0>A0\nD0>A0"),
        ("horizon: 1", "horizon: 0"),
        ("gamma: 1.0", 'gamma: "fast"'),
        ("D0[2, type=float]", "D0[100000000, type=float]"),
        ("D0=[0.5,0.5]", "D0=[0.5,0.7]"),
        ("GNN v1.1", "GNN v2.0"),
        (
            "E=[0.3333333333333333,0.3333333333333333,0.3333333333333333]",
            "E=[NaN,0.5,0.5]",
        ),
    ],
)
def test_connections_resources_probabilities(tmp_path, before, after):
    p = tmp_path / "model.md"
    p.write_text(
        (ROOT / "configs/gnn/fixture_sensor.md").read_text().replace(before, after)
    )
    with pytest.raises(GNNError, match="GNN-E"):
        compile_gnn(p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_labels", [["wrong"], ["idle", "static", "dynamic"]]),
        ("A", [1, 2]),
        ("source_sha256", 1),
        ("model_digest", "0" * 64),
        ("extra_execution", True),
    ],
)
def test_export_tampering_fails(field, value):
    m = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    payload = m.to_dict()
    payload[field] = value
    with pytest.raises(GNNError, match="GNN-E"):
        CompiledModel.from_dict(payload)


def test_policy_resource_limit_without_integer_overflow():
    m = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    with pytest.raises(GNNError, match="GNN-E103"):
        replace(
            m,
            factors=tuple(f"f{i}" for i in range(65)),
            state_labels=(("s0", "s1"),) * 65,
            actions=(("a0", "a1"),) * 65,
            B=(np.ones((2, 2, 2)) * 0.5,) * 65,
            D=(np.array([0.5, 0.5]),) * 65,
        )


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("D", 0, ["0.5", "0.5"]),
        ("D", 1, [True, False, False]),
    ],
)
def test_export_rejects_implicit_numeric_coercion(field, index, value):
    model = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    payload = model.to_dict()
    payload[field][index] = value
    with pytest.raises(GNNError, match="GNN-E"):
        CompiledModel.from_dict(payload)
