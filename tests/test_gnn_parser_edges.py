"""Real-file adversarial controls for the bounded executable GNN contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gnn_model import CompiledModel, GNNError
from src.gnn_parser import compile_gnn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/gnn/fixture_sensor.md"


@pytest.mark.parametrize(
    ("before", "after", "code"),
    [
        ("## GNNSection", "## Unknown", "GNN-E100"),
        ("## ModelName", "## GNNSection", "GNN-E100"),
        ("## ModelAnnotation", "## OtherAnnotation", "GNN-E100"),
        ("A0[5,2,3, type=float]", "A0[N,2,3, type=float]", "GNN-E100"),
        ("A0[5,2,3, type=float]", "A0[5,2,3, type=int]", "GNN-E100"),
        (
            "A0[5,2,3, type=float]",
            "A0[5,2,3, type=float]\nA0[5,2,3, type=float]",
            "GNN-E004",
        ),
        ("D0>A0", "D9>A0", "GNN-E003"),
        ("D0>A0", "D0 -> A0", "GNN-E005"),
        ("D0>A0", "D0>A0\nD0>A0", "GNN-E004"),
        ("C0=[0.0,0.0,0.0,0.0,0.0]", "C0=[0.0,0.0]", "GNN-E002"),
        ("C0=[0.0,0.0,0.0,0.0,0.0]", "C0=[1e309,0.0,0.0,0.0,0.0]", "GNN-E101"),
        ("C0=[0.0,0.0,0.0,0.0,0.0]", "C0=[NaN,0.0,0.0,0.0,0.0]", "GNN-E100"),
        ("D0=[0.5,0.5]", "D0=[0.5,]", "GNN-E100"),
        ("D0=[0.5,0.5]", "D0=[0.5,0.5]\nD0=[0.5,0.5]", "GNN-E102"),
        ("D0=[0.5,0.5]", "D9=[0.5,0.5]", "GNN-E102"),
        ("gamma: 1.0", "gamma: no_python_eval", "GNN-E100"),
        ("gamma: 1.0", "gamma: 1.0\ngamma: 2.0", "GNN-E100"),
        ("gamma: 1.0", "unknown: 1.0", "GNN-E100"),
        ("gamma: 1.0", "gamma 1.0", "GNN-E100"),
        ("gamma: 1.0", "gamma: NaN", "GNN-E102"),
        ("gamma: 1.0", "gamma: 0.0", "GNN-E102"),
        ('factors: ["hypothesis", "sensor_mode"]', "factors: null", "GNN-E102"),
        ("A_dependencies: [[0, 1], [1]]", "A_dependencies: null", "GNN-E102"),
        ("horizon: 1", "horizon: 4", "GNN-E103"),
        ("D0[2, type=float]", "D0[0, type=float]", "GNN-E103"),
        ("D0[2, type=float]", "D0[1,1,1,1,1, type=float]", "GNN-E103"),
        ("seed: 0", "seed: 4294967296", "GNN-E102"),
    ],
)
def test_real_file_rejects_unsupported_execution(before, after, code, tmp_path):
    text = SOURCE.read_text() + "\n## ModelAnnotation\nBounded literal profile\n"
    assert before in text
    path = tmp_path / "invalid.md"
    path.write_text(text.replace(before, after))
    with pytest.raises(GNNError, match=code):
        compile_gnn(path)


@pytest.mark.parametrize(
    "case",
    [
        "outside",
        "missing",
        "reordered",
        "too_large",
        "invalid_utf8",
        "incomplete",
        "extra_tensor",
    ],
)
def test_layout_inventory_and_byte_resource_boundaries(case, tmp_path):
    text = SOURCE.read_text()
    if case == "outside":
        text = "execute_me\n" + text
    elif case == "missing":
        text = text.replace("## Metadata", "## ModelAnnotation")
    elif case == "reordered":
        text = text.replace(
            "## GNNSection\nActInfPOMDP\n## GNNVersionAndFlags\nGNN v1.1",
            "## GNNVersionAndFlags\nGNN v1.1\n## GNNSection\nActInfPOMDP",
        )
    elif case == "too_large":
        text += "\n#" + ("x" * 65536)
    elif case == "incomplete":
        text = text.replace("seed: 0", "")
    elif case == "extra_tensor":
        text = text.replace(
            "E[3, type=float]", "E[3, type=float]\nC9[1, type=float]"
        ).replace("C0=[0.0,0.0,0.0,0.0,0.0]", "C0=[0.0,0.0,0.0,0.0,0.0]\nC9=[0.0]")
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe" if case == "invalid_utf8" else text.encode())
    with pytest.raises(GNNError, match="GNN-E"):
        compile_gnn(path)


def test_executable_string_is_rejected_without_side_effect(tmp_path):
    marker = tmp_path / "must_not_exist"
    text = SOURCE.read_text().replace(
        "D0=[0.5,0.5]", f"D0=[__import__('pathlib').Path('{marker}').touch(),0.5]"
    )
    path = tmp_path / "malicious.md"
    path.write_text(text)
    with pytest.raises(GNNError, match="GNN-E100"):
        compile_gnn(path)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"name": ""}, "GNN-E102"),
        ({"source_sha256": "invalid"}, "GNN-E102"),
        ({"gamma": float("inf")}, "GNN-E102"),
        ({"A": (np.array([1.0]),)}, "GNN-E102"),
        ({"C": (np.array(1.0), np.zeros(3))}, "GNN-E103"),
        ({"C": (np.zeros(5), np.array([0.0, np.nan, 0.0]))}, "GNN-E101"),
        ({"D": (np.array([-0.1, 1.1]), np.array([1.0, 0.0, 0.0]))}, "GNN-E101"),
        ({"A_dependencies": ((), (1,))}, "GNN-E102"),
        ({"A_dependencies": ((0, 0), (1,))}, "GNN-E102"),
        ({"A_dependencies": ((0, 2), (1,))}, "GNN-E102"),
        ({"A_dependencies": ((0, True), (1,))}, "GNN-E102"),
        ({"E": np.array([1.0, 0.0])}, "GNN-E101"),
        ({"E": np.array([1.0, np.inf, 0.0])}, "GNN-E101"),
    ],
)
def test_ir_rejects_nonfinite_axes_and_invalid_distributions(change, code):
    original = compile_gnn(SOURCE)
    with pytest.raises(GNNError, match=code):
        replace(original, **change)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "dialect",
        "policies",
        "B_dependencies",
        "B_action_dependencies",
        "D",
    ],
)
def test_export_rejects_schema_policy_axes_and_missing_fields(field):
    model = compile_gnn(SOURCE)
    payload = model.to_dict()
    if field in ("schema_version", "dialect"):
        payload[field] = "unsupported"
    elif field == "policies":
        payload[field] = [[[0, 2]], [[0, 1]], [[0, 0]]]
    elif field == "B_dependencies":
        payload[field] = [[1], [0]]
    elif field == "B_action_dependencies":
        payload[field] = [[1], [0]]
    else:
        del payload[field]
    with pytest.raises(GNNError, match="GNN-E"):
        CompiledModel.from_dict(payload)


def test_annotation_comments_and_serialized_roundtrip_remain_executable(tmp_path):
    path = tmp_path / "annotated.md"
    path.write_text(
        SOURCE.read_text()
        + "\n## ModelAnnotation\nA finite fixture model\n# ignored comment\n"
    )
    model = compile_gnn(path)
    restored = CompiledModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.annotations == "A finite fixture model"
    assert restored.tensor_digest == model.tensor_digest
    np.testing.assert_array_equal(restored.C[1], [0.0, -0.01, -0.1])
    np.testing.assert_array_equal(restored.policies, [[[0, 0]], [[0, 1]], [[0, 2]]])
