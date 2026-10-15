from pathlib import Path
from dataclasses import replace

import pytest

from src.gnn_parser import compile_gnn
from src.gnn_experiments import run_passive

ROOT = Path(__file__).resolve().parents[1]


def test_passive_run_uses_model_seed_unless_explicitly_overridden():
    model = replace(compile_gnn(ROOT / "configs/gnn/passive_phase.md"), seed=17)
    default = run_passive(model, steps=1)
    explicit = run_passive(model, steps=1, seed=23)
    assert default["seed"] == 17
    assert explicit["seed"] == 23
    assert default["final_posterior"] == explicit["final_posterior"]


def test_specs_are_executable():
    model = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    assert model.policies.shape == (3, 1, 2)
    assert model.C[1][-1] == -0.1
    assert not model.A[0].flags.writeable


def test_numeric_edit_and_invalid_distribution(tmp_path):
    source = (ROOT / "configs/gnn/fixture_sensor.md").read_text()
    path = tmp_path / "model.md"
    path.write_text(source.replace("-0.1", "-2.0"))
    changed = compile_gnn(path)
    assert changed.C[1][-1] == -2
    assert (
        changed.tensor_digest
        != compile_gnn(ROOT / "configs/gnn/fixture_sensor.md").tensor_digest
    )
    path.write_text(source.replace("[0.5,0.5]", "[0.5,0.7]"))
    with pytest.raises(ValueError, match="GNN-E101"):
        compile_gnn(path)
