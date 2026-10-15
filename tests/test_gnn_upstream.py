import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.gnn_model import GNNError
from src.gnn_upstream import compile_upstream

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/gnn/fixture_sensor.md"


@pytest.fixture(scope="session")
def upstream_root():
    configured = os.environ.get("GNN_ROOT")
    if not configured:
        pytest.skip("real upstream integration requires GNN_ROOT")
    return Path(configured).resolve()


@pytest.fixture()
def upstream_snapshot(tmp_path, upstream_root):
    receipt = compile_upstream(SOURCE, upstream_root).receipt
    root = tmp_path / "upstream"
    root.mkdir()
    for name in receipt["import_hashes"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream_root / name, target)
    for args in (
        ["init", "--initial-branch=main"],
        ["remote", "add", "origin", receipt["repository"]],
        ["add", "src"],
        [
            "-c",
            "user.name=GNN test",
            "-c",
            "user.email=gnn@example.invalid",
            "commit",
            "-m",
            "Retain real upstream parser sources",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.mark.parametrize("name", ["fixture_sensor", "passive_phase"])
def test_real_upstream_tensors_roundtrip(name, upstream_root):
    result = compile_upstream(ROOT / f"configs/gnn/{name}.md", upstream_root)
    assert result.receipt["model_digest"] == result.model.tensor_digest
    assert "src/gnn/parsers/markdown_parser.py" in result.receipt["import_hashes"]
    assert all(len(value) == 64 for value in result.receipt["import_hashes"].values())


def test_upstream_numeric_change_controls_runtime_tensor(tmp_path, upstream_root):
    source = tmp_path / "changed.md"
    source.write_text(SOURCE.read_text().replace("-0.1", "-2.0"))
    result = compile_upstream(source, upstream_root)
    np.testing.assert_array_equal(result.model.C[1], [0.0, -0.01, -2.0])
    assert (
        result.model.tensor_digest
        != compile_upstream(SOURCE, upstream_root).model.tensor_digest
    )


@pytest.mark.parametrize("change", ["dirty", "untracked", "origin"])
def test_upstream_source_admission_rejects_drift(change, upstream_snapshot):
    parser = upstream_snapshot / "src/gnn/parsers/markdown_parser.py"
    if change == "dirty":
        parser.write_text(parser.read_text() + "\n")
    elif change == "untracked":
        (parser.parent / "unexpected.py").write_text("unapproved = True\n")
    else:
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream_snapshot),
                "remote",
                "set-url",
                "origin",
                "https://example.invalid/other.git",
            ],
            check=True,
        )
    with pytest.raises(GNNError, match="GNN_UPSTREAM_SOURCE"):
        compile_upstream(SOURCE, upstream_snapshot)


def test_real_parser_tensor_corruption_is_rejected(upstream_snapshot):
    parser = upstream_snapshot / "src/gnn/parsers/markdown_parser.py"
    before = parser.read_text()
    after = before.replace(
        "result.model.raw_sections = dict(sections.items())",
        "result.model.raw_sections = dict(sections.items())\n"
        "            result.model.parameters[0].value = [0.5, 0.5]",
    )
    assert after != before
    parser.write_text(after)
    subprocess.run(["git", "-C", str(upstream_snapshot), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream_snapshot),
            "-c",
            "user.name=GNN test",
            "-c",
            "user.email=gnn@example.invalid",
            "commit",
            "-m",
            "Inject a parser regression",
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(GNNError, match="GNN_UPSTREAM_TENSOR"):
        compile_upstream(SOURCE, upstream_snapshot)


def test_missing_upstream_cli_has_no_outputs(tmp_path):
    output = tmp_path / "absent"
    environment = {key: value for key, value in os.environ.items() if key != "GNN_ROOT"}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_gnn_inference.py"),
            "study",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 2
    assert "GNN_UPSTREAM_REQUIRED" in result.stderr
    assert not output.exists()


def test_real_cli_upstream_validate(upstream_root):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_gnn_inference.py"),
            "validate",
            "--gnn-root",
            str(upstream_root),
            "--model",
            str(SOURCE),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["parser"] == "gnn.parsers.markdown_parser.MarkdownGNNParser"
    assert (
        receipt["model_digest"]
        == compile_upstream(SOURCE, upstream_root).model.tensor_digest
    )


def test_worker_emits_actual_parser_results_and_source_witnesses(
    upstream_root, monkeypatch, capsys
):
    from src.gnn_upstream_worker import main

    before_path = sys.path[:]
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "gnn" or name.startswith("gnn.")
    }
    monkeypatch.setattr(sys, "argv", ["gnn-worker", str(upstream_root), str(SOURCE)])
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    try:
        assert main() == 0
        result = json.loads(capsys.readouterr().out)
    finally:
        sys.path[:] = before_path
        for name in list(sys.modules):
            if (
                name == "gnn" or name.startswith("gnn.")
            ) and name not in before_modules:
                del sys.modules[name]
        sys.modules.update(before_modules)
    qualified = compile_upstream(SOURCE, upstream_root)
    assert result["success"] is True
    assert result["errors"] == result["warnings"] == []
    assert result["name"] == qualified.model.name
    assert result["source_sha256"] == qualified.model.source_sha256
    assert result["import_hashes"] == qualified.receipt["import_hashes"]
    assert {row["name"] for row in result["parameters"]} == {
        row["name"] for row in result["variables"]
    }
