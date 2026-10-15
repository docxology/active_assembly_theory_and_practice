import json
from pathlib import Path

import pytest

from src.gnn_cli import main
from src.rxinfer_bridge import julia_available

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["passive_phase", "fixture_sensor"])
def test_public_cli_run_and_seed_provenance(tmp_path, gnn_study, capsys, name):
    _, upstream, _ = gnn_study
    arguments = [
        "--gnn-root",
        str(upstream),
        "--model",
        str(ROOT / f"configs/gnn/{name}.md"),
    ]
    assert main(["validate", *arguments]) == 0
    assert json.loads(capsys.readouterr().out)["model_digest"]
    assert (
        main(
            [
                "run",
                *arguments,
                "--fixture",
                "x86_accumulator",
                "--steps",
                "1",
                "--seed",
                "23",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["seed"] == 23
    assert run["steps"] == 1
    assert run["backend"] == "pymdp.Agent"
    assert (tmp_path / "compiled_model.json").is_file()
    assert (
        json.loads((tmp_path / "upstream.json").read_text())["model_digest"]
        == run["model_digest"]
    )


def test_public_cli_audit_and_missing_input_exit_codes(gnn_study, tmp_path, capsys):
    output, upstream, _ = gnn_study
    assert (
        main(["audit", "--gnn-root", str(upstream), "--output-dir", str(output)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "validated_bounded"
    assert (
        main(["audit", "--gnn-root", str(upstream), "--output-dir", str(tmp_path)]) == 1
    )
    assert json.loads(capsys.readouterr().out)["errors"]
    assert (
        main(
            [
                "validate",
                "--gnn-root",
                str(upstream),
                "--model",
                str(tmp_path / "absent.md"),
            ]
        )
        == 2
    )
    assert "absent.md" in capsys.readouterr().err


def test_public_cli_missing_upstream_never_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GNN_ROOT", raising=False)
    target = tmp_path / "not-created"
    assert main(["study", "--output-dir", str(target)]) == 2
    assert "GNN_UPSTREAM_REQUIRED" in capsys.readouterr().err
    assert not target.exists()


@pytest.mark.skipif(not julia_available(), reason="native CLI requires Julia")
def test_public_cli_study_and_native_produce_auditable_artifacts(tmp_path, gnn_study):
    _, upstream, _ = gnn_study
    arguments = ["--gnn-root", str(upstream), "--output-dir", str(tmp_path)]
    assert main(["study", *arguments]) == 0
    assert main(["native", *arguments]) == 0
    assert main(["audit", *arguments, "--require-native"]) == 0
    native = json.loads((tmp_path / "native_comparison.json").read_text())
    assert native["max_abs_error"] < 1e-5
