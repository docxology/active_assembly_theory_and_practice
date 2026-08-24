"""Direct failure-mode tests for the generated Python-to-Lean bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.lean_bridge as lean_bridge


def test_generated_fixture_is_deterministic_and_elan_is_required(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = lean_bridge.write_generated_lean_fixture(tmp_path)
    source = tmp_path / lean_bridge.GENERATED_RELATIVE_PATH
    assert source.is_file()
    assert metadata["python_reference_final_rax"] == lean_bridge.BRIDGE_EXPECTED_RAX
    assert metadata["source_sha256"]
    monkeypatch.setattr(lean_bridge.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="elan is required"):
        lean_bridge._elan_executable()


def test_generated_bridge_reports_compiler_failure_and_value_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "formal").mkdir()
    monkeypatch.setattr(lean_bridge, "_elan_executable", lambda: "elan")
    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="compiler failure"
        ),
    )
    with pytest.raises(RuntimeError, match="bridge failed"):
        lean_bridge.run_generated_lean_fixture(tmp_path)

    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        lean_bridge,
        "write_generated_lean_fixture",
        lambda root: {
            "python_reference_final_rax": 1,
            "lean_expected_final_rax": 2,
        },
    )
    with pytest.raises(RuntimeError, match="values disagree"):
        lean_bridge.run_generated_lean_fixture(tmp_path)
