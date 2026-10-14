"""Smoke tests for thin project scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_preflight_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/00_preflight.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "capstone" in completed.stdout
    assert "pymdp" in completed.stdout
    assert "unicorn" in completed.stdout


def test_analysis_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_assembly_inference.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "sample: x86_accumulator" in completed.stdout
    assert "final_rax: 5" in completed.stdout


def test_validation_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/validate_analysis_outputs.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "validation_failed: 0" in completed.stdout
    assert "validation_total:" in completed.stdout


def test_manuscript_variable_script_runs_and_hydrates_manuscript() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/z_generate_manuscript_variables.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PROJECT_TITLE=Active Assembly Theory and Practice" in completed.stdout
    hydrated_abstract = PROJECT_ROOT / "output/manuscript/00_abstract.md"
    assert hydrated_abstract.exists()
    assert "{{" not in hydrated_abstract.read_text(encoding="utf-8")


def test_full_verification_orchestrator_is_valid_shell() -> None:
    script = PROJECT_ROOT / "scripts/run_full_verification.sh"
    assert script.exists()
    assert os.access(script, os.X_OK)
    # Syntax-check the orchestrator without executing any gate.
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0


def test_existing_output_validation_mode_is_explicit_in_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/validate_analysis_outputs.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--existing-only" in completed.stdout
    assert "without regenerating scientific inputs or figures" in " ".join(
        completed.stdout.split()
    )


def test_full_verification_modes_are_explicit_and_conflicts_fail():
    script = PROJECT_ROOT / "scripts/run_full_verification.sh"
    help_result = subprocess.run(
        ["bash", str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert "--post-render-only" in help_result.stdout
    assert "native template" in help_result.stdout
    for args in (
        ["--post-render-only", "--analysis-only"],
        ["--post-render-only", "--julia"],
        ["--unknown"],
    ):
        result = subprocess.run(
            ["bash", str(script), *args], capture_output=True, text=True, check=False
        )
        assert result.returncode == 2
