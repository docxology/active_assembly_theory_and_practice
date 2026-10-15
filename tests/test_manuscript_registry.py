"""Fail-closed manuscript value and registry contracts."""

import json

import pytest

from src.manuscript_registry import build_variable_registry
from src.validation import _manuscript_variable_coverage_checks


def _surface(root, text, values):
    (root / "manuscript").mkdir()
    (root / "manuscript/00_abstract.md").write_text(text)
    (root / "output/data").mkdir(parents=True)
    (root / "output/data/manuscript_variables.json").write_text(json.dumps(values))
    registry = build_variable_registry(values, root)
    (root / "output/data/manuscript_variable_registry.json").write_text(
        json.dumps(registry)
    )
    return registry


def test_unused_compatibility_pending_does_not_bypass_registry(tmp_path):
    registry = _surface(
        tmp_path,
        "{{RESULT_COUNT}}",
        {"RESULT_COUNT": "2", "VALIDATION_FIGURE": "pending"},
    )
    assert registry["summary"]["no_pending_values"]
    assert registry["summary"]["deferred_compatibility_tokens"] == ["VALIDATION_FIGURE"]
    assert all(check.passed for check in _manuscript_variable_coverage_checks(tmp_path))
    registry["records"] = []
    (tmp_path / "output/data/manuscript_variable_registry.json").write_text(
        json.dumps(registry)
    )
    checks = _manuscript_variable_coverage_checks(tmp_path)
    assert checks[0].passed
    assert not checks[1].passed
    assert "record/generated token mismatch" in checks[1].detail


def test_unconsumed_docx_measurements_can_wait_for_native_export(tmp_path):
    registry = _surface(
        tmp_path,
        "{{RESULT_COUNT}}",
        {
            "RESULT_COUNT": "2",
            "DOCX_RENDER_STATUS": "pending_render",
            "DOCX_RENDER_FIGURE_COUNT": "pending",
            "DOCX_RENDER_CAPTION_COUNT": "pending",
        },
    )
    assert registry["summary"]["no_pending_values"]
    assert registry["summary"]["no_unused_generated_tokens"]
    assert registry["summary"]["deferred_compatibility_tokens"] == [
        "DOCX_RENDER_CAPTION_COUNT",
        "DOCX_RENDER_FIGURE_COUNT",
    ]
    assert all(check.passed for check in _manuscript_variable_coverage_checks(tmp_path))


@pytest.mark.parametrize("token", ["RESULT_COUNT", "DOCX_RENDER_FIGURE_COUNT"])
def test_consumed_pending_value_fails_closed(tmp_path, monkeypatch, token):
    monkeypatch.delenv("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER", raising=False)
    registry = _surface(tmp_path, "{{" + token + "}}", {token: "pending"})
    assert not registry["summary"]["no_pending_values"]
    assert not any(
        check.passed for check in _manuscript_variable_coverage_checks(tmp_path)
    )


def test_validation_summary_bootstrap_is_explicit(tmp_path):
    _surface(
        tmp_path, "{{VALIDATION_TOTAL_COUNT}}", {"VALIDATION_TOTAL_COUNT": "pending"}
    )
    checks = _manuscript_variable_coverage_checks(tmp_path)
    assert all(check.passed for check in checks)
    assert "deferred=[" in checks[0].detail
    assert "VALIDATION_TOTAL_COUNT" in checks[0].detail


def test_pre_render_docx_deferral_keeps_registry_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER", "1")
    registry = _surface(
        tmp_path,
        "{{DOCX_RENDER_FIGURE_COUNT}}",
        {"DOCX_RENDER_FIGURE_COUNT": "pending"},
    )
    assert all(check.passed for check in _manuscript_variable_coverage_checks(tmp_path))
    registry["records"] = []
    (tmp_path / "output/data/manuscript_variable_registry.json").write_text(
        json.dumps(registry)
    )
    assert not _manuscript_variable_coverage_checks(tmp_path)[1].passed
