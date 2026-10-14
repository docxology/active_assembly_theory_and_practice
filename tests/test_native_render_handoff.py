"""Execute a minimal native renderer CLI to verify the registered handoff."""

import json
import subprocess
import sys

import pytest

from src.document_rendering import render_manuscript


def _project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    template = tmp_path / "template"
    (project / "manuscript").mkdir(parents=True)
    (project / "manuscript/00_abstract.md").write_text("# {{TITLE}}\n")
    (project / "output/manuscript").mkdir(parents=True)
    (project / "output/manuscript/00_abstract.md").write_text("# Test\n")
    (project / "output/data").mkdir()
    (project / "output/data/manuscript_variables.json").write_text('{"TITLE": "Test"}')
    (template / "infrastructure/rendering").mkdir(parents=True)
    (template / "infrastructure/rendering/manuscript_injection.py").write_text("")
    (template / "projects/working").mkdir(parents=True)
    (template / "projects/working/project").symlink_to(
        project, target_is_directory=True
    )
    (template / "scripts/pipeline").mkdir(parents=True)
    (template / "scripts/pipeline/stage_03_render.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        'Path("invocation.json").write_text(json.dumps({"argv":sys.argv[1:],"cwd":str(Path.cwd())}))\n'
    )
    (template / ".venv/bin").mkdir(parents=True)
    (template / ".venv/bin/python").symlink_to(sys.executable)
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    monkeypatch.delenv("TEMPLATE_PYTHON", raising=False)
    return project, template


def test_native_handoff_preserves_hydration_and_uses_registered_slot(
    tmp_path, monkeypatch
):
    project, template = _project(tmp_path, monkeypatch)
    render_manuscript(project)
    invocation = json.loads((template / "invocation.json").read_text())
    assert invocation == {
        "argv": ["--project", "working/project", "--skip-manuscript-hydration"],
        "cwd": str(template),
    }
    assert (project / "output/manuscript/00_abstract.md").read_text() == "# Test\n"
    monkeypatch.setenv("TEMPLATE_PYTHON", sys.executable)
    render_manuscript(project)


def test_stale_hydration_never_reaches_template(tmp_path, monkeypatch):
    project, template = _project(tmp_path, monkeypatch)
    (project / "manuscript/00_abstract.md").write_text("# Changed\n")
    with pytest.raises(RuntimeError, match="hydration contract failed"):
        render_manuscript(project)
    assert not (template / "invocation.json").exists()


def test_wrong_or_missing_working_slot_is_rejected(tmp_path, monkeypatch):
    project, template = _project(tmp_path, monkeypatch)
    slot = template / "projects/working/project"
    slot.unlink()
    with pytest.raises(RuntimeError, match="working slot"):
        render_manuscript(project)
    slot.symlink_to(template, target_is_directory=True)
    with pytest.raises(RuntimeError, match="working slot"):
        render_manuscript(project)


def test_missing_runtime_and_native_failure_are_not_success(tmp_path, monkeypatch):
    project, template = _project(tmp_path, monkeypatch)
    monkeypatch.setenv("TEMPLATE_PYTHON", str(template / "absent"))
    with pytest.raises(RuntimeError, match="template Python environment"):
        render_manuscript(project)
    monkeypatch.setenv("TEMPLATE_PYTHON", sys.executable)
    (template / "scripts/pipeline/stage_03_render.py").write_text(
        "raise SystemExit(7)\n"
    )
    with pytest.raises(subprocess.CalledProcessError) as failure:
        render_manuscript(project)
    assert failure.value.returncode == 7
