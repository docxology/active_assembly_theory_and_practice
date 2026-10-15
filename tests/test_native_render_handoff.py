import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from src.document_rendering import render_manuscript


def _project(tmp_path):
    project = tmp_path / "project"
    (project / "manuscript").mkdir(parents=True)
    (project / "manuscript/00_abstract.md").write_text("# {{TITLE}}\n")
    (project / "manuscript/config.yaml").write_text(
        "paper:\n  title: Native handoff regression\n"
    )
    (project / "output/manuscript").mkdir(parents=True)
    (project / "output/manuscript/00_abstract.md").write_text("# CURRENT_ROOT_7319\n")
    (project / "output/manuscript/config.yaml").write_bytes(
        (project / "manuscript/config.yaml").read_bytes()
    )
    (project / "output/data").mkdir()
    (project / "output/data/manuscript_variables.json").write_text(
        '{"TITLE": "CURRENT_ROOT_7319"}'
    )
    return project


def test_native_handoff_preserves_hydration_and_uses_explicit_root(tmp_path):
    project = _project(tmp_path)
    before = (project / "output/manuscript/00_abstract.md").read_bytes()
    render_manuscript(project)
    report = json.loads(
        (project / "output/data/project_render_receipt.json").read_text()
    )
    assert report["project_root"] == str(project.resolve())
    assert report["template_root"] == str(
        Path(os.environ["TEMPLATE_REPO_ROOT"]).resolve()
    )
    assert report["skip_manuscript_hydration"] is True
    assert set(report["formats"]) == {"pdf", "html", "docx", "slides"}
    assert len(report["rendered_hashes"]) == 4
    assert (project / "output/manuscript/00_abstract.md").read_bytes() == before
    assert "CURRENT_ROOT_7319" in (project / "output/web/index.html").read_text()
    with zipfile.ZipFile(
        project / "output/docx/active_assembly_theory_and_practice_combined.docx"
    ) as archive:
        assert b"CURRENT_ROOT_7319" in archive.read("word/document.xml")


def test_stale_hydration_never_reaches_template(tmp_path):
    project = _project(tmp_path)
    (project / "manuscript/00_abstract.md").write_text("# Changed\n")
    with pytest.raises(RuntimeError, match="hydration contract failed"):
        render_manuscript(project)
    assert not (project / "output/reports/project_render.json").exists()


def test_incomplete_template_cannot_relabel_existing_exports(tmp_path, monkeypatch):
    project = _project(tmp_path)
    template = tmp_path / "template"
    marker = template / "infrastructure/rendering/manuscript_injection.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    target = project / "output/docx/prior.docx"
    target.parent.mkdir()
    target.write_bytes(b"prior document")
    with pytest.raises(subprocess.CalledProcessError):
        render_manuscript(project)
    assert target.read_bytes() == b"prior document"
    assert not (project / "output/data/project_render_receipt.json").exists()


def test_missing_runtime_and_native_failure_are_not_success(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setenv("TEMPLATE_PYTHON", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="template Python environment"):
        render_manuscript(project)
    runtime = tmp_path / "failed-runtime"
    runtime.write_text("#!/bin/sh\nexit 7\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("TEMPLATE_PYTHON", str(runtime))
    with pytest.raises(subprocess.CalledProcessError) as failure:
        render_manuscript(project)
    assert failure.value.returncode == 7


def test_successful_process_without_fresh_receipt_is_rejected(tmp_path, monkeypatch):
    project = _project(tmp_path)
    runtime = tmp_path / "no-output-runtime"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("TEMPLATE_PYTHON", str(runtime))
    report = project / "output/reports/project_render.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"project_root": str(project.resolve())}))
    sealed = project / "output/data/project_render_receipt.json"
    sealed.write_bytes(b"prior sealed receipt")
    before = report.read_bytes(), sealed.read_bytes()
    with pytest.raises(RuntimeError, match="RENDER_REPORT_NOT_FRESH"):
        render_manuscript(project)
    assert (report.read_bytes(), sealed.read_bytes()) == before


@pytest.fixture(scope="module")
def exported_project(tmp_path_factory):
    project = _project(tmp_path_factory.mktemp("native-parent-export"))
    render_manuscript(project)
    return project


@pytest.mark.parametrize(
    "field",
    [
        "project_root",
        "template_root",
        "source_hashes",
        "hydrated_hashes",
        "variables_sha256",
        "rendered_hashes",
        "formats",
        "stale_output",
    ],
)
def test_parent_rejects_fresh_but_corrupted_export_reports(
    tmp_path, monkeypatch, exported_project, field
):
    project = tmp_path / "project"
    shutil.copytree(exported_project, project)
    report_path = project / "output/reports/project_render.json"
    report = json.loads(report_path.read_text())
    report["project_root"] = str(project.resolve())
    outputs = list(report["rendered_hashes"])
    if field in {"project_root", "template_root"}:
        report[field] = str(tmp_path / "foreign")
    elif field in {"source_hashes", "hydrated_hashes", "rendered_hashes"}:
        report[field] = {}
    elif field == "variables_sha256":
        report[field] = "0" * 64
    elif field == "formats":
        report[field] = ["html"]
    if field == "stale_output":
        os.utime(project / outputs[0], ns=(1, 1))
        outputs = outputs[1:]
    runtime = tmp_path / "corrupt-export-runtime"
    runtime.write_text(
        f"#!{sys.executable}\nimport os\nfrom pathlib import Path\n"
        + "\n".join(
            f"os.utime({str(project / relative)!r}, None)" for relative in outputs
        )
        + f"\nPath({str(report_path)!r}).write_text({json.dumps(report)!r})\n"
    )
    runtime.chmod(0o755)
    monkeypatch.setenv("TEMPLATE_PYTHON", str(runtime))
    sealed = project / "output/data/project_render_receipt.json"
    before = sealed.read_bytes()
    with pytest.raises(RuntimeError, match="RENDER_"):
        render_manuscript(project)
    assert sealed.read_bytes() == before
