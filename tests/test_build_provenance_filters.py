"""Executable document filters must invalidate build provenance on change."""

from src.build_provenance import verify_build_provenance, write_build_provenance
import pytest


@pytest.mark.parametrize("filename", ["uv.lock", "pyproject.toml", ".python-version"])
def test_template_environment_change_invalidates_build_receipt(
    tmp_path, monkeypatch, filename
):
    project = tmp_path / "project"
    project.mkdir()
    template = tmp_path / "template"
    source = template / filename
    template.mkdir()
    marker = template / "infrastructure/rendering/manuscript_injection.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("pass\n")
    source.write_text("version-one\n")
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    write_build_provenance(project)
    assert verify_build_provenance(project) == []
    source.write_text("version-two\n")
    assert "template implementation hashes mismatch" in verify_build_provenance(project)


def test_project_lua_filter_change_invalidates_build_receipt(tmp_path, monkeypatch):
    project = tmp_path / "project"
    template = tmp_path / "template"
    template.mkdir()
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    source = project / "src/docx_cover.lua"
    source.parent.mkdir(parents=True)
    source.write_text("return {}\n")
    write_build_provenance(project)
    assert verify_build_provenance(project) == []
    source.write_text("return {Pandoc = function(doc) return doc end}\n")
    assert "source_hashes stale=src/docx_cover.lua" in verify_build_provenance(project)


def test_template_lua_filter_change_invalidates_build_receipt(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    template = tmp_path / "template"
    source = template / "infrastructure/rendering/filter.lua"
    source.parent.mkdir(parents=True)
    (source.parent / "manuscript_injection.py").write_text(
        '"""Transparent hydration marker for the filesystem lineage test."""\n'
    )
    source.write_text("return {}\n")
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    write_build_provenance(project)
    assert verify_build_provenance(project) == []
    source.write_text("return {Pandoc = function(doc) return doc end}\n")
    assert "template implementation hashes mismatch" in verify_build_provenance(project)
