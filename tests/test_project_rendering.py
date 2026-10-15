import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from src.artifact_contract import write_json_artifact
from src.build_provenance import build_provenance_payload
from src.document_rendering import render_docx_with_cover
from src.project_rendering import render_project
from src.template_hydration import load_template_injection

ROOT = Path(__file__).resolve().parents[1]


def project(root):
    (root / "manuscript").mkdir()
    (root / "manuscript/00_body.md").write_text("# Current root\n\n{{SENTINEL}}\n")
    (root / "manuscript/config.yaml").write_text(
        "paper:\n  title: Explicit-root regression\n"
    )
    write_json_artifact(
        root / "output/data/manuscript_variables.json",
        {"SENTINEL": "CURRENT_ROOT_7319"},
        project_root=root,
    )
    load_template_injection(root)(root, {"SENTINEL": "CURRENT_ROOT_7319"})


def render_child(root, environment=None):
    executable = os.environ.get("TEMPLATE_PYTHON")
    if not executable:
        pytest.skip("real native export requires prepared TEMPLATE_PYTHON")
    code = (
        "import sys,json; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); "
        "from src.project_rendering import render_project; "
        "before=sys.path[:]; bytecode=sys.dont_write_bytecode; "
        "result=render_project(Path(sys.argv[2]),formats=('html','docx'),"
        "project_name='stable-name'); "
        "assert sys.path==before and sys.dont_write_bytecode==bytecode; "
        "print(json.dumps(result))"
    )
    return subprocess.run(
        [executable, "-I", "-B", "-c", code, str(ROOT), str(root)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=120,
    )


def test_actual_template_exports_current_root_without_registration(tmp_path):
    project(tmp_path)
    result = render_child(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["project_root"] == str(tmp_path.resolve())
    assert receipt["skip_manuscript_hydration"] is True
    assert len(receipt["rendered_hashes"]) == 2
    assert "CURRENT_ROOT_7319" in (tmp_path / "output/web/index.html").read_text()
    with zipfile.ZipFile(tmp_path / "output/docx/stable-name_combined.docx") as archive:
        assert b"CURRENT_ROOT_7319" in archive.read("word/document.xml")


def test_preamble_stays_bound_without_entering_reader_or_slide_compositions(tmp_path):
    project(tmp_path)
    assets = tmp_path / "manuscript/assets"
    assets.mkdir()
    cover = assets / "cover.png"
    shutil.copyfile(ROOT / "manuscript/assets/active_assembly_cover.png", cover)
    (tmp_path / "manuscript/config.yaml").write_text(
        "paper:\n  title: Preamble regression\n  cover:\n"
        f"    image: {json.dumps(str(cover))}\n    alt: Owned cover alternative\n"
    )
    preamble = tmp_path / "manuscript/preamble.md"
    preamble.write_text(
        "# PREAMBLE_INTERNAL_8391\n\n```latex\n\\hypersetup{pdfdisplaydoctitle=true}\n```\n"
    )
    before = preamble.read_bytes()
    load_template_injection(tmp_path)(tmp_path, {"SENTINEL": "CURRENT_ROOT_7319"})
    report = render_project(
        tmp_path,
        formats=("pdf", "html", "docx", "slides"),
        project_name="active_assembly_theory_and_practice",
    )
    assert len(report["rendered_hashes"]) == 4
    assert "manuscript/preamble.md" in report["source_hashes"]
    assert "output/manuscript/preamble.md" in report["hydrated_hashes"]
    assert not (tmp_path / "output/slides/preamble_slides.pdf").exists()
    assert (
        "PREAMBLE_INTERNAL_8391" not in (tmp_path / "output/web/index.html").read_text()
    )
    target = render_docx_with_cover(tmp_path)
    with zipfile.ZipFile(target) as archive:
        body = archive.read("word/document.xml")
        assert b"CURRENT_ROOT_7319" in body
        assert b"PREAMBLE_INTERNAL_8391" not in body
        assert b"Owned cover alternative" in body
    assert preamble.read_bytes() == before


def test_direct_native_export_preserves_imports_and_current_root(tmp_path):
    project(tmp_path)
    before = sys.path[:]
    bytecode = sys.dont_write_bytecode
    scripts = sys.modules.get("scripts")
    receipt = render_project(
        tmp_path, formats=("html", "docx"), project_name="direct-edition"
    )
    assert sys.path == before
    assert sys.dont_write_bytecode == bytecode
    assert sys.modules.get("scripts") is scripts
    assert len(receipt["rendered_hashes"]) == 2
    assert "CURRENT_ROOT_7319" in (tmp_path / "output/web/index.html").read_text()
    with zipfile.ZipFile(
        tmp_path / "output/docx/direct-edition_combined.docx"
    ) as archive:
        assert b"CURRENT_ROOT_7319" in archive.read("word/document.xml")


def test_real_missing_pandoc_cannot_reuse_existing_edition(tmp_path):
    project(tmp_path)
    first = render_child(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    receipt = tmp_path / "output/reports/project_render.json"
    before = receipt.read_bytes()
    environment = {**os.environ, "PANDOC_PATH": str(tmp_path / "absent-pandoc")}
    failed = render_child(tmp_path, environment)
    assert failed.returncode != 0
    assert "FileNotFoundError" in failed.stderr
    assert receipt.read_bytes() == before


def test_stale_hydration_fails_before_output_write(tmp_path):
    project(tmp_path)
    (tmp_path / "output/manuscript/00_body.md").write_text("wrong root")
    with pytest.raises(RuntimeError, match="hydration contract failed"):
        render_project(tmp_path, formats=("html",))
    assert not (tmp_path / "output/web/index.html").exists()


@pytest.mark.parametrize("formats", [(), ("unknown",), ("html", "html")])
def test_invalid_formats_fail_before_output_write(tmp_path, formats):
    with pytest.raises(ValueError, match="RENDER_FORMAT_INVALID"):
        render_project(tmp_path, formats=formats)
    assert list(tmp_path.iterdir()) == []


def test_docx_requires_current_composition(tmp_path):
    with pytest.raises(ValueError, match="RENDER_DOCX_REQUIRES_COMPOSITION"):
        render_project(tmp_path, formats=("docx",))


def test_output_escape_fails_before_foreign_write(tmp_path, tmp_path_factory):
    project(tmp_path)
    foreign = tmp_path_factory.mktemp("foreign-render")
    (foreign / "sentinel").write_bytes(b"preserved")
    (tmp_path / "output/web").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RuntimeError, match="RENDER_PATH_ESCAPE"):
        render_project(tmp_path, formats=("html",))
    assert list(foreign.iterdir()) == [foreign / "sentinel"]
    assert (foreign / "sentinel").read_bytes() == b"preserved"


@pytest.mark.parametrize("name", ["..", "../foreign"])
def test_invalid_logical_name_fails(tmp_path, name):
    with pytest.raises(ValueError, match="RENDER_PROJECT_NAME_INVALID"):
        render_project(tmp_path, project_name=name)


def test_template_loading_preserves_project_import_path(tmp_path):
    before = sys.path[:]
    bytecode = sys.dont_write_bytecode
    scripts = sys.modules.get("scripts")
    load_template_injection(tmp_path)
    assert sys.path == before
    assert sys.dont_write_bytecode == bytecode
    assert sys.modules.get("scripts") is scripts


def test_gnn_sources_and_outputs_enter_build_lineage(tmp_path):
    source = tmp_path / "configs/gnn/model.md"
    source.parent.mkdir(parents=True)
    source.write_text("model-v1")
    output = tmp_path / "output/gnn/model.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    first = build_provenance_payload(tmp_path)["lineage"]
    source.write_text("model-v2")
    second = build_provenance_payload(tmp_path)["lineage"]
    assert (
        first["source_hashes"]["configs/gnn/model.md"]
        != second["source_hashes"]["configs/gnn/model.md"]
    )
    assert (
        first["generated_hashes"]["output/gnn/model.json"]
        == second["generated_hashes"]["output/gnn/model.json"]
    )
