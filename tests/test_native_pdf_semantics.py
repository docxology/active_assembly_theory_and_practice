import hashlib
import json
from pathlib import Path
import shutil

import pytest
from pypdf import PdfWriter
from pypdf.generic import NameObject

from src.artifact_contract import write_json_artifact
from src.pdf_accessibility import (
    _walk_structure,
    audit_pdf,
    write_pdf_accessibility_report,
)
from src.project_rendering import render_project
from src.template_hydration import load_template_injection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def semantic_pdf(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-pdf-semantics")
    assets = root / "manuscript/assets"
    assets.mkdir(parents=True)
    cover = assets / "cover.png"
    shutil.copyfile(ROOT / "manuscript/assets/active_assembly_cover.png", cover)
    figures = root / "output/figures"
    figures.mkdir(parents=True)
    image = figures / "owned.png"
    shutil.copyfile(cover, image)
    (root / "manuscript/config.yaml").write_text(
        "paper:\n  title: Native PDF semantic regression\n  cover:\n"
        f"    image: {json.dumps(str(cover))}\n    alt: Owned cover alternative\n"
        "metadata:\n  language: en\n  tagged_pdf: true\n"
    )
    (root / "manuscript/00_body.md").write_text(
        '---\nlang: en\npdfstandard:\n  version: "2.0"\n'
        "  standards: [UA-2]\n  tagging: true\n---\n\n"
        "# {{TITLE}}\n\n## Evidence\n\n### Detail\n\n"
        f"![Owned fixture image]({image}){{#fig:owned width=35%}}\n\n"
        "See @fig:owned and [the source](https://example.invalid/owned).\n\n"
        "$$\\frac{1}{2}+\\frac{1}{2}=1$$\n\n"
        "| State | Probability |\n| --- | --- |\n| first | 0.5 |\n| second | 0.5 |\n"
    )
    data = root / "output/data"
    data.mkdir()
    (data / "figure_registry.json").write_text(
        json.dumps({"figures": [{"alt_text": "Owned fixture image"}]})
    )
    values = {"TITLE": "Owned semantic surface"}
    write_json_artifact(data / "manuscript_variables.json", values, project_root=root)
    load_template_injection(root)(root, values)
    render_project(root, formats=("pdf",), project_name="semantics")
    return root, root / "output/pdf/semantics_combined.pdf"


def test_actual_native_pdf_has_complete_semantic_surface(semantic_pdf):
    root, pdf = semantic_pdf
    report = audit_pdf(pdf, project_root=root)
    assert report["summary"]["all_passed"], report["checks"]
    assert report["metrics"]["figure_tag_count"] >= 2
    assert report["metrics"]["formula_tag_count"] > 0
    assert report["metrics"]["link_annotation_count"] > 0
    target = root / "output/data/pdf_accessibility_report.json"
    write_pdf_accessibility_report(pdf, target, project_root=root)
    assert json.loads(target.read_text())["summary"]["all_passed"]


@pytest.mark.parametrize("case", ["figure", "formula", "heading", "link", "table"])
def test_real_pdf_semantic_loss_is_rejected(semantic_pdf, tmp_path, case):
    root, original = semantic_pdf
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    writer = PdfWriter(clone_from=original)
    structure = writer._root_object["/StructTreeRoot"].get_object()
    nodes = list(_walk_structure(structure))
    if case == "figure":
        node = next(n for n in nodes if str(n.get("/Alt", "")) == "Owned fixture image")
        del node["/Alt"]
        failed = "figure_alternatives"
    elif case == "formula":
        node = next(n for n in nodes if n.get("/S") == "/Formula")
        del node["/AF"]
        failed = "formula_associated_files"
    elif case == "heading":
        structure["/RoleMap"].get_object()[NameObject("/section")] = NameObject("/H3")
        failed = "heading_hierarchy"
    elif case == "table":
        for node in nodes:
            if node.get("/S") == "/Table":
                node[NameObject("/S")] = NameObject("/P")
        failed = "table_structure"
    else:
        link = next(
            ref.get_object()
            for page in writer.pages
            for ref in page.get("/Annots", [])
            if ref.get_object().get("/Subtype") == "/Link"
        )
        del link["/StructParent"]
        failed = "structured_links"
    changed = tmp_path / "changed.pdf"
    with changed.open("wb") as handle:
        writer.write(handle)
    report = audit_pdf(changed, project_root=root)
    rows = {row["name"]: row for row in report["checks"]}
    assert rows[failed]["passed"] is False
    assert report["summary"]["all_passed"] is False
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_missing_pdf_is_a_failed_report(tmp_path):
    report = audit_pdf(tmp_path / "absent.pdf", project_root=tmp_path)
    assert report["summary"] == {"passed": 0, "failed": 1, "all_passed": False}
