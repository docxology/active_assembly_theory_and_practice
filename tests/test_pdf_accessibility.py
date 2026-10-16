"""Tests for the source-bound rendered-PDF accessibility audit."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import BooleanObject, DictionaryObject, NameObject, TextStringObject

from src.pdf_accessibility import audit_pdf


@pytest.mark.parametrize("marked", [False, True])
def test_pdf_audit_rejects_language_or_mark_flag_as_accessibility_proof(
    tmp_path, marked
):
    # A valid PDF, even one claiming tagging, needs an actual semantic tree.
    pdf = tmp_path / "document.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=200)
    writer._root_object[NameObject("/Lang")] = TextStringObject("en")
    writer._root_object[NameObject("/MarkInfo")] = DictionaryObject(
        {NameObject("/Marked"): BooleanObject(marked)}
    )
    with pdf.open("wb") as handle:
        writer.write(handle)
    report = audit_pdf(pdf, project_root=tmp_path)
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["pdf_opened"]["passed"] is True
    assert checks["document_language"]["passed"] is True
    assert checks["marked_pdf"]["passed"] is marked
    assert checks["structure_tree"]["passed"] is False
    assert checks["parent_tree"]["passed"] is False
    assert report["summary"]["all_passed"] is False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "output/pdf/active_assembly_theory_and_practice_combined.pdf"


@pytest.mark.post_render
def test_rendered_pdf_has_audited_semantic_accessibility_surface() -> None:
    assert PDF_PATH.is_file(), "render the current native manuscript before this gate"
    report = audit_pdf(PDF_PATH, project_root=PROJECT_ROOT)

    assert report["summary"]["all_passed"], report["checks"]
    assert report["metrics"]["figure_tag_count"] >= 32
    assert report["metrics"]["formula_tag_count"] > 0
    assert report["metrics"]["link_annotation_count"] > 0
