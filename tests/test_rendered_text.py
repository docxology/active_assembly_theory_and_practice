from pathlib import Path
import subprocess

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.pdf_accessibility import rendered_pdf_text_errors


def _text_pdf(path: Path, text: str, x: int = 30, y: int = 100) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf {x} {y} Td ({escaped}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as handle:
        writer.write(handle)


@pytest.mark.parametrize(
    ("text", "x", "y", "expected"),
    [
        ("Readable bounded text", 30, 100, None),
        ("Overflow", 295, 100, "word outside paper"),
        ("Overflow", -2, 100, "word outside paper"),
        ("Overflow", 30, 199, "word outside paper"),
        ("Overflow", 30, 1, "word outside paper"),
        ("((def:model-evidence?))", 30, 100, "unresolved reference"),
        ("Section sec:measured-sensing", 30, 100, "unresolved reference"),
        ("(prop:transition-collapse?)", 30, 100, "unresolved reference"),
    ],
)
def test_real_pdf_word_endings_and_reference_controls(tmp_path, text, x, y, expected):
    pdf = tmp_path / "text.pdf"
    _text_pdf(pdf, text, x, y)
    errors = rendered_pdf_text_errors(pdf)
    if expected is None:
        assert errors == []
    else:
        assert any(expected in error for error in errors), errors


def test_pdf_text_extraction_failure_is_not_a_pass(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        rendered_pdf_text_errors(tmp_path / "absent.pdf")


@pytest.mark.post_render
def test_current_manuscript_and_all_slide_text_fit_without_unresolved_references():
    root = Path(__file__).resolve().parents[1]
    slides = sorted((root / "output/slides").glob("*_slides.pdf"))
    assert len(slides) == 8
    for pdf in [
        root / "output/pdf/active_assembly_theory_and_practice_combined.pdf",
        *slides,
    ]:
        assert pdf.is_file(), pdf
        assert (errors := rendered_pdf_text_errors(pdf)) == [], (pdf, errors)
