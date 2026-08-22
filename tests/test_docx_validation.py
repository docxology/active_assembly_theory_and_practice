"""Tests for the source-bound, post-render DOCX integrity lane."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.docx_validation import (
    DOCX_ARTIFACT,
    DOCX_RECEIPT_ARTIFACT,
    docx_render_payload,
    validate_docx_render_payload,
    validate_docx_render_receipt,
    write_docx_render_receipt,
)


def _write_fake_docx(
    project_root: Path, figure_count: int = 2, *, cover: bool = False
) -> None:
    """Create a minimal transparent OOXML package for contract tests."""

    output = project_root / "output"
    (output / "data").mkdir(parents=True)
    (output / "manuscript").mkdir(parents=True)
    (output / "docx").mkdir(parents=True)
    (output / "manuscript" / "00_abstract.md").write_text(
        "hydrated source\n", encoding="utf-8"
    )
    (output / "data" / "figure_registry.json").write_text(
        json.dumps(
            {"figures": [{"filename": f"image{i}.png"} for i in range(figure_count)]}
        ),
        encoding="utf-8",
    )
    (output / "data" / "manuscript_variable_registry.json").write_text(
        json.dumps(
            {
                "summary": {
                    "all_source_tokens_generated": True,
                    "no_pending_values": True,
                },
                "template_hydration": {
                    "status": "hydrated",
                    "template_contract": "infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree",
                    "template_repo_root": "/template",
                    "template_revision": "abc123",
                    "unresolved_tokens": [],
                },
            }
        ),
        encoding="utf-8",
    )
    if cover:
        cover_path = project_root / "manuscript" / "assets" / "cover.png"
        cover_path.parent.mkdir(parents=True)
        cover_path.write_bytes(b"COVER PNG")
        (project_root / "manuscript" / "config.yaml").write_text(
            "paper:\n"
            "  cover:\n"
            "    image: assets/cover.png\n"
            "    alt: Test cover alternative.\n",
            encoding="utf-8",
        )
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    paragraphs = [
        "Active Assembly Theory and Practice",
        "Cross-architecture adapter falsification surface",
        "Formal description",
        "Exact finite-horizon information-per-cost analysis",
        *[f"Figure {i}: A transparent caption." for i in range(1, figure_count + 1)],
    ]
    cover_xml = (
        '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        '<w:r><w:drawing><wp:inline><wp:docPr descr="Test cover alternative." />'
        "</wp:inline></w:drawing></w:r></w:p>"
        if cover
        else ""
    )
    body = "".join(
        f'<w:p xmlns:w="{namespace}"><w:r><w:t>{text}</w:t></w:r></w:p>'
        for text in paragraphs
    )
    document = (
        f'<w:document xmlns:w="{namespace}"><w:body>{cover_xml}{body}</w:body></w:document>'
    ).encode("utf-8")
    relationship_namespace = (
        "http://schemas.openxmlformats.org/package/2006/relationships"
    )
    media_names = (["cover.png"] if cover else []) + [
        f"image{i}.png" for i in range(figure_count)
    ]
    relationships = "".join(
        f'<Relationship xmlns="{relationship_namespace}" Id="rId{i}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/{name}"/>'
        for i, name in enumerate(media_names)
    )
    relationship_xml = (
        f'<Relationships xmlns="{relationship_namespace}">{relationships}</Relationships>'
    ).encode("utf-8")
    artifact = project_root / DOCX_ARTIFACT
    with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("_rels/.rels", "<Relationships/>")
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", relationship_xml)
        archive.writestr("word/styles.xml", "<styles/>")
        archive.writestr("docProps/core.xml", "<coreProperties/>")
        for name in media_names:
            archive.writestr(
                f"word/media/{name}",
                b"COVER PNG" if name == "cover.png" else b"PNG fixture",
            )


def test_docx_contract_accepts_complete_fake_package(tmp_path: Path) -> None:
    _write_fake_docx(tmp_path)

    payload = docx_render_payload(tmp_path)

    assert payload["valid"] is True
    assert validate_docx_render_payload(payload) == []
    assert payload["embedded_media_count"] == 2
    assert payload["image_relationship_count"] == 2
    assert payload["caption_count"] == 2
    assert payload["unresolved_tokens"] == []

    receipt_path = write_docx_render_receipt(tmp_path)
    assert receipt_path == tmp_path / DOCX_RECEIPT_ARTIFACT
    assert validate_docx_render_receipt(tmp_path, payload) == []

    registry_path = tmp_path / "output/data/figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["generated_at"] = "2030-01-01T00:00:00+00:00"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert validate_docx_render_receipt(tmp_path, docx_render_payload(tmp_path)) == []


def test_docx_contract_requires_and_binds_configured_cover(tmp_path: Path) -> None:
    """A configured cover is counted separately and bound by bytes and alt text."""

    _write_fake_docx(tmp_path, cover=True)

    payload = docx_render_payload(tmp_path)

    assert payload["valid"] is True
    assert payload["expected_media_count"] == 3
    assert payload["embedded_media_count"] == 3
    assert payload["cover_media_count"] == 1
    assert payload["cover_alt_present"] is True
    assert validate_docx_render_payload(payload) == []


def test_docx_contract_rejects_cover_and_media_drift(tmp_path: Path) -> None:
    """Cover and figure parity failures remain fail-closed after rendering."""

    _write_fake_docx(tmp_path, cover=True)
    payload = docx_render_payload(tmp_path)
    payload.update(
        {
            "expected_media_count": 3,
            "embedded_media_count": 2,
            "cover_media_count": 0,
            "cover_alt_present": False,
            "image_relationship_count": 2,
            "caption_count": 0,
        }
    )

    errors = validate_docx_render_payload(payload)

    assert "embedded media count mismatch: observed=2 expected=3" in errors
    assert "configured cover media mismatch: observed=0 expected=1" in errors
    assert "configured cover alternative text is missing from DOCX" in errors
    assert "caption count mismatch: observed=0 expected=2" in errors


def test_docx_receipt_detects_hydrated_source_drift(tmp_path: Path) -> None:
    _write_fake_docx(tmp_path)
    write_docx_render_receipt(tmp_path)

    source_path = tmp_path / "output/manuscript/00_abstract.md"
    source_path.write_text("changed hydrated source\n", encoding="utf-8")
    current = docx_render_payload(tmp_path)

    errors = validate_docx_render_receipt(tmp_path, current)

    assert "DOCX receipt stale=hydrated_manuscript_sha256" in errors


def test_docx_contract_fails_closed_when_artifact_is_missing(tmp_path: Path) -> None:
    payload = docx_render_payload(tmp_path)

    assert payload["valid"] is False
    assert f"missing {DOCX_ARTIFACT}" in validate_docx_render_payload(payload)
    assert validate_docx_render_receipt(tmp_path, payload) == [
        f"missing {DOCX_RECEIPT_ARTIFACT}"
    ]
