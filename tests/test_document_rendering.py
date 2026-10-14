"""Real Pandoc checks for the project-owned DOCX cover projection."""

import hashlib
from pathlib import Path
import zipfile

import pytest

from src.document_rendering import render_docx_with_cover
from src.docx_validation import DOCX_ARTIFACT


def test_missing_shared_source_leaves_existing_docx_intact(tmp_path):
    target = tmp_path / DOCX_ARTIFACT
    target.parent.mkdir(parents=True)
    target.write_bytes(b"prior document")
    with pytest.raises(FileNotFoundError, match="shared manuscript first"):
        render_docx_with_cover(tmp_path)
    assert target.read_bytes() == b"prior document"


def test_missing_cover_alternative_leaves_existing_docx_intact(tmp_path):
    (tmp_path / "output/web").mkdir(parents=True)
    (tmp_path / "output/web/_combined_manuscript.md").write_text("# Abstract\n")
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/config.yaml").write_text("paper: {title: Test}\n")
    target = tmp_path / DOCX_ARTIFACT
    target.parent.mkdir()
    target.write_bytes(b"prior document")
    with pytest.raises(ValueError, match="cover and alternative"):
        render_docx_with_cover(tmp_path)
    assert target.read_bytes() == b"prior document"


def test_native_docx_cover_preserves_body_and_scientific_numbering(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cover = root / "manuscript/assets/active_assembly_cover.png"
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/config.yaml").write_text(
        f'paper:\n  title: Cover regression\n  cover:\n    image: "{cover}"\n    alt: Cover regression alternative\n'
    )
    (tmp_path / "output/web").mkdir(parents=True)
    (tmp_path / "output/web/_combined_manuscript.md").write_text(
        f"# Abstract\n\nRetained manuscript body.\n\n![Scientific image]({cover}){{#fig:scientific}}\n\nSee @fig:scientific.\n"
    )
    _bind_shared_composition(tmp_path)
    target = render_docx_with_cover(tmp_path)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        from src.docx_validation import (
            HYDRATED_SOURCE_BINDING_PREFIX,
            _hydrated_manuscript_hash,
        )

        assert (
            f"{HYDRATED_SOURCE_BINDING_PREFIX}{_hydrated_manuscript_hash(tmp_path)}"
            in archive.read("docProps/core.xml").decode()
        )

        xml = archive.read("word/document.xml").decode()
        assert "Cover regression alternative" in xml
        assert "Retained manuscript body." in xml
        assert "Figure 1: Scientific image" in xml
        assert "Figure 2:" not in xml
        assert 'w:type="page"' in xml
        media = [
            archive.read(name)
            for name in archive.namelist()
            if name.startswith("word/media/")
        ]
        assert hashlib.sha256(cover.read_bytes()).digest() in [
            hashlib.sha256(data).digest() for data in media
        ]
    # Re-rendering from the shared source adds one cover again, never stacks
    # an additional cover onto an already exported document.
    render_docx_with_cover(tmp_path)
    with zipfile.ZipFile(target) as archive:
        xml = archive.read("word/document.xml").decode()
        assert xml.count('descr="Cover regression alternative"') == 1


def _bind_shared_composition(project_root):
    from importlib import import_module
    from src.template_hydration import load_template_injection

    source = project_root / "output/web/_combined_manuscript.md"
    hydrated = project_root / "output/manuscript/00_abstract.md"
    hydrated.parent.mkdir(parents=True)
    hydrated.write_text(source.read_text())
    load_template_injection(project_root)
    import_module(
        "infrastructure.rendering.manuscript_composition"
    ).write_manuscript_composition(
        project_root,
        "regression",
        [hydrated],
        source,
        algorithm="shared-combined-markdown-v1",
    )


@pytest.mark.parametrize("change", ["body", "combined", "section"])
def test_stale_shared_composition_cannot_relabel_existing_docx(tmp_path, change):
    root = Path(__file__).resolve().parents[1]
    cover = root / "manuscript/assets/active_assembly_cover.png"
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/config.yaml").write_text(
        f'paper:\n  title: Regression\n  cover:\n    image: "{cover}"\n    alt: Regression cover\n'
    )
    combined = tmp_path / "output/web/_combined_manuscript.md"
    combined.parent.mkdir(parents=True)
    combined.write_text("# Abstract\n\nOriginal body.\n")
    _bind_shared_composition(tmp_path)
    target = tmp_path / DOCX_ARTIFACT
    target.parent.mkdir()
    target.write_bytes(b"prior document")
    if change == "body":
        (tmp_path / "output/manuscript/00_abstract.md").write_text("Changed body.\n")
    elif change == "combined":
        combined.write_text("Changed combined source.\n")
    else:
        (tmp_path / "output/manuscript/01_introduction.md").write_text("New section.\n")
    with pytest.raises(ValueError, match="current hydrated manuscript"):
        render_docx_with_cover(tmp_path)
    assert target.read_bytes() == b"prior document"


def test_native_docx_uses_canonical_images_when_snapshots_shadow_resource_paths(
    tmp_path,
):
    import base64

    root = Path(__file__).resolve().parents[1]
    cover = root / "manuscript/assets/active_assembly_cover.png"
    canonical = tmp_path / "output/figures/scientific.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(cover.read_bytes())
    stale = tmp_path / "figures/scientific.png"
    stale.parent.mkdir()
    stale.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/config.yaml").write_text(
        f'paper:\n  title: Canonical image regression\n  cover:\n    image: "{cover}"\n    alt: Source cover\n'
    )
    combined = tmp_path / "output/web/_combined_manuscript.md"
    combined.parent.mkdir()
    combined.write_text(
        "# Abstract\n\n![Scientific image](../figures/scientific.png){#fig:scientific}\n"
    )
    _bind_shared_composition(tmp_path)
    target = render_docx_with_cover(tmp_path)
    with zipfile.ZipFile(target) as archive:
        digests = {
            hashlib.sha256(archive.read(name)).digest()
            for name in archive.namelist()
            if name.startswith("word/media/")
        }
        assert hashlib.sha256(canonical.read_bytes()).digest() in digests
        assert hashlib.sha256(stale.read_bytes()).digest() not in digests
        assert (
            "Figure 1: Scientific image" in archive.read("word/document.xml").decode()
        )
