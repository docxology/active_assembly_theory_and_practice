"""Post-render integrity checks for the template-generated DOCX artifact.

The DOCX is a format-specific delivery surface, not a substitute for the
tagged PDF audit.  This module therefore checks only properties that can be
established from the OOXML package itself: ZIP integrity, required parts,
source-bound hydration metadata, embedded-figure parity, caption parity,
resolved template tokens, and a small set of manuscript anchors.  A receipt
also binds the document to the hydrated manuscript and figure registry that
were present when the template renderer ran.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile
from xml.etree import ElementTree

import yaml

from .artifact_contract import write_json_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCX_SCHEMA_VERSION = "1.0.0"
DOCX_ARTIFACT = "output/docx/active_assembly_theory_and_practice_combined.docx"
DOCX_RECEIPT_ARTIFACT = "output/data/docx_render_receipt.json"

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_TEXT_TAG = f"{{{_WORD_NAMESPACE}}}t"
_PARAGRAPH_TAG = f"{{{_WORD_NAMESPACE}}}p"
_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_CAPTION_RE = re.compile(r"^Figure\s+\d+:\s+")

REQUIRED_DOCX_PARTS: tuple[str, ...] = (
    "[Content_Types].xml",
    "_rels/.rels",
    "word/document.xml",
    "word/_rels/document.xml.rels",
    "word/styles.xml",
    "docProps/core.xml",
)

REQUIRED_TEXT_ANCHORS: dict[str, str] = {
    "title": "Active Assembly Theory and Practice",
    "cross_architecture": "Cross-architecture adapter falsification surface",
    "formalization": "Formal description",
    "information_policy": "Exact finite-horizon information-per-cost analysis",
}

# These two figures summarize validator state and therefore change while the
# validation receipt is being regenerated. Including either in the DOCX
# lineage digest would create a self-referential cycle: validating the receipt
# would rewrite the figure, which would invalidate the receipt. Their own
# nonblankness, caption, and registry checks remain active elsewhere.
_VALIDATOR_PRODUCED_FIGURES = frozenset(
    {"artifact_hash_manifest", "validation_summary"}
)


def _sha256_file(path: Path) -> str | None:
    """Return a file digest, or ``None`` when the file is absent."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_named_files(paths: tuple[Path, ...], root: Path) -> str | None:
    """Hash a deterministic named-file collection with path boundaries."""

    if not paths or not all(path.is_file() for path in paths):
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _hydrated_manuscript_hash(project_root: Path) -> str | None:
    """Hash the hydrated Markdown tree used by the renderer."""

    paths = tuple(sorted((project_root / "output" / "manuscript").glob("*.md")))
    return _sha256_named_files(paths, project_root)


def _figure_registry_hash(project_root: Path) -> str | None:
    """Hash the canonical figure registry consumed by template rendering."""

    path = project_root / "output" / "data" / "figure_registry.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # The registry timestamp records production time but is not document
    # content. Excluding only this volatile field keeps the receipt sensitive
    # to figure bytes, captions, alternatives, inputs, and checks while
    # remaining stable across an otherwise identical regeneration.
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("generated_at", None)
        figures = payload.get("figures")
        if isinstance(figures, list):
            payload["figures"] = [
                row
                for row in figures
                if isinstance(row, dict)
                and row.get("figure_id") not in _VALIDATOR_PRODUCED_FIGURES
            ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _figure_registry_count(project_root: Path) -> tuple[int | None, str | None]:
    """Return the canonical registry count and a parse diagnostic."""

    path = project_root / "output" / "data" / "figure_registry.json"
    if not path.is_file():
        return None, "missing output/data/figure_registry.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    figures = payload.get("figures")
    if not isinstance(figures, list) or not figures:
        return None, "figure registry has no non-empty figures list"
    return len(figures), None


def _configured_cover(project_root: Path) -> tuple[Path | None, str | None]:
    """Resolve the source-configured cover image and its alternative text."""

    config_path = project_root / "manuscript" / "config.yaml"
    if not config_path.is_file():
        return None, None
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, None
    if not isinstance(config, dict):
        return None, None
    for section_name in ("book", "paper"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        cover = section.get("cover")
        if not isinstance(cover, dict) or not cover.get("image"):
            continue
        raw_image = Path(str(cover["image"]))
        candidates = (
            raw_image if raw_image.is_absolute() else config_path.parent / raw_image,
            project_root / raw_image,
            project_root / "output" / raw_image,
        )
        image_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
        alt = cover.get("alt")
        return image_path, str(alt).strip() if isinstance(alt, str) else None
    return None, None


def _hydration_snapshot(project_root: Path) -> dict[str, Any]:
    """Return the template-hydration facts recorded by the variable registry."""

    path = project_root / "output" / "data" / "manuscript_variable_registry.json"
    if not path.is_file():
        return {
            "status": "missing",
            "template_contract": "unavailable",
            "template_repo_root": "unavailable",
            "template_revision": "unavailable",
            "unresolved_tokens": [],
            "summary": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unreadable",
            "template_contract": "unavailable",
            "template_repo_root": "unavailable",
            "template_revision": "unavailable",
            "unresolved_tokens": [str(exc)],
            "summary": {},
        }
    hydration = payload.get("template_hydration", {})
    return {
        "status": hydration.get("status", "unknown"),
        "template_contract": hydration.get("template_contract", "unknown"),
        "template_repo_root": hydration.get("template_repo_root", "unknown"),
        "template_revision": hydration.get("template_revision", "unknown"),
        "unresolved_tokens": hydration.get("unresolved_tokens", []),
        "summary": payload.get("summary", {}),
    }


def _document_text(document_xml: bytes) -> tuple[str, tuple[str, ...]]:
    """Extract paragraph text and joined text nodes from ``document.xml``."""

    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(_PARAGRAPH_TAG):
        paragraphs.append(
            "".join(node.text or "" for node in paragraph.iter(_TEXT_TAG))
        )
    text_nodes = tuple(node.text or "" for node in root.iter(_TEXT_TAG))
    return " ".join(paragraphs), text_nodes


def _image_relationship_count(relationships_xml: bytes) -> int:
    """Count document relationships targeting embedded media."""

    root = ElementTree.fromstring(relationships_xml)
    count = 0
    for relationship in root.findall(f"{{{_RELATIONSHIP_NAMESPACE}}}Relationship"):
        target = str(relationship.attrib.get("Target", ""))
        rel_type = str(relationship.attrib.get("Type", ""))
        if (
            "/media/" in target
            or target.startswith("media/")
            or rel_type.endswith("/image")
        ):
            count += 1
    return count


def docx_render_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Inspect the current DOCX and return a JSON-safe diagnostic payload."""

    artifact_path = project_root / DOCX_ARTIFACT
    expected_figure_count, figure_registry_error = _figure_registry_count(project_root)
    cover_path, cover_alt = _configured_cover(project_root)
    cover_sha256 = _sha256_file(cover_path) if cover_path is not None else None
    expected_media_count = (
        expected_figure_count + int(cover_path is not None)
        if expected_figure_count is not None
        else None
    )
    hydration = _hydration_snapshot(project_root)
    payload: dict[str, Any] = {
        "schema_version": DOCX_SCHEMA_VERSION,
        "artifact_path": DOCX_ARTIFACT,
        "artifact_exists": artifact_path.is_file(),
        "artifact_bytes": artifact_path.stat().st_size
        if artifact_path.is_file()
        else 0,
        "artifact_sha256": _sha256_file(artifact_path),
        "zip_integrity": False,
        "required_parts": {part: False for part in REQUIRED_DOCX_PARTS},
        "embedded_media_count": 0,
        "image_relationship_count": 0,
        "expected_figure_count": expected_figure_count,
        "expected_media_count": expected_media_count,
        "configured_cover_artifact": (
            str(cover_path.relative_to(project_root))
            if cover_path is not None and cover_path.is_relative_to(project_root)
            else str(cover_path)
            if cover_path is not None
            else None
        ),
        "configured_cover_sha256": cover_sha256,
        "cover_media_count": 0,
        "cover_alt_present": False,
        "caption_count": 0,
        "document_text_length": 0,
        "document_text_sha256": None,
        "unresolved_tokens": [],
        "text_anchors": {name: False for name in REQUIRED_TEXT_ANCHORS},
        "hydrated_manuscript_sha256": _hydrated_manuscript_hash(project_root),
        "figure_registry_sha256": _figure_registry_hash(project_root),
        "excluded_registry_figures": sorted(_VALIDATOR_PRODUCED_FIGURES),
        "hydration": hydration,
        "figure_registry_error": figure_registry_error,
    }
    if not artifact_path.is_file():
        payload["errors"] = validate_docx_render_payload(payload)
        payload["valid"] = False
        return payload

    try:
        with zipfile.ZipFile(artifact_path) as archive:
            payload["zip_integrity"] = archive.testzip() is None
            names = set(archive.namelist())
            payload["required_parts"] = {
                part: part in names for part in REQUIRED_DOCX_PARTS
            }
            payload["embedded_media_count"] = sum(
                name.startswith("word/media/") and not name.endswith("/")
                for name in names
            )
            document_xml = archive.read("word/document.xml")
            relationship_xml = archive.read("word/_rels/document.xml.rels")
            document_text, text_nodes = _document_text(document_xml)
            payload["document_text_length"] = len(document_text)
            payload["document_text_sha256"] = hashlib.sha256(
                document_text.encode("utf-8")
            ).hexdigest()
            media_names = sorted(
                name
                for name in names
                if name.startswith("word/media/") and not name.endswith("/")
            )
            if cover_sha256 is not None:
                payload["cover_media_count"] = sum(
                    hashlib.sha256(archive.read(name)).hexdigest() == cover_sha256
                    for name in media_names
                )
            if cover_alt:
                payload["cover_alt_present"] = cover_alt in document_xml.decode(
                    "utf-8", errors="replace"
                )
            # Captions are paragraphs; the joined document string is useful
            # for anchors but would lose paragraph boundaries for this count.
            root = ElementTree.fromstring(document_xml)
            paragraph_texts = [
                "".join(node.text or "" for node in paragraph.iter(_TEXT_TAG))
                for paragraph in root.iter(_PARAGRAPH_TAG)
            ]
            payload["caption_count"] = sum(
                bool(_CAPTION_RE.match(paragraph)) for paragraph in paragraph_texts
            )
            xml_text = "\n".join(
                archive.read(name).decode("utf-8", errors="replace")
                for name in names
                if name.endswith(".xml")
            )
            payload["unresolved_tokens"] = sorted(set(_TOKEN_RE.findall(xml_text)))
            payload["text_anchors"] = {
                name: anchor in document_text
                for name, anchor in REQUIRED_TEXT_ANCHORS.items()
            }
            payload["image_relationship_count"] = _image_relationship_count(
                relationship_xml
            )
            payload["text_node_count"] = len(text_nodes)
    except (
        OSError,
        KeyError,
        ValueError,
        zipfile.BadZipFile,
        ElementTree.ParseError,
    ) as exc:
        payload["zip_error"] = str(exc)

    errors = validate_docx_render_payload(payload)
    payload["errors"] = errors
    payload["valid"] = not errors
    return payload


def validate_docx_render_payload(payload: dict[str, Any]) -> list[str]:
    """Return structural and source-lineage errors for one DOCX payload."""

    errors: list[str] = []
    if payload.get("schema_version") != DOCX_SCHEMA_VERSION:
        errors.append("schema version mismatch")
    if not payload.get("artifact_exists"):
        errors.append(f"missing {DOCX_ARTIFACT}")
    if not payload.get("zip_integrity"):
        errors.append("DOCX ZIP integrity check failed")
    missing_parts = sorted(
        part
        for part, present in payload.get("required_parts", {}).items()
        if not present
    )
    if missing_parts:
        errors.append(f"missing DOCX parts={missing_parts}")
    if int(payload.get("embedded_media_count", 0)) <= 0:
        errors.append("DOCX contains no embedded media")
    expected = payload.get("expected_figure_count")
    expected_media = payload.get("expected_media_count")
    if expected is None:
        errors.append(
            str(payload.get("figure_registry_error", "missing figure registry"))
        )
    elif expected_media is None:
        expected_media = expected
    if (
        expected_media is not None
        and payload.get("embedded_media_count") != expected_media
    ):
        errors.append(
            "embedded media count mismatch: "
            f"observed={payload.get('embedded_media_count')} expected={expected_media}"
        )
    if payload.get("configured_cover_artifact") is not None:
        if payload.get("configured_cover_sha256") is None:
            errors.append("configured cover image is missing or unreadable")
        if payload.get("cover_media_count") != 1:
            errors.append(
                "configured cover media mismatch: "
                f"observed={payload.get('cover_media_count')} expected=1"
            )
        if not payload.get("cover_alt_present"):
            errors.append("configured cover alternative text is missing from DOCX")
    if payload.get("image_relationship_count") != payload.get("embedded_media_count"):
        errors.append(
            "image relationship/media mismatch: "
            f"relationships={payload.get('image_relationship_count')} "
            f"media={payload.get('embedded_media_count')}"
        )
    if expected is not None and payload.get("caption_count") != expected:
        errors.append(
            "caption count mismatch: "
            f"observed={payload.get('caption_count')} expected={expected}"
        )
    if int(payload.get("document_text_length", 0)) <= 0:
        errors.append("DOCX document text is empty")
    unresolved = payload.get("unresolved_tokens", [])
    if unresolved:
        errors.append(f"unresolved DOCX tokens={unresolved}")
    missing_anchors = sorted(
        name for name, present in payload.get("text_anchors", {}).items() if not present
    )
    if missing_anchors:
        errors.append(f"missing DOCX text anchors={missing_anchors}")
    hydration = payload.get("hydration", {})
    summary = hydration.get("summary", {})
    if hydration.get("status") != "hydrated":
        errors.append(f"template hydration status={hydration.get('status')}")
    if hydration.get("unresolved_tokens"):
        errors.append("template hydration registry reports unresolved tokens")
    if not summary.get("all_source_tokens_generated", False):
        errors.append("template hydration did not generate every source token")
    if not summary.get("no_pending_values", False):
        errors.append("template hydration registry contains pending values")
    if not payload.get("hydrated_manuscript_sha256"):
        errors.append("missing hydrated manuscript hash")
    if not payload.get("figure_registry_sha256"):
        errors.append("missing figure registry hash")
    return errors


def validate_docx_render_receipt(
    project_root: Path = PROJECT_ROOT,
    payload: dict[str, Any] | None = None,
) -> list[str]:
    """Return freshness errors for the receipt associated with the current DOCX."""

    current = payload or docx_render_payload(project_root)
    receipt_path = project_root / DOCX_RECEIPT_ARTIFACT
    if not receipt_path.is_file():
        return [f"missing {DOCX_RECEIPT_ARTIFACT}"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [str(exc)]
    errors: list[str] = []
    if receipt.get("schema_version") != DOCX_SCHEMA_VERSION:
        errors.append("DOCX receipt schema version mismatch")
    if receipt.get("artifact_path") != DOCX_ARTIFACT:
        errors.append("DOCX receipt artifact path mismatch")
    if receipt.get("valid") is not True or receipt.get("errors"):
        errors.append("DOCX receipt records an invalid artifact")
    if receipt.get("excluded_registry_figures") != sorted(_VALIDATOR_PRODUCED_FIGURES):
        errors.append("DOCX receipt excluded-figure contract mismatch")
    for key in (
        "artifact_sha256",
        "artifact_bytes",
        "hydrated_manuscript_sha256",
        "figure_registry_sha256",
        "expected_media_count",
        "embedded_media_count",
        "caption_count",
        "configured_cover_artifact",
        "configured_cover_sha256",
        "cover_media_count",
        "cover_alt_present",
    ):
        if receipt.get(key) != current.get(key):
            errors.append(f"DOCX receipt stale={key}")
    return errors


def write_docx_render_receipt(project_root: Path = PROJECT_ROOT) -> Path:
    """Write a post-render DOCX receipt and return its absolute path."""

    payload = docx_render_payload(project_root)
    receipt = {
        "schema_version": DOCX_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": DOCX_ARTIFACT,
        "artifact_sha256": payload.get("artifact_sha256"),
        "artifact_bytes": payload.get("artifact_bytes", 0),
        "hydrated_manuscript_sha256": payload.get("hydrated_manuscript_sha256"),
        "figure_registry_sha256": payload.get("figure_registry_sha256"),
        "expected_media_count": payload.get("expected_media_count"),
        "configured_cover_artifact": payload.get("configured_cover_artifact"),
        "configured_cover_sha256": payload.get("configured_cover_sha256"),
        "cover_media_count": payload.get("cover_media_count", 0),
        "cover_alt_present": payload.get("cover_alt_present", False),
        "excluded_registry_figures": payload.get("excluded_registry_figures", []),
        "embedded_media_count": payload.get("embedded_media_count", 0),
        "caption_count": payload.get("caption_count", 0),
        "zip_integrity": payload.get("zip_integrity", False),
        "template_hydration": payload.get("hydration", {}),
        "valid": payload.get("valid", False),
        "errors": payload.get("errors", []),
    }
    path = project_root / DOCX_RECEIPT_ARTIFACT
    write_json_artifact(
        path,
        receipt,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path
