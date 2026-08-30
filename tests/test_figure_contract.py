"""Regression tests for the canonical source-bound visual atlas."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from src.figure_contract import (
    FIGURE_REGISTRY_SCHEMA_VERSION,
    _format_number,
    _load,
    _registry_envelope,
    _relative,
    _root_for,
    _safe_value,
    _visual_qa,
    _write_raster_svg,
    build_figure_specs,
    validate_figure_bundle,
    write_figure_bundle,
)
from src.figures import FIGURE_CAPTIONS, plot_quotient_classifier
from src.visual_atlas import NARRATIVE_FIGURE_IDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def test_canonical_registry_has_exactly_62_unique_figures() -> None:
    specs = build_figure_specs(OUTPUT_DIR)

    assert len(specs) == 62
    assert len({spec.figure_id for spec in specs}) == 62
    assert len({spec.label for spec in specs}) == 62
    assert set(NARRATIVE_FIGURE_IDS).issubset(FIGURE_CAPTIONS)
    assert set(NARRATIVE_FIGURE_IDS).issubset({spec.figure_id for spec in specs})


def test_figure_specs_have_current_source_hashes_and_accessibility_metadata() -> None:
    for spec in build_figure_specs(OUTPUT_DIR):
        assert spec.input_artifacts
        assert set(spec.input_artifacts) == set(spec.input_hashes)
        for artifact in spec.input_artifacts:
            path = PROJECT_ROOT / artifact
            assert path.is_file(), (
                f"missing source artifact for {spec.figure_id}: {artifact}"
            )
            assert (
                spec.input_hashes[artifact]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
        assert spec.caption.strip()
        assert 120 <= len(spec.caption.split()) <= 260
        assert spec.alt_text.strip()
        assert len(spec.alt_text.split()) <= 45
        assert 150 <= len(spec.long_description.split()) <= 360
        assert spec.caption_word_count == len(spec.caption.split())
        assert spec.alt_text_word_count == len(spec.alt_text.split())
        assert spec.long_description_word_count == len(spec.long_description.split())
        assert spec.palette_contract
        assert spec.table_companion_path.endswith(f"#{spec.figure_id}-data-table")
        assert spec.source_note.startswith("Source receipts:")
        assert spec.visual_qa_status == spec.visual_qa["status"]
        assert spec.claim_ids
        assert spec.validation_checks


def test_figure_bundle_and_offline_atlas_are_closed() -> None:
    assert validate_figure_bundle(OUTPUT_DIR) == []
    registry = (OUTPUT_DIR / "data" / "figure_registry.json").read_text(
        encoding="utf-8"
    )
    assert f'"schema_version": "{FIGURE_REGISTRY_SCHEMA_VERSION}"' in registry
    atlas = (OUTPUT_DIR / "figure_atlas" / "index.html").read_text(encoding="utf-8")
    assert atlas.count("<figure>") == 62
    assert atlas.count("<figcaption>") == 62
    assert atlas.count('alt="') == 62
    assert atlas.count("<details>") == 62
    assert not re.search(r"(?:https?:)?//|<script|<link", atlas, re.IGNORECASE)


def test_canonical_registry_keeps_compatibility_and_protected_boundaries() -> None:
    payload = json.loads(
        (OUTPUT_DIR / "data" / "figure_registry.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "2.1"
    assert "2.0" in payload["compatible_schema_versions"]
    assert all(
        {
            "png_path",
            "input_hashes",
            "population",
            "sample_hierarchy",
            "metrics",
            "uncertainty",
            "scholarship_keys",
            "evidence_status",
            "receipt_status",
        }.issubset(row)
        for row in payload["figures"]
    )
    assert all(
        {
            "caption_word_count",
            "alt_text_word_count",
            "long_description_word_count",
            "palette_contract",
            "table_companion_path",
            "source_note",
            "visual_qa_status",
        }.issubset(row)
        for row in payload["figures"]
    )
    replay = next(
        row for row in payload["figures"] if row["figure_id"] == "replay_failure_audit"
    )
    assert not any("private/gold_labels" in path for path in replay["input_artifacts"])


def test_registry_aggregate_hashes_exclude_validator_feedback_inputs() -> None:
    """Validator outputs must not create a self-invalidating registry cycle."""

    payload = _registry_envelope(PROJECT_ROOT, build_figure_specs(OUTPUT_DIR))
    assert "output/data/validation_report.json" not in payload["input_hashes"]
    assert "output/data/reproducibility_manifest.json" not in payload["input_hashes"]


def test_observation_semantics_is_source_bound_and_caption_defects_are_absent() -> None:
    specs = {spec.figure_id: spec for spec in build_figure_specs(OUTPUT_DIR)}
    observation = specs["observation_semantics"]
    assert observation.input_artifacts == (
        "output/data/replication_manifest.json",
        "output/data/research_protocol.json",
        "output/data/typed_observations.json",
    )
    assert "missing" in observation.caption.lower()
    assert "unseen" in observation.caption.lower()
    assert "unsupported" in observation.caption.lower()
    assert "unavailable" in observation.caption.lower()
    assert "numeric zero" in observation.caption.lower()
    all_captions = " ".join(spec.caption for spec in specs.values()).lower()
    assert re.search(r"\b(the|a|an|configured)\s+\1\b", all_captions) is None
    assert "configured configured" not in all_captions


def test_quotient_classifier_figure_uses_receipt_intervals_and_deltas(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (OUTPUT_DIR / "data" / "quotient_classifier.json").read_text(encoding="utf-8")
    )
    path = plot_quotient_classifier(payload, tmp_path / "quotient_classifier.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_figure_pngs_and_vector_companions_are_nonblank_and_parseable() -> None:
    for spec in build_figure_specs(OUTPUT_DIR):
        png = PROJECT_ROOT / spec.png_path
        image = mpimg.imread(png)
        assert image.size > 0
        assert float(image.var()) > 0.00001
        svg = PROJECT_ROOT / spec.svg_path
        pdf = PROJECT_ROOT / spec.pdf_path
        assert b"<svg" in svg.read_bytes()[:2048].lower()
        assert pdf.read_bytes()[:4] == b"%PDF"


def test_narrative_figure_tokens_are_consumed_by_authored_manuscript() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "manuscript").glob("*.md"))
    )
    for figure_id in NARRATIVE_FIGURE_IDS:
        prefix = f"FIGURE_{figure_id.upper()}"
        assert f"{{{{{prefix}_CAPTION}}}}" in manuscript
        assert f"{{{{{prefix}_PATH}}}}" in manuscript


def test_figure_contract_defensive_helpers_preserve_status_and_path_boundaries(
    tmp_path: Path,
) -> None:
    """Exercise malformed-input and path-boundary branches explicitly."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    assert _root_for(output_dir) == tmp_path
    assert _root_for(tmp_path) == tmp_path
    assert _root_for(tmp_path / "unrelated") == PROJECT_ROOT

    outside = tmp_path / "outside.txt"
    assert _relative(outside, tmp_path / "different-root") == str(outside)
    assert _safe_value(None) == "not observed"
    assert _safe_value(None, "unavailable") == "unavailable"
    assert _format_number(True) == "not observed"
    assert _format_number("not numeric") == "not observed"
    assert _format_number(1.23456) == "1.235"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        _load(malformed)

    missing_qa = _visual_qa(tmp_path / "missing.png", tmp_path / "vectors")
    assert missing_qa["status"] == "missing"
    assert missing_qa["sha256"] is None

    unreadable = tmp_path / "unreadable.png"
    unreadable.write_bytes(b"not an image")
    fallback_svg = tmp_path / "vectors" / "unreadable.svg"
    _write_raster_svg(unreadable, fallback_svg)
    assert fallback_svg.read_text(encoding="utf-8").startswith("<svg")
    unreadable_qa = _visual_qa(unreadable, fallback_svg.parent)
    assert unreadable_qa["status"] == "failed"
    assert unreadable_qa["pixel_variance"] == 0.0


def test_figure_bundle_validation_reports_spec_build_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed source context must fail closed rather than vanish."""

    def raise_spec_error(_output_dir: Path) -> list[object]:
        raise ValueError("synthetic figure-spec drift")

    monkeypatch.setattr("src.figure_contract.build_figure_specs", raise_spec_error)
    errors = validate_figure_bundle(OUTPUT_DIR)
    assert errors == ["figure_spec_build: synthetic figure-spec drift"]


def test_figure_bundle_rebuilds_vector_and_atlas_surfaces_from_producers() -> None:
    """Exercise the direct bundle path used by the public figure CLI."""

    result = write_figure_bundle(OUTPUT_DIR, formats=("svg", "pdf", "atlas"))

    assert result["registry"] == "output/data/figure_registry.json"
    assert result["atlas"] == "output/figure_atlas/index.html"
    assert (OUTPUT_DIR / "figures" / "vector" / "model_comparison.svg").is_file()
    assert (OUTPUT_DIR / "figures" / "vector" / "model_comparison.pdf").is_file()
    assert (OUTPUT_DIR / "figure_atlas" / "index.html").is_file()
