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


def test_canonical_registry_has_exactly_69_unique_figures() -> None:
    specs = build_figure_specs(OUTPUT_DIR)

    assert len(specs) == 69
    assert len({spec.figure_id for spec in specs}) == 69
    assert len({spec.label for spec in specs}) == 69
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
        assert 30 <= len(spec.caption.split()) <= 260
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
    assert atlas.count("<figure>") == 69
    assert atlas.count("<figcaption>") == 69
    assert atlas.count('alt="') == 69
    assert atlas.count("<details>") == 69
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
    assert "3 label-blind fields" in observation.caption
    assert "5 forbidden fields" in observation.caption
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


def test_counterexample_search_figure_renders_from_receipt(tmp_path: Path) -> None:
    """The counterexample-search figure renders from its real receipt."""

    from src.figures import plot_counterexample_search

    payload = _load(OUTPUT_DIR / "data" / "counterexample_search.json")
    path = plot_counterexample_search(payload, tmp_path / "counterexample_search.png")
    assert path.is_file() and path.stat().st_size > 0


def test_counterexample_search_figure_rejects_witnessless_falsified(
    tmp_path: Path,
) -> None:
    """A falsified candidate without a witness rank fails closed."""

    from src.figures import plot_counterexample_search

    payload = _load(OUTPUT_DIR / "data" / "counterexample_search.json")
    falsified = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["status"] == "falsified_by_counterexample"
    )
    falsified["witness"] = {
        key: value
        for key, value in falsified["witness"].items()
        if key != "witness_size"
    }
    with pytest.raises(ValueError, match="witness_size"):
        plot_counterexample_search(payload, tmp_path / "broken.png")


def test_sensing_stress_figure_renders_from_receipt(tmp_path: Path) -> None:
    """The sensing-stress figure renders from its real receipt."""

    from src.figures import plot_sensing_stress

    payload = _load(OUTPUT_DIR / "data" / "sensing_stress.json")
    path = plot_sensing_stress(payload, tmp_path / "sensing_stress.png")
    assert path.is_file() and path.stat().st_size > 0


def test_replay_failure_audit_renders_not_observed_state(tmp_path: Path) -> None:
    """An absent receipt renders the documented not-observed audit state."""

    from src.visual_atlas import plot_replay_failure_audit

    path = plot_replay_failure_audit({}, tmp_path / "replay_failure_audit.png")
    assert path.is_file() and path.stat().st_size > 0
    from xml.etree import ElementTree

    svg = path.parent / "vector/replay_failure_audit.svg"
    tree = ElementTree.fromstring(svg.read_text())
    panel = next(element for element in tree.iter() if element.get("id") == "axes_2")
    assert not any(element.tag.endswith("}image") for element in panel.iter())
    assert "Protected counts not observed" in svg.read_text()


@pytest.mark.parametrize("bad", [None, -1, 0.5, float("nan")])
def test_replay_confusion_panel_rejects_incomplete_or_invalid_counts(tmp_path, bad):
    from src.visual_atlas import plot_replay_failure_audit

    confusion = dict(
        true_positive=1, false_negative=1, false_positive=1, true_negative=bad
    )
    with pytest.raises(ValueError, match="confusion counts"):
        plot_replay_failure_audit(
            {"protected_score": {"confusion_matrix": confusion}},
            tmp_path / "invalid.png",
        )


def test_counterexample_search_figure_rejects_structural_drift(
    tmp_path: Path,
) -> None:
    """Empty, malformed, and unknown-status payloads fail closed."""

    from src.figures import plot_counterexample_search

    with pytest.raises(ValueError, match="must be non-empty"):
        plot_counterexample_search({"candidates": []}, tmp_path / "empty.png")

    payload = _load(OUTPUT_DIR / "data" / "counterexample_search.json")
    payload["candidates"] = [
        {
            "property_id": "x",
            "status": "mystery_status",
            "searched_domain_size": 3,
            "witness": None,
        }
    ]
    with pytest.raises(ValueError, match="unknown candidate status"):
        plot_counterexample_search(payload, tmp_path / "status.png")

    payload = _load(OUTPUT_DIR / "data" / "counterexample_search.json")
    payload["candidates"] = ["not-an-object"]
    with pytest.raises(ValueError, match="not an object"):
        plot_counterexample_search(payload, tmp_path / "row.png")


def test_counterexample_search_figure_switches_to_log_domain_scale(
    tmp_path: Path,
) -> None:
    """A four-orders-of-magnitude domain span renders on a log axis."""

    from src.figures import plot_counterexample_search

    payload = {
        "candidates": [
            {
                "property_id": "tiny_domain",
                "status": "holds_in_searched_domain",
                "searched_domain_size": 1,
                "witness": None,
            },
            {
                "property_id": "huge_domain",
                "status": "holds_in_searched_domain",
                "searched_domain_size": 1_000_000,
                "witness": None,
            },
        ]
    }
    path = plot_counterexample_search(payload, tmp_path / "log_scale.png")
    assert path.is_file() and path.stat().st_size > 0


def test_sensing_stress_figure_rejects_structural_drift(tmp_path: Path) -> None:
    """Empty grid, negative regret, and negative gaps fail closed."""

    from src.figures import plot_sensing_stress

    payload = _load(OUTPUT_DIR / "data" / "sensing_stress.json")

    with pytest.raises(ValueError, match="grid must be non-empty"):
        plot_sensing_stress({**payload, "grid": []}, tmp_path / "empty.png")

    drifted = {
        **payload,
        "grid": [{**payload["grid"][0], "regret_vs_exact": -1.0}],
    }
    with pytest.raises(ValueError, match="regret is negative"):
        plot_sensing_stress(drifted, tmp_path / "regret.png")

    drifted = {
        **payload,
        "budget_value_curve": [
            {**row, "unattained_information_gain_gap": -1.0}
            for row in payload["budget_value_curve"]
        ],
    }
    with pytest.raises(ValueError, match="gap is negative"):
        plot_sensing_stress(drifted, tmp_path / "gap.png")


def test_replay_failure_audit_renders_populated_score(tmp_path: Path) -> None:
    """A receipt-backed protected score renders every audit panel."""

    from src.visual_atlas import plot_replay_failure_audit

    payload = {
        "control_scope": "same_host_separately_controlled_guest",
        "network_disabled": True,
        "protected_score": {
            "rows": 40,
            "confusion_matrix": {
                "true_positive": 10,
                "false_negative": 10,
                "false_positive": 18,
                "true_negative": 2,
            },
            "balanced_accuracy": 0.3,
            "macro_f1": 0.28,
            "brier_score": 0.61,
            "expected_calibration_error": 0.19,
            "selective_risk_coverage": {"coverage": 0.5, "risk": 0.5},
            "false_equivalence_rate": 0.9,
            "near_miss_separation_rate": 0.1,
        },
    }
    path = plot_replay_failure_audit(payload, tmp_path / "audit_populated.png")
    assert path.is_file() and path.stat().st_size > 0


def test_narrative_vector_export_retains_readable_editable_type(tmp_path):
    from xml.etree import ElementTree
    from src.visual_atlas import plot_research_architecture

    path = plot_research_architecture({}, tmp_path / "architecture.png")
    svg = path.parent / "vector/architecture.svg"
    tree = ElementTree.fromstring(svg.read_text())
    texts = [node for node in tree.iter() if node.tag.endswith("}text")]
    assert len(texts) >= 20
    sizes = [
        float(size)
        for node in texts
        for size in re.findall(r"font-size:\s*([0-9.]+)px", node.get("style", ""))
    ]
    assert len(sizes) == len(texts)
    assert min(sizes) >= 10


def test_bulk_png_reader_preserves_decoded_pixels(tmp_path):
    """A real PNG keeps identical values through the bulk-reading path."""
    import numpy as np
    from src.figures import read_png_image

    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    path = tmp_path / "pixels.png"
    mpimg.imsave(path, pixels)
    assert np.array_equal(read_png_image(path), mpimg.imread(path))
