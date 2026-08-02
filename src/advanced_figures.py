"""Generated figures for the v0.2 temporal / active / evaluation layer.

The v0.2 manuscript describes a temporal HMM, an active EFE policy, and a
quantitative decoder comparison, but the original figure set only covered the
v0.1 observation/posterior/RxInfer surface. This module closes that gap with
three honest, measured visualizations plus a single caption ledger that
documents every figure the project emits.

All numbers plotted here come from artifacts already produced by
:mod:`advanced_inference` and :mod:`evaluation`; nothing is hand-stamped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .figures import (
    plot_active_efe_trajectory,
    plot_ambiguity_ablation,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_corpus_stability,
    plot_decoder_calibration,
    plot_generative_model,
    plot_model_comparison_bars,
    plot_model_evidence,
    plot_observation_informativeness,
    plot_phase_recovery,
    plot_segmentation_ribbons,
    plot_temporal_beliefs_heatmap,
    write_figure_captions,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def write_temporal_belief_figures(output_dir: Path = OUTPUT_DIR) -> list[str]:
    """Write one temporal-belief heatmap per corpus fixture; return relative paths."""

    figure_dir = output_dir / "figures"
    temporal_path = output_dir / "data" / "temporal_inference.json"
    if not temporal_path.exists():
        raise FileNotFoundError(f"missing temporal inference artifact: {temporal_path}")
    payload = json.loads(temporal_path.read_text(encoding="utf-8"))
    hidden_state_labels = payload["model"]["hidden_state_labels"]
    relative_paths: list[str] = []
    for fixture_name, fixture in payload["per_fixture"].items():
        figure_path = figure_dir / f"temporal_beliefs_{fixture_name}.png"
        plot_temporal_beliefs_heatmap(
            fixture_name,
            list(fixture["observation_ids"]),
            [list(row) for row in fixture["filter"]["filtered_posteriors"]],
            list(hidden_state_labels),
            figure_path,
        )
        relative_paths.append(str(figure_path.relative_to(PROJECT_ROOT)))
    return relative_paths


def write_stats_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write calibration-curve, confusion-matrix, and ambiguity-ablation figures.

    Returns:
        A dict with keys ``calibration_curve``, ``confusion_matrix``, and
        ``ambiguity_ablation`` mapping to their project-relative PNG paths.
    """

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir / "data" / "evaluation_report.json"
    evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))

    calibration_path = figure_dir / "calibration_curve.png"
    plot_calibration_curve(
        evaluation_report["pooled_calibration_bins"], calibration_path
    )

    confusion_path = figure_dir / "confusion_matrix.png"
    plot_confusion_matrix(evaluation_report["pooled_confusion"], confusion_path)

    ablation_path = figure_dir / "ambiguity_ablation.png"
    plot_ambiguity_ablation(evaluation_report["ambiguity_ablation"], ablation_path)

    ribbons_path = figure_dir / "segmentation_ribbons.png"
    plot_segmentation_ribbons(evaluation_report, ribbons_path)

    stability_path = figure_dir / "corpus_stability.png"
    plot_corpus_stability(evaluation_report["corpus_stability"], stability_path)

    phase_recovery_path = figure_dir / "phase_recovery.png"
    plot_phase_recovery(evaluation_report["phase_recovery"], phase_recovery_path)

    calibration_path = figure_dir / "decoder_calibration.png"
    plot_decoder_calibration(evaluation_report["decoder_calibration"], calibration_path)

    generative_path = figure_dir / "generative_model.png"
    plot_generative_model(generative_path)

    informativeness_path = figure_dir / "observation_informativeness.png"
    plot_observation_informativeness(
        evaluation_report["observation_informativeness"], informativeness_path
    )

    evidence_path = figure_dir / "model_evidence.png"
    plot_model_evidence(evaluation_report["model_evidence"], evidence_path)

    return {
        "calibration_curve": str(calibration_path.relative_to(PROJECT_ROOT)),
        "confusion_matrix": str(confusion_path.relative_to(PROJECT_ROOT)),
        "ambiguity_ablation": str(ablation_path.relative_to(PROJECT_ROOT)),
        "segmentation_ribbons": str(ribbons_path.relative_to(PROJECT_ROOT)),
        "corpus_stability": str(stability_path.relative_to(PROJECT_ROOT)),
        "phase_recovery": str(phase_recovery_path.relative_to(PROJECT_ROOT)),
        "decoder_calibration": str(calibration_path.relative_to(PROJECT_ROOT)),
        "generative_model": str(generative_path.relative_to(PROJECT_ROOT)),
        "observation_informativeness": str(
            informativeness_path.relative_to(PROJECT_ROOT)
        ),
        "model_evidence": str(evidence_path.relative_to(PROJECT_ROOT)),
    }


def write_advanced_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the v0.2 figure set and the caption ledger; return relative paths.

    Returns:
        A dict with keys ``temporal_beliefs`` (first fixture, for the manuscript
        token), ``model_comparison``, ``active_efe``, ``calibration_curve``,
        ``confusion_matrix``, and ``captions``.
    """

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = output_dir / "data" / "evaluation_report.json"
    evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))

    temporal_paths = write_temporal_belief_figures(output_dir)
    model_comparison_path = figure_dir / "model_comparison.png"
    plot_model_comparison_bars(evaluation_report, model_comparison_path)

    active_path = output_dir / "data" / "active_selection.json"
    active_report = json.loads(active_path.read_text(encoding="utf-8"))
    first_fixture = next(iter(active_report["per_fixture"]))
    active_efe_path = figure_dir / "active_efe.png"
    plot_active_efe_trajectory(
        active_report["per_fixture"][first_fixture]["selection"],
        active_efe_path,
    )

    stats_paths = write_stats_figures(output_dir)
    captions_path = write_figure_captions(output_dir)

    return {
        "temporal_beliefs": temporal_paths[0] if temporal_paths else "",
        "model_comparison": str(model_comparison_path.relative_to(PROJECT_ROOT)),
        "active_efe": str(active_efe_path.relative_to(PROJECT_ROOT)),
        "calibration_curve": stats_paths["calibration_curve"],
        "confusion_matrix": stats_paths["confusion_matrix"],
        "ambiguity_ablation": stats_paths["ambiguity_ablation"],
        "segmentation_ribbons": stats_paths["segmentation_ribbons"],
        "corpus_stability": stats_paths["corpus_stability"],
        "phase_recovery": stats_paths["phase_recovery"],
        "decoder_calibration": stats_paths["decoder_calibration"],
        "generative_model": stats_paths["generative_model"],
        "observation_informativeness": stats_paths["observation_informativeness"],
        "model_evidence": stats_paths["model_evidence"],
        "captions": str(captions_path.relative_to(PROJECT_ROOT)),
    }
