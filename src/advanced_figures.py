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
    plot_model_comparison_bars,
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


def write_advanced_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the v0.2 figure set and the caption ledger; return relative paths.

    Returns:
        A dict with keys ``temporal_beliefs`` (first fixture, for the manuscript
        token), ``model_comparison``, ``active_efe``, and ``captions``.
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

    captions_path = write_figure_captions(output_dir)

    return {
        "temporal_beliefs": temporal_paths[0] if temporal_paths else "",
        "model_comparison": str(model_comparison_path.relative_to(PROJECT_ROOT)),
        "active_efe": str(active_efe_path.relative_to(PROJECT_ROOT)),
        "captions": str(captions_path.relative_to(PROJECT_ROOT)),
    }
