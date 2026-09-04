"""Publication figures for the assembly Active Inference demonstrator."""

from __future__ import annotations

import json
import hashlib
import os
import textwrap
from pathlib import Path
from typing import Any, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from .active_inference_bridge import InferenceResult
from .artifact_contract import write_json_artifact
from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
INK = "#243447"
MUTED = "#52606D"
GRID = "#D9E2EC"
LIGHT = "#F5F7FA"
FIGURE_DPI = 300
AXIS_TITLE_PAD = 6
AXIS_SUBTITLE_Y = 1.16


def _style_axes(ax: Any, *, grid_axis: str = "y") -> None:
    """Apply a restrained, print-safe visual grammar to an axes."""

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)
    if grid_axis in {"x", "both"}:
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    if grid_axis in {"y", "both"}:
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)


def _save_figure(fig: Any, output_path: Path) -> Path:
    """Save a figure atomically with a stable white background.

    The registry is generated immediately after the figure suite. Writing to a
    process-specific temporary PNG and replacing the destination only after
    ``savefig`` succeeds prevents a failed render or concurrent producer from
    exposing a zero-byte/truncated image to the registry reader.
    """

    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )
    try:
        fig.savefig(
            temporary_path,
            format="png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        temporary_path.replace(output_path)
    finally:
        plt.close(fig)
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def plot_observation_pipeline(
    observations: Sequence[AssemblyObservation],
    output_path: Path,
) -> Path:
    """Plot the decoded instruction stream as an auditable categorical map.

    A connected line is deliberately avoided: observation IDs are labels, not
    measurements on an ordered numeric scale. Each instruction is therefore
    represented by a card containing its address, mnemonic/operands, and the
    category consumed by the inference model.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic observation_id -> category map (first occurrence wins). The
    # map is a contract, not a plotting convenience: an ID cannot silently
    # refer to two semantic channels.
    category_by_id: dict[int, str] = {}
    for observation in observations:
        existing = category_by_id.get(observation.observation_id)
        if existing is not None and existing != observation.category:
            raise ValueError(
                "observation_id must map to one category, got "
                f"{observation.observation_id}: {existing!r} and "
                f"{observation.category!r}"
            )
        category_by_id.setdefault(observation.observation_id, observation.category)

    present_categories = [category_by_id[key] for key in sorted(category_by_id)]
    category_colors = {
        category: PALETTE[index % len(PALETTE)]
        for index, category in enumerate(sorted(set(present_categories)))
    }
    x = np.arange(len(observations), dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for index, observation in enumerate(observations):
        color = category_colors[observation.category]
        edge = INK if observation.executed else "#9AA5B1"
        linestyle = "-" if observation.executed else "--"
        ax.add_patch(
            Rectangle(
                (index - 0.43, 0.12),
                0.86,
                0.68,
                facecolor=color,
                edgecolor=edge,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.88,
            )
        )
        instruction = observation.mnemonic
        if observation.operand_text:
            instruction = f"{instruction} {observation.operand_text}"
        ax.text(
            index,
            0.58,
            instruction,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            family="DejaVu Sans Mono",
            wrap=True,
        )
        ax.text(
            index,
            0.31,
            observation.category.replace("_", " "),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
        ax.text(
            index,
            0.91,
            f"{index + 1} · 0x{observation.address:x}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )
    ax.set_xlim(-0.55, len(observations) - 0.45)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"step {index + 1}" for index in range(len(observations))])
    ax.set_yticks([])
    ax.set_xlabel(
        "Static instruction order; solid border = observed during bounded emulation"
    )
    ax.set_title(
        "Decoded x86-64 instructions mapped to categorical observations",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "The category label is the model input; the mnemonic/operand text is the auditable assembly evidence.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    _style_axes(ax, grid_axis="none")
    return _save_figure(fig, output_path)


def plot_posterior_beliefs(
    inference: InferenceResult,
    model: GenerativeModelSpec,
    output_path: Path,
) -> Path:
    """Plot posterior beliefs over latent program states."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(inference.posterior_by_step, dtype=float)
    x = np.arange(matrix.shape[0])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for state_index, state_label in enumerate(model.hidden_state_labels):
        ax.plot(
            x,
            matrix[:, state_index],
            marker="o",
            linewidth=2,
            label=state_label.replace("_", " "),
            color=PALETTE[state_index % len(PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xlabel("Observation step")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Memoryless posterior beliefs over latent program phases",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Each column is normalized; markers show the posterior after one decoded observation.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    _style_axes(ax, grid_axis="y")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.16, top=0.84)
    return _save_figure(fig, output_path)


def plot_rxinfer_free_energy(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "RxInfer Bethe Free Energy trajectory",
) -> Path:
    """Plot one model's RxInfer Bethe Free Energy trajectory."""

    if not records:
        raise ValueError("free-energy records must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(record["iteration"]) for record in records]
    values = [float(record["value"]) for record in records]
    if iterations != sorted(iterations) or len(set(iterations)) != len(iterations):
        raise ValueError("RxInfer iterations must be unique and increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("RxInfer free-energy values must be finite")

    initial = values[0]
    final = values[-1]
    delta = final - initial
    spread = max(values) - min(values)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(
        iterations,
        values,
        marker="o",
        markersize=5,
        linewidth=2,
        color=PALETTE[2],
        markeredgecolor=INK,
        markeredgewidth=0.5,
    )
    if len(values) == 1:
        ax.scatter(iterations, values, color=PALETTE[2], zorder=3)
    if spread == 0.0:
        pad = max(abs(initial) * 0.02, 0.01)
        ax.set_ylim(initial - pad, initial + pad)
        subtitle = (
            f"{len(values)} iterations · all values identical to recorded precision · "
            f"Δ = {delta:+.3f}"
        )
    else:
        pad = max(spread * 0.12, 0.02)
        ax.set_ylim(min(values) - pad, max(values) + pad)
        subtitle = (
            f"{len(values)} iterations · initial = {initial:.3f} · "
            f"final = {final:.3f} · Δ = {delta:+.3f}"
        )
    ax.set_xlabel("RxInfer iteration")
    ax.set_ylabel("Bethe Free Energy (model-specific units)")
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=AXIS_TITLE_PAD)
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.annotate(
        f"final {final:.3f}",
        xy=(iterations[-1], final),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK,
    )
    _style_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.80)
    return _save_figure(fig, output_path)


def plot_rxinfer_categorical_free_energy(
    records: list[dict[str, Any]], output_path: Path
) -> Path:
    """Plot the namespaced categorical RxInfer companion diagnostic."""

    return plot_rxinfer_free_energy(
        records,
        output_path,
        title="RxInfer Bethe Free Energy: categorical phase-HMM",
    )


def plot_package_provenance(
    provenance_rows: list[dict[str, object]], output_path: Path
) -> Path:
    """Plot required and optional assembly package adapter availability.

    Availability is shown with both marker geometry and direct text labels. An
    empty bar is visually ambiguous (it can mean zero, missing, or not
    applicable), so this view treats the probe as a categorical status rather
    than a pseudo-continuous measurement.
    """

    if not provenance_rows:
        raise ValueError("package provenance rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["name"]) for row in provenance_rows]
    available = [bool(row["available"]) for row in provenance_rows]

    fig, ax = plt.subplots(figsize=(9.0, max(3.8, 0.42 * len(names) + 1.8)))
    y = np.arange(len(names))
    ax.hlines(y, 0.0, 1.0, color=GRID, linewidth=1.0, zorder=0)
    for row_index, (row, is_available) in enumerate(
        zip(provenance_rows, available, strict=True)
    ):
        color = PALETTE[2] if is_available else "#7B8794"
        marker = "o" if is_available else "x"
        if is_available:
            ax.scatter(
                [1.0],
                [row_index],
                s=90,
                marker=marker,
                color=color,
                linewidths=1.0,
                edgecolors=INK,
                zorder=3,
            )
        else:
            ax.scatter(
                [0.0],
                [row_index],
                s=90,
                marker=marker,
                color=color,
                linewidths=2.0,
                zorder=3,
            )
        ax.text(
            1.08 if is_available else 0.08,
            row_index,
            "available" if is_available else "not found",
            ha="left" if is_available else "left",
            va="center",
            fontsize=8,
            color=INK,
        )
        ax.text(
            1.39,
            row_index,
            "optional" if row["optional"] else "required",
            ha="left",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.28, 1.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not found", "available"])
    ax.set_xlabel("Local package probe")
    ax.set_title("Assembly adapter availability and provenance", pad=AXIS_TITLE_PAD)
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Filled circles and direct labels identify found adapters; x markers identify absent probes.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="none")
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.15, top=0.82)
    return _save_figure(fig, output_path)


def plot_trace_feature_matrix(
    observations: list[AssemblyObservation],
    output_path: Path,
) -> Path:
    """Plot trace-derived feature semantics for each decoded instruction."""

    if not observations:
        raise ValueError("observations must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = np.array(
        [
            [
                float(observation.executed),
                float(observation.has_immediate_operand),
                float(observation.uses_accumulator),
            ]
            for observation in observations
        ],
        dtype=float,
    )
    fig, (ax_flags, ax_codes) = plt.subplots(
        1,
        2,
        figsize=(11.4, max(4.6, 0.55 * len(observations) + 1.8)),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    ax_flags.imshow(
        flags,
        aspect="auto",
        cmap=ListedColormap(["#F1F5F9", PALETTE[0]]),
        vmin=0.0,
        vmax=1.0,
    )
    ax_flags.set_xticks(np.arange(flags.shape[1]))
    ax_flags.set_xticklabels(
        ["executed", "immediate", "uses EAX/RAX"],
        rotation=25,
        ha="right",
    )
    ax_flags.set_yticks(np.arange(len(observations)))
    ax_flags.set_yticklabels(
        [
            f"step {observation.index + 1}: {observation.mnemonic}"
            for observation in observations
        ]
    )
    for row in range(flags.shape[0]):
        for column in range(flags.shape[1]):
            ax_flags.text(
                column,
                row,
                str(int(flags[row, column])),
                ha="center",
                va="center",
                fontsize=8,
            )
    ax_flags.set_title("Evidence-backed instruction features", pad=12)
    ax_flags.set_xlabel(
        "0 = absent; 1 = present; immediate and register flags use Capstone detail"
    )
    _style_axes(ax_flags, grid_axis="none")

    ax_codes.axis("off")
    code_rows = [
        [
            f"step {observation.index + 1}",
            str(observation.observation_id),
            str(observation.observation_id),
            observation.category.replace("_", " "),
            "yes" if observation.has_immediate_operand else "no",
            "yes" if observation.uses_accumulator else "no",
        ]
        for observation in observations
    ]
    table = ax_codes.table(
        cellText=code_rows,
        colLabels=["step", "obs id", "cat. id", "semantic label", "imm.", "EAX/RAX"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.16, 1.0, 0.68],
        colWidths=[0.14, 0.12, 0.13, 0.31, 0.14, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row > 0:
            cell.set_facecolor(LIGHT if row % 2 == 0 else "white")
    ax_codes.set_title("Categorical codes (labels, not a continuous scale)", pad=12)
    fig.suptitle(
        "Trace features used to construct observation semantics", color=INK, fontsize=12
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.84, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_inference_diagnostics(
    pymdp_free_energy: list[float],
    rxinfer_free_energy: list[float],
    output_path: Path,
) -> Path:
    """Plot pymdp comparison diagnostics beside RxInfer Bethe Free Energy."""

    if not pymdp_free_energy or not rxinfer_free_energy:
        raise ValueError("both inference diagnostic series must be non-empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pymdp_steps = np.arange(1, len(pymdp_free_energy) + 1)
    rxinfer_steps = np.arange(1, len(rxinfer_free_energy) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), sharex=False)
    axes[0].plot(
        pymdp_steps,
        pymdp_free_energy,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
    )
    axes[0].set_title("pymdp comparison variational diagnostic")
    axes[0].set_xlabel("Observation step")
    axes[0].set_ylabel("Diagnostic value")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        rxinfer_steps,
        rxinfer_free_energy,
        marker="o",
        linewidth=2,
        color=PALETTE[2],
    )
    axes[1].set_title("RxInfer Bethe Free Energy")
    axes[1].set_xlabel("RxInfer iteration")
    axes[1].set_ylabel("Bethe Free Energy")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Inference diagnostics recorded as separate objectives", color=INK, fontsize=12
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.11, top=0.88, hspace=0.42)
    for axis in axes:
        _style_axes(axis, grid_axis="y")
    return _save_figure(fig, output_path)


def plot_validation_summary(checks: list[dict[str, object]], output_path: Path) -> Path:
    """Plot pass/fail status for analysis validation checks."""

    if not checks:
        raise ValueError("validation checks must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_order = [
        "trace",
        "assembly",
        "symbolic",
        "rxinfer",
        "figures",
        "scholarship",
        "manuscript",
        "advanced evaluation",
        "held-out estimation",
        "typed observations",
        "factorized observations",
        "corpus design",
        "CFG evidence",
        "memory evidence",
        "cover",
        "claim register",
        "documentation",
        "reproducibility",
        "provenance",
    ]

    def group_name(raw_name: str) -> str:
        if raw_name.startswith("figure_"):
            return "figures"
        if raw_name.startswith("advanced_"):
            return "advanced evaluation"
        if raw_name.startswith("held_out_"):
            return "held-out estimation"
        if raw_name.startswith("typed_"):
            return "typed observations"
        if raw_name.startswith("factorized_"):
            return "factorized observations"
        if raw_name.startswith("corpus_coverage"):
            return "corpus design"
        if raw_name.startswith("cfg_"):
            return "CFG evidence"
        if raw_name.startswith("memory_"):
            return "memory evidence"
        if raw_name == "build_provenance_current":
            return "provenance"
        if raw_name in {"claim_register_contract"}:
            return "claim register"
        if raw_name == "reproducibility_artifact_manifest":
            return "reproducibility"
        if raw_name == "cover_image_contract":
            return "cover"
        if raw_name == "documentation_contract_surface":
            return "documentation"
        return raw_name.split("_", 1)[0]

    totals: dict[str, int] = {}
    passed: dict[str, int] = {}
    for check in checks:
        group = group_name(str(check["name"]))
        totals[group] = totals.get(group, 0) + 1
        if bool(check["passed"]):
            passed[group] = passed.get(group, 0) + 1
    groups = [group for group in group_order if group in totals]
    groups.extend(sorted(set(totals) - set(groups)))
    y = np.arange(len(groups))
    pass_values = [passed.get(group, 0) for group in groups]
    fail_values = [totals[group] - passed.get(group, 0) for group in groups]
    total_passed = sum(pass_values)
    total_checks = sum(totals.values())

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.34 * len(groups) + 1.8)))
    ax.barh(y, pass_values, color=PALETTE[0], label="passed")
    ax.barh(
        y,
        fail_values,
        left=pass_values,
        color="#D55E00",
        label="failed",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{group} ({passed.get(group, 0)}/{totals[group]})" for group in groups]
    )
    ax.set_xlim(0.0, max(totals.values(), default=1) * 1.08)
    ax.set_xlabel("Checks in domain")
    ax.set_title(
        f"Validation gate summary: {total_passed}/{total_checks} checks passed",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Compact domain view; individual check names and details remain in validation_report.json.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for yi, value in enumerate(pass_values):
        if value:
            ax.text(
                value - 0.12,
                yi,
                str(value),
                ha="right",
                va="center",
                fontsize=8,
                color="white",
            )
        if fail_values[yi]:
            ax.text(
                value + fail_values[yi] + 0.12,
                yi,
                str(fail_values[yi]),
                ha="left",
                va="center",
                fontsize=8,
                color="#D55E00",
            )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.14, top=0.82)
    return _save_figure(fig, output_path)


def plot_artifact_hash_manifest(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot tracked artifact bytes by artifact class."""

    if not rows:
        raise ValueError("artifact manifest rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_by_type: dict[str, int] = {}
    count_by_type: dict[str, int] = {}
    missing_by_type: dict[str, int] = {}
    for row in rows:
        artifact_type = str(row["artifact_type"])
        bytes_by_type[artifact_type] = bytes_by_type.get(artifact_type, 0) + int(
            row["bytes"]
        )
        count_by_type[artifact_type] = count_by_type.get(artifact_type, 0) + 1
        if row["status"] != "present":
            missing_by_type[artifact_type] = missing_by_type.get(artifact_type, 0) + 1

    labels = sorted(bytes_by_type, key=lambda label: bytes_by_type[label], reverse=True)
    values_kib = [bytes_by_type[label] / 1024.0 for label in labels]
    colors = [
        "#D55E00" if missing_by_type.get(label, 0) else PALETTE[0] for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.34 * len(labels) + 1.8)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values_kib, color=colors, edgecolor=INK, linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([label.replace("_", " ") for label in labels])
    ax.set_xlabel("Tracked bytes (KiB)")
    ax.set_title(
        f"Reproducibility manifest by artifact class ({sum(count_by_type.values())} files)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Bar length is byte size; labels report file count and missing entries are orange.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for bar, label in zip(bars, labels, strict=True):
        text = f"{values_kib[labels.index(label)]:.1f} KiB · {count_by_type[label]} file(s)"
        if missing_by_type.get(label, 0):
            text += f" · {missing_by_type[label]} missing"
        ax.text(
            bar.get_width() + max(values_kib, default=1.0) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            text,
            ha="left",
            va="center",
            fontsize=7.5,
            color="#D55E00" if missing_by_type.get(label, 0) else INK,
        )
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.14, top=0.82)
    return _save_figure(fig, output_path)


def plot_corpus_coverage(payload: dict[str, object], output_path: Path) -> Path:
    """Plot static/dynamic path size and evidence-regime coverage.

    The two panels deliberately use different visual grammars: instruction
    counts are quantitative bars, while memory, alternate-path, repeated-phase,
    and branch-target presence are binary design properties.  This avoids
    implying that categorical coverage flags are measurements on a continuous
    scale.
    """

    rows = payload.get("per_fixture", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("corpus coverage payload must contain fixture rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_names = [str(row["fixture"]) for row in rows]
    y = np.arange(len(rows))
    static_counts = [int(row["static_instruction_count"]) for row in rows]
    executed_counts = [int(row["executed_instruction_count"]) for row in rows]

    fig, (ax_counts, ax_flags) = plt.subplots(
        1,
        2,
        figsize=(11.2, max(4.8, 0.42 * len(rows) + 2.2)),
        gridspec_kw={"width_ratios": (1.28, 1.0)},
    )
    bar_height = 0.34
    ax_counts.barh(
        y - bar_height / 2,
        static_counts,
        height=bar_height,
        color="#B8C4CE",
        edgecolor=INK,
        linewidth=0.5,
        label="static decoded",
    )
    ax_counts.barh(
        y + bar_height / 2,
        executed_counts,
        height=bar_height,
        color=PALETTE[0],
        edgecolor=INK,
        linewidth=0.5,
        label="bounded executed",
    )
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(fixture_names)
    ax_counts.set_xlabel("Instruction rows")
    ax_counts.set_title("Static span versus bounded path", loc="left", fontsize=10)
    ax_counts.legend(loc="lower right", frameon=False, fontsize=8)
    ax_counts.set_xlim(0.0, max(static_counts, default=1) * 1.22)
    for index, (static_count, executed_count) in enumerate(
        zip(static_counts, executed_counts, strict=True)
    ):
        ax_counts.text(
            static_count + 0.08,
            index - bar_height / 2,
            str(static_count),
            va="center",
            fontsize=7.5,
            color=INK,
        )
        ax_counts.text(
            executed_count + 0.08,
            index + bar_height / 2,
            str(executed_count),
            va="center",
            fontsize=7.5,
            color=PALETTE[0],
        )
    _style_axes(ax_counts, grid_axis="x")

    flag_labels = (
        "memory",
        "alternate\npath",
        "repeated\nphase",
        "branch\nedge",
    )
    flag_matrix = np.asarray(
        [
            [
                bool(row["has_memory_variation"]),
                bool(row["has_alternate_dynamic_path"]),
                int(row.get("repeated_phase_count", 0)) > 0,
                bool(row.get("branch_target_classes", [])),
            ]
            for row in rows
        ],
        dtype=float,
    )
    ax_flags.imshow(
        flag_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([LIGHT, PALETTE[2]]),
        vmin=0.0,
        vmax=1.0,
    )
    ax_flags.set_xticks(np.arange(len(flag_labels)))
    ax_flags.set_xticklabels(flag_labels)
    ax_flags.set_yticks(y)
    ax_flags.set_yticklabels(fixture_names)
    ax_flags.set_title("Evidence-regime coverage", loc="left", fontsize=10)
    for row_index in range(flag_matrix.shape[0]):
        for column_index in range(flag_matrix.shape[1]):
            present = bool(flag_matrix[row_index, column_index])
            ax_flags.text(
                column_index,
                row_index,
                "yes" if present else "—",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if present else MUTED,
                fontweight="bold" if present else "normal",
            )
    ax_flags.tick_params(axis="x", labelsize=7)
    _style_axes(ax_flags, grid_axis="")
    for spine in ax_flags.spines.values():
        spine.set_visible(False)

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    fig.suptitle(
        "Transparent corpus design coverage",
        x=0.06,
        y=0.995,
        ha="left",
        va="top",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.06,
        0.94,
        (
            f"{summary.get('fixture_count', len(rows))} fixtures · "
            f"{summary.get('memory_fixture_count', 0)} memory · "
            f"{summary.get('alternate_path_fixture_count', 0)} alternate-path · "
            f"{summary.get('repeated_phase_fixture_count', 0)} repeated-phase"
        ),
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.015,
        "Counts are fixture-design diagnostics; labels and rows are not independent binary samples.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.82, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_action_sensing(payload: dict[str, object], output_path: Path) -> Path:
    """Plot held-out action-conditioned sensing policies and their costs.

    The left panel reports the censored action budget actually used by each
    policy, while the right panel reports the final posterior confidence and
    the configured threshold.  A censored budget is shown when a policy does
    not reach the threshold within the common action limit; this keeps random
    failures visible instead of dropping them from a mean.
    """

    summary = payload.get("summary")
    policies = payload.get("policies")
    if not isinstance(summary, dict) or not isinstance(policies, list) or not policies:
        raise ValueError("action-sensing payload must contain summary and policies")
    mean_actions = summary.get("mean_actions_used")
    mean_confidence = summary.get("mean_final_confidence")
    confidence_rate = summary.get("confidence_rate")
    if not all(
        isinstance(value, dict)
        for value in (mean_actions, mean_confidence, confidence_rate)
    ):
        raise ValueError("action-sensing summary metrics are missing")
    policy_names = [str(policy) for policy in policies]
    display_names = {
        "active": "active",
        "fixed": "fixed order",
        "random": "seeded random",
        "oracle": "oracle upper bound",
    }
    colors = {
        "active": PALETTE[0],
        "fixed": PALETTE[1],
        "random": PALETTE[3],
        "oracle": PALETTE[2],
    }
    labels = [display_names.get(policy, policy) for policy in policy_names]
    x = np.arange(len(policy_names), dtype=float)
    action_values = [
        float(cast(Any, cast(dict[str, object], mean_actions)[policy]))
        for policy in policy_names
    ]
    confidence_values = [
        float(cast(Any, cast(dict[str, object], mean_confidence)[policy]))
        for policy in policy_names
    ]
    rate_values = [
        float(cast(Any, cast(dict[str, object], confidence_rate)[policy]))
        for policy in policy_names
    ]
    threshold = float(cast(Any, payload.get("confidence_threshold", 0.85)))
    max_steps = int(cast(Any, payload.get("max_steps", 0)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    bars = axes[0].bar(
        x,
        action_values,
        color=[colors.get(policy, PALETTE[0]) for policy in policy_names],
        edgecolor=INK,
        linewidth=0.5,
    )
    for bar, value, rate in zip(bars, action_values, rate_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f}\n{rate:.0%} reached",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("mean actions used (censored at budget)")
    axes[0].set_ylim(0.0, max(1.0, max_steps) * 1.25)
    axes[0].set_title("Evidence acquisition budget", loc="left", fontsize=10)

    bars = axes[1].bar(
        x,
        confidence_values,
        color=[colors.get(policy, PALETTE[0]) for policy in policy_names],
        edgecolor=INK,
        linewidth=0.5,
    )
    axes[1].axhline(
        threshold,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold = {threshold:.2f}",
    )
    for bar, value in zip(bars, confidence_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            min(1.03, value + 0.025),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel("mean final posterior confidence")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_title("Confidence after the common budget", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")

    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fixture_count = int(summary.get("fixture_count", 0))
    fig.suptitle(
        "Action-conditioned typed-field sensing",
        x=0.07,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.07,
        0.91,
        (
            f"{fixture_count} held-out fixtures · {len(policy_names)} policies · "
            f"equal normalized action cost · oracle is inadmissible"
        ),
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.015,
        "Active choices use predictive field likelihoods before the held-out value is revealed; random failures remain censored at the same budget.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.78, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_parameter_learning(payload: dict[str, object], output_path: Path) -> Path:
    """Plot held-out observation-only EM and penalized robustness comparators."""

    summary = payload.get("summary")
    rows = payload.get("per_fixture")
    if not isinstance(summary, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("parameter-learning payload must contain summary and rows")
    methods = (
        "weakly_supervised_em",
        "supervised_map",
        "fixed_reference",
        "penalized_dirichlet_map",
    )
    labels = {
        "weakly_supervised_em": "ordered EM",
        "supervised_map": "supervised MAP",
        "fixed_reference": "fixed reference",
        "penalized_dirichlet_map": "penalized MAP",
    }
    colors = {
        "weakly_supervised_em": PALETTE[0],
        "supervised_map": PALETTE[1],
        "fixed_reference": PALETTE[3],
        "penalized_dirichlet_map": PALETTE[4],
    }
    markers = {
        "weakly_supervised_em": "o",
        "supervised_map": "s",
        "fixed_reference": "^",
        "penalized_dirichlet_map": "D",
    }
    fixture_names = [str(row["held_out_fixture"]).removeprefix("x86_") for row in rows]
    x = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 5.2),
        gridspec_kw={"width_ratios": (1.25, 1.15, 0.95)},
    )
    means_by_method: dict[str, float] = {}
    for method in methods:
        values = [
            float(cast(float, cast(dict[str, object], row[method])["accuracy"]))
            for row in rows
        ]
        means_by_method[method] = float(np.mean(values))
        axes[0].plot(
            x,
            values,
            color=colors[method],
            marker=markers[method],
            markersize=5.5,
            linewidth=1.4,
            alpha=0.95,
            label=labels[method],
        )
        axes[0].plot(
            [x[-1] + 0.16, x[-1] + 0.42],
            [means_by_method[method], means_by_method[method]],
            color=colors[method],
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )
        axes[0].text(
            x[-1] + 0.46,
            means_by_method[method],
            f"{means_by_method[method]:.3f}",
            color=colors[method],
            fontsize=7,
            va="center",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fixture_names, rotation=22, ha="right", fontsize=8)
    axes[0].set_xlim(-0.35, len(rows) - 0.15 + 0.55)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("held-out temporal accuracy")
    axes[0].set_title("Complete-fixture holdouts", loc="left", fontsize=10)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
    )

    histories: list[list[float]] = []
    for row in rows:
        diagnostics = cast(dict[str, object], row["fit_diagnostics"])
        history = cast(list[dict[str, object]], diagnostics["history"])
        values = [float(cast(float, item["log_evidence_after"])) for item in history]
        histories.append(values)
        axes[1].plot(
            np.arange(1, len(values) + 1),
            values,
            color=PALETTE[0],
            linewidth=0.8,
            alpha=0.28,
        )
    max_length = max(len(values) for values in histories)
    means = [
        float(np.mean([values[index] for values in histories if len(values) > index]))
        for index in range(max_length)
    ]
    axes[1].plot(
        np.arange(1, max_length + 1),
        means,
        color=INK,
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="holdout mean",
    )
    axes[1].set_xlabel("EM iteration")
    axes[1].set_ylabel("training log evidence")
    axes[1].set_title("Observation-only EM convergence", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")

    initialization_order = (
        "reference",
        "uniform",
        "random_11",
        "random_23",
        "random_47",
    )
    initialization_labels = {
        "reference": "reference",
        "uniform": "uniform",
        "random_11": "random 11",
        "random_23": "random 23",
        "random_47": "random 47",
    }
    initialization_colors = {
        name: PALETTE[index % len(PALETTE)]
        for index, name in enumerate(initialization_order)
    }
    for index, initialization in enumerate(initialization_order):
        start_values: list[float] = []
        for row in rows:
            starts = row.get("multi_start")
            if not isinstance(starts, list):
                raise ValueError("parameter-learning row lacks multi-start records")
            match = next(
                (
                    start
                    for start in starts
                    if isinstance(start, dict)
                    and start.get("initialization") == initialization
                ),
                None,
            )
            if not isinstance(match, dict):
                raise ValueError(f"missing multi-start {initialization}")
            held_out_score = match.get("held_out_score")
            accuracy = (
                held_out_score.get("accuracy")
                if isinstance(held_out_score, dict)
                else None
            )
            if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
                raise ValueError("multi-start held-out accuracy is not numeric")
            start_values.append(float(accuracy))
        jitter = (
            np.arange(len(start_values), dtype=float) - (len(start_values) - 1) / 2
        ) * 0.018
        x_position = np.full(len(start_values), float(index)) + jitter
        axes[2].scatter(
            x_position,
            start_values,
            s=24,
            color=initialization_colors[initialization],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean_value = float(np.mean(start_values))
        axes[2].plot(
            [index - 0.22, index + 0.22],
            [mean_value, mean_value],
            color=initialization_colors[initialization],
            linewidth=2.0,
            zorder=4,
        )
    axes[2].set_xticks(
        np.arange(len(initialization_order)),
        [initialization_labels[name] for name in initialization_order],
        rotation=28,
        ha="right",
        fontsize=7.5,
    )
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_ylabel("held-out temporal accuracy")
    axes[2].set_title("Initialization sensitivity", loc="left", fontsize=10)
    axes[2].text(
        0.02,
        0.03,
        "dots = holdouts; bar = mean",
        transform=axes[2].transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    for ax in axes:
        _style_axes(ax, grid_axis="y")

    figure_count = int(summary.get("fixture_count", len(rows)))
    converged = int(summary.get("converged_holdout_count", 0))
    fig.suptitle(
        "Leakage-safe HMM parameter learning",
        x=0.07,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.07,
        0.91,
        (
            f"{figure_count} complete-fixture holdouts · observation IDs only · "
            f"{converged}/{figure_count} EM fits converged"
        ),
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    multi_start_count = int(summary.get("multi_start_count", 0))
    fig.text(
        0.07,
        0.015,
        (
            "Accuracy is held-out; training log evidence is an optimization diagnostic. "
            "Penalized MAP uses fixed Dirichlet pseudocounts and posterior-predictive checks "
            "use observation summaries only. "
            f"The sensitivity panel retains {multi_start_count} declared-start fits; "
            "none is a population uncertainty interval."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.24, top=0.78, wspace=0.34)
    return _save_figure(fig, output_path)


def plot_scholarship_source_roles(
    sources: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot scholarship source types used by the manuscript."""

    if not sources:
        raise ValueError("scholarship sources must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for source in sources:
        source_type = str(source["source_type"])
        counts[source_type] = counts.get(source_type, 0) + 1
    labels = sorted(counts, key=lambda label: counts[label], reverse=True)
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8.6, max(4.4, 0.42 * len(labels) + 1.8)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=PALETTE[0], edgecolor=INK, linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([label.replace("_", " ") for label in labels])
    ax.set_xlabel("Sources in curated ledger")
    ax.set_title(
        f"Scholarship ledger composition ({len(sources)} sources)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Source-type counts describe curation roles; they are not weights of empirical evidence.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            ha="left",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.14, top=0.82)
    return _save_figure(fig, output_path)


def plot_scholarship_evidence_matrix(
    sources: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot source counts by project stage and evidence tier."""

    if not sources:
        raise ValueError("scholarship sources must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stages = ["implemented", "comparison", "roadmap", "context"]
    tier_order = [
        "implemented_backend",
        "implemented_dependency",
        "comparison_baseline",
        "assembly_formalism",
        "statistical_method",
        "integration_guidance",
        "binary_analysis_context",
        "theory",
        "roadmap_adapter",
    ]
    present_tiers = {str(source["evidence_tier"]) for source in sources}
    tiers = [tier for tier in tier_order if tier in present_tiers]
    tiers.extend(sorted(present_tiers - set(tiers)))
    tier_index = {tier: index for index, tier in enumerate(tiers)}
    stage_index = {stage: index for index, stage in enumerate(stages)}
    matrix = np.zeros((len(tiers), len(stages)), dtype=float)
    for source in sources:
        stage = str(source["project_stage"])
        tier = str(source["evidence_tier"])
        if stage in stage_index and tier in tier_index:
            matrix[tier_index[tier], stage_index[stage]] += 1.0

    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.46 * len(tiers) + 1.6)))
    image = ax.imshow(
        matrix,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=max(1.0, float(matrix.max())),
    )
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(tiers)))
    ax.set_yticklabels([tier.replace("_", " ") for tier in tiers], fontsize=8)
    ax.set_title(
        "Scholarship evidence tiers by implementation stage",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Cells are source counts; zero cells are shown explicitly so roadmap and implemented evidence stay distinct.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlabel("Project stage")
    ax.set_ylabel("Evidence tier")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= max(1, int(matrix.max() * 0.55)) else INK,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Source count")
    _style_axes(ax, grid_axis="none")
    fig.subplots_adjust(left=0.30, right=0.90, bottom=0.22, top=0.82)
    return _save_figure(fig, output_path)


def plot_claim_evidence_map(
    claims: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot claim coverage by source, artifact, and validation-link counts."""

    if not claims:
        raise ValueError("claim records must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(claim["claim_id"]) for claim in claims]
    source_counts = [len(cast(list, claim["support_keys"])) for claim in claims]
    artifact_counts = [len(cast(list, claim["artifact_paths"])) for claim in claims]
    validation_counts = [
        len(cast(list, claim["validation_checks"])) for claim in claims
    ]
    y = np.arange(len(claims), dtype=float)
    series = (
        ("sources", source_counts, PALETTE[0], "o", 0.14),
        ("artifacts", artifact_counts, PALETTE[2], "s", 0.0),
        ("checks", validation_counts, PALETTE[1], "^", -0.14),
    )

    fig, ax = plt.subplots(figsize=(9.4, max(6.4, 0.36 * len(labels) + 2.0)))
    for label, values, color, marker, y_offset in series:
        row_y = y + y_offset
        ax.scatter(
            values, row_y, s=42, color=color, marker=marker, label=label, zorder=3
        )
        for yi, value in zip(row_y, values, strict=True):
            ax.text(value + 0.12, yi, str(value), va="center", fontsize=7, color=INK)
    for yi in y:
        ax.axhline(yi, color=GRID, linewidth=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of linked sources, artifacts, or validation checks")
    ax.set_title(
        f"Claim-register traceability ({len(claims)} bounded claims)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Each claim row has three vertically separated evidence markers; counts audit traceability, not independent replication.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlim(
        -0.45,
        max([*source_counts, *artifact_counts, *validation_counts], default=1) + 1.5,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3)
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.82)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# temporal / active / evaluation visualizations
# ---------------------------------------------------------------------------


def plot_temporal_beliefs_heatmap(
    fixture_name: str,
    observation_ids: list[int],
    filtered_posteriors: list[list[float]],
    hidden_state_labels: list[str],
    output_path: Path,
) -> Path:
    """Plot a single fixture's forward-filter belief trajectory as a heatmap.

    Rows are observation steps (labelled by decoded observation category); columns
    are the five latent program-phase beliefs. Each row is a normalized belief, so
    columns sum to one row-wise.
    """

    if not filtered_posteriors:
        raise ValueError("filtered_posteriors must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(filtered_posteriors, dtype=float)
    step_labels = [
        f"{index + 1}: obs {obs_id}" for index, obs_id in enumerate(observation_ids)
    ]

    fig, ax = plt.subplots(figsize=(8.1, max(2.8, 0.42 * len(step_labels) + 1.4)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(step_labels)))
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.set_xticks(np.arange(len(hidden_state_labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in hidden_state_labels],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Observation step")
    ax.set_xlabel("Latent program-phase belief")
    ax.set_title(
        f"Temporal phase beliefs — {fixture_name}",
        loc="left",
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Rows are normalized posterior distributions; the cell text reports probability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Posterior probability")
    _style_axes(ax, grid_axis="none")
    fig.subplots_adjust(left=0.19, right=0.88, bottom=0.18, top=0.84)
    return _save_figure(fig, output_path)


def plot_model_comparison_bars(
    evaluation_report: dict[str, object],
    output_path: Path,
) -> Path:
    """Plot per-fixture decoding accuracy and pooled accuracy with 95% Wilson CIs.

    Left panel: identity-transition vs memoryless vs temporal accuracy per corpus fixture.
    Right panel: the pooled (micro) accuracy of each decoder with its 95% Wilson
    confidence interval, so the headline uncertainty is visible rather than just
    a point estimate.
    """

    compare = cast(
        dict[str, dict[str, float]], evaluation_report.get("compare_models", {})
    )
    if not compare:
        raise ValueError("evaluation_report must contain compare_models rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_names = list(compare.keys())
    identity_transition = [
        compare[name].get("identity_transition_accuracy", 0.0) for name in fixture_names
    ]
    memoryless = [
        compare[name].get("memoryless_accuracy", 0.0) for name in fixture_names
    ]
    temporal = [compare[name].get("temporal_accuracy", 0.0) for name in fixture_names]
    x = np.arange(len(fixture_names))
    width = 0.26

    summary = cast(dict[str, object], evaluation_report.get("summary", {}))
    pooled_step_count = int(cast(int, summary.get("pooled_step_count", 0)))
    pooled_accs = [
        cast(float, summary.get("pooled_identity_transition_accuracy", 0.0)),
        cast(float, summary.get("pooled_memoryless_accuracy", 0.0)),
        cast(float, summary.get("pooled_temporal_accuracy", 0.0)),
    ]
    ci_lo = [
        cast(list[float], summary.get("identity_transition_ci95", [0.0, 0.0]))[0],
        cast(list[float], summary.get("memoryless_ci95", [0.0, 0.0]))[0],
        cast(list[float], summary.get("temporal_ci95", [0.0, 0.0]))[0],
    ]
    ci_hi = [
        cast(list[float], summary.get("identity_transition_ci95", [0.0, 0.0]))[1],
        cast(list[float], summary.get("memoryless_ci95", [0.0, 0.0]))[1],
        cast(list[float], summary.get("temporal_ci95", [0.0, 0.0]))[1],
    ]
    decoder_labels = ["identity transition", "memoryless", "temporal HMM"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=False)
    ax = axes[0]
    ax.bar(
        x - width,
        identity_transition,
        width=width,
        label="identity transition",
        color=PALETTE[0],
    )
    ax.bar(x, memoryless, width=width, label="memoryless", color=PALETTE[1])
    ax.bar(x + width, temporal, width=width, label="temporal HMM", color=PALETTE[2])
    ax.axhline(
        0.2, color="#333333", linewidth=0.8, linestyle="--", label="chance (0.20)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("x86_", "") for name in fixture_names], rotation=15, ha="right"
    )
    ax.set_ylabel("Phase-decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-fixture accuracy")
    for bars in ax.containers:
        if isinstance(bars, BarContainer):
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    _style_axes(ax, grid_axis="y")

    ax2 = axes[1]
    ypos = np.arange(len(decoder_labels))
    ax2.errorbar(
        pooled_accs,
        ypos,
        xerr=np.array(
            [
                [p - lo for p, lo in zip(pooled_accs, ci_lo)],
                [hi - p for p, hi in zip(pooled_accs, ci_hi)],
            ]
        ),
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.5,
    )
    for yi, (p, color) in enumerate(zip(pooled_accs, colors)):
        ax2.plot(p, yi, "o", color=color, markersize=8)
    pooled_hits = [
        int(
            cast(
                int,
                summary.get(
                    "identity_transition_hits",
                    round(pooled_accs[0] * pooled_step_count),
                ),
            )
        ),
        int(
            cast(
                int,
                summary.get(
                    "memoryless_hits", round(pooled_accs[1] * pooled_step_count)
                ),
            )
        ),
        int(
            cast(
                int,
                summary.get("temporal_hits", round(pooled_accs[2] * pooled_step_count)),
            )
        ),
    ]
    for yi, (p, hits) in enumerate(zip(pooled_accs, pooled_hits, strict=True)):
        ax2.annotate(
            f"{p:.3f}\n{hits}/{pooled_step_count}",
            xy=(p, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax2.axvline(
        0.2, color="#333333", linewidth=0.8, linestyle="--", label="chance (0.20)"
    )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(decoder_labels)
    ax2.set_xlabel("Pooled accuracy (95% Wilson CI)")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title("Pooled accuracy with descriptive 95% CI")
    _style_axes(ax2, grid_axis="x")
    fig.suptitle(
        "Phase-decoder accuracy across the labelled fixture corpus",
        color=INK,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.94,
        f"Left: fixture-level point estimates. Right: pooled hit rate with descriptive Wilson interval; n = {pooled_step_count} labelled steps; chance floor = 0.20.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.82, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_active_efe_trajectory(
    selection: dict[str, object],
    output_path: Path,
    predictive_selection: dict[str, object] | None = None,
) -> Path:
    """Plot realized and predictive EFE trajectories with belief entropy."""

    efe_by_step = cast(list[float], selection.get("efe_by_step"))
    entropy_by_step = cast(list[float], selection.get("belief_entropy_by_step"))
    if not efe_by_step or not entropy_by_step:
        raise ValueError(
            "selection must contain efe_by_step and belief_entropy_by_step"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(efe_by_step) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)
    axes[0].plot(
        steps,
        efe_by_step,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
        label="decoded-category EFE",
    )
    if predictive_selection is not None:
        predictive_efe = cast(list[float], predictive_selection.get("efe_by_step", []))
        if len(predictive_efe) != len(efe_by_step):
            raise ValueError(
                "predictive selection must have the same trajectory length"
            )
        axes[0].plot(
            steps,
            predictive_efe,
            marker="^",
            linewidth=1.6,
            linestyle="--",
            color=PALETTE[1],
            label="predictive contribution EFE",
        )
    axes[0].set_ylabel("EFE score (nats)")
    axes[0].set_title("Expected free energy", loc="left", fontsize=10, color=INK)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[1].plot(steps, entropy_by_step, marker="s", linewidth=2, color=PALETTE[2])
    axes[1].set_ylabel("Belief entropy (nats)")
    axes[1].set_xlabel("Active-selection step")
    axes[1].set_title(
        "Posterior uncertainty after selection", loc="left", fontsize=10, color=INK
    )
    for axis in axes:
        _style_axes(axis, grid_axis="y")
    fig.suptitle("Active-disassembly trajectory", color=INK, fontsize=12)
    fig.text(
        0.5,
        0.94,
        "Realized EFE uses the decoded category; predictive EFE weights its information contribution by prior category probability.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.84, hspace=0.48)
    return _save_figure(fig, output_path)


def plot_assembly_cfg(payload: dict[str, object], output_path: Path) -> Path:
    """Draw the bounded direct-target CFG emitted from real Capstone events.

    The figure is intentionally a static evidence diagram rather than a generic
    graph layout: block order is deterministic, every block retains its address
    range and decoded mnemonics, and out-of-span branch targets remain explicit
    external nodes. Dynamic execution is shown as a border treatment, not as a
    claim that the observed path exhausts the CFG.
    """

    blocks = cast(list[dict[str, object]], payload.get("basic_blocks", []))
    edges = cast(list[dict[str, object]], payload.get("cfg_edges", []))
    instructions = cast(
        list[dict[str, object]], payload.get("static_instruction_events", [])
    )
    if not blocks or not instructions:
        raise ValueError("CFG payload must contain blocks and instructions")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instruction_by_index = {index: row for index, row in enumerate(instructions)}
    dynamic_addresses = {
        int(cast(int, address))
        for address in cast(list[object], payload.get("dynamic_addresses", []))
    }
    positions = {
        str(block["block_id"]): (float(index) * 2.85, 0.0)
        for index, block in enumerate(blocks)
    }
    external_ids = sorted(
        {
            str(edge["target_block_id"])
            for edge in edges
            if str(edge.get("target_block_id", "")).startswith("external_")
        }
    )
    for index, external_id in enumerate(external_ids):
        positions[external_id] = (float(len(blocks) + index) * 2.85, 0.0)

    fig, ax = plt.subplots(figsize=(max(9.0, 2.85 * len(positions)), 4.8))
    node_width = 2.15
    node_height = 1.45
    for block in blocks:
        block_id = str(block["block_id"])
        x, y = positions[block_id]
        indexes = [
            int(cast(int, value))
            for value in cast(list[object], block["instruction_indexes"])
        ]
        rows = [instruction_by_index[index] for index in indexes]
        lines = [
            block_id,
            f"0x{int(cast(int, block['start_address'])):x}–"
            f"0x{int(cast(int, block['end_address'])):x}",
        ]
        lines.extend(f"{row['mnemonic']} {row['op_str']}".rstrip() for row in rows)
        dynamic = any(
            int(cast(int, row["address"])) in dynamic_addresses for row in rows
        )
        patch = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor="#E8F1F8" if dynamic else "#F5F7FA",
            edgecolor=PALETTE[0] if dynamic else "#9AA5B1",
            linewidth=2.0 if dynamic else 1.2,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            family="DejaVu Sans Mono",
        )
        ax.text(
            x,
            y - node_height / 2 - 0.18,
            "dynamic path" if dynamic else "static only",
            ha="center",
            va="top",
            fontsize=7,
            color=PALETTE[0] if dynamic else MUTED,
        )

    for external_id in external_ids:
        x, y = positions[external_id]
        patch = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor="#FFF4DE",
            edgecolor=PALETTE[1],
            linewidth=1.5,
            linestyle="--",
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            f"{external_id}\noutside decoded span",
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            family="DejaVu Sans Mono",
        )

    for edge in edges:
        source = str(edge["source_block_id"])
        target = str(edge["target_block_id"])
        if source not in positions or target not in positions:
            continue
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        edge_type = str(edge.get("edge_type", "edge"))
        color = PALETTE[1] if edge_type == "taken" else PALETTE[2]
        arrow = FancyArrowPatch(
            (source_x + node_width / 2, source_y),
            (target_x - node_width / 2, target_y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color=color,
            connectionstyle=(
                "arc3,rad=0.22" if edge_type == "taken" else "arc3,rad=-0.22"
            ),
            zorder=1,
        )
        ax.add_patch(arrow)
        ax.text(
            (source_x + target_x) / 2,
            0.16 if edge_type == "fallthrough" else -0.23,
            edge_type,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )

    ax.set_xlim(-1.55, float(len(positions) - 1) * 2.85 + 1.55)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.set_title(
        f"Static CFG evidence for {payload.get('sample_id', 'fixture')}",
        loc="left",
        color=INK,
        fontsize=12,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Capstone-derived blocks and direct edges; blue border = observed by bounded Unicorn execution; amber dashed node = explicit external target.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.0,
        -0.10,
        "edge colors: green = fall-through · amber = branch-taken possibility",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    return _save_figure(fig, output_path)


def plot_memory_access_timeline(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the static-to-dynamic memory-evidence join for one fixture.

    The left panel keeps instruction order discrete and marks the Capstone
    memory operands.  The right panel is an event table ordered by Unicorn's
    memory-hook sequence; addresses are shown relative to the captured stack
    pointer so the plot exposes the effective-address calculation without
    pretending that absolute mapped addresses are biologically meaningful.
    """

    instructions = cast(
        list[dict[str, object]], payload.get("static_instruction_events", [])
    )
    events = cast(list[dict[str, object]], payload.get("dynamic_memory_events", []))
    if not instructions or not events:
        raise ValueError("memory payload must contain instructions and events")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    memory_indexes = {
        int(index)
        for index in cast(
            list[int], payload.get("static_memory_instruction_indexes", [])
        )
    }
    instruction_labels = [
        f"{index}: {row['mnemonic']} {row['op_str']}".rstrip()
        for index, row in enumerate(instructions)
    ]
    colors = [
        PALETTE[1] if index in memory_indexes else "#D9E2EC"
        for index in range(len(instructions))
    ]

    fig, (ax_static, ax_events) = plt.subplots(
        1,
        2,
        figsize=(12.0, max(4.8, 0.55 * len(instructions) + 2.0)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    y_static = np.arange(len(instructions))[::-1]
    ax_static.barh(
        y_static, np.ones(len(instructions)), color=colors, edgecolor=INK, linewidth=0.6
    )
    ax_static.set_yticks(y_static)
    ax_static.set_yticklabels(
        instruction_labels, fontfamily="DejaVu Sans Mono", fontsize=8
    )
    ax_static.set_xlim(0.0, 1.0)
    ax_static.set_xticks([])
    ax_static.set_xlabel("Capstone static operand classification")
    ax_static.set_title("Static memory operands", loc="left", color=INK, fontsize=10)
    for y, index in zip(y_static, range(len(instructions)), strict=True):
        label = (
            "memory operand" if index in memory_indexes else "register/immediate only"
        )
        ax_static.text(
            0.98,
            y,
            label,
            ha="right",
            va="center",
            fontsize=7,
            color=INK if index in memory_indexes else MUTED,
        )
    _style_axes(ax_static, grid_axis="x")

    event_rows = list(reversed(events))
    y_events = np.arange(len(event_rows))
    event_colors = [
        PALETTE[0] if str(row["access"]) == "read" else PALETTE[1] for row in event_rows
    ]
    ax_events.barh(
        y_events,
        np.ones(len(event_rows)),
        color=event_colors,
        edgecolor=INK,
        linewidth=0.6,
    )
    ax_events.set_yticks(y_events)
    ax_events.set_yticklabels(
        [
            f"#{int(cast(int, row['event_index']))}  {str(row['access']).upper()}  "
            f"idx {int(cast(int, row['instruction_index']))}"
            for row in event_rows
        ],
        fontfamily="DejaVu Sans Mono",
        fontsize=8,
    )
    ax_events.set_xlim(0.0, 1.0)
    ax_events.set_xticks([])
    ax_events.set_xlabel("Unicorn memory-hook order")
    ax_events.set_title("Dynamic access events", loc="left", color=INK, fontsize=10)
    for y, row in zip(y_events, event_rows, strict=True):
        ax_events.text(
            0.98,
            y,
            f"rsp{int(cast(int, row['stack_relative_address'])):+d}  "
            f"value={int(cast(int, row['value']))}",
            ha="right",
            va="center",
            fontsize=7,
            color=INK,
        )
    _style_axes(ax_events, grid_axis="x")

    summary = cast(dict[str, object], payload.get("summary", {}))
    fig.suptitle(
        f"Static-to-dynamic memory provenance for {payload.get('sample_id', 'fixture')}",
        color=INK,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.935,
        f"{summary.get('read_count', 0)} read · {summary.get('write_count', 0)} write · "
        f"{summary.get('distinct_addresses', 0)} distinct effective address(es); values are observed fixture bytes",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.82, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_calibration_curve(bins: list[dict[str, object]], output_path: Path) -> Path:
    """Plot a reliability (calibration) diagram from per-bin statistics.

    Each bin's mean confidence (max posterior) is drawn against its observed
    accuracy, with the identity line for perfect calibration. Point labels show
    the number of steps in each bin.
    """

    if not bins:
        raise ValueError("calibration bins must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .evaluation import wilson_ci

    confidences = [cast(float, bin_["confidence"]) for bin_ in bins]
    accuracies = [cast(float, bin_["accuracy"]) for bin_ in bins]
    counts = [cast(int, bin_["count"]) for bin_ in bins]
    bin_indices = [
        int(cast(int, bin_.get("bin_index", index))) for index, bin_ in enumerate(bins)
    ]
    configured_bin_count = max(bin_indices, default=len(bins) - 1) + 1
    if any(count <= 0 for count in counts):
        raise ValueError("calibration bins must contain positive counts")
    if any(not 0.0 <= accuracy <= 1.0 for accuracy in accuracies):
        raise ValueError("calibration accuracies must lie in [0, 1]")
    intervals = [
        wilson_ci(int(round(accuracy * count)), count)
        for accuracy, count in zip(accuracies, counts, strict=True)
    ]
    lower_error = np.asarray(
        [
            accuracy - lower
            for accuracy, (lower, _) in zip(accuracies, intervals, strict=True)
        ]
    )
    upper_error = np.asarray(
        [
            upper - accuracy
            for accuracy, (_, upper) in zip(accuracies, intervals, strict=True)
        ]
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    ax.errorbar(
        confidences,
        accuracies,
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        linewidth=0.0,
        color=PALETTE[0],
        capsize=3,
        markersize=5,
        label="Forward filter",
    )
    for cx, cy, count, bin_index in zip(
        confidences, accuracies, counts, bin_indices, strict=True
    ):
        ax.annotate(
            f"b{bin_index + 1}\nn={count}",
            (cx, cy),
            textcoords="offset points",
            xytext=(4, -14 if cy > 0.9 else 4),
            fontsize=8,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence (mean max posterior)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(
        "Reliability diagram for the temporal forward filter",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"{len(bins)} non-empty bins among {configured_bin_count} configured equal-width bins; b labels the bin and n its denominator; whiskers are 95% Wilson intervals around bin hit counts (descriptive, n={sum(counts)}). Empty bins are not connected.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(loc="lower right", fontsize=8)
    _style_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.82)
    return _save_figure(fig, output_path)


def plot_confusion_matrix(pooled: dict[str, object], output_path: Path) -> Path:
    """Plot a pooled phase-decoding confusion matrix as an annotated heatmap."""

    labels = cast(list[str], pooled.get("labels"))
    matrix = np.asarray(cast(list[list[int]], pooled.get("matrix")), dtype=float)
    if matrix.size == 0 or matrix.shape != (len(labels), len(labels)):
        raise ValueError("confusion matrix must be square and match label count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_count = int(matrix.sum())

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels], rotation=30, ha="right"
    )
    ax.set_yticklabels([label.replace("_", " ") for label in labels])
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            row_total = matrix[row].sum()
            percent = value / row_total if row_total else 0.0
            ax.text(
                column,
                row,
                f"{value}\n({percent:.0%})",
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=9,
            )
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_title(
        f"Pooled temporal phase-decoding confusion matrix (n={total_count})",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Cells show count and row percentage; rows are true phases and columns are predicted phases.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    _style_axes(ax, grid_axis="none")
    fig.subplots_adjust(left=0.23, right=0.90, bottom=0.18, top=0.83)
    return _save_figure(fig, output_path)


def plot_ambiguity_ablation(ablation: dict[str, object], output_path: Path) -> Path:
    """Plot decoder accuracy, segmentation F1, and EFE-active advantage vs ambiguity.

    Three panels. Left: mean step-decoding accuracy of the temporal,
    memoryless, identity-transition, and Viterbi decoders as the likelihood is
    blended toward uniform —
    per-step decoding degrades and the temporal transition structure carries the
    information. Middle: pooled phase-boundary F1 for the temporal (smoothed) and
    memoryless segmentations. Right: how many fixtures reach confidence in fewer
    steps under the EFE-active policy than under a fixed scan — an advantage that
    appears only in an intermediate-ambiguity band where observations become
    differentially informative. The crossover (temporal first beating memoryless)
    is annotated on every panel, and the figure shows all three inference layers
    gaining their mechanism exactly under observation ambiguity.
    """

    rows = cast(list[dict[str, object]], ablation.get("rows"))
    if not rows:
        raise ValueError("ambiguity ablation must contain rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blends = [cast(float, row["blend"]) for row in rows]
    crossover: float | None = cast(float | None, ablation.get("crossover_blend"))

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2), sharex=True)
    ax_left, ax_mid, ax_right = axes

    accuracy_series = {
        "temporal HMM": ("temporal_accuracy", PALETTE[2], "-"),
        "memoryless": ("memoryless_accuracy", PALETTE[1], "--"),
        "identity transition": ("identity_transition_accuracy", PALETTE[0], ":"),
        "viterbi": ("viterbi_accuracy", "#CC79A7", "-."),
    }
    for label, (key, color, linestyle) in accuracy_series.items():
        ax_left.plot(
            blends,
            [cast(float, row[key]) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    if crossover is not None:
        ax_left.axvline(
            crossover,
            color="#333333",
            linewidth=1.0,
            linestyle=":",
            label=f"crossover {crossover:.1f}",
        )
    ax_left.set_xlabel("Likelihood blend toward uniform")
    ax_left.set_ylabel("Mean decoding accuracy")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_title("Step decoding accuracy")
    _style_axes(ax_left, grid_axis="both")

    ax_mid.plot(
        blends,
        [cast(float, row.get("temporal_segmentation_f1", 0.0)) for row in rows],
        marker="s",
        linewidth=2,
        color=PALETTE[2],
        linestyle="-",
        label="temporal segmentation",
    )
    ax_mid.plot(
        blends,
        [cast(float, row.get("memoryless_segmentation_f1", 0.0)) for row in rows],
        marker="s",
        linewidth=2,
        color="#555555",
        linestyle="--",
        label="memoryless segmentation",
    )
    if crossover is not None:
        ax_mid.axvline(crossover, color="#333333", linewidth=1.0, linestyle=":")
    ax_mid.set_xlabel("Likelihood blend toward uniform")
    ax_mid.set_ylabel("Pooled boundary F1")
    ax_mid.set_ylim(0.0, 1.0)
    ax_mid.set_title("Phase-segmentation boundary F1")
    _style_axes(ax_mid, grid_axis="both")

    active_counts = [cast(int, row.get("active_advantage_count", 0)) for row in rows]
    fixture_count = int(
        cast(int, ablation.get("fixture_count", max(active_counts, default=1)))
    )
    step = (
        (blends[-1] - blends[0]) / max(1, len(blends) - 1) if len(blends) > 1 else 0.05
    )
    ax_right.bar(
        blends,
        active_counts,
        width=0.7 * step,
        color=PALETTE[2],
        alpha=0.85,
        label="# fixtures where EFE active reaches confidence first",
    )
    if crossover is not None:
        ax_right.axvline(crossover, color="#333333", linewidth=1.0, linestyle=":")
    ax_right.set_xlabel("Likelihood blend toward uniform")
    ax_right.set_ylabel("Fixtures with EFE-active advantage")
    ax_right.set_ylim(0.0, max(fixture_count, 1) + 0.6)
    ax_right.set_yticks(range(0, max(fixture_count, 1) + 1))
    ax_right.set_title("Active-policy advantage")
    for blend, count in zip(blends, active_counts, strict=True):
        ax_right.text(
            blend,
            count + 0.12,
            f"{count}/{fixture_count}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    _style_axes(ax_right, grid_axis="y")
    axes[0].set_xticks(blends)
    axes[0].set_xticklabels([f"{blend:.1f}" for blend in blends])
    handles, labels = ax_left.get_legend_handles_labels()
    handles_mid, labels_mid = ax_mid.get_legend_handles_labels()
    fig.suptitle(
        "Decoder and active-policy behavior under likelihood ambiguity",
        color=INK,
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        "Blend 0 is the authored likelihood and blend 1 is uniform; the same labelled fixtures are reused at every blend.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.legend(
        handles + handles_mid,
        labels + labels_mid,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.73, wspace=0.28)
    return _save_figure(fig, output_path)


def _draw_segment_runs(
    ax: Any, y: float, sequence: Sequence[str], phase_colors: dict[str, Any]
) -> None:
    """Draw a phase sequence as contiguous colored runs on one ribbon row."""

    i = 0
    while i < len(sequence):
        j = i
        while j < len(sequence) and sequence[j] == sequence[i]:
            j += 1
        ax.barh(
            y,
            j - i,
            left=i,
            height=0.8,
            color=phase_colors.get(sequence[i], "#cccccc"),
            edgecolor="white",
            linewidth=0.5,
        )
        i = j


def plot_segmentation_ribbons(
    evaluation_report: dict[str, object], output_path: Path
) -> Path:
    """Plot per-fixture ground-truth vs predicted phase-segmentation ribbons.

    For each fixture two horizontal ribbons are drawn: the hand-authored
    ground-truth phase sequence (top) and the phase sequence decoded from the
    smoothed forward-backward beliefs (bottom). Contiguous segments sharing a
    phase are merged into single colored runs, so phase boundaries appear as
    color changes. This makes the measured segmentation (CLM-015) visual.
    """

    segmented = cast(dict[str, object], evaluation_report.get("segmentation"))
    per_fixture = cast(dict[str, dict[str, object]], segmented.get("per_fixture"))
    if not per_fixture:
        raise ValueError("segmentation per_fixture must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    for entry in per_fixture.values():
        for label in cast(list[str], entry.get("truth_labels", [])) + cast(
            list[str], entry.get("predicted_labels", [])
        ):
            if label not in labels:
                labels.append(label)
    phase_colors = {
        label: PALETTE[index % len(PALETTE)] for index, label in enumerate(labels)
    }

    max_len = max(
        len(cast(list[str], entry.get("truth_labels", [])))
        for entry in per_fixture.values()
    )
    n_fixtures = len(per_fixture)
    fig, ax = plt.subplots(figsize=(0.55 * max_len + 2.0, 1.2 * n_fixtures + 1.5))

    y = n_fixtures * 2 - 1
    yticks: list[int] = []
    yticklabels: list[str] = []
    for name, entry in per_fixture.items():
        truth = cast(list[str], entry.get("truth_labels", []))
        predicted = cast(list[str], entry.get("predicted_labels", []))
        _draw_segment_runs(ax, y, truth, phase_colors)
        yticks.append(y)
        yticklabels.append(f"{name.replace('x86_', '')} · truth")
        y -= 1
        _draw_segment_runs(ax, y, predicted, phase_colors)
        yticks.append(y)
        yticklabels.append(f"{name.replace('x86_', '')} · predicted")
        y -= 1

    pooled = cast(dict[str, object], segmented.get("pooled", {}))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlim(0, max_len)
    ax.set_xticks(range(0, max_len + 1))
    ax.set_xlabel("Instruction position within fixture (0 = first decoded instruction)")
    ax.set_title(
        "Phase segmentation: authored truth versus smoothed prediction",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"Pooled boundary F1 = {float(cast(float, pooled.get('boundary_f1', 0.0))):.3f}; colors encode phase and each truth/prediction pair shares one fixture axis.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for i in range(1, max_len):
        ax.axvline(i, color="#dddddd", linewidth=0.5, zorder=0)
    proxies = [
        Rectangle(
            (0, 0),
            1,
            1,
            color=phase_colors[label],
            label=label.replace("_", " "),
        )
        for label in labels
    ]
    ax.legend(
        handles=proxies,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8,
        ncol=3,
        frameon=False,
    )
    ax.set_ylim(-1, n_fixtures * 2)
    _style_axes(ax, grid_axis="none")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.82)
    return _save_figure(fig, output_path)


def plot_corpus_stability(stability: dict[str, object], output_path: Path) -> Path:
    """Plot leave-one-out pooled temporal accuracy to quantify corpus stability.

    Bars show the pooled temporal decoding accuracy on the full corpus and after
    holding out each single fixture; the dashed line is the pooled
    identity-transition
    accuracy and the dotted line is chance. The headline temporal result surviving
    every single-fixture removal is made visual, and the span is annotated so the
    small-corpus fragility is explicit rather than hidden.
    """

    reference = cast(dict[str, object], stability.get("reference", {}))
    per_holdout = cast(list[dict[str, object]], stability.get("per_holdout", []))
    if not per_holdout:
        raise ValueError("corpus stability must contain per_holdout rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["full corpus"] + [
        cast(str, row["held_out_fixture"]).replace("x86_", "") for row in per_holdout
    ]
    temporal = [cast(float, reference.get("temporal", 0.0))] + [
        cast(float, row["temporal"]) for row in per_holdout
    ]
    identity_transition = [cast(float, reference.get("identity_transition", 0.0))] + [
        cast(float, row["identity_transition"]) for row in per_holdout
    ]
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(
        x, temporal, width=0.55, color=PALETTE[2], label="temporal HMM pooled accuracy"
    )
    ax.plot(
        x,
        identity_transition,
        marker="x",
        linewidth=1.5,
        linestyle="--",
        color=PALETTE[0],
        label="identity-transition pooled",
    )
    chance = float(cast(float, stability.get("chance_floor", 0.2)))
    ax.axhline(
        chance,
        color="#555555",
        linewidth=1.0,
        linestyle=":",
        label=f"chance ({chance:.1f})",
    )
    pooled_range = cast(dict[str, object], stability.get("temporal_pooled_range", {}))
    span = float(cast(float, pooled_range.get("span", 0.0)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pooled decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Leave-one-fixture-out pooled temporal accuracy",
        loc="left",
        color=INK,
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"{len(per_holdout)} held-out fixtures · observed temporal span = {span:.2f}; bars are pooled over remaining steps.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for xi, value in zip(x, temporal, strict=True):
        ax.text(
            xi,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.legend(loc="lower right", fontsize=8)
    _style_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.82)
    return _save_figure(fig, output_path)


def plot_phase_recovery(phase_recovery: dict[str, object], output_path: Path) -> Path:
    """Plot per-phase recall and F1 to show which phases are recoverable.

    Phases are ordered by F1 so the hardest (underdetermined) phases are on the
    left; recall and F1 are grouped bars, and the worst phase is highlighted. This
    makes quantitative — rather than asserted — which phases the temporal forward
    filter recovers (e.g. that ``exit`` and especially ``guard_check`` are
    observation-underdetermined).
    """

    per_phase = cast(dict[str, dict[str, object]], phase_recovery.get("per_phase"))
    if not per_phase:
        raise ValueError("phase recovery must contain per_phase rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(
        list(per_phase.keys()), key=lambda p: float(cast(float, per_phase[p]["f1"]))
    )
    f1s = [float(cast(float, per_phase[p]["f1"])) for p in labels]
    recalls = [float(cast(float, per_phase[p]["recall"])) for p in labels]
    true_counts = [int(cast(int, per_phase[p].get("true_count", 0))) for p in labels]
    x = np.arange(len(labels))
    hardest = cast(str, phase_recovery.get("hardest_phase", labels[0]))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(
        x - 0.18,
        recalls,
        width=0.36,
        color="#9ecae1",
        edgecolor=INK,
        linewidth=0.4,
        label="recall",
    )
    f1_bars = ax.bar(
        x + 0.18,
        f1s,
        width=0.36,
        color=PALETTE[2],
        edgecolor=INK,
        linewidth=0.4,
        label="F1",
    )
    for phase, bar in zip(labels, f1_bars, strict=True):
        if phase == hardest:
            bar.set_hatch("//")
            bar.set_edgecolor(INK)
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label.replace('_', ' ')}\n(n={count})"
            for label, count in zip(labels, true_counts, strict=True)
        ],
        fontsize=8,
    )
    ax.set_ylabel("Recovery score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Per-phase recovery of the temporal forward filter",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"Macro F1 = {float(cast(float, phase_recovery.get('macro_f1', 0.0))):.2f}; hatched F1 bar = hardest phase ({hardest.replace('_', ' ')}).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for bars in ax.containers:
        if isinstance(bars, BarContainer):
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    _style_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.23, top=0.82)
    return _save_figure(fig, output_path)


def plot_decoder_calibration(
    calibration_payload: dict[str, object], output_path: Path
) -> Path:
    """Plot ECE and Brier for each decoder to compare calibration quality.

    Shows expected calibration error (ECE) and Brier score for the temporal,
    identity-transition, and memoryless decoders (ordered by ECE). This makes
    the honest result visual: on this corpus calibration tracks accuracy — the
    memoryless decoder is best calibrated and the identity-transition comparator
    worst.
    """

    decoders = cast(dict[str, dict[str, object]], calibration_payload.get("decoders"))
    if not decoders:
        raise ValueError("decoder calibration must contain decoders")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = ["memoryless", "temporal", "identity_transition"]
    eces = [float(cast(float, decoders[name]["ece"])) for name in names]
    briers = [float(cast(float, decoders[name]["brier"])) for name in names]
    best = cast(str, calibration_payload.get("best_calibrated", names[0]))
    worst = cast(str, calibration_payload.get("worst_calibrated", names[-1]))
    x = np.arange(len(names))

    n_steps = max(int(cast(int, decoders[name].get("n_steps", 0))) for name in names)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=False)
    axes[0].bar(x, eces, color=PALETTE[2])
    axes[0].set_title("Expected calibration error", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("Error (lower is better)")
    axes[1].bar(x, briers, color="#9ecae1")
    axes[1].set_title("Brier score", loc="left", fontsize=10, color=INK)
    axes[1].set_ylabel("Score (lower is better)")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        _style_axes(axis, grid_axis="y")
    ece_max = max([*eces, 0.1])
    brier_max = max([*briers, 0.1])
    axes[0].set_ylim(0.0, ece_max * 1.25)
    axes[1].set_ylim(0.0, brier_max * 1.25)
    for axis, values, limit in zip(
        axes, (eces, briers), (ece_max, brier_max), strict=True
    ):
        for index, value in enumerate(values):
            axis.text(
                index, value + limit * 0.04, f"{value:.3f}", ha="center", fontsize=7
            )
    fig.suptitle("Decoder calibration metrics", color=INK, fontsize=12)
    fig.text(
        0.5,
        0.93,
        f"Best by ECE: {best}; worst by ECE: {worst}. n={n_steps} labelled steps; panels use independent y-scales because ECE and Brier are distinct metrics.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.80, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_generative_model(output_path: Path) -> Path:
    """Bayesian-graph (plate) diagram of the generative model.

    Schematic of the discrete generative model rendered from constants (no data),
    so it is deterministic. Hidden phase states ``s_t`` form a left-to-right
    Markov chain (transition ``B``) that emits categorical observations ``o_t``
    (likelihood ``A``); ``D`` is the initial prior. The repeating ``s``-``o``
    block is drawn as a plate. An active layer (policy ``pi``, expected free
    energy ``G(pi)``, and prior preference ``C``) extends the model to
    observation selection.
    """

    from matplotlib.patches import FancyArrowPatch, Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def node(x: float, y: float, label: str, fill: str = "#EAF1FB") -> None:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=fill, ec=INK, lw=1.0),
        )

    def edge(
        x1: float, y1: float, x2: float, y2: float, label: str | None = None
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", color="#333333", lw=1.2
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0 + 0.03,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
            )

    s_x = [0.25, 0.56, 0.87]
    o_y = 0.25
    s_y = 0.64

    # hidden states and their observations
    for x in s_x:
        node(x, s_y, "")
    node(s_x[0], s_y, "$s_{t-1}$")
    node(s_x[1], s_y, "$s_t$")
    node(s_x[2], s_y, "$s_{t+1}$")
    node(s_x[0], o_y, "$o_{t-1}$", "#FEF3D9")
    node(s_x[1], o_y, "$o_t$", "#FEF3D9")
    node(s_x[2], o_y, "$o_{t+1}$", "#FEF3D9")
    # transition B between consecutive states
    edge(s_x[0], s_y, s_x[1], s_y, "B")
    edge(s_x[1], s_y, s_x[2], s_y, "B")
    # likelihood A from each state to its observation
    for x in s_x:
        edge(x, s_y, x, o_y, "A")
    # initial prior D
    node(0.10, s_y, "$D$", "#E8F5E9")
    edge(0.10, s_y, s_x[0], s_y)
    ax.text(0.17, s_y + 0.10, "prior D", fontsize=8, ha="center", color=MUTED)
    # plate over the repeating s-o block
    ax.add_patch(
        Rectangle(
            (s_x[1] - 0.14, o_y - 0.11),
            s_x[2] - s_x[1] + 0.30,
            s_y - o_y + 0.22,
            fill=False,
            edgecolor="#777777",
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(
        s_x[2] - 0.04,
        s_y + 0.12,
        "t = 1, …, T",
        fontsize=8,
        ha="right",
        color="#555555",
    )
    # --- active / EFE layer ---
    ax.axvline(1.20, color="#B8C2CC", linewidth=0.8, linestyle=":")
    ax.text(
        1.58,
        0.95,
        "active-selection extension",
        fontsize=9,
        style="italic",
        color=MUTED,
        ha="center",
    )
    node(1.58, 0.78, "candidate policy $\\pi$", "#F3E5F5")
    node(1.58, 0.54, "score $G(\\pi)$", "#FCE8D5")
    node(1.58, 0.28, "preference $C$", "#E8F5E9")
    node(1.58, 0.08, "select next $o$", "#FEF3D9")
    edge(1.58, 0.70, 1.58, 0.62)
    edge(1.58, 0.46, 1.58, 0.36)
    edge(1.58, 0.20, 1.58, 0.14)
    ax.annotate(
        "beliefs",
        xy=(s_x[2], s_y),
        xytext=(1.34, 0.79),
        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, linestyle="--"),
        fontsize=7,
        color=MUTED,
    )
    ax.text(
        0.57,
        0.04,
        "B: transition matrix · A: categorical likelihood · D: initial prior",
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
    )
    fig.suptitle(
        "Discrete generative model and active-selection extension",
        color=INK,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.935,
        "The left panel is the temporal evidence model; the right panel is a realized-information-gain policy layer.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.84)
    return _save_figure(fig, output_path)


def plot_observation_informativeness(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot per-category information gain to show which instructions matter.

    Horizontal bars of the realised information gain of each observation category
    (decoded-instruction channel) over the prior ``D``, sorted descending and
    coloured by the phase each category most strongly indicates. This visualizes
    the generative model's likelihood: categories with high gain (e.g.
    ``load_immediate``, ``arithmetic_update``) are strongly phase-diagnostic,
    while low-gain categories (e.g. the ``exit``-indicating ``control_transfer``)
    leave the phase underdetermined.
    """

    from matplotlib.patches import Patch

    observations = cast(list[dict[str, object]], payload.get("observations"))
    if not observations:
        raise ValueError("observation informativeness must contain observations")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [str(row["observation"]) for row in observations]
    gains = [float(cast(float, row["info_gain"])) for row in observations]
    bests = [str(row["best_phase"]) for row in observations]
    most = cast(str, payload.get("most_informative", names[0]))
    least = cast(str, payload.get("least_informative", names[-1]))

    phases = sorted(set(bests))
    phase_color = {
        phase: PALETTE[index % len(PALETTE)] for index, phase in enumerate(phases)
    }
    # Highest gain at the top (readability): reverse the y order.
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.barh(y, gains, color=[phase_color[phase] for phase in bests])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.0, max(gains) * 1.1)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("information gain (nats) over the prior given this observation")
    ax.set_title(
        "Realized information gain by observation category",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"Most informative: {most}; least informative: {least}. Gain is measured in nats relative to prior D.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(
        handles=[Patch(color=phase_color[p], label=p) for p in phases],
        title="best-indicated phase",
        fontsize=7,
        loc="lower right",
    )
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.15, top=0.82)
    return _save_figure(fig, output_path)


def plot_model_evidence(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the log marginal likelihood of each generative model.

    Bars of the pooled log marginal likelihood for the temporal HMM, identity-transition, and
    memoryless generative assumptions (higher is better), with the best model
    highlighted and the temporal margin over its best alternative annotated. This
    shows the principled model-selection result: the temporal HMM best explains
    the observed traces even where the memoryless decoder wins on per-step
    accuracy.
    """

    evidence = cast(dict[str, object], payload.get("log_evidence"))
    if not evidence:
        raise ValueError("model evidence must contain log_evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = ["temporal", "identity_transition", "memoryless"]
    values = [float(cast(float, evidence[name])) for name in names]
    best = cast(str, payload.get("best_model", "temporal"))
    margin = float(
        cast(float, payload.get("temporal_margin_over_best_alternative", 0.0))
    )
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.barh(
        x,
        values,
        color=[PALETTE[2] if name == best else "#B8CDE0" for name in names],
    )
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels([name.replace("_", " ") for name in names])
    ax.axvline(0.0, color=INK, linewidth=0.8)
    ax.set_xlabel("Pooled log marginal likelihood (higher is better)")
    ax.set_title("Fixed-parameter model evidence", pad=AXIS_TITLE_PAD)
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"Best model: {best}; temporal margin over best alternative = {margin:.2f}.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.invert_yaxis()
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.16, top=0.82)
    return _save_figure(fig, output_path)


def plot_transition_ablation(payload: dict[str, object], output_path: Path) -> Path:
    """Plot mean temporal accuracy as the transition matrix B -> uniform.

    Line of the temporal decoder's mean accuracy across the transition blend,
    with the constant memoryless and identity-transition references as horizontal lines. This
    shows the second axis of the mechanism: as the transition structure is removed
    (blend -> 1, uniform B) the temporal decoder converges exactly to the
    memoryless per-step decoder; the peak (if off-zero) is annotated to record
    honestly that accuracy and generative-optimality need not coincide.
    """

    rows = cast(list[dict[str, object]], payload.get("rows"))
    if not rows:
        raise ValueError("transition ablation must contain rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blends = [float(cast(float, row["blend"])) for row in rows]
    temps = [float(cast(float, row["temporal_mean_accuracy"])) for row in rows]
    memoryless_mean = float(cast(float, payload.get("memoryless_mean_accuracy", 0.0)))
    identity_transition_mean = float(
        cast(float, payload.get("identity_transition_mean_accuracy", 0.0))
    )
    convergence = payload.get("convergence_blend")
    peak_blend = float(cast(float, payload.get("peak_blend", 0.0)))
    peak_acc = float(cast(float, payload.get("peak_temporal_accuracy", 0.0)))

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(blends, temps, marker="o", color=PALETTE[2], label="temporal HMM")
    ax.axhline(
        memoryless_mean,
        color="#8b5a00",
        linestyle="--",
        linewidth=1.0,
        label=f"memoryless ({memoryless_mean:.3f})",
    )
    ax.axhline(
        identity_transition_mean,
        color="#999999",
        linestyle=":",
        linewidth=1.0,
        label=f"identity transition ({identity_transition_mean:.3f})",
    )
    if convergence is not None:
        convergence_float = float(cast(float, convergence))
        ax.axvline(convergence_float, color="#555555", linestyle=":", linewidth=0.8)
        ax.annotate(
            "converges to memoryless",
            xy=(convergence_float, memoryless_mean),
            xytext=(convergence_float - 0.02, memoryless_mean + 0.06),
            fontsize=7,
            color="#555555",
        )
    ax.plot(peak_blend, peak_acc, marker="*", color="#c62828", markersize=12)
    ax.set_xlabel("transition blend toward uniform")
    ax.set_ylabel("mean decoding accuracy")
    ax.set_title(
        "Temporal accuracy as transition structure is removed",
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"Peak = {peak_acc:.3f} at blend {peak_blend:.1f}; the uniform-transition endpoint is the memoryless limit.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.82)
    return _save_figure(fig, output_path)


def plot_generative_adequacy(payload: dict[str, object], output_path: Path) -> Path:
    """Plot observed fixture structure against fixed-HMM simulation intervals.

    The two panels show the observed phase-transition rate and temporal decoder
    recovery accuracy for each fixture. Error bars are central 95% simulation
    intervals from same-length sequences generated by the fixed ``A/B/D`` model;
    they are not posterior or population uncertainty intervals.
    """

    rows = cast(list[dict[str, object]], payload.get("per_fixture"))
    if not rows:
        raise ValueError("generative adequacy must contain per_fixture rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixture_labels = [str(row["fixture"]).replace("x86_", "") for row in rows]
    metric_specs = (
        (
            "transition_rate",
            "Latent-state transition rate",
            "fraction of adjacent steps",
        ),
        (
            "temporal_recovery_accuracy",
            "Temporal recovery on simulated paths",
            "accuracy against simulated latent states",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharex=True)
    x = np.arange(len(rows), dtype=float)
    for ax, (key, title, ylabel) in zip(axes, metric_specs, strict=True):
        observed_values = [
            float(cast(float, cast(dict[str, object], row["observed"])[key]))
            for row in rows
        ]
        predictive = [
            cast(dict[str, float], cast(dict[str, object], row["predictive"])[key])
            for row in rows
        ]
        medians = np.asarray([interval["median"] for interval in predictive])
        lower = np.asarray([interval["lower"] for interval in predictive])
        upper = np.asarray([interval["upper"] for interval in predictive])
        ax.errorbar(
            x,
            medians,
            yerr=np.vstack([medians - lower, upper - medians]),
            fmt="o",
            color=PALETTE[2],
            capsize=4,
            linewidth=1.5,
            label="fixed-HMM simulation median ± 95% interval",
        )
        ax.scatter(
            x,
            observed_values,
            marker="D",
            s=42,
            color=INK,
            zorder=3,
            label="observed fixture",
        )
        for index, value in enumerate(observed_values):
            ax.annotate(
                f"{value:.2f}",
                (x[index], value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=INK,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(fixture_labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, pad=AXIS_TITLE_PAD)
        _style_axes(ax, grid_axis="y")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(
        "Generative adequacy of the fixed phase HMM", x=0.08, ha="left", fontsize=12
    )
    fig.text(
        0.08,
        0.91,
        f"{payload.get('replicates', 'configured')} same-length simulations per fixture; intervals are descriptive model-checking intervals, not posterior uncertainty.",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.78, wspace=0.26)
    return _save_figure(fig, output_path)


def _short_typed_value(value: object) -> str:
    """Format a typed categorical value for a compact figure table."""

    if value is None:
        return "—"
    if isinstance(value, list):
        return "∅" if not value else "+".join(str(item) for item in value)
    text = str(value)
    replacements = {
        "data_transfer": "data",
        "control_transfer": "control",
        "base_plus_displacement": "base+disp",
        "stack_relative": "stack",
        "not_executed": "not exec",
        "__MISSING__": "missing",
        "__UNKNOWN__": "unknown",
    }
    return replacements.get(text, text)


def plot_typed_observation_matrix(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot the typed observation schema as an annotated categorical table.

    The display uses text and light group shading rather than mapping category
    tokens to an artificial numeric heatmap.  This makes the distinction
    between a categorical field and a measurement visible in the publication
    artifact itself.
    """

    per_fixture = cast(dict[str, object], payload.get("per_fixture"))
    if not per_fixture:
        raise ValueError("typed observation payload must contain per_fixture")
    sample_name = next(iter(per_fixture))
    sample = cast(dict[str, object], per_fixture[sample_name])
    records = cast(list[dict[str, object]], sample.get("observations"))
    fields = [str(field) for field in cast(list[object], payload.get("fields", []))]
    if not records or not fields:
        raise ValueError("typed observation payload must contain fields and rows")

    field_labels = {
        "instruction_family": "instruction family",
        "operand_class": "operand class",
        "register_read_set": "registers read",
        "register_write_set": "registers written",
        "flag_effect": "flag effect",
        "memory_addressing_mode": "memory mode",
        "branch_target_class": "branch target",
        "dynamic_status": "dynamic status",
    }
    column_labels = [
        f"step {index + 1}\n{row['mnemonic']}\n0x{int(cast(int, row['address'])):x}"
        for index, row in enumerate(records)
    ]
    cell_text = [
        [_short_typed_value(record.get(field)) for record in records]
        for field in fields
    ]
    fig, ax = plt.subplots(figsize=(11.4, max(4.8, 0.38 * len(fields) + 2.4)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[
            field_labels.get(field, field.replace("_", " ")) for field in fields
        ],
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        loc="center",
        bbox=cast(Any, [0.10, 0.08, 0.88, 0.74]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    group_by_field = {
        field: group
        for group, group_fields in cast(
            dict[str, list[object]], payload.get("field_groups", {})
        ).items()
        for field in group_fields
    }
    group_colors = {
        "instruction_semantics": "#EAF1FB",
        "register_context": "#FFF3D6",
        "control_flow": "#E7F5EE",
        "dynamic_context": "#F4EAF5",
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif column == -1:
            field = fields[row - 1]
            cell.set_facecolor(group_colors.get(group_by_field.get(field, ""), LIGHT))
            cell.get_text().set_color(INK)
            cell.get_text().set_fontweight("bold")
        elif row > 0:
            field = fields[row - 1]
            cell.set_facecolor(group_colors.get(group_by_field.get(field, ""), "white"))
    ax.set_title(
        f"Typed observation record: {sample_name}",
        loc="left",
        color=INK,
        fontsize=12,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.10,
        0.88,
        "Text preserves categorical values; shaded row bands identify evidence families. An em dash is missing data, not zero.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.10,
        0.01,
        "Static fields come from Capstone detail; dynamic status comes from bounded Unicorn execution.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.06, top=0.88)
    return _save_figure(fig, output_path)


def plot_factorized_observation_comparison(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot held-out typed-model comparison, field ablation, and missingness."""

    rows = cast(list[dict[str, object]], payload.get("per_fixture"))
    ablation = cast(dict[str, object], payload.get("field_group_ablation"))
    if not rows or not ablation:
        raise ValueError("factorized payload must contain rows and field ablation")
    labels = [str(row["held_out_fixture"]).replace("x86_", "") for row in rows]
    methods = (
        ("collapsed", "collapsed", PALETTE[4]),
        ("static_factorized", "static typed", PALETTE[1]),
        ("factorized", "factorized", PALETTE[0]),
        ("hierarchical", "hierarchical", PALETTE[2]),
    )
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2))

    width = 0.18
    for offset, (key, label, color) in enumerate(methods):
        values = [
            float(cast(float, cast(dict[str, object], row[key])["accuracy"]))
            for row in rows
        ]
        bars = axes[0].bar(
            x + (offset - 1.5) * width, values, width, label=label, color=color
        )
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=22, ha="right")
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_ylabel("held-out accuracy")
    axes[0].set_title("Leave-one-fixture-out comparison", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")

    group_labels = [group.replace("_", "\n") for group in ablation]
    ablation_values = [
        float(cast(float, cast(dict[str, object], values)["mean_accuracy"]))
        for values in ablation.values()
    ]
    full_mean = float(
        cast(
            float,
            cast(dict[str, object], payload["summary"])["mean_factorized_accuracy"],
        )
    )
    ax = axes[1]
    group_x = np.arange(len(group_labels) + 1, dtype=float)
    group_values = [full_mean, *ablation_values]
    bars = ax.bar(
        group_x,
        group_values,
        color=[PALETTE[0], *["#B8CDE0"] * len(ablation_values)],
        edgecolor=INK,
        linewidth=0.4,
    )
    ax.set_xticks(group_x)
    ax.set_xticklabels(["full\ntyped", *group_labels], fontsize=7)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean held-out accuracy")
    ax.set_title("Field-family ablation", loc="left", fontsize=10)
    for bar, value in zip(bars, group_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    full_values = [
        float(cast(float, cast(dict[str, object], row["factorized"])["accuracy"]))
        for row in rows
    ]
    missing_values = [
        float(
            cast(
                float,
                cast(dict[str, object], row["missing_memory_field"])["accuracy"],
            )
        )
        for row in rows
    ]
    ax = axes[2]
    # Several fixtures intentionally tie on this tiny corpus.  Stagger the
    # direct labels by a small, deterministic amount and draw a connector so
    # tied points remain readable without implying a change in the measured
    # accuracy.
    label_offsets = (0.018, -0.018, 0.018, -0.018)
    for index, (full, missing) in enumerate(
        zip(full_values, missing_values, strict=True)
    ):
        ax.plot([0, 1], [full, missing], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter([0], [full], color=PALETTE[0], s=42, zorder=2)
        ax.scatter([1], [missing], color=PALETTE[1], s=42, zorder=2)
        label_y = min(
            0.96, max(0.04, missing + label_offsets[index % len(label_offsets)])
        )
        ax.plot([1.01, 1.035], [missing, label_y], color=GRID, linewidth=0.8)
        ax.text(1.04, label_y, labels[index], va="center", fontsize=7, color=INK)
    ax.set_xlim(-0.2, 1.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["all typed fields", "memory field masked"], rotation=22, ha="right"
    )
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Explicit missing-field policy", loc="left", fontsize=10)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE[0],
                label="full",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE[1],
                label="masked",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )

    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Typed evidence is an evaluated companion, not an automatic improvement",
        x=0.07,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.07,
        0.91,
        "Each model is fit on the other fixtures; missing values map to __MISSING__ and unseen training values to __UNKNOWN__.",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.78, wspace=0.34)
    return _save_figure(fig, output_path)


def plot_representation_invariance_benchmark(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot the underpowered representation-invariance pilot endpoints.

    The chart intentionally presents a bounded pilot diagnostic rather than a
    model score: each bar is a binary endpoint aggregated over source-program
    families, and each whisker is the descriptive Wilson interval stored in the
    receipt. The scaled co-primary classifier benchmark is a separate receipt.
    """

    summary = payload.get("summary")
    negative_control = payload.get("negative_control")
    if not isinstance(summary, dict) or not isinstance(negative_control, dict):
        raise ValueError("representation benchmark summaries must be objects")
    negative_summary = negative_control.get("summary")
    if not isinstance(negative_summary, dict):
        raise ValueError("representation benchmark negative summary must be an object")

    endpoint_names = (
        "equivalence_output_consistency",
        "near_miss_separation_rate",
    )

    def endpoint_record(
        record: dict[str, object], endpoint: str
    ) -> tuple[float, float, float, int]:
        """Extract and validate one stored endpoint and its Wilson interval."""

        value = record.get(endpoint)
        uncertainty = record.get("uncertainty")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{endpoint} value is not numeric")
        if not isinstance(uncertainty, dict):
            raise ValueError(f"{endpoint} uncertainty is missing")
        interval_record = uncertainty.get(endpoint)
        if not isinstance(interval_record, dict):
            raise ValueError(f"{endpoint} interval record is missing")
        successes = interval_record.get("successes")
        trials = interval_record.get("trials")
        interval = interval_record.get("interval")
        if (
            not isinstance(successes, int)
            or isinstance(successes, bool)
            or not isinstance(trials, int)
            or isinstance(trials, bool)
            or trials <= 0
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(bound, (int, float)) for bound in interval)
        ):
            raise ValueError(f"{endpoint} interval record is malformed")
        lower, upper = float(interval[0]), float(interval[1])
        numeric_value = float(value)
        if (
            not 0 <= successes <= trials
            or not all(
                np.isfinite(bound) and 0.0 <= bound <= 1.0 for bound in (lower, upper)
            )
            or not np.isfinite(numeric_value)
            or not 0.0 <= numeric_value <= 1.0
            or lower > upper
            or not lower <= numeric_value <= upper
        ):
            raise ValueError(f"{endpoint} interval bounds are invalid")
        return numeric_value, lower, upper, trials

    primary_values = [endpoint_record(summary, endpoint) for endpoint in endpoint_names]
    negative_values = [
        endpoint_record(negative_summary, endpoint) for endpoint in endpoint_names
    ]
    if [record[3] for record in primary_values] != [
        record[3] for record in negative_values
    ]:
        raise ValueError(
            "primary and negative-control intervals use different family counts"
        )
    family_count = int(primary_values[0][3])
    unresolved_count = sum(
        not bool(pair.get("near_miss_separated"))
        for pair in cast(list[object], negative_control.get("pairs", []))
        if isinstance(pair, dict) and pair.get("variant_role") == "near_miss"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3), sharey=True)
    labels = ("Equivalent\nconsistency", "Near-miss\nseparation")
    colors = (PALETTE[0], PALETTE[1])
    for index, (ax, records, title) in enumerate(
        zip(
            axes,
            (primary_values, negative_values),
            ("Pilot four-state catalog", "Pilot negative control: one state"),
            strict=True,
        )
    ):
        values = [record[0] for record in records]
        lower_errors = [record[0] - record[1] for record in records]
        upper_errors = [record[2] - record[0] for record in records]
        bars = ax.bar(
            np.arange(len(labels)),
            values,
            color=colors,
            width=0.62,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            np.arange(len(labels)),
            values,
            yerr=[lower_errors, upper_errors],
            fmt="none",
            ecolor=INK,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.035, 1.065),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
        ax.set_title(title, pad=AXIS_TITLE_PAD, fontsize=10)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylim(0.0, 1.12)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        if index == 0:
            ax.set_ylabel("Protocol endpoint (proportion of families)")
        if index == 1:
            ax.text(
                1.0,
                0.03,
                f"{unresolved_count} unresolved near-miss families",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )
        ax.text(
            0.5,
            AXIS_SUBTITLE_Y,
            f"n = {family_count} source families; Wilson 95% interval",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Representation-invariance pilot diagnostics",
        y=1.02,
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.5,
        -0.01,
        "Pilot intervals are family-clustered and descriptive; no fitted semantic model or population claim is shown.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.82, wspace=0.12)
    return _save_figure(fig, output_path)


def plot_intervention_power(payload: dict[str, object], output_path: Path) -> Path:
    """Plot exact finite intervention-subset separation power.

    The first panel preserves every enumerated subset as a point while the band
    summarizes the minimum/mean/maximum near-miss separation at each cardinality.
    The second panel isolates singleton interventions. This is an exact finite
    combinatorial visualization and therefore has no sampling-error bars.
    """

    catalog = payload.get("intervention_catalog")
    subset_rows = payload.get("subset_evaluations")
    curve = payload.get("subset_curve")
    pair_count = payload.get("pair_count")
    if (
        not isinstance(catalog, list)
        or not catalog
        or not isinstance(subset_rows, list)
        or not subset_rows
        or not isinstance(curve, list)
        or not curve
        or not isinstance(pair_count, int)
        or isinstance(pair_count, bool)
        or pair_count <= 0
    ):
        raise ValueError("intervention-power payload is incomplete")

    def numeric(row: dict[str, object], key: str) -> float:
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"intervention-power field {key!r} is not numeric")
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f"intervention-power field {key!r} is outside [0, 1]")
        return numeric_value

    subset_points: list[tuple[int, float]] = []
    for raw_row in subset_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("intervention-power subset row is not an object")
        values = raw_row.get("values")
        size = raw_row.get("size")
        if (
            not isinstance(values, list)
            or not values
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size != len(values)
        ):
            raise ValueError("intervention-power subset row has malformed values")
        subset_points.append((size, numeric(raw_row, "near_miss_separation")))

    curve_rows = sorted(
        (row for row in curve if isinstance(row, dict)),
        key=lambda row: int(row.get("size", -1)),
    )
    if len(curve_rows) != len(curve):
        raise ValueError("intervention-power curve contains a non-object row")
    sizes = np.asarray([int(row["size"]) for row in curve_rows], dtype=float)
    minimum = np.asarray(
        [numeric(row, "min_near_miss_separation") for row in curve_rows], dtype=float
    )
    mean = np.asarray(
        [numeric(row, "mean_near_miss_separation") for row in curve_rows], dtype=float
    )
    maximum = np.asarray(
        [numeric(row, "max_near_miss_separation") for row in curve_rows], dtype=float
    )

    singleton_rows = [
        row for row in subset_rows if isinstance(row, dict) and row.get("size") == 1
    ]
    singleton_rows.sort(key=lambda row: int(cast(list[int], row["values"])[0]))
    if len(singleton_rows) != len(catalog):
        raise ValueError("intervention-power singleton catalog is incomplete")

    def intervention_label(value: int) -> str:
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        if value == 2**32 - 1:
            return "2³²−1"
        if value == 2**64 - 1:
            return "2⁶⁴−1"
        return f"{value:,}"

    singleton_labels = [
        intervention_label(cast(list[int], row["values"])[0]) for row in singleton_rows
    ]
    singleton_values = np.asarray(
        [numeric(row, "near_miss_separation") for row in singleton_rows], dtype=float
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.6), gridspec_kw={"width_ratios": (1.35, 1.0)}
    )
    point_sizes = np.asarray([point[0] for point in subset_points], dtype=float)
    point_values = np.asarray([point[1] for point in subset_points], dtype=float)
    axes[0].scatter(
        point_sizes,
        point_values,
        s=20,
        color=PALETTE[1],
        alpha=0.42,
        edgecolors="none",
        label="individual subset",
        zorder=3,
    )
    axes[0].fill_between(
        sizes,
        minimum,
        maximum,
        color=PALETTE[0],
        alpha=0.15,
        label="minimum--maximum",
        zorder=1,
    )
    axes[0].plot(
        sizes,
        mean,
        color=PALETTE[0],
        marker="o",
        linewidth=1.8,
        label="mean subset separation",
        zorder=4,
    )
    axes[0].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    axes[0].set_xticks(sizes)
    axes[0].set_xlabel("Number of interventions in subset")
    axes[0].set_ylabel("Near-miss separation (exact proportion)")
    axes[0].set_ylim(-0.02, 1.08)
    axes[0].set_title("All non-empty subsets", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")

    colors = [PALETTE[0] if value >= 1.0 else PALETTE[1] for value in singleton_values]
    bars = axes[1].barh(
        np.arange(len(singleton_labels)),
        singleton_values,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    axes[1].set_yticks(np.arange(len(singleton_labels)), singleton_labels)
    axes[1].set_xlabel("Near-miss separation")
    axes[1].set_title("Singleton intervention power", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[1].set_xlim(0.0, 1.08)
    axes[1].axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    for bar, value in zip(bars, singleton_values, strict=True):
        axes[1].text(
            min(value + 0.025, 1.015),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color=INK,
        )

    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Finite intervention power in the declared quotient catalog",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.08,
        0.01,
        (
            f"Exact enumeration of {len(subset_rows)} subsets over {pair_count} pair outcomes; "
            "no population uncertainty or universal identifiability claim is shown."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.80, wspace=0.34)
    return _save_figure(fig, output_path)


def plot_trace_refinement(payload: dict[str, object], output_path: Path) -> Path:
    """Plot stepwise symbolic/Unicorn agreement for executed fixture steps.

    The left panel uses small horizontal offsets so exact-overlap markers remain
    visible.  The right panel counts only the EFLAGS bits that the bounded
    transfer functions define.  Static instructions that Unicorn did not visit
    are intentionally omitted from both panels.
    """

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("trace-refinement events must be a list")
    events = [
        event
        for event in raw_events
        if isinstance(event, dict) and event.get("executed")
    ]
    if not events:
        raise ValueError("trace-refinement plot requires executed events")
    required = ("post_rax", "unicorn_post_rax", "mnemonic", "address")
    if any(any(key not in event for key in required) for event in events):
        raise ValueError("trace-refinement events are missing post-step fields")

    def event_integer(event: dict[str, object], key: str) -> int:
        value = event.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"trace-refinement field {key!r} is not an integer")
        return value

    def payload_integer(key: str) -> int:
        value = payload.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"trace-refinement count {key!r} is not an integer")
        return value

    symbolic_values = [event_integer(event, "post_rax") for event in events]
    unicorn_values = [event_integer(event, "unicorn_post_rax") for event in events]
    if symbolic_values != unicorn_values:
        raise ValueError("trace-refinement plot received a stepwise RAX mismatch")
    flag_agreements = payload_integer("modeled_flag_agreements")
    flag_mismatches = payload_integer("modeled_flag_mismatches")
    flag_comparisons = payload_integer("modeled_flag_comparisons")
    if flag_comparisons <= 0 or flag_agreements + flag_mismatches != flag_comparisons:
        raise ValueError("trace-refinement modeled-flag counts are inconsistent")

    x = np.arange(len(events), dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 4.1),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
    )
    symbolic_offset = x - 0.055
    unicorn_offset = x + 0.055
    axes[0].plot(
        symbolic_offset,
        symbolic_values,
        color=PALETTE[0],
        marker="o",
        markersize=5,
        linewidth=1.2,
        label="transfer function",
        zorder=3,
    )
    axes[0].plot(
        unicorn_offset,
        unicorn_values,
        color=PALETTE[1],
        marker="x",
        markersize=6,
        linewidth=1.0,
        linestyle="--",
        label="Unicorn",
        zorder=4,
    )
    axes[0].set_title("Post-instruction RAX", pad=AXIS_TITLE_PAD, fontsize=10)
    axes[0].set_ylabel("Unsigned register value")
    axes[0].set_xticks(
        x, [f"{i}\n{event['mnemonic']}" for i, event in enumerate(events)]
    )
    axes[0].set_xlabel("Executed instruction step")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[0].text(
        0.02,
        0.98,
        f"{len(events)}/{len(events)} post-step values agree",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )

    bars = axes[1].bar(
        ["agree", "mismatch"],
        [flag_agreements, flag_mismatches],
        color=(PALETTE[0], PALETTE[1]),
        width=0.58,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for bar, count in zip(bars, (flag_agreements, flag_mismatches), strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            count + max(flag_comparisons * 0.025, 0.15),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    axes[1].set_title("Modeled EFLAGS bits", pad=AXIS_TITLE_PAD, fontsize=10)
    axes[1].set_ylabel("Bit comparisons")
    axes[1].set_ylim(0, max(flag_comparisons * 1.18, 1.0))
    axes[1].text(
        0.5,
        0.98,
        f"{flag_comparisons} defined bits compared",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Bounded transfer functions refine the observed execution trace",
        x=0.07,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.07,
        0.01,
        "Executed fixture steps only; the comparison is not a complete x86-64 semantics proof.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.80, wspace=0.30)
    return _save_figure(fig, output_path)


def plot_cross_architecture_benchmark(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot real decoding/execution outcomes for the three-ISA fixture set."""

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cross-architecture rows must be non-empty")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("cross-architecture rows must be objects")
    labels = [str(row["architecture"]) for row in rows]
    static_counts = [len(cast(list[object], row["instructions"])) for row in rows]
    execution = [cast(dict[str, object], row["execution"]) for row in rows]
    executed_counts = [
        int(cast(int, row["executed_instruction_count"])) for row in execution
    ]
    final_values = [int(cast(int, row["final_accumulator"])) for row in execution]
    expected_values = [int(cast(int, row["expected_accumulator"])) for row in execution]
    if final_values != expected_values or len(set(final_values)) != 1:
        raise ValueError("cross-architecture fixture results are not invariant")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 4.0), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        static_counts,
        width=width,
        color=PALETTE[0],
        label="Capstone decoded",
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].bar(
        x + width / 2,
        executed_counts,
        width=width,
        color=PALETTE[1],
        label="Unicorn executed",
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Instruction count")
    axes[0].set_title("Adapter coverage", pad=AXIS_TITLE_PAD, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for index, value in enumerate(static_counts):
        axes[0].text(
            index - width / 2,
            value + 0.08,
            str(value),
            ha="center",
            fontsize=8,
            color=INK,
        )
    for index, value in enumerate(executed_counts):
        axes[0].text(
            index + width / 2,
            value + 0.08,
            str(value),
            ha="center",
            fontsize=8,
            color=INK,
        )

    axes[1].axhline(
        expected_values[0],
        color=INK,
        linestyle="--",
        linewidth=1.0,
        label=f"declared result = {expected_values[0]}",
    )
    axes[1].scatter(
        x,
        final_values,
        color=PALETTE[2],
        edgecolor=INK,
        linewidth=0.8,
        s=55,
        zorder=3,
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Accumulator after two instructions")
    axes[1].set_title("Bounded numeric outcome", pad=AXIS_TITLE_PAD, fontsize=10)
    axes[1].set_ylim(0, max(expected_values[0] + 2, 10))
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for index, value in enumerate(final_values):
        axes[1].text(
            index, value + 0.28, str(value), ha="center", fontsize=8, color=INK
        )

    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Cross-architecture executable fixture benchmark",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.08,
        0.91,
        "Same declared numeric operation; real Capstone decoding and bounded Unicorn execution; n = 3 architectures.",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.text(
        0.5,
        -0.01,
        "The shared result is a local adapter check, not a cross-ISA semantic-equivalence theorem.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.78, wspace=0.30)
    return _save_figure(fig, output_path)


def plot_formal_verification_receipt(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot the proof-backed fragment alongside its explicit evidence boundary."""

    if payload.get("endpoint_status", payload.get("status")) != "locally_verified":
        raise ValueError("formal verification receipt is not locally verified")
    theorem_names = payload.get("theorems")
    if not isinstance(theorem_names, list) or not theorem_names:
        raise ValueError("formal theorem manifest must be non-empty")
    bridge = payload.get("cross_language_bridge")
    bridge_checked = isinstance(bridge, dict) and bool(bridge.get("kernel_checked"))
    unsupported = payload.get("unsupported_cases")
    underpowered = payload.get("underpowered_cases")
    unsupported_count = len(unsupported) if isinstance(unsupported, list) else None
    underpowered_count = len(underpowered) if isinstance(underpowered, list) else None
    status_rows = [
        ["proof-backed", str(len(theorem_names)), "Lean kernel checked"],
        [
            "cross-language bridge",
            "1 fixture" if bridge_checked else "not observed",
            "generated trace; bounded scope",
        ],
        ["computational-only", "not recorded", "not promoted by this receipt"],
        ["falsified candidate", "not recorded", "see theorem status map"],
        ["unresolved candidate", "not recorded", "see theorem status map"],
        [
            "unsupported / excluded",
            str(unsupported_count) if unsupported_count else "none recorded",
            "outside declared fragment",
        ],
        [
            "underpowered",
            str(underpowered_count) if underpowered_count else "none recorded",
            "not a positive result",
        ],
    ]
    status_colors = (
        PALETTE[2],
        PALETTE[1],
        MUTED,
        PALETTE[3],
        PALETTE[4],
        MUTED,
        MUTED,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.6, 6.7))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.12, 1.0), wspace=0.24)
    ax_status = fig.add_subplot(grid[0, 0])
    ax_status.axis("off")
    table = ax_status.table(
        cellText=status_rows,
        colLabels=("evidence status", "receipt value", "interpretation"),
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=cast(Any, (0.0, 0.03, 1.0, 0.82)),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.18)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT)
            if col == 0:
                cell.set_facecolor(status_colors[row - 1] + "36")
                cell.get_text().set_fontweight("bold")
            if col == 1:
                cell.get_text().set_fontweight("bold")
    ax_status.set_title("Evidence status ledger", loc="left", fontsize=10, color=INK)
    ax_status.text(
        0.0,
        0.92,
        "The formal receipt reports what this Lean build checked and keeps absent evidence visible.",
        transform=ax_status.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )

    ax_scope = fig.add_subplot(grid[0, 1])
    ax_scope.axis("off")
    checked = payload.get("checked_fragment")
    checked_operations = (
        checked.get("operations", []) if isinstance(checked, dict) else []
    )
    operations_text = (
        "\n".join(f"• {str(operation)}" for operation in checked_operations)
        or "• not recorded"
    )
    ax_scope.add_patch(
        Rectangle(
            (0.03, 0.22),
            0.94,
            0.62,
            facecolor=PALETTE[2] + "18",
            edgecolor=PALETTE[2],
            linewidth=1.0,
            hatch="..",
        )
    )
    ax_scope.text(
        0.07,
        0.79,
        "Checked fragment",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    ax_scope.text(
        0.07,
        0.73,
        operations_text,
        fontsize=7.4,
        va="top",
        color=INK,
        linespacing=1.25,
    )
    relation = (
        checked.get("relation", "not recorded")
        if isinstance(checked, dict)
        else "not recorded"
    )
    state = (
        checked.get("state", "not recorded")
        if isinstance(checked, dict)
        else "not recorded"
    )
    ax_scope.text(
        0.07,
        0.15,
        "Formal boundary\n"
        + textwrap.fill(f"state: {state}; relation: {relation}", 54)
        + "\n\n"
        + "Unsupported ISA effects, complete x86-64 semantics, and population-level claims remain outside this receipt.",
        fontsize=7.7,
        va="top",
        color=MUTED,
        linespacing=1.3,
    )
    fig.suptitle(
        "Formal verification receipt: checked fragment and evidence boundary",
        x=0.06,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.06,
        0.93,
        f"Lean toolchain {payload.get('toolchain', 'unknown')} · {len(theorem_names)} declared theorems · endpoint: {payload.get('endpoint_status', 'not observed')}",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.025,
        "Kernel-checked rows are not a complete x86-64 semantics proof; computational bridge evidence and absent theorem statuses remain explicitly separated.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.83)
    return _save_figure(fig, output_path)


def plot_active_information_frontier(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot the exact finite-horizon information-per-cost frontier."""

    frontier = payload.get("frontier")
    if not isinstance(frontier, list) or not frontier:
        raise ValueError("active-information frontier must be non-empty")
    grouped: dict[float, list[dict[str, object]]] = {}
    for raw_row in frontier:
        if not isinstance(raw_row, dict):
            raise ValueError("active-information rows must be objects")
        cost_weight = float(cast(float, raw_row["cost_weight"]))
        grouped.setdefault(cost_weight, []).append(raw_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.7, 4.1))
    styles = ("-", "--", "-.", ":")
    plotted_gains: list[tuple[float, tuple[float, ...]]] = []
    for index, (cost_weight, rows) in enumerate(sorted(grouped.items())):
        rows = sorted(rows, key=lambda row: int(cast(int, row["horizon"])))
        horizons = [int(cast(int, row["horizon"])) for row in rows]
        gains = [float(cast(float, row["expected_information_gain"])) for row in rows]
        if any(not np.isfinite(value) or value < -1e-8 for value in gains):
            raise ValueError("frontier contains invalid information gain")
        plotted_gains.append((cost_weight, tuple(gains)))
        ax.plot(
            horizons,
            gains,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=PALETTE[index % len(PALETTE)],
            linestyle=styles[index % len(styles)],
            label=f"cost weight = {cost_weight:g}",
        )
    ax.set_xticks(sorted({int(cast(int, row["horizon"])) for row in frontier}))
    ax.set_xlabel("Finite sensing horizon")
    ax.set_ylabel("Expected information gain (nats)")
    ax.set_title(
        "Exact information-per-cost policy frontier",
        loc="left",
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Dynamic programming enumerates all sensor/outcome branches for the declared finite sensor family.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if len({gains for _, gains in plotted_gains}) == 1 and len(plotted_gains) > 1:
        ax.text(
            0.99,
            0.96,
            "All cost-weight curves coincide\nfor this finite sensor contract",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=MUTED,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
            },
        )
    ax.text(
        1.0,
        -0.16,
        "Costs are model-derived unit proxies; the curve is a mechanism result, not a wall-clock law.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    _style_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.24, top=0.80)
    return _save_figure(fig, output_path)


_COUNTEREXAMPLE_STATUS_COLORS = {
    "falsified_by_counterexample": PALETTE[1],
    "holds_in_searched_domain": PALETTE[2],
}


def plot_counterexample_search(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the smallest-first exhaustive counterexample search receipt."""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("counterexample-search candidates must be non-empty")
    ids: list[str] = []
    sizes: list[float] = []
    colors: list[str] = []
    witness_sizes: list[int | None] = []
    for raw_row in candidates:
        if not isinstance(raw_row, dict):
            raise ValueError("counterexample-search candidate is not an object")
        status = str(raw_row["status"])
        if status not in _COUNTEREXAMPLE_STATUS_COLORS:
            raise ValueError(f"unknown candidate status: {status}")
        domain_size = raw_row.get("searched_domain_size")
        if not isinstance(domain_size, int) or isinstance(domain_size, bool):
            raise ValueError("searched_domain_size must be an integer")
        if domain_size <= 0:
            raise ValueError("searched_domain_size must be positive")
        witness_size = None
        raw_witness = raw_row.get("witness")
        if isinstance(raw_witness, dict):
            witness_size = raw_witness.get("witness_size")
        if status == "falsified_by_counterexample":
            if not isinstance(witness_size, int) or isinstance(witness_size, bool):
                raise ValueError("falsified candidates must record witness_size")
            if witness_size < 1:
                raise ValueError("witness_size must be a positive rank")
        ids.append(str(raw_row["property_id"]))
        sizes.append(float(domain_size))
        colors.append(_COUNTEREXAMPLE_STATUS_COLORS[status])
        witness_sizes.append(
            witness_size if status == "falsified_by_counterexample" else None
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(ids), dtype=float)
    ax.barh(y, sizes, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, ids)
    ax.invert_yaxis()
    if max(sizes) >= 100 * max(min(sizes), 1.0):
        ax.set_xscale("log")
    for position, size, witness in zip(y, sizes, witness_sizes, strict=True):
        label = f"domain={int(size)}"
        if witness is not None:
            label += f", witness rank={witness}"
        ax.text(
            size,
            position,
            f"  {label}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=MUTED,
        )
    ax.set_xlabel("Enumerated inputs before resolution (smallest-first order)")
    ax.set_title(
        "Formal counterexample search over the declared bounded model",
        loc="left",
        fontsize=11,
        pad=AXIS_TITLE_PAD,
    )
    ax.text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Every falsified witness is re-verified by an independent second interpretation.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    handles = [
        Rectangle((0, 0), 1, 1, color=color, label=label.replace("_", " "))
        for label, color in _COUNTEREXAMPLE_STATUS_COLORS.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.text(
        1.0,
        -0.18,
        "A falsified candidate narrows the claim; a holds result is bounded to the declared search domain only.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    _style_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.26, top=0.78)
    return _save_figure(fig, output_path)


def plot_sensing_stress(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the active-sensing stress grid and calibration stability."""

    grid = payload.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("sensing-stress grid must be non-empty")
    curve = payload.get("budget_value_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("sensing-stress budget value curve must be non-empty")
    stability = payload.get("calibration_stability")
    if not isinstance(stability, dict):
        raise ValueError("sensing-stress calibration stability is missing")
    max_regret: dict[str, float] = {}
    for raw_row in grid:
        if not isinstance(raw_row, dict):
            raise ValueError("sensing-stress grid row is not an object")
        policy = str(raw_row["policy"])
        regret = float(raw_row["regret_vs_exact"])
        if regret < 0.0:
            raise ValueError("sensing-stress regret is negative")
        max_regret[policy] = max(max_regret.get(policy, 0.0), regret)
    budgets = [int(cast(int, row["budget"])) for row in curve]
    upper = [float(cast(float, row["upper_bound_information_gain"])) for row in curve]
    achieved = [float(cast(float, row["achieved_information_gain"])) for row in curve]
    gaps = [float(cast(float, row["unattained_information_gain_gap"])) for row in curve]
    if any(gap < -1e-12 for gap in gaps):
        raise ValueError("sensing-stress value-curve gap is negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": (1.2, 1.0)}
    )
    policies = sorted(max_regret)
    x = np.arange(len(policies), dtype=float)
    colors = [
        PALETTE[2] if policy == "exact_oracle" else PALETTE[0] for policy in policies
    ]
    axes[0].bar(
        x,
        [max_regret[policy] for policy in policies],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axes[0].set_xticks(x, [policy.replace("_", "\n") for policy in policies])
    axes[0].set_ylabel("Maximum regret vs exact reference (objective units)")
    axes[0].set_title(
        "Policy controls vs exact dynamic program", fontsize=10, pad=AXIS_TITLE_PAD
    )
    axes[1].plot(
        budgets,
        upper,
        linestyle="--",
        color=MUTED,
        linewidth=1.4,
        marker="s",
        markersize=4,
        label="unconstrained exact bound",
    )
    axes[1].plot(
        budgets,
        achieved,
        color=PALETTE[0],
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="achieved at budget",
    )
    axes[1].fill_between(
        budgets, achieved, upper, color=PALETTE[1], alpha=0.25, label="unattained gap"
    )
    axes[1].set_xticks(budgets)
    axes[1].set_xlabel("Action budget (sensing actions)")
    axes[1].set_ylabel("Expected information gain (nats)")
    axes[1].set_title("Budget-constrained value curve", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.text(
        0.01,
        0.97,
        "Active-sensing stress study",
        fontsize=11,
        color=INK,
        ha="left",
    )
    fig.text(
        0.01,
        0.925,
        "Exact-enumeration references and fixed-seed rollouts; the timing-stability panel is a local-machine receipt, not a universal law.",
        fontsize=8,
        color=MUTED,
        ha="left",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.86, wspace=0.24)
    return _save_figure(fig, output_path)


def plot_quotient_classifier(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the quotient-classifier portfolio with uncertainty and deltas.

    Dot-and-interval marks keep the two co-primary endpoints visible without
    implying an ordered ranking.  A second panel exposes the paired,
    source-family-clustered difference from the typed surface baseline, and a
    third panel retains calibration and selective-coverage diagnostics.  The
    receipt is allowed to omit intervals for calibration quantities; in that
    case the panel labels them as point estimates instead of inventing
    uncertainty.
    """

    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("quotient-classifier models must be an object")
    names = (
        "calibrated_interpretable",
        "temporal_probabilistic",
        "neural_sequence",
        "surface_baseline",
    )
    if any(not isinstance(models.get(name), dict) for name in names):
        raise ValueError("quotient-classifier portfolio is incomplete")

    def metric(name: str, key: str) -> float:
        value = cast(dict[str, object], models[name]).get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"quotient-classifier metric {name}.{key} is invalid")
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"quotient-classifier metric {name}.{key} is out of range")
        return numeric

    labels = {
        "calibrated_interpretable": "interpretable",
        "temporal_probabilistic": "temporal",
        "neural_sequence": "neural",
        "surface_baseline": "surface baseline",
    }

    def interval(name: str, key: str) -> tuple[float, float] | None:
        model = cast(dict[str, object], models[name])
        intervals = model.get("confidence_intervals")
        if not isinstance(intervals, dict):
            return None
        record = intervals.get(key)
        if not isinstance(record, dict):
            return None
        raw_interval = record.get("interval")
        if (
            not isinstance(raw_interval, list)
            or len(raw_interval) != 2
            or not all(isinstance(item, (int, float)) for item in raw_interval)
        ):
            raise ValueError(f"malformed confidence interval for {name}.{key}")
        lo, hi = float(raw_interval[0]), float(raw_interval[1])
        value = metric(name, key)
        if not 0.0 <= lo <= value <= hi <= 1.0:
            raise ValueError(f"confidence interval does not contain {name}.{key}")
        return lo, hi

    def paired_record(name: str, key: str) -> dict[str, object] | None:
        comparisons = payload.get("paired_comparisons")
        if not isinstance(comparisons, dict):
            return None
        raw = comparisons.get(f"{name}:surface_baseline:{key}")
        return raw if isinstance(raw, dict) else None

    def paired_delta(name: str, key: str) -> tuple[float, tuple[float, float] | None]:
        record = paired_record(name, key)
        if record is None:
            model_comparisons = payload.get("model_comparisons")
            if isinstance(model_comparisons, dict):
                model_record = model_comparisons.get(name)
                if isinstance(model_record, dict):
                    metric_record = model_record.get(key)
                    if isinstance(metric_record, dict):
                        difference = metric_record.get("difference_vs_surface_baseline")
                        if isinstance(difference, (int, float)) and not isinstance(
                            difference, bool
                        ):
                            return float(difference), None
            raise ValueError(f"paired comparison is missing for {name}.{key}")
        observed = record.get("observed_difference")
        if not isinstance(observed, dict):
            raise ValueError(f"paired difference is missing for {name}.{key}")
        difference = observed.get(key)
        if not isinstance(difference, (int, float)) or isinstance(difference, bool):
            raise ValueError(f"paired difference is malformed for {name}.{key}")
        confidence_intervals = record.get("confidence_intervals")
        if not isinstance(confidence_intervals, dict):
            return float(difference), None
        interval_record = confidence_intervals.get(key)
        if not isinstance(interval_record, dict):
            return float(difference), None
        raw_interval = interval_record.get("interval")
        if (
            not isinstance(raw_interval, list)
            or len(raw_interval) != 2
            or not all(isinstance(item, (int, float)) for item in raw_interval)
        ):
            raise ValueError(f"malformed paired interval for {name}.{key}")
        return float(difference), (float(raw_interval[0]), float(raw_interval[1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 6.2),
        gridspec_kw={"width_ratios": (1.15, 1.15, 0.95)},
    )
    # Panel A: co-primary endpoint estimates.  Two marks per model preserve
    # the portfolio view while making the interval and the model identity
    # legible in grayscale.
    y = np.arange(len(names), dtype=float)
    endpoint_keys = (
        ("balanced_accuracy", "balanced accuracy", PALETTE[0], -0.12),
        ("macro_f1", "macro-F1", PALETTE[1], 0.12),
    )
    for key, label, color, offset in endpoint_keys:
        values = [metric(name, key) for name in names]
        xerr_low: list[float] = []
        xerr_high: list[float] = []
        for name, value in zip(names, values, strict=True):
            bounds = interval(name, key)
            if bounds is None:
                xerr_low.append(0.0)
                xerr_high.append(0.0)
            else:
                xerr_low.append(value - bounds[0])
                xerr_high.append(bounds[1] - value)
        axes[0].errorbar(
            values,
            y + offset,
            xerr=[xerr_low, xerr_high],
            fmt="o",
            markersize=5.5,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            color=color,
            ecolor=INK,
            elinewidth=1.1,
            capsize=3,
            label=label,
            zorder=3,
        )
        for value, position in zip(values, y + offset, strict=True):
            axes[0].text(
                min(value + 0.025, 1.04),
                position,
                f"{value:.2f}",
                va="center",
                fontsize=7.3,
                color=INK,
            )
    axes[0].axvline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    axes[0].set_yticks(y, [labels[name] for name in names])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.16)
    axes[0].set_xlabel("Held-out score")
    axes[0].set_title("Co-primary endpoints", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].text(
        0.0,
        AXIS_SUBTITLE_Y,
        f"{payload.get('held_out_family_count', 'n/a')} held-out families; 95% clustered intervals",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )

    # Panel B: paired model-minus-baseline deltas.  The interval is read from
    # the same source-family bootstrap receipt as the primary endpoint, so a
    # favorable point estimate cannot be mistaken for a promoted comparison.
    comparison_names = names[:-1]
    delta_y = np.arange(len(comparison_names), dtype=float)
    for key, label, color, offset in endpoint_keys:
        deltas: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for name in comparison_names:
            difference, bounds = paired_delta(name, key)
            deltas.append(difference)
            if bounds is None:
                lows.append(0.0)
                highs.append(0.0)
            else:
                lows.append(difference - bounds[0])
                highs.append(bounds[1] - difference)
        axes[1].errorbar(
            deltas,
            delta_y + offset,
            xerr=[lows, highs],
            fmt="o",
            markersize=5.5,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            color=color,
            ecolor=INK,
            elinewidth=1.1,
            capsize=3,
            label=label,
            zorder=3,
        )
        for difference, position in zip(deltas, delta_y + offset, strict=True):
            axes[1].text(
                difference + (0.018 if difference >= 0 else -0.018),
                position,
                f"{difference:+.2f}",
                ha="left" if difference >= 0 else "right",
                va="center",
                fontsize=7.2,
                color=INK,
            )
    axes[1].axvline(0.0, color=MUTED, linestyle="--", linewidth=0.9)
    axes[1].set_yticks(delta_y, [labels[name] for name in comparison_names])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.55, 0.38)
    axes[1].set_xlabel("Difference from surface baseline")
    axes[1].set_title("Paired family-level deltas", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[1].text(
        0.0,
        AXIS_SUBTITLE_Y,
        "Positive values favor the model; intervals use the frozen paired bootstrap",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )

    # Panel C: probability calibration and selective diagnostics.  Current
    # receipts provide point estimates for these quantities, so no fabricated
    # interval is drawn.  Coverage/risk is printed beneath the plot as a
    # compact, auditable companion to the calibration marks.
    diagnostic_y = np.arange(len(names), dtype=float)
    diagnostic_keys = (
        ("brier_score", "Brier", PALETTE[2], -0.12),
        ("expected_calibration_error", "ECE", PALETTE[3], 0.12),
    )
    for key, label, color, offset in diagnostic_keys:
        values = [metric(name, key) for name in names]
        axes[2].errorbar(
            values,
            diagnostic_y + offset,
            fmt="o",
            markersize=5.5,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            color=color,
            label=label,
            zorder=3,
        )
        for value, position in zip(values, diagnostic_y + offset, strict=True):
            axes[2].text(
                min(value + 0.025, 1.02),
                position,
                f"{value:.2f}",
                va="center",
                fontsize=7.2,
                color=INK,
            )
    axes[2].set_yticks(diagnostic_y, [labels[name] for name in names])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 1.12)
    axes[2].set_xlabel("Lower is better")
    axes[2].set_title("Calibration diagnostics", fontsize=10, pad=AXIS_TITLE_PAD)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    coverage_lines: list[str] = []
    for name in names:
        model = cast(dict[str, object], models[name])
        selective = model.get("selective_risk_coverage")
        if isinstance(selective, dict):
            coverage = selective.get("coverage")
            risk = selective.get("risk")
            if isinstance(coverage, (int, float)) and isinstance(risk, (int, float)):
                coverage_lines.append(
                    f"{labels[name]}: cov {coverage:.2f}, risk {risk:.2f}"
                )
            elif isinstance(coverage, (int, float)):
                coverage_lines.append(f"{labels[name]}: cov {coverage:.2f}, risk n/a")
    for ax in axes:
        _style_axes(ax, grid_axis="x")
    fig.suptitle(
        "Formal-quotient membership under source-family holdout",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.70,
        0.185,
        "Point estimates; no clustered CI supplied for Brier/ECE.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.text(
        0.70,
        0.145,
        "Selective coverage/risk (receipt threshold): "
        + "; ".join(
            line.replace(": cov ", " ").replace(", risk ", "/")
            for line in coverage_lines
        ),
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.01,
        "Labels come from the declared intervention relation; typed features omit family, transformation, bytes, and target labels. No post-hoc winner is selected.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.79, wspace=0.45)
    return _save_figure(fig, output_path)


def plot_compiler_transfer(payload: dict[str, object], output_path: Path) -> Path:
    """Plot status-aware static compiler and decoder capability evidence."""

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("compiler-transfer rows must be non-empty")
    normalized: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("compiler-transfer row is not an object")
        status = str(raw.get("status", "not_observed"))
        if status == "validated_static_object":
            count = raw.get("decoded_instruction_count")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("validated transfer row lacks decoded count")
        elif status not in {"unavailable", "failed", "unsupported", "underpowered"}:
            raise ValueError(f"compiler-transfer row has unknown status: {status}")
        normalized.append(raw)

    source_ids = sorted(
        {str(row.get("source_id", "not_observed")) for row in normalized}
    )
    configurations = sorted(
        {
            (
                str(row.get("target_id", "not_observed")),
                str(row.get("optimization", "not_observed")),
            )
            for row in normalized
        }
    )
    config_labels = [
        f"{target}\n{optimization}" for target, optimization in configurations
    ]
    by_key = {
        (
            str(row.get("source_id")),
            str(row.get("target_id")),
            str(row.get("optimization")),
        ): row
        for row in normalized
    }
    status_text = {
        "validated_static_object": "VALID\nstatic",
        "unavailable": "UNAVAILABLE",
        "failed": "FAILED",
        "unsupported": "UNSUPPORTED",
        "underpowered": "UNDERPOWERED",
    }
    status_color = {
        "validated_static_object": PALETTE[2],
        "unavailable": LIGHT,
        "failed": PALETTE[4] + "30",
        "unsupported": PALETTE[3] + "30",
        "underpowered": PALETTE[1] + "30",
    }
    status_hatch = {
        "validated_static_object": "",
        "unavailable": "//",
        "failed": "xx",
        "unsupported": "..",
        "underpowered": "\\\\",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )
    axes[0].axis("off")
    axes[0].set_title(
        "Capability status by source × target × optimization",
        loc="left",
        fontsize=10,
        pad=AXIS_TITLE_PAD,
    )
    table_values: list[list[str]] = []
    for source_id in source_ids:
        cells: list[str] = []
        for target_id, optimization in configurations:
            row = by_key.get((source_id, target_id, optimization))
            status = str(row.get("status", "not_observed")) if row else "not_observed"
            cells.append(status_text.get(status, status.upper()))
        table_values.append(cells)
    table = axes[0].table(
        cellText=table_values,
        rowLabels=source_ids,
        colLabels=config_labels,
        loc="center",
        cellLoc="center",
        rowLoc="left",
        bbox=(0.0, 0.10, 1.0, 0.78),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.55)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row_index == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col_index == -1:
            cell.set_facecolor(LIGHT)
            cell.get_text().set_fontweight("bold")
        else:
            source_id = source_ids[row_index - 1]
            target_id, optimization = configurations[col_index]
            row = by_key.get((source_id, target_id, optimization))
            status = str(row.get("status", "not_observed")) if row else "not_observed"
            cell.set_facecolor(status_color.get(status, "white"))
            cell.set_hatch(status_hatch.get(status, "///"))
    axes[0].text(
        0.0,
        0.03,
        "Text and hatching carry status; unavailable/failed/unsupported cells are not zero performance.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )

    valid_rows = [
        row for row in normalized if row.get("status") == "validated_static_object"
    ]
    x = np.arange(len(configurations), dtype=float)
    for source_index, source_id in enumerate(source_ids):
        for config_index, (target_id, optimization) in enumerate(configurations):
            row = by_key.get((source_id, target_id, optimization))
            if row is None or row.get("status") != "validated_static_object":
                continue
            count_value = row.get("decoded_instruction_count")
            if not isinstance(count_value, (int, float)):
                continue
            count = int(count_value)
            offset = (source_index - (len(source_ids) - 1) / 2) * 0.04
            axes[1].scatter(
                config_index + offset,
                count,
                s=34,
                color=PALETTE[source_index % len(PALETTE)],
                edgecolor=INK,
                linewidth=0.45,
                zorder=3,
            )
            axes[1].text(
                config_index + offset,
                count + 0.55,
                str(count),
                ha="center",
                va="bottom",
                fontsize=6.1,
                color=INK,
            )
    axes[1].set_xticks(x, config_labels)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].set_ylabel("Decoded instructions per validated object")
    axes[1].set_title(
        "Static decoded-count distribution",
        loc="left",
        fontsize=10,
        pad=AXIS_TITLE_PAD,
    )
    axes[1].text(
        0.0,
        -0.20,
        f"{len(valid_rows)} validated rows shown; no emitted object was executed. A missing configuration has no count to plot.",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )
    _style_axes(axes[1], grid_axis="y")
    fig.suptitle(
        "Compiler/source/architecture transfer capability receipt",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.24, top=0.80, wspace=0.28)
    return _save_figure(fig, output_path)


def plot_formal_correspondence(payload: dict[str, object], output_path: Path) -> Path:
    """Plot field-level correspondence numerators and denominators."""

    rows = payload.get("step_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("formal-correspondence rows must be non-empty")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("formal-correspondence row is not an object")
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    labels = sorted(grouped)
    if any(
        not any(
            isinstance(row.get("reference_steps"), list) and row.get("reference_steps")
            for row in grouped[label]
        )
        for label in labels
    ):
        raise ValueError("formal-correspondence fixture has no reference steps")

    trace_count = len(rows)
    reference_step_counts = [
        sum(
            len(cast(list[object], row.get("reference_steps", [])))
            for row in grouped[label]
        )
        for label in labels
    ]
    unicorn_step_counts = [
        sum(
            len(cast(list[object], row.get("unicorn_steps", [])))
            for row in grouped[label]
        )
        for label in labels
    ]
    total_reference_steps = sum(reference_step_counts)
    field_checks = (
        ("step agreement", "matched_at_every_step", "step rows"),
        ("decoded address", "addresses_equal", "traces"),
        ("register file", "register_file_equal", "traces"),
        ("flags", "modeled_flags_equal", "traces"),
        ("memory effects", "memory_effects_equal", "traces"),
        ("RIP / target", "rip_and_control_targets_equal", "traces"),
    )
    field_counts: list[tuple[str, int, int, str]] = []
    for label, key, denominator_kind in field_checks:
        denominator = (
            total_reference_steps if denominator_kind == "step rows" else trace_count
        )
        numerator = 0
        for row in rows:
            checks = row.get("checks", {})
            if not isinstance(checks, dict):
                continue
            if bool(checks.get(key)):
                if denominator_kind == "step rows":
                    numerator += len(cast(list[object], row.get("reference_steps", [])))
                else:
                    numerator += 1
        field_counts.append((label, numerator, denominator, denominator_kind))
    terminal_agreements = 0
    unsupported_rows = 0
    bound_rows = 0
    for row in rows:
        checks = row.get("checks", {})
        if isinstance(checks, dict):
            unsupported = checks.get("unsupported")
            unsupported_rows += int(unsupported is not None)
        reference_steps = row.get("reference_steps", [])
        if isinstance(reference_steps, list):
            bound_rows += sum(
                int(
                    isinstance(step, dict)
                    and str(step.get("status", "ok"))
                    in {"bound_exceeded", "exception", "unsupported"}
                )
                for step in reference_steps
            )
        final = row.get("unicorn_final", {})
        if (
            isinstance(final, dict)
            and isinstance(reference_steps, list)
            and reference_steps
        ):
            last = reference_steps[-1]
            if isinstance(last, dict) and final.get("final_rax") == last.get(
                "post_rax"
            ):
                terminal_agreements += 1
    field_counts.append(("terminal RAX", terminal_agreements, trace_count, "traces"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(10.1, 5.0), gridspec_kw={"width_ratios": (1.0, 1.35)}
    )
    y = np.arange(len(field_counts), dtype=float)
    proportions = [
        numerator / denominator if denominator else np.nan
        for _label, numerator, denominator, _kind in field_counts
    ]
    axes[0].barh(
        y,
        proportions,
        color=[PALETTE[0] if value == 1.0 else PALETTE[1] for value in proportions],
        edgecolor=INK,
        linewidth=0.6,
        alpha=0.88,
    )
    axes[0].set_yticks(y, [row[0] for row in field_counts])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.18)
    axes[0].set_xlabel("Agreement proportion")
    axes[0].set_title("Compared fields", fontsize=10, pad=AXIS_TITLE_PAD)
    for index, (label, numerator, denominator, kind) in enumerate(field_counts):
        axes[0].text(
            min(proportions[index] + 0.025, 1.02),
            index,
            f"{numerator}/{denominator} {kind}",
            va="center",
            fontsize=7.4,
            color=INK,
        )
    axes[0].axvline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    axes[1].bar(
        x - width / 2,
        reference_step_counts,
        width=width,
        color=PALETTE[0],
        label="Python reference steps",
        zorder=3,
    )
    axes[1].bar(
        x + width / 2,
        unicorn_step_counts,
        width=width,
        color=PALETTE[1],
        label="Unicorn decoded steps",
        zorder=3,
    )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Step rows")
    axes[1].set_title(
        "Workload and explicit boundary rows", fontsize=10, pad=AXIS_TITLE_PAD
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Python reference ↔ Capstone ↔ bounded Unicorn correspondence",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=7.5,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.895),
        ncol=2,
    )
    fig.text(
        0.12,
        0.105,
        f"{trace_count} intervention traces; {total_reference_steps} reference step rows; no inferential interval.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    fig.text(
        0.55,
        0.105,
        f"Boundary rows: {unsupported_rows} unsupported traces; {bound_rows} bound/exception/unsupported steps.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.01,
        "Agreement is reported as numerator/denominator at the declared field grain; the receipt is bounded, not complete x86-64 semantics.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.80, wspace=0.32)
    return _save_figure(fig, output_path)


def plot_sensor_costs(payload: dict[str, object], output_path: Path) -> Path:
    """Plot auditable deterministic sensor vectors beside local calibration."""

    sensors = payload.get("sensors")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("sensor-cost rows must be non-empty")
    required = (
        "sensor_invocations",
        "decoded_instructions",
        "bounded_execution_steps",
        "register_flag_observations",
        "branch_observations",
        "memory_hook_events",
        "serialized_bytes",
        "unsupported_attempts",
    )
    component_labels = (
        "sensor\ncalls",
        "decoded\ninstructions",
        "bounded\nsteps",
        "register / flag\nobservations",
        "branch\nobservations",
        "memory-hook\nevents",
        "serialized\nbytes",
        "unsupported\nattempts",
    )
    labels = [str(cast(dict[str, object], row)["sensor_id"]) for row in sensors]
    display_labels = [
        label.replace("instruction_family", "instruction\nfamily")
        .replace("operand_class", "operand\nclass")
        .replace("register_read_set", "register\nread")
        .replace("register_write_set", "register\nwrite")
        for label in labels
    ]
    values: list[list[float]] = []
    for row in sensors:
        if not isinstance(row, dict):
            raise ValueError("sensor-cost row is not an object")
        cost_vector = cast(dict[str, object], row["deterministic_cost_vector"])
        row_values: list[float] = []
        for key in required:
            value = cost_vector.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"sensor-cost component {key} is malformed")
            if value < 0:
                raise ValueError(f"sensor-cost component {key} is negative")
            row_values.append(float(value))
        values.append(row_values)
    value_matrix = np.asarray(values, dtype=float)
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for row in sensors:
        if not isinstance(row, dict):
            raise ValueError("sensor-cost row is not an object")
        timing = row.get("wall_clock_nanoseconds_per_query")
        if not isinstance(timing, dict):
            raise ValueError("sensor-cost timing is missing")
        median = timing.get("median")
        minimum = timing.get("min")
        maximum = timing.get("max")
        if not isinstance(median, (int, float)):
            raise ValueError("sensor-cost median is malformed")
        if not isinstance(minimum, (int, float)):
            raise ValueError("sensor-cost minimum is malformed")
        if not isinstance(maximum, (int, float)):
            raise ValueError("sensor-cost timing is malformed")
        medians.append(float(median))
        lows.append(float(minimum))
        highs.append(float(maximum))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 5.1), gridspec_kw={"width_ratios": (1.45, 1.0)}
    )
    image = axes[0].imshow(
        np.log1p(value_matrix),
        aspect="auto",
        cmap=ListedColormap(("#F5F7FA", PALETTE[0], PALETTE[2], PALETTE[1])),
    )
    del image
    axes[0].set_xticks(np.arange(len(required)), component_labels)
    axes[0].tick_params(axis="x", labelrotation=22, labelsize=7.2)
    axes[0].set_yticks(np.arange(len(labels)), display_labels)
    axes[0].set_xlabel(
        "Deterministic component (log-scaled shading; labels are exact counts)",
        labelpad=4,
    )
    axes[0].set_title("Primary cost vector", fontsize=10, pad=AXIS_TITLE_PAD)
    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[1]):
            axes[0].text(
                j,
                i,
                str(int(value_matrix[i, j])),
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
            )
    x = np.arange(len(labels), dtype=float)
    axes[1].bar(
        x, medians, color=PALETTE[2], edgecolor="white", linewidth=0.8, zorder=3
    )
    axes[1].errorbar(
        x,
        medians,
        yerr=[
            np.asarray(medians) - np.asarray(lows),
            np.asarray(highs) - np.asarray(medians),
        ],
        fmt="none",
        ecolor=INK,
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        zorder=4,
    )
    axes[1].set_xticks(x, display_labels)
    axes[1].set_ylabel("Nanoseconds per calibrated query")
    axes[1].set_title("Local wall-clock calibration", fontsize=10, pad=AXIS_TITLE_PAD)
    for ax in axes:
        _style_axes(ax, grid_axis="y")
    fig.suptitle(
        "Measured sensing costs and finite policy certificates",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    certificates = payload.get("policy_certificates", [])
    certificate_count = len(certificates) if isinstance(certificates, list) else 0
    calibration = payload.get("calibration", {})
    repetitions = (
        calibration.get("repetitions") if isinstance(calibration, dict) else None
    )
    trials = (
        calibration.get("trials_per_repetition")
        if isinstance(calibration, dict)
        else None
    )
    fig.text(
        0.11,
        0.105,
        "All 8 deterministic receipt components shown; cell labels are counts.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    fig.text(
        0.55,
        0.105,
        f"Bars are medians; whiskers: {repetitions if repetitions is not None else 'not observed'} repetitions × {trials if trials is not None else 'not observed'} trials.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.025,
        f"{certificate_count} exact dynamic-program certificates; optimality is scoped to the frozen finite sensor space.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.30, top=0.80, wspace=0.32)
    return _save_figure(fig, output_path)


def plot_replication_manifest(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the public/gold/receipt separation of the local harness."""

    count = payload.get("public_row_count")
    gold_count = payload.get("gold_row_count")
    separation = payload.get("label_separated")
    if not isinstance(count, int) or not isinstance(gold_count, int):
        raise ValueError("replication counts are missing")
    if separation is not True:
        raise ValueError("replication labels are not separated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.7, 3.9))
    # The protected label boundary is a first-class visual object.  It is not
    # enough to place a gold-label box beside the public input: the diagram
    # must show which material is unavailable to the label-blind model path.
    ax.add_patch(
        Rectangle(
            (0.265, 0.16),
            0.22,
            0.66,
            transform=ax.transAxes,
            facecolor=PALETTE[1] + "12",
            edgecolor=PALETTE[1],
            linewidth=1.0,
            linestyle="--",
            hatch="///",
            zorder=0,
        )
    )
    ax.text(
        0.375,
        0.84,
        "protected owner boundary",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=PALETTE[1],
        weight="bold",
    )
    boxes = (
        (0.04, "Public input\n% s rows" % count, "features only", PALETTE[0]),
        (
            0.29,
            "Gold labels\n% s rows" % gold_count,
            "held outside public input",
            PALETTE[1],
        ),
        (0.54, "Predictions\nCLI output", "label-separated scoring", PALETTE[2]),
        (0.79, "Receipts\nhashes + protocol", "replay validation", PALETTE[3]),
    )
    for index, (left, title, subtitle, color) in enumerate(boxes):
        patch = FancyBboxPatch(
            (left, 0.35),
            0.16,
            0.35,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor=INK,
            linewidth=0.8,
            alpha=0.92,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            left + 0.08,
            0.56,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
        ax.text(
            left + 0.08,
            0.25,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=MUTED,
        )
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (left + 0.17, 0.525),
                    (boxes[index + 1][0] - 0.015, 0.525),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.1,
                    color=INK,
                )
            )
    ax.text(
        0.5,
        0.08,
        "Public filenames, ordering, and metadata carry no family or quotient label; hashes invalidate stale replay.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
    )
    ax.set_axis_off()
    fig.suptitle(
        "Frozen local replication harness and provenance boundary",
        x=0.05,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.05,
        0.02,
        "The gold-label boundary is protected; the observed controlled replay is separate from independent physical replication or reimplementation.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.08)
    return _save_figure(fig, output_path)


def plot_external_source_provenance(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Plot the pinned-source safety boundary and promotion gates."""

    manifest = payload.get("manifest")
    gate = payload.get("promotion_gate")
    if not isinstance(manifest, dict) or not isinstance(gate, dict):
        raise ValueError("external-source provenance receipt is incomplete")
    status = str(payload.get("status", "unknown"))
    source_count = len(manifest.get("sources", []))
    gate_rows = (
        ("five independent projects", "project_count_pass"),
        ("two revisions per project", "revision_count_pass"),
        ("≥100 labelable examples", "labelable_example_count_pass"),
        ("project-held-out test", "project_held_out_pass"),
        ("license + provenance", "license_provenance_pass"),
        ("static build + decode", "reproducible_static_build_pass"),
        ("static-only safety", "static_only_pass"),
        ("separate replay", "independent_replay_pass"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.55), gridspec_kw={"width_ratios": (1.0, 1.3)}
    )
    ax = axes[0]
    ax.set_axis_off()
    stages = (
        ("Pinned manifest", "license · revision · hashes"),
        ("Local acquisition", "explicit network opt-in"),
        ("Static object analysis", "decode and hash; never execute"),
    )
    colors = (PALETTE[0], PALETTE[2], PALETTE[3])
    for index, (title, subtitle) in enumerate(stages):
        y = 0.78 - index * 0.25
        patch = FancyBboxPatch(
            (0.08, y),
            0.84,
            0.14,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=colors[index],
            edgecolor=INK,
            linewidth=0.8,
            alpha=0.92,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            0.50,
            y + 0.087,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
        ax.text(
            0.50,
            y + 0.035,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
        )
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.50, y - 0.015),
                    (0.50, y - 0.09),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.0,
                    color=INK,
                )
            )
    ax.text(
        0.50,
        0.08,
        f"status: {status}\nmanifested source families: {source_count}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_title("Static-only provenance path", fontsize=10, pad=AXIS_TITLE_PAD)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(gate_rows) + 1.5)
    ax.text(
        0.03,
        len(gate_rows) + 1.15,
        "Promotion gate",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    ax.text(0.76, len(gate_rows) + 1.15, "state", fontsize=10, weight="bold", color=INK)
    for index, (label, gate_key) in enumerate(gate_rows):
        state = "pass" if bool(gate.get(gate_key)) else "open"
        y = len(gate_rows) - index - 0.05
        color = PALETTE[2] if state == "pass" else PALETTE[4]
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.32),
                0.94,
                0.48,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor="#F5F7FA",
                edgecolor="#D6DCE5",
                linewidth=0.6,
            )
        )
        ax.text(0.05, y - 0.08, label, fontsize=8.2, va="center", color=INK)
        ax.text(
            0.78, y - 0.08, state, fontsize=8.2, va="center", color=color, weight="bold"
        )
    ax.set_title("P13 application-validity gates", fontsize=10, pad=AXIS_TITLE_PAD)
    fig.suptitle(
        "Licensed-source provenance and application-validity gate",
        x=0.08,
        ha="left",
        fontsize=12,
        color=INK,
    )
    static_validated_value = payload.get("static_validated_row_count")
    static_validated_count = (
        str(int(static_validated_value))
        if isinstance(static_validated_value, (int, float))
        and not isinstance(static_validated_value, bool)
        else "not observed"
    )
    fig.text(
        0.08,
        0.018,
        "\n".join(
            [
                f"P13 pinned rows: {source_count}; validated static rows: {static_validated_count}. No arbitrary external binary is executed.",
                "Open or unavailable gates remain status values; P15 controlled replay is a separate evidence surface, never a zero-valued performance result.",
            ]
        ),
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
        linespacing=1.15,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.80, wspace=0.20)
    return _save_figure(fig, output_path)


def plot_external_label_coverage(payload: dict[str, object], output_path: Path) -> Path:
    """Plot the source-defined label census and frozen split allocation.

    This figure is intentionally a provenance census rather than a performance
    chart. Accepted rows are grouped by the source-contract role and by
    project/revision split; explicit dashes in the matrix mean that no
    accepted row occupies that cell, while excluded candidates remain outside
    the accepted-row denominator and are stated in the footer.
    """

    raw_contracts = payload.get("contracts")
    if not isinstance(raw_contracts, list):
        raise ValueError("external label coverage receipt contracts must be a list")
    contracts: list[dict[str, str]] = []
    for index, raw_row in enumerate(raw_contracts):
        if not isinstance(raw_row, dict):
            raise ValueError(f"external label coverage row {index} is not an object")
        project = raw_row.get("project_id")
        revision = raw_row.get("revision_label")
        role = raw_row.get("role", raw_row.get("variant_role"))
        split = raw_row.get("split")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (project, revision, role, split)
        ):
            raise ValueError(f"external label coverage row {index} is incomplete")
        project = cast(str, project)
        revision = cast(str, revision)
        role = cast(str, role)
        split = cast(str, split)
        if role not in {"equivalent", "near_miss", "negative_control"}:
            raise ValueError(f"external label coverage row {index} has unknown role")
        if split not in {"train", "development", "test"}:
            raise ValueError(f"external label coverage row {index} has unknown split")
        contracts.append(
            {
                "project": project,
                "revision": revision,
                "role": role,
                "split": split,
            }
        )

    role_order = ("equivalent", "near_miss", "negative_control")
    role_labels = {
        "equivalent": "equivalence witness",
        "near_miss": "near miss",
        "negative_control": "identity control",
    }
    role_colors = {
        "equivalent": PALETTE[0],
        "near_miss": PALETTE[1],
        "negative_control": PALETTE[2],
    }
    role_hatches = {
        "equivalent": "///",
        "near_miss": "...",
        "negative_control": "xx",
    }
    split_order = ("train", "development", "test")
    split_colors = {split: PALETTE[index] for index, split in enumerate(split_order)}
    split_hatches = {"train": "///", "development": "...", "test": "xx"}

    projects = sorted({row["project"] for row in contracts})
    revisions = sorted({(row["project"], row["revision"]) for row in contracts})
    summary = payload.get("summary", {})
    summary_test_projects = (
        summary.get("test_projects", []) if isinstance(summary, dict) else []
    )
    test_projects = {
        str(project) for project in summary_test_projects if isinstance(project, str)
    }
    if not test_projects:
        test_projects = {row["project"] for row in contracts if row["split"] == "test"}
    exclusions_value = payload.get("exclusions", [])
    exclusion_count = len(exclusions_value) if isinstance(exclusions_value, list) else 0
    status = str(payload.get("status", "not_yet_observed"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, max(4.8, 2.9 + 0.34 * len(revisions))),
        gridspec_kw={"width_ratios": (1.05, 1.15)},
    )

    ax = axes[0]
    _style_axes(ax, grid_axis="x")
    y_positions = np.arange(len(projects), dtype=float)
    left = np.zeros(len(projects), dtype=float)
    for role in role_order:
        counts = np.asarray(
            [
                sum(
                    row["project"] == project and row["role"] == role
                    for row in contracts
                )
                for project in projects
            ],
            dtype=float,
        )
        bars = ax.barh(
            y_positions,
            counts,
            left=left,
            color=role_colors[role],
            edgecolor="white",
            linewidth=0.7,
            hatch=role_hatches[role],
            alpha=0.9,
            label=role_labels[role],
        )
        for bar, count, start in zip(bars, counts, left, strict=True):
            if count > 0:
                ax.text(
                    float(start + count / 2),
                    float(bar.get_y() + bar.get_height() / 2),
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )
        left += counts
    for y, total in zip(y_positions, left, strict=True):
        ax.text(
            float(total + 0.2),
            float(y),
            f"n={int(total)}",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(projects, fontsize=8)
    ax.set_xlabel("accepted source-contract rows (count)")
    ax.set_xlim(0, max(1.0, float(max(left, default=0.0)) * 1.32))
    ax.set_title("Roles by project", pad=AXIS_TITLE_PAD)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        fontsize=7.5,
    )
    ax.invert_yaxis()

    ax = axes[1]
    _style_axes(ax, grid_axis="x")
    ax.set_xlim(-0.5, len(split_order) - 0.5)
    ax.set_ylim(-0.5, max(0.5, len(revisions) - 0.5))
    for y, (project, revision) in enumerate(revisions):
        for x, split in enumerate(split_order):
            count = sum(
                row["project"] == project
                and row["revision"] == revision
                and row["split"] == split
                for row in contracts
            )
            present = count > 0
            ax.add_patch(
                Rectangle(
                    (x - 0.43, y - 0.31),
                    0.86,
                    0.62,
                    facecolor=split_colors[split] if present else LIGHT,
                    edgecolor=INK if present else GRID,
                    linewidth=0.7,
                    hatch=split_hatches[split] if present else "",
                    alpha=0.9 if present else 1.0,
                )
            )
            ax.text(
                x,
                y,
                str(count) if present else "—",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if present else MUTED,
                weight="bold" if present else "normal",
            )
    y_labels = [
        f"{project} · {revision}{' · held-out' if project in test_projects else ''}"
        for project, revision in revisions
    ]
    ax.set_xticks(np.arange(len(split_order), dtype=float))
    ax.set_xticklabels(split_order, fontsize=8)
    ax.set_yticks(np.arange(len(revisions), dtype=float))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("accepted rows per frozen split; — = none")
    ax.set_title("Revision allocation", pad=AXIS_TITLE_PAD)
    ax.invert_yaxis()

    fig.suptitle(
        "Source-bound external label coverage",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.06,
        0.018,
        (
            f"Receipt status: {status}; accepted rows: {len(contracts)}; explicit exclusions: {exclusion_count}. "
            "Roles are source-defined controls; rows are static-only and no emitted object is executed. "
            "This census is not an accuracy estimate or a universal semantic-equivalence result."
        ),
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
        wrap=True,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.32)
    return _save_figure(fig, output_path)


def plot_artifact_receipt_closure(
    program_payload: dict[str, object],
    output_path: Path,
    *,
    artifact_count: int,
    artifact_error_count: int,
) -> Path:
    """Plot the protocol-to-document provenance boundary.

    This is an audit visualization, not a performance plot.  The pipeline
    stages are intentionally labelled in text so that status colors are never
    the only carrier of meaning.  ``artifact_count`` and
    ``artifact_error_count`` are computed from the project-owned JSON surface
    by the producer; they are passed in rather than inferred from the image.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = program_payload.get("status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    protocol_id = str(program_payload.get("protocol_id", "unavailable"))
    protocol_hash = str(program_payload.get("protocol_sha256", "unavailable"))
    protocol_label = (
        "\n".join(
            [
                *textwrap.wrap(protocol_id, width=17),
                f"sha256 {protocol_hash[:12]}…",
            ]
        )
        if protocol_hash not in {"", "unavailable"}
        else "\n".join(textwrap.wrap(protocol_id, width=17))
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = (
        ("Protocol v2", protocol_label, PALETTE[0]),
        ("Producers", "source modules\n+ atomic writes", PALETTE[1]),
        ("JSON receipts", f"N = {artifact_count}\nsealed artifacts", PALETTE[2]),
        ("Audit", f"errors = {artifact_error_count}\nfail closed", PALETTE[3]),
        ("Publication", "variables → template\nDOCX / tagged PDF", PALETTE[4]),
    )
    x_positions = np.linspace(0.02, 0.80, len(stages))
    box_width = 0.155
    for index, ((title, subtitle, color), x) in enumerate(zip(stages, x_positions)):
        ax.add_patch(
            FancyBboxPatch(
                (float(x), 0.47),
                box_width,
                0.22,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=color,
                edgecolor=INK,
                linewidth=0.8,
                alpha=0.95,
            )
        )
        ax.text(
            float(x) + box_width / 2,
            0.61,
            title,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
        ax.text(
            float(x) + box_width / 2,
            0.535,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.2 if index == 0 else 7.2,
            color="white",
        )
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (float(x) + box_width + 0.008, 0.58),
                    (float(x_positions[index + 1]) - 0.008, 0.58),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.0,
                    color=INK,
                )
            )
    ax.text(
        0.02,
        0.31,
        "The receipt layer preserves each domain schema while binding protocol, producer,\n"
        "input hashes, toolchain versions, limitations, and unsupported/underpowered states.",
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
    )
    ax.text(
        0.02,
        0.12,
        "Audit result: "
        + (
            "all project-owned JSON artifacts satisfy the common envelope."
            if artifact_error_count == 0
            else f"{artifact_error_count} artifact(s) fail the common envelope."
        ),
        ha="left",
        va="top",
        fontsize=9,
        color=PALETTE[2] if artifact_error_count == 0 else "#B42318",
        weight="bold",
    )
    ax.set_title(
        "Source-bound producer and rendering path", fontsize=10, pad=AXIS_TITLE_PAD
    )

    ax = axes[1]
    status_order = (
        "implemented_local",
        "active",
        "blocked_by_evidence",
        "planned",
    )
    labels = {
        "implemented_local": "implemented locally",
        "active": "active",
        "blocked_by_evidence": "blocked by evidence",
        "planned": "planned",
    }
    colors = (PALETTE[2], PALETTE[0], PALETTE[3], PALETTE[4])
    values = [int(status_counts.get(status, 0)) for status in status_order]
    y = np.arange(len(status_order))
    ax.barh(y, values, color=colors, edgecolor=INK, linewidth=0.5)
    ax.set_yticks(y, [labels[status] for status in status_order])
    ax.invert_yaxis()
    ax.set_xlabel("roadmap milestones (count)")
    ax.set_xlim(0, max(values + [1]) + 1)
    for index, value in enumerate(values):
        ax.text(value + 0.08, index, str(value), va="center", fontsize=9, color=INK)
    _style_axes(ax, grid_axis="x")
    ax.set_title(
        "Evidence status retained in the same receipt", fontsize=10, pad=AXIS_TITLE_PAD
    )
    ax.text(
        0.0,
        -0.23,
        "Counts describe the roadmap contract; they are not scientific effect sizes.\n"
        "Blocked, unavailable, and underpowered evidence is never recoded as zero performance.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.suptitle(
        "Protocol-v2 artifact receipt closure",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.06,
        0.02,
        "Deterministic provenance and status audit; no inferential uncertainty is applicable.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.82, wspace=0.28)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure caption registry
# ---------------------------------------------------------------------------
# Captions are written next to every output figure so the rendered manuscript and
# any figure inventory stay consistent with what each PNG actually shows. Keys
# match the generated PNG stem under ``output/figures/``.

FIGURE_CAPTIONS: dict[str, str] = {
    "generative_model": (
        "Conceptual graph of the discrete generative model and its active-selection extension. "
        "The left side shows hidden phases linked by transition B and emitting categorical observations "
        "through likelihood A from prior D; the right side shows candidate policy pi, score G(pi), "
        "preference C, and the selected next observation. This is a deterministic schematic of "
        "source-controlled model constants, not an empirical network diagram."
    ),
    "observation_informativeness": (
        "Realized information gain for each decoded-instruction category, sorted from largest to "
        "smallest and colored by the phase receiving the largest posterior contribution. The range "
        "is computed from the fixed likelihood and prior in evaluation_report.json; {most_informative} "
        "is the largest-gain category and {least_informative} the smallest. Because the plot uses "
        "the category actually revealed, it is a realized-information diagnostic rather than a full "
        "posterior-predictive expected-free-energy calculation."
    ),
    "model_evidence": (
        "Pooled log marginal likelihood under three fixed-parameter generative assumptions: temporal "
        "HMM, identity-transition, and memoryless. Higher values indicate better in-sample "
        "predictive normalization for these labelled traces; they are not posterior probabilities "
        "of the models. The highlighted {best_model} model has the largest value, while the companion "
        "accuracy figures report the separate discriminative comparison."
    ),
    "transition_ablation": (
        "Transition-structure ablation across the likelihood blend grid. The temporal decoder's mean "
        "per-step accuracy is plotted against constant memoryless and identity-transition references while B is "
        "moved toward a uniform transition matrix. At the uniform endpoint the temporal update "
        "collapses to the memoryless update, isolating the role of B; the marked peak records the "
        "accuracy optimum for this small fixed corpus rather than a claim that the peak transition is "
        "generatively correct."
    ),
    "generative_adequacy": (
        "Fixed-parameter generative adequacy check. Diamonds show the observed phase-transition rate "
        "and temporal recovery accuracy for each fixture; circles and intervals show the median and "
        "central 95% simulation interval from same-length sequences generated by the fixed A/B/D HMM. "
        "Intervals are descriptive model-checking intervals, not posterior or population uncertainty; "
        "a mismatch identifies a concrete model/data discrepancy for the next iteration [@gelman2013bda]."
    ),
    "observation_pipeline": (
        "Source-bound assembly evidence map for the selected x86-64 fixture. Each card shows the "
        "decoded address, mnemonic/operands, and categorical channel consumed by the model; the "
        "instruction order is static, and a solid border marks an address observed during bounded "
        "Unicorn execution. Observation IDs are labels, not continuous measurements."
    ),
    "assembly_cfg": (
        "Static control-flow evidence for the branch-bearing x86_guard fixture. Each block preserves "
        "its decoded address range and Capstone mnemonic sequence; green fall-through and amber "
        "taken-branch edges are direct-target relations, blue borders mark blocks observed on the "
        "bounded Unicorn path, and the dashed amber external node records an out-of-span target. "
        "This is a finite fixture CFG, not whole-binary reachability or decompilation."
    ),
    "assembly_memory_timeline": (
        "Static-to-dynamic memory provenance for the transparent x86_memory fixture. The left panel "
        "marks which Capstone-decoded instructions contain explicit memory operands; the right panel "
        "orders the real Unicorn memory-hook events and reports access type, triggering instruction, "
        "stack-relative effective address, and observed value. The write/read pair demonstrates a "
        "bounded store-load join, not general heap recovery, alias analysis, or arbitrary-binary tracing."
    ),
    "posterior_beliefs": (
        "Memoryless posterior beliefs over the {hidden_state_count} hand-authored latent program "
        "phases at each of the {observation_step_count} decoded observation steps. Each line is a "
        "normalized probability from the fixed comparison path; the visualization shows belief updates "
        "and does not supply independent phase ground truth."
    ),
    "package_provenance": (
        "Package-provenance audit for the required Capstone and Unicorn adapters plus optional roadmap "
        "slots. Filled circles and direct labels mark packages found by the local probe; x markers mark "
        "packages not found. Required/optional status is printed beside each row. Optional rows are "
        "recorded as provenance and are not treated as evidence streams."
    ),
    "trace_feature_matrix": (
        "Trace-feature audit for every decoded instruction. The left panel contains binary flags for "
        "dynamic execution, an immediate operand, and an accumulator operand; the right table preserves "
        "step, raw observation ID, categorical code, semantic label, and the two operand flags. The "
        "codes are annotated as categories and must not be read as a continuous scale."
    ),
    "rxinfer_free_energy": (
        "Bethe Free Energy values emitted by the native Julia/RxInfer backend for the Gaussian "
        "event-code model. Points are backend iterations from the selected trace bundle; this is "
        "the authoritative RxInfer diagnostic for the implemented model, not the pymdp comparison "
        "objective and not evidence of semantic understanding."
    ),
    "rxinfer_categorical_free_energy": (
        "Bethe Free Energy values emitted by the native Julia/RxInfer categorical phase-HMM "
        "companion for the canonical trace bundle. Points are backend iterations for the fixed "
        "categorical model; the recorded trajectory is {rxinfer_categorical_flat} with "
        "delta {rxinfer_categorical_delta}. This namespaced diagnostic is model-specific and "
        "must not be compared as an absolute scale with the Gaussian event-code trajectory."
    ),
    "inference_diagnostics": (
        "Two inference diagnostics kept on separate axes and scales: the pymdp comparison variational "
        "diagnostic above and the native RxInfer Bethe Free Energy below. The traces share the project "
        "input lineage but are distinct objectives and are not numerically interchangeable."
    ),
    "scholarship_source_roles": (
        "Counts of sources in the curated scholarship ledger by source type. The ledger separates "
        "peer-reviewed theory and binary-analysis context from software papers, package documentation, "
        "and repository-level roadmap references; counts describe support roles, not evidence strength "
        "by themselves."
    ),
    "scholarship_evidence_matrix": (
        "Scholarship-source counts crossed by project stage and evidence tier. Implemented and comparison "
        "rows identify sources used to contextualize executed paths; roadmap and context rows remain "
        "explicitly outside the currently executed evidence-producing surface."
    ),
    "claim_evidence_map": (
        "Claim-register coverage map. For each bounded claim, vertically separated markers count linked "
        "source keys, generated artifacts, and validation checks. These counts audit traceability; they "
        "do not turn a linked claim into independent scientific replication."
    ),
    "artifact_hash_manifest": (
        "Reproducibility-manifest coverage by artifact class. Bar height is the total tracked file size "
        "in KiB and the annotation gives the number of files; the manifest independently records each "
        "file's SHA-256 and byte size so stale or replaced artifacts fail validation."
    ),
    "validation_summary": (
        "Generated validation checks for trace structure, assembly-listing lineage, inference artifacts, "
        "figures, scholarship, manuscript hydration, provenance, and reproducibility. A pass means the "
        "specific project oracle evaluated true; it is not a substitute for scientific peer review."
    ),
    "temporal_beliefs_x86_accumulator": (
        "Temporal forward-filter beliefs for the {sample_name} fixture. Each of the {observation_step_count} "
        "rows is a normalized posterior over {hidden_state_count} hand-authored program phases after "
        "one observation; cell labels expose the probability rather than hiding it in color alone."
    ),
    "temporal_beliefs_x86_bitwise": (
        "Temporal forward-filter beliefs for the x86_bitwise fixture. Each row is a normalized posterior "
        "over {hidden_state_count} hand-authored phases after one decoded observation; observation IDs "
        "are row labels and are not numeric magnitudes."
    ),
    "temporal_beliefs_x86_guard": (
        "Temporal forward-filter beliefs for the x86_guard fixture. Each row is a normalized posterior "
        "over {hidden_state_count} hand-authored phases after one decoded observation; the heatmap is "
        "a compact view of filtering uncertainty, not a discovered control-flow graph."
    ),
    "temporal_beliefs_x86_long_accumulator": (
        "Temporal forward-filter beliefs for the x86_long_accumulator fixture. Each row is a normalized "
        "posterior over {hidden_state_count} hand-authored phases after one decoded observation; the "
        "longer fixture makes state persistence visible without changing the inference contract."
    ),
    "temporal_beliefs_x86_memory": (
        "Temporal forward-filter beliefs for the x86_memory fixture. Each row is a normalized posterior "
        "over {hidden_state_count} hand-authored phases after one decoded observation; the fixture adds "
        "bounded stack-memory operands to the typed evidence surface without claiming memory semantics."
    ),
    "temporal_beliefs_x86_guard_taken": (
        "Temporal forward-filter beliefs for the x86_guard_taken fixture. Each row is a normalized "
        "posterior over {hidden_state_count} hand-authored phases after one decoded observation; the "
        "taken branch leaves one statically decoded instruction outside the bounded dynamic path."
    ),
    "corpus_coverage": (
        "Transparent fixture-design coverage across the {fixture_count}-fixture corpus. The left panel "
        "separates static decoded span from bounded executed path length; the right panel records the "
        "presence of memory operands, alternate dynamic paths, repeated authored phases, and direct "
        "branch edges. The corpus contains {memory_fixture_count} memory fixture(s), "
        "{alternate_path_fixture_count} alternate-path fixture(s), and "
        "{repeated_phase_fixture_count} repeated-phase fixture(s). These are design diagnostics, not "
        "independent binary samples."
    ),
    "action_sensing": (
        "Held-out action-conditioned sensing over typed assembly fields. The left panel reports the "
        "mean number of actions used before the configured confidence threshold, with failures retained "
        "at the common budget; the right panel reports mean final posterior confidence. Active choices "
        "use predictive training-only field likelihoods before the held-out value is revealed, fixed and "
        "seeded-random policies use the same candidate pool, and the oracle is an explicitly inadmissible "
        "upper bound. Equal normalized action cost is a mechanism contract, not a wall-clock measurement."
    ),
    "parameter_learning": (
        "Leakage-safe ordered-HMM parameter learning. The left panel compares held-out temporal accuracy "
        "for observation-only ordered Baum--Welch EM, supervised MAP fitting, a fixed hand-authored "
        "reference, and a fixed-penalty Dirichlet MAP comparator on complete-fixture holdouts; the middle "
        "panel shows training log-evidence convergence for the primary EM fit and its mean; the right panel "
        "shows held-out accuracy for five deterministic initializations, with dots for holdouts and bars for "
        "within-start means. The receipt also records label-free posterior-predictive checks for the penalized "
        "fit. Phase labels are reserved for scoring and are not used in fitting; training evidence, predictive "
        "checks, and start-spread diagnostics are optimization/robustness diagnostics rather than held-out or "
        "population uncertainty [@dempster1977em;@rabiner1989tutorial]."
    ),
    "model_comparison": (
        "Phase-decoder accuracy across the {fixture_count}-fixture labelled corpus. The left panel shows "
        "fixture-level accuracy for identity-transition, memoryless, and temporal decoders; the right panel shows pooled "
        "per-step accuracy with {ci_level} descriptive Wilson intervals around hit counts over "
        "{corpus_observation_count} labelled steps. Steps within a fixture are dependent, so the intervals "
        "are descriptive rather than population-level uncertainty estimates."
    ),
    "active_efe": (
        "Greedy active-disassembly trajectory for the selected fixture. The upper panel reports both "
        "the decoded-category EFE score and its predictive information-contribution companion in nats; "
        "the lower panel reports posterior entropy on a separate axis. Predictive contributions weight "
        "a category's realized entropy reduction by its prior predictive probability, and their sum is "
        "the one-step channel information gain. The machine-checked corpus-level never-worse result "
        "for the realized policy is {active_never_worse}; a tie on short fixtures is not presented as "
        "an active-policy gain."
    ),
    "calibration_curve": (
        "Reliability diagram for the temporal forward filter over {corpus_observation_count} labelled "
        "observation steps. The plot shows {calibration_nonempty_bin_count} non-empty bins from "
        "{calibration_bin_count} equal-width confidence bins configured by the protocol; each point plots mean maximum "
        "posterior against observed "
        "accuracy; n labels the bin denominator, vertical bars are 95% Wilson intervals around bin hit "
        "counts, and the dashed diagonal is perfect calibration. Points below the diagonal indicate "
        "overconfidence for this fixture corpus [@guo2017calibration;@brown2001binomial]."
    ),
    "confusion_matrix": (
        "Pooled phase-decoding confusion matrix for the temporal forward filter across the labelled "
        "fixtures. Each cell shows the integer count followed by the percentage of its true-phase row; "
        "diagonal cells are correct recoveries and off-diagonal mass identifies confusable phase labels."
    ),
    "ambiguity_ablation": (
        "Three linked ablations on the same {fixture_count} transparent fixtures. Left: mean per-step "
        "decoder accuracy as the fixed categorical likelihood is blended toward uniform. Middle: pooled "
        "phase-boundary F1 for smoothed temporal and memoryless segmentations. Right: the number of "
        "fixtures in which the realized-EFE policy reaches the confidence threshold sooner than a fixed "
        "scan. Blend 0 is authored and blend 1 is uniform; the sweep demonstrates a mechanism under "
        "synthetic ambiguity, not a measurement of real-world binary noise."
    ),
    "segmentation_ribbons": (
        "Per-fixture phase-segmentation ribbons. For each fixture, the upper row is the hand-authored "
        "phase sequence and the lower row is the sequence decoded from smoothed forward-backward beliefs. "
        "Contiguous runs share a color, so a boundary disagreement is a displaced color transition. The "
        "pooled boundary F1 is reported in the figure subtitle and computed against the authored labels."
    ),
    "corpus_stability": (
        "Leave-one-fixture-out stability analysis. Bars show pooled temporal accuracy on the full corpus "
        "and after removing each fixture; the dashed line is the pooled identity-transition reference and the dotted "
        "line is the configured chance floor. The span in the title is the observed range across the "
        "leave-one-out fits, making small-corpus sensitivity explicit."
    ),
    "phase_recovery": (
        "Per-phase recovery for the temporal forward filter, ordered by F1. Grouped bars report recall "
        "and F1 for each of the {hidden_state_count} hand-authored phase labels; the hardest phase and "
        "macro F1 are stated in the figure subtitle. These are corpus-specific label-recovery metrics, "
        "not evidence that the model has discovered universal program phases."
    ),
    "decoder_calibration": (
        "Decoder calibration on the labelled corpus. Separate panels report expected calibration error "
        "and Brier score for memoryless, temporal, and identity-transition decoders; both metrics are dimensionless and "
        "lower is better. The ECE ordering identifies {best_calibrated} as best for {corpus_observation_count} "
        "labelled steps and is descriptive of these hand-authored traces, not a deployment-reliability "
        "guarantee [@guo2017calibration;@brown2001binomial]."
    ),
    "typed_observation_matrix": (
        "Typed observation record for the canonical fixture. Text cells preserve the instruction family, "
        "operand class, Capstone register read/write sets, flag effect, memory addressing mode, branch "
        "target class, and bounded dynamic status; shaded bands identify field families. An em dash "
        "means missing data rather than zero, and the table is categorical rather than a numeric heatmap "
        "[@intel_sdm;@capstone;@unicorn]."
    ),
    "factorized_observation_comparison": (
        "Leave-one-fixture-out comparison of the collapsed categorical HMM with static typed, full "
        "factorized, and hierarchical composite observation scores. The left panel reports held-out "
        "accuracy per fixture, the middle panel removes one typed field family at a time, and the right "
        "panel masks memory addressing metadata under the explicit missing-field policy. Every fit excludes "
        "the displayed fixture; no bar is a population estimate. The hierarchical score balances evidence "
        "groups and is not interpreted as a normalized joint likelihood [@rabiner1989tutorial;@gelman2013bda]."
    ),
    "representation_invariance_benchmark": (
        "Primary and negative-control endpoints for the separate, underpowered "
        "source-family-disjoint representation-invariance pilot. Bars report final-RAX "
        "equivalence consistency and near-miss separation over {representation_family_count} "
        "source-program families; whiskers are descriptive 95% Wilson intervals computed at "
        "the family level. The single-zero-state catalog is intentionally insufficient and "
        "leaves {representation_negative_control_unresolved} near-miss families unresolved. "
        "These are bounded Capstone/Unicorn protocol diagnostics, not fitted semantic-model "
        "performance, population uncertainty, or evidence of universal separation. The scaled "
        "64-family co-primary classifier benchmark is reported separately."
    ),
    "intervention_power": (
        "Exact finite intervention-power analysis for the source-family-disjoint quotient catalog. "
        "The left panel shows every one of the {intervention_power_subset_count} non-empty intervention "
        "subsets as a point and the exact minimum--maximum envelope by subset cardinality; the right "
        "panel resolves singleton interventions. Across {intervention_power_pair_count} equivalent and "
        "one-change near-miss pair outcomes, the smallest fully separating cardinality is "
        "{intervention_power_minimal_cardinality} with {intervention_power_minimal_set_count} tied set(s). "
        "The all-zero negative control separates {intervention_power_single_state_separation} of near-misses "
        "and leaves {intervention_power_single_state_unresolved_count} unresolved. No sampling uncertainty "
        "is shown because this is exact enumeration of a finite receipt, not a population estimate or a "
        "semantic-identifiability theorem."
    ),
    "trace_refinement": (
        "Stepwise trace-refinement check for the canonical bounded x86-64 execution. The left panel overlays "
        "post-instruction RAX values from the closed-world transfer functions and real Unicorn, with one marker "
        "per executed instruction; the right panel counts agreement and mismatch for only the EFLAGS bits defined "
        "by the transfer functions. Static instructions not visited by Unicorn are omitted. Exact stepwise agreement "
        "strengthens the terminal-value check but remains a fixture-bounded instrument validation, not a complete "
        "x86-64 semantics theorem [@intel_sdm;@dasgupta2019x86semantics;@unicorn]."
    ),
    "cross_architecture_benchmark": (
        "Cross-architecture executable fixture benchmark for the same declared numeric "
        "accumulator operation. The left panel reports the three Capstone-decoded "
        "instructions and the two non-returning instructions actually executed by Unicorn "
        "for x86-64, AArch64, and RISC-V64; the right panel reports the resulting accumulator "
        "value and the declared value of 8. All rows pass the bounded adapter contract. "
        "The shared numeric result is a local runtime check, not a cross-ISA semantic "
        "equivalence theorem or compiler-generalization result [@capstone;@unicorn]."
    ),
    "formal_verification_receipt": (
        "Kernel-checked theorem manifest for the declared Word64 accumulator fragment. "
        "Each marker records one theorem present in the source-bound Lean receipt under "
        "toolchain {formal_toolchain}; the fragment covers modular mov, add, sub, inc, and "
        "nop transfer functions plus an explicit near-miss counterexample. The receipt "
        "does not certify the Python pipeline, unsupported ISA effects, or complete x86 "
        "semantics [@lean_reference;@dasgupta2019x86semantics]."
    ),
    "active_information_frontier": (
        "Exact finite-horizon information-per-cost analysis over the declared "
        "label-blind typed-field sensor family of {information_policy_sensor_count} "
        "channels. Lines show expected information "
        "gain in nats as the sensing horizon grows under four cost weights; every "
        "point is obtained by exhaustive dynamic programming over sensor/outcome "
        "branches, with no Monte Carlo approximation. Curves may coincide when the "
        "finite contract selects the same information frontier at each weight. The "
        "plotted frontier uses unit action costs from the finite model channel; the separate sensor-cost receipt "
        "reports deterministic vectors and local timing calibration, so this panel is a mechanism frontier rather than "
        "a deployment cost law "
        "[@friston2017activeinference;@parr2022activeinference]."
    ),
    "quotient_classifier": (
        "Co-primary binary classification of same-formal-quotient membership for a source-family-disjoint "
        "transparent benchmark. The left panel reports balanced accuracy and macro-F1 with clustered 95% "
        "percentile intervals over held-out source families for the calibrated interpretable, temporal "
        "probabilistic, neural sequence, and surface-baseline models; the right panel reports Brier score "
        "and ten-bin expected calibration error. The benchmark contains {quotient_family_count} families, "
        "{quotient_row_count} pair rows, and a three-way training/development/held-out split. Six additional "
        "byte/opcode/mnemonic/structure baselines and source-family-paired Holm comparisons are retained in the receipt. Labels are "
        "generated from the declared terminal-observation relation and are not supplied as model features; "
        "these results are bounded endpoint evidence, not complete semantic equivalence or application performance."
    ),
    "compiler_transfer": (
        "Compiler/source/architecture capability receipt for {compiler_validated_count} locally compiled and "
        "statically disassembled objects. A check mark means the exact Clang target command produced an object "
        "whose text bytes were decoded by Capstone; a dash means the requested target was unavailable or failed "
        "and was retained as an explicit missing cell. The matrix includes {compiler_architecture_count} validated "
        "architecture target(s) and {compiler_unavailable_count} unavailable row(s); the line panel counts "
        "validated source-family-by-optimization rows. No emitted object was executed, "
        "so the figure supports capability and provenance claims only, not compiler correctness or semantic transfer."
    ),
    "formal_correspondence": (
        "Step-level correspondence receipt for the declared transparent x86-64 subset. The left panel reports "
        "whether every intervention trace for each fixture matched; the right panel reports reference-step workload "
        "across {formal_correspondence_fixture_count} fixtures and {formal_correspondence_intervention_count} bounded "
        "interventions per fixture. Matching compares decoded addresses, modeled RAX/RSP state, declared EFLAGS bits, "
        "and terminal output between the Python reference and bounded Unicorn; a generated Word64 boundary fixture is also "
        "kernel-checked in Lean. The result is a mechanized fixture-bounded correspondence, not a complete proof of x86-64 semantics."
    ),
    "sensor_costs": (
        "Two-layer sensing-cost receipt for the finite active-sensing benchmark. The left panel displays four of the "
        "eight deterministic cost-vector components as exact counts; the right panel displays local wall-clock medians "
        "with five-repetition min--max whiskers; the receipt additionally records IQR, p95, timing-overhead correction, "
        "and a deterministic-vector calibration regression. The package also contains {sensor_certificate_count} exact dynamic-program "
        "certificates with zero measured regret under the frozen finite state/action space; fixed, greedy, seeded-random, "
        "and abstaining policy rows are also reported. Deterministic counts are the "
        "primary auditable endpoint; calibrated nanoseconds are machine-specific and are not generalized across hardware."
    ),
    "replication_manifest": (
        "Provenance boundary for the frozen no-network local replication harness. Public feature rows and the separately "
        "held-out gold-label rows each contain {replication_row_count} records, while predictions and machine-readable "
        "receipts carry hashes tied to the protocol. Arrows show the local generation, scoring, and validation path; "
        "public filenames, ordering, and metadata are label-free. This bundle demonstrates replayability and blind separation "
        "inside the repository, not independent external replication."
    ),
    "external_source_provenance": (
        "Licensed-source provenance and application gate for the separate external-validation track. The left panel shows "
        "the pinned-manifest, explicitly opt-in acquisition, and static-only object-analysis path; the right panel lists "
        "the immutable revision, license/hash, safety, project/revision/example, and independent-replay requirements. "
        "The current receipt is {external_source_status}: {external_project_count} projects are pinned, "
        "{external_static_validated_count} of {external_static_row_count} static rows decode successfully, and "
        "the source-bound contract is {external_label_status} with {external_labelable_count} accepted rows "
        "({external_label_equivalent_count} lexical-equivalence witnesses, {external_label_near_miss_count} "
        "contract-directed near misses, and {external_label_negative_count} identity controls; "
        "{external_label_exclusion_count} explicit exclusions; held-out project(s): {external_label_test_projects}). "
        "Lexical witnesses and near misses are source-contract controls, not universal semantic proofs or executed "
        "behavioral labels. No application-performance value is plotted and unavailable evidence is not treated as zero. "
        "This is a source-bound safety and promotion diagnostic, not evidence of external validity."
    ),
    "external_label_coverage": (
        "Source-bound external label-contract census for {external_labelable_count} accepted rows across "
        "{external_label_project_count} projects and {external_label_revision_count} project revisions. "
        "The left panel counts the source-defined equivalence witnesses, contract-directed near misses, and "
        "identity controls by project; the right panel locates accepted rows in the frozen train, development, "
        "and held-out test allocation, marking held-out project(s) explicitly. Counts are deterministic provenance "
        "diagnostics with no inferential uncertainty; an em dash means no accepted row in that cell, while explicit "
        "exclusions remain outside the accepted-row denominator. Rows are static-only and no emitted object is executed. "
        "This figure does not estimate model accuracy or prove universal semantic equivalence."
    ),
    "artifact_receipt_closure": (
        "Protocol-v2 artifact receipt closure for {artifact_receipt_count} project-owned JSON artifacts. "
        "The flow binds the canonical protocol to producer modules, atomically written domain payloads, "
        "surface validation, and the hydrated manuscript/DOCX/tagged-PDF boundary; the companion bars retain "
        "the {scientific_program_milestone_count}-milestone evidence-status contract. The audit found "
        "{artifact_receipt_error_count} receipt error(s). Counts are deterministic provenance checks with no "
        "inferential uncertainty; blocked, unavailable, and underpowered evidence remains labelled rather than "
        "rendered as zero performance, and this figure does not establish scientific promotion [@w3cprov2013;@rocrate2023;@sandve2013repro;@acm2020artifact]."
    ),
}

# The visual-atlas tranche adds nine narrative figures.  Their canonical
# metadata, source hashes, and fully expanded captions are produced by
# ``src.figure_contract``; these compatibility entries keep the long-standing
# ``FIGURE_CAPTIONS`` callers and validation oracles complete during the
# transition.
FIGURE_CAPTIONS.update(
    {
        "research_architecture": (
            "Research architecture for the source-bound assembly-to-inference pipeline. "
            "The diagram follows bytes from static Capstone decoding through normalized "
            "instruction representation, Python reference semantics, bounded Unicorn "
            "cross-checks, the declared Lean proof boundary, typed observations, finite "
            "interventions and quotient labels, the co-primary model portfolio, sensing "
            "costs, receipts, claim registration, and template hydration. Boundary labels "
            "distinguish kernel-checked, computationally cross-checked, inferential, and "
            "provenance-preserving edges. This is an architecture and evidence map, not "
            "a claim that every edge proves the next one or that the package implements "
            "complete x86-64 semantics."
        ),
        "assembly_state_plate": (
            "Assembly-state plate for a transparent x86-64 fixture. The listing is decoded "
            "from the hashed fixture and paired with normalized opcode families, operand "
            "widths, flag effects, bounded register-state transitions, program-counter "
            "updates, and memory effects where present. A separate unsupported or bound "
            "failure branch is retained to make the closed-world boundary visible. State "
            "rows are exact observations from the declared fixture and interventions; they "
            "are not a universal architectural trace and do not establish behavior for "
            "unsupported instructions, exceptions, aliases, or arbitrary external code."
        ),
        "quotient_construction": (
            "Finite quotient construction used by the source-bound benchmark. Mechanically "
            "documented source pairs are assigned equivalent-transformation, one-change "
            "near-miss, or non-separating-control roles; declared initial-state interventions "
            "are applied; normalized traces are compared; and the resulting endpoint relation "
            "defines the benchmark label before model fitting. The held-out source-family "
            "split is shown as a separate boundary. Agreement under this finite intervention "
            "catalog means agreement on the declared observation and output relation; it is "
            "not unrestricted program equivalence, semantic discovery, or universal "
            "identifiability."
        ),
        "semantic_coverage_matrix": (
            "Declared semantic coverage across instruction families and evidence layers. "
            "Rows are grouped by the supported normalized x86-64 subset and the matrix "
            "records width, flag, memory, control-flow, Lean, Python, Capstone, and bounded "
            "Unicorn coverage. Text symbols and hatching distinguish supported, checked, "
            "computationally cross-checked, unsupported, and not-applicable cells; an absent "
            "cell is never treated as a zero score. The matrix is a receipt of the declared "
            "formal and adapter boundary, not a complete-ISA specification or a claim that "
            "static decoding alone validates dynamic semantics."
        ),
        "statistical_hierarchy": (
            "Statistical hierarchy and leakage boundary for the source-bound evaluations. "
            "Projects and source families contain revisions, revisions contain transformation "
            "pairs, pairs are evaluated under interventions, and interventions produce traces "
            "with instruction steps. The family is the primary clustered bootstrap unit; pair "
            "rows and instruction steps remain dependent observations. Training, development, "
            "and test family partitions are fixed before model fitting, while development data "
            "alone controls selection and stopping. The diagram therefore explains what the "
            "reported intervals quantify and why row-count inflation cannot be read as extra "
            "independent scientific samples."
        ),
        "replay_failure_audit": (
            "Controlled replay failure audit for the public label-blind interface. Public rows "
            "cross a no-network guest boundary without gold labels; predictions return through "
            "identifier validation; protected scoring remains on the owner host; and the "
            "receipt records a same-host separately controlled guest rather than an independent "
            "physical-machine replication or independent reimplementation. The confusion, "
            "calibration, and selective-risk panels reproduce the protected negative result: "
            "the observed model predicts equivalence for both classes and therefore has chance "
            "balanced accuracy with poor calibration. This result is a reproducibility and "
            "failure-boundary audit, not an efficacy result or a new independent sample."
        ),
        "policy_certificate_frontier": (
            "Finite active-sensing policy frontier under the declared sensor and action space. "
            "Points combine exact dynamic-program information gain, deterministic receipt cost, "
            "local calibrated wall-clock cost when available, regret, and optimality gap for "
            "implemented, fixed, greedy, seeded-random, abstaining, and exact-oracle policies. "
            "Exact enumeration has no sampling confidence interval; timing summaries are "
            "machine-specific calibration diagnostics with their own trial counts and spread. "
            "The tolerance and certificate apply only to the frozen finite horizon, state, and "
            "sensor catalog, not to an information-per-cost law for arbitrary binaries or "
            "hardware."
        ),
        "theorem_status_map": (
            "Theorem and countertheorem status map for the formal program. Each candidate is "
            "located as kernel-checked, computationally validated, empirically supported, "
            "falsified, or unresolved, with its state and memory bounds, supported instruction "
            "subset, intervention and observation assumptions, and proof or counterexample "
            "artifact. A checked Lean theorem is kept separate from a Python, Capstone, or "
            "Unicorn cross-check; a bounded computational result is kept separate from a broad "
            "mathematical claim. Falsified and unresolved rows remain first-class results, and "
            "the manuscript promotion language is generated from these statuses rather than "
            "from the existence of a theorem-shaped file."
        ),
        "observation_semantics": (
            "Observation-semantics diagram for the typed, label-blind interface. The visual "
            "distinguishes observed values, explicit missing fields, unseen categorical values, "
            "unsupported operations, unavailable capabilities, and measured numeric zero; it also "
            "shows the boundary between public rows, predictions, protected labels, and source-bound "
            "receipts. These are schema and provenance categories, not interchangeable encodings. "
            "The diagram is descriptive and computationally constructed from the frozen protocol and "
            "replication manifests. It does not estimate model performance, recover protected labels, "
            "establish complete semantic coverage, or imply universal identifiability."
        ),
    }
)

# Keep the compatibility caption mapping synchronized with the redesigned
# high-risk figures.  The canonical figure contract adds the richer
# alternative text and long description; these entries remain available to
# legacy callers and are hydrated through the same receipt context.
FIGURE_CAPTIONS.update(
    {
        "formal_correspondence": (
            "Field-level correspondence receipt for the declared transparent x86-64 subset. "
            "The left panel uses numerator/denominator marks for step agreement, decoded "
            "addresses, register state, modeled flags, memory effects, control targets, and "
            "terminal RAX; the right panel reports Python-reference and Unicorn step workloads "
            "by fixture. The receipt covers {formal_correspondence_fixture_count} fixtures and "
            "{formal_correspondence_intervention_count} bounded interventions per fixture, with "
            "explicit unsupported and bound/exception rows. These are exact fixture-level checks, "
            "so inferential intervals are not applicable. A generated Word64 boundary fixture is "
            "also kernel-checked in Lean. The result is a bounded correspondence instrument, not "
            "a complete proof of x86-64 semantics."
        ),
        "quotient_classifier": (
            "Co-primary binary classification of the declared finite quotient relation under "
            "source-family holdout. The left panel shows balanced accuracy and macro-F1 as "
            "model-specific dot-and-interval estimates; the middle panel shows paired model-minus-"
            "surface-baseline deltas with source-family-clustered 95% bootstrap intervals; and the "
            "right panel shows Brier score and ten-bin expected calibration error as point diagnostics, "
            "with selective coverage and risk reported below. The benchmark contains "
            "{quotient_family_count} families, {quotient_row_count} pair rows, and fixed "
            "training/development/held-out family partitions. Primary intervals use source-family "
            "clusters; Brier/ECE have no clustered interval in the current receipt. Positive deltas "
            "do not by themselves authorize a superiority claim, and no post-hoc model winner is "
            "selected. This is bounded endpoint evidence, not unrestricted semantic equivalence or "
            "application performance."
        ),
        "compiler_transfer": (
            "Compiler/source/architecture capability receipt for {compiler_validated_count} locally "
            "compiled and statically disassembled objects. The left panel is a status matrix: a "
            "check mark means the exact compiler command produced an object whose text bytes decoded "
            "with Capstone, while a dash denotes an unavailable or failed capability retained as an "
            "explicit missing state rather than a zero score. The matrix includes "
            "{compiler_architecture_count} validated architecture target(s) and "
            "{compiler_unavailable_count} unavailable row(s); the right panel counts validated "
            "source-by-optimization rows. Counts are deterministic provenance diagnostics with no "
            "population uncertainty. No emitted object was executed, so the visual supports local "
            "capability and provenance claims only, not compiler correctness, behavioral transfer, "
            "or architecture-general performance."
        ),
        "sensor_costs": (
            "Two-layer sensing-cost receipt for the finite active-sensing benchmark. The left panel "
            "shows all eight deterministic cost-vector components as exact cell labels with shading "
            "used only for magnitude; the right panel shows machine-specific wall-clock medians and "
            "min--max whiskers in nanoseconds per query. The deterministic units are sensor calls, "
            "decoded instructions, bounded steps, register/flag observations, branch observations, "
            "memory-hook events, serialized bytes, and unsupported attempts. Timing uses the recorded "
            "calibration repetitions and trials, and the receipt also stores IQR, p95, overhead, and "
            "regression diagnostics. {sensor_certificate_count} exact dynamic-program certificates "
            "are scoped to the frozen finite state/action space; no cross-machine cost law is implied."
        ),
        "replication_manifest": (
            "Public/private provenance boundary for the frozen no-network replication harness. The "
            "diagram explicitly encloses the {replication_row_count}-row gold-label surface inside a "
            "protected owner boundary, while label-blind public rows, predictions, and hash-bound "
            "receipts remain on the replay path. Identifier validation and protocol hashes are shown "
            "as gates rather than as scientific outcomes. Counts are deterministic and no sampling "
            "uncertainty is applicable to this manifest audit. The package demonstrates local replay "
            "and blind separation; the controlled same-host guest result is not independent physical "
            "replication or independent reimplementation."
        ),
        "external_source_provenance": (
            "Licensed-source provenance and P13 application-validity gate for the separate external "
            "validation track. The left panel shows pinned acquisition, explicit network opt-in, and "
            "static-only compilation/disassembly; the right panel lists the immutable revision, "
            "license/hash, safety, project/revision/example, and separate-acquisition-replay gates. "
            "The current receipt is {external_source_status}: {external_project_count} project(s) are "
            "pinned and {external_static_validated_count} of {external_static_row_count} static rows "
            "decode successfully, with {external_labelable_count} source-contract rows accepted. "
            "Lexical witnesses, directed near misses, and identity controls are provenance labels, "
            "not universal behavioral equivalence. P13 remains distinct from the observed P15 "
            "controlled replay; no unavailable evidence is rendered as zero and no application claim "
            "is promoted by this diagnostic alone."
        ),
        "external_label_coverage": (
            "Source-bound external label-contract census for {external_labelable_count} accepted rows "
            "across {external_label_project_count} projects and {external_label_revision_count} "
            "project revisions. The left panel counts source-defined equivalence witnesses, directed "
            "near misses, and identity controls by project; the right panel shows the frozen train, "
            "development, and held-out allocation. An em dash means that no accepted row exists in a "
            "cell; explicit exclusions remain outside the denominator. Counts are deterministic static "
            "provenance diagnostics with no inferential interval, and no emitted object is executed. "
            "The figure does not estimate model accuracy or prove unrestricted semantic equivalence."
        ),
    }
)

FIGURE_CAPTIONS.update(
    {
        "counterexample_search": (
            "Smallest-first exhaustive counterexample search over the declared bounded "
            "x86-64 reference model. Each bar is one declared candidate theorem resolved "
            "inside its declared finite domain; orange bars were falsified by a verified "
            "minimal witness and green bars held across every enumerated input. Witness "
            "ranks are re-checked by an independent second interpretation of the same "
            "reference semantics. Falsified candidates remain part of the evidence and "
            "narrow the claim; a holds result is bounded to the searched domain and "
            "establishes no complete-ISA theorem."
        ),
        "sensing_stress": (
            "Active-sensing stress study over the declared label-blind typed-field "
            "sensor channels. The left panel shows each policy control's maximum regret "
            "against the exact finite-horizon dynamic-programming reference across the "
            "horizon, budget, and cost-weight grid; the right panel shows the "
            "budget-constrained value curve with the exact unconstrained bound and the "
            "unattained gap that finite budgets forgo. Regret and gaps are "
            "exact-enumeration quantities; the accompanying calibration-stability rows "
            "repeat the local timing regression and are machine-specific measurements, "
            "not a universal information-per-cost law."
        ),
    }
)


FIGURE_CITATIONS: dict[str, tuple[str, ...]] = {
    "generative_model": ("friston2017activeinference", "parr2022activeinference"),
    "observation_informativeness": ("friston2017activeinference",),
    "model_evidence": ("kass1995bayes", "rabiner1989tutorial"),
    "transition_ablation": ("rabiner1989tutorial", "friston2017activeinference"),
    "generative_adequacy": ("rabiner1989tutorial", "gelman2013bda"),
    "observation_pipeline": (
        "capstone",
        "quynh2014capstone",
        "intel_sdm",
    ),
    "assembly_cfg": (
        "capstone",
        "quynh2014capstone",
        "unicorn",
        "quynh2015unicorn",
        "intel_sdm",
    ),
    "assembly_memory_timeline": (
        "capstone",
        "quynh2014capstone",
        "unicorn",
        "quynh2015unicorn",
        "intel_sdm",
    ),
    "posterior_beliefs": ("friston2017activeinference",),
    "package_provenance": ("capstone", "unicorn"),
    "trace_feature_matrix": ("capstone", "quynh2014capstone"),
    "rxinfer_free_energy": (
        "bagaev2023rxinfer",
        "rxinfer_static_docs",
        "rxinfer_python_docs",
    ),
    "rxinfer_categorical_free_energy": (
        "bagaev2023rxinfer",
        "rxinfer_static_docs",
    ),
    "inference_diagnostics": ("bagaev2023rxinfer", "heins2022pymdp"),
    "scholarship_source_roles": (),
    "scholarship_evidence_matrix": (),
    "claim_evidence_map": (),
    "artifact_hash_manifest": (),
    "validation_summary": (),
    "temporal_beliefs_x86_accumulator": (
        "friston2017activeinference",
        "rabiner1989tutorial",
    ),
    "temporal_beliefs_x86_bitwise": (
        "friston2017activeinference",
        "rabiner1989tutorial",
    ),
    "temporal_beliefs_x86_guard": (
        "friston2017activeinference",
        "rabiner1989tutorial",
    ),
    "temporal_beliefs_x86_long_accumulator": (
        "friston2017activeinference",
        "rabiner1989tutorial",
    ),
    "temporal_beliefs_x86_memory": (
        "friston2017activeinference",
        "capstone",
        "unicorn",
    ),
    "temporal_beliefs_x86_guard_taken": (
        "friston2017activeinference",
        "capstone",
        "unicorn",
    ),
    "corpus_coverage": ("capstone", "unicorn", "rabiner1989tutorial"),
    "action_sensing": (
        "friston2017activeinference",
        "parr2022activeinference",
        "rabiner1989tutorial",
    ),
    "parameter_learning": ("dempster1977em", "rabiner1989tutorial"),
    "model_comparison": ("rabiner1989tutorial", "brown2001binomial"),
    "active_efe": ("friston2017activeinference", "parr2022activeinference"),
    "calibration_curve": ("guo2017calibration", "brown2001binomial"),
    "confusion_matrix": ("rabiner1989tutorial",),
    "ambiguity_ablation": ("friston2017activeinference", "parr2022activeinference"),
    "segmentation_ribbons": ("friston2017activeinference", "rabiner1989tutorial"),
    "corpus_stability": ("rabiner1989tutorial",),
    "phase_recovery": ("rabiner1989tutorial",),
    "decoder_calibration": ("guo2017calibration", "brown2001binomial"),
    "typed_observation_matrix": ("intel_sdm", "capstone", "unicorn"),
    "factorized_observation_comparison": (
        "rabiner1989tutorial",
        "gelman2013bda",
    ),
    "representation_invariance_benchmark": (
        "capstone",
        "unicorn",
        "brown2001binomial",
    ),
    "intervention_power": ("capstone", "unicorn", "intel_sdm"),
    "trace_refinement": ("intel_sdm", "dasgupta2019x86semantics", "unicorn"),
    "cross_architecture_benchmark": ("capstone", "unicorn"),
    "formal_verification_receipt": (
        "lean_reference",
        "dasgupta2019x86semantics",
    ),
    "active_information_frontier": (
        "friston2017activeinference",
        "parr2022activeinference",
    ),
    "quotient_classifier": (
        "rabiner1989tutorial",
        "guo2017calibration",
        "brown2001binomial",
    ),
    "compiler_transfer": (
        "capstone",
        "clang_users_manual",
        "llvm_objdump",
        "intel_sdm",
        "necula2000translationvalidation",
    ),
    "formal_correspondence": (
        "capstone",
        "unicorn",
        "lean_reference",
        "intel_sdm",
    ),
    "sensor_costs": ("friston2017activeinference", "parr2022activeinference"),
    "replication_manifest": ("wilkinson2016fair",),
    "external_source_provenance": ("wilkinson2016fair",),
    "external_label_coverage": ("wilkinson2016fair",),
    "artifact_receipt_closure": (
        "w3cprov2013",
        "rocrate2023",
        "sandve2013repro",
        "acm2020artifact",
    ),
}

FIGURE_CITATIONS.update(
    {
        "research_architecture": (
            "capstone",
            "unicorn",
            "lean_reference",
            "wilkinson2016fair",
        ),
        "assembly_state_plate": (
            "intel_sdm",
            "capstone",
            "unicorn",
            "lean_reference",
        ),
        "quotient_construction": (
            "intel_sdm",
            "capstone",
            "unicorn",
            "necula2000translationvalidation",
        ),
        "semantic_coverage_matrix": (
            "intel_sdm",
            "dasgupta2019x86semantics",
            "lean_reference",
        ),
        "statistical_hierarchy": (
            "gelman2013bda",
            "brown2001binomial",
            "wilkinson2016fair",
        ),
        "replay_failure_audit": ("wilkinson2016fair", "guo2017calibration"),
        "policy_certificate_frontier": (
            "friston2017activeinference",
            "parr2022activeinference",
        ),
        "theorem_status_map": (
            "lean_reference",
            "wilkinson2016fair",
        ),
        "observation_semantics": (
            "intel_sdm",
            "wilkinson2016fair",
        ),
    }
)


def _caption_context(output_dir: Path) -> dict[str, str]:
    """Load source-bound counts used by captions, with safe pre-generation defaults."""

    context = {
        "hidden_state_count": "configured",
        "observation_step_count": "configured",
        "corpus_observation_count": "configured",
        "fixture_count": "configured",
        "sample_name": "the selected",
        "ci_level": "95%",
        "most_informative": "configured",
        "least_informative": "configured",
        "best_model": "configured",
        "best_calibrated": "configured",
        "active_never_worse": "configured",
        "calibration_bin_count": "configured",
        "calibration_nonempty_bin_count": "configured",
        "rxinfer_categorical_delta": "configured",
        "rxinfer_categorical_flat": "the status is not yet available",
        "memory_fixture_count": "configured",
        "alternate_path_fixture_count": "configured",
        "repeated_phase_fixture_count": "configured",
        "representation_family_count": "configured",
        "representation_negative_control_unresolved": "configured",
        "intervention_power_subset_count": "configured",
        "intervention_power_pair_count": "configured",
        "intervention_power_minimal_cardinality": "not available",
        "intervention_power_minimal_set_count": "configured",
        "intervention_power_single_state_separation": "configured",
        "intervention_power_single_state_unresolved_count": "configured",
        "cross_architecture_count": "configured",
        "formal_theorem_count": "configured",
        "formal_toolchain": "configured",
        "information_policy_sensor_count": "configured",
        "quotient_family_count": "configured",
        "quotient_row_count": "configured",
        "compiler_validated_count": "configured",
        "compiler_unavailable_count": "configured",
        "compiler_architecture_count": "configured",
        "formal_correspondence_fixture_count": "configured",
        "formal_correspondence_intervention_count": "configured",
        "sensor_certificate_count": "configured",
        "replication_row_count": "configured",
        "external_source_status": "not_yet_observed",
        "external_project_count": "0",
        "external_labelable_count": "0",
        "external_label_status": "not_yet_observed",
        "external_label_equivalent_count": "0",
        "external_label_near_miss_count": "0",
        "external_label_negative_count": "0",
        "external_label_exclusion_count": "0",
        "external_label_test_projects": "none",
        "external_label_project_count": "0",
        "external_label_revision_count": "0",
        "external_static_validated_count": "0",
        "external_static_row_count": "0",
        "artifact_receipt_count": "configured",
        "artifact_receipt_error_count": "configured",
        "scientific_program_milestone_count": "configured",
    }
    data_dir = output_dir / "data"
    try:
        trace = json.loads((data_dir / "trace_bundle.json").read_text(encoding="utf-8"))
        context["hidden_state_count"] = str(len(trace.get("hidden_state_labels", [])))
        context["observation_step_count"] = str(len(trace.get("observations", [])))
        context["sample_name"] = str(trace.get("sample_id", context["sample_name"]))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        coverage = json.loads(
            (data_dir / "corpus_coverage.json").read_text(encoding="utf-8")
        )
        coverage_summary = coverage.get("summary", {})
        context["memory_fixture_count"] = str(
            coverage_summary.get(
                "memory_fixture_count", context["memory_fixture_count"]
            )
        )
        context["alternate_path_fixture_count"] = str(
            coverage_summary.get(
                "alternate_path_fixture_count", context["alternate_path_fixture_count"]
            )
        )
        context["repeated_phase_fixture_count"] = str(
            coverage_summary.get(
                "repeated_phase_fixture_count", context["repeated_phase_fixture_count"]
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        evaluation = json.loads(
            (data_dir / "evaluation_report.json").read_text(encoding="utf-8")
        )
        context["most_informative"] = str(
            evaluation.get("observation_informativeness", {}).get(
                "most_informative", context["most_informative"]
            )
        )
        context["least_informative"] = str(
            evaluation.get("observation_informativeness", {}).get(
                "least_informative", context["least_informative"]
            )
        )
        context["best_model"] = str(
            evaluation.get("model_evidence", {}).get(
                "best_model", context["best_model"]
            )
        )
        context["best_calibrated"] = str(
            evaluation.get("decoder_calibration", {}).get(
                "best_calibrated", context["best_calibrated"]
            )
        )
        context["active_never_worse"] = str(
            evaluation.get("ambiguity_ablation", {}).get(
                "active_never_worse", context["active_never_worse"]
            )
        ).lower()
        summary = evaluation.get("summary", {})
        context["corpus_observation_count"] = str(
            summary.get(
                "observation_count",
                evaluation.get("model_evidence", {}).get(
                    "observation_count", context["corpus_observation_count"]
                ),
            )
        )
        context["fixture_count"] = str(
            summary.get("fixture_count", context["fixture_count"])
        )
        context["calibration_bin_count"] = str(
            evaluation.get("calibration_bin_count", context["calibration_bin_count"])
        )
        context["calibration_nonempty_bin_count"] = str(
            len(evaluation.get("pooled_calibration_bins", []))
        )
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        categorical = json.loads(
            (
                output_dir / "rxinfer_categorical" / "categorical_diagnostics.json"
            ).read_text(encoding="utf-8")
        )
        delta = float(
            categorical.get("free_energy_final", 0.0)
            - categorical.get("free_energy_initial", 0.0)
        )
        context["rxinfer_categorical_delta"] = f"{delta:.3f}"
        context["rxinfer_categorical_flat"] = (
            "all values are identical to recorded precision"
            if abs(delta) < 1e-12
            else "values change across iterations"
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        representation = json.loads(
            (data_dir / "representation_invariance_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        context["representation_family_count"] = str(
            representation.get("family_count", context["representation_family_count"])
        )
        context["representation_negative_control_unresolved"] = str(
            sum(
                not bool(pair.get("near_miss_separated"))
                for pair in representation.get("negative_control", {}).get("pairs", [])
                if isinstance(pair, dict) and pair.get("variant_role") == "near_miss"
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        intervention = json.loads(
            (data_dir / "intervention_power.json").read_text(encoding="utf-8")
        )
        context["intervention_power_subset_count"] = str(
            len(intervention.get("subset_evaluations", []))
        )
        context["intervention_power_pair_count"] = str(
            intervention.get("pair_count", context["intervention_power_pair_count"])
        )
        minimal_cardinality = intervention.get("minimal_separating_cardinality")
        context["intervention_power_minimal_cardinality"] = (
            str(minimal_cardinality)
            if minimal_cardinality is not None
            else "not available"
        )
        context["intervention_power_minimal_set_count"] = str(
            len(intervention.get("minimal_separating_sets", []))
        )
        negative = intervention.get("negative_control", {})
        context["intervention_power_single_state_separation"] = str(
            negative.get("near_miss_separation", "not available")
        )
        context["intervention_power_single_state_unresolved_count"] = str(
            len(negative.get("unresolved_near_miss_families", []))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        cross_architecture = json.loads(
            (data_dir / "cross_architecture_fixtures.json").read_text(encoding="utf-8")
        )
        context["cross_architecture_count"] = str(
            cross_architecture.get("summary", {}).get(
                "architecture_count", context["cross_architecture_count"]
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        formal = json.loads(
            (data_dir / "formal_verification.json").read_text(encoding="utf-8")
        )
        context["formal_theorem_count"] = str(len(formal.get("theorems", [])))
        context["formal_toolchain"] = str(
            formal.get("toolchain", context["formal_toolchain"])
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        information = json.loads(
            (data_dir / "active_information_frontier.json").read_text(encoding="utf-8")
        )
        context["information_policy_sensor_count"] = str(
            len(information.get("sensors", []))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        quotient = json.loads(
            (data_dir / "quotient_classifier.json").read_text(encoding="utf-8")
        )
        context["quotient_family_count"] = str(
            quotient.get("family_count", context["quotient_family_count"])
        )
        context["quotient_row_count"] = str(
            quotient.get("row_count", context["quotient_row_count"])
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        compiler = json.loads(
            (data_dir / "compiler_transfer.json").read_text(encoding="utf-8")
        )
        summary = compiler.get("capability_summary", {})
        context["compiler_validated_count"] = str(
            summary.get("validated_static_objects", context["compiler_validated_count"])
        )
        context["compiler_unavailable_count"] = str(
            summary.get("unavailable_rows", context["compiler_unavailable_count"])
        )
        context["compiler_architecture_count"] = str(
            len(summary.get("architectures_validated", []))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        correspondence = json.loads(
            (data_dir / "formal_trace_correspondence.json").read_text(encoding="utf-8")
        )
        context["formal_correspondence_fixture_count"] = str(
            correspondence.get(
                "fixture_count", context["formal_correspondence_fixture_count"]
            )
        )
        context["formal_correspondence_intervention_count"] = str(
            correspondence.get(
                "intervention_count",
                context["formal_correspondence_intervention_count"],
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        sensor = json.loads(
            (data_dir / "sensor_cost_receipt.json").read_text(encoding="utf-8")
        )
        context["sensor_certificate_count"] = str(
            len(sensor.get("policy_certificates", []))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        replication = json.loads(
            (data_dir / "replication_manifest.json").read_text(encoding="utf-8")
        )
        context["replication_row_count"] = str(
            replication.get("public_row_count", context["replication_row_count"])
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        labels = json.loads(
            (data_dir / "external_label_contract.json").read_text(encoding="utf-8")
        )
        label_rows = labels.get("contracts", [])
        if not isinstance(label_rows, list):
            label_rows = []
        context["external_label_status"] = str(
            labels.get("status", context["external_label_status"])
        )
        context["external_label_equivalent_count"] = str(
            sum(
                isinstance(row, dict)
                and row.get("role", row.get("variant_role")) == "equivalent"
                for row in label_rows
            )
        )
        context["external_label_near_miss_count"] = str(
            sum(
                isinstance(row, dict)
                and row.get("role", row.get("variant_role")) == "near_miss"
                for row in label_rows
            )
        )
        context["external_label_negative_count"] = str(
            sum(
                isinstance(row, dict)
                and row.get("role", row.get("variant_role")) == "negative_control"
                for row in label_rows
            )
        )
        context["external_label_exclusion_count"] = str(
            len(labels.get("exclusions", []))
        )
        label_summary = labels.get("summary", {})
        if isinstance(label_summary, dict):
            context["external_labelable_count"] = str(
                label_summary.get("labelable_example_count", len(label_rows))
            )
            project_count = label_summary.get("project_count")
            if isinstance(project_count, int) and not isinstance(project_count, bool):
                context["external_label_project_count"] = str(project_count)
            revisions = label_summary.get("revisions_by_project", {})
            if isinstance(revisions, dict):
                context["external_label_revision_count"] = str(
                    sum(
                        len(value)
                        for value in revisions.values()
                        if isinstance(value, list)
                    )
                )
            test_projects = label_summary.get("test_projects", [])
            if isinstance(test_projects, list) and test_projects:
                context["external_label_test_projects"] = ", ".join(
                    str(project) for project in test_projects
                )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        external = json.loads(
            (data_dir / "external_source_validation.json").read_text(encoding="utf-8")
        )
        context["external_source_status"] = str(
            external.get("status", context.get("external_source_status", "unknown"))
        )
        context["external_project_count"] = str(
            external.get("project_count", context.get("external_project_count", "0"))
        )
        context["external_labelable_count"] = str(
            external.get(
                "labelable_example_count", context.get("external_labelable_count", "0")
            )
        )
        context["external_static_validated_count"] = str(
            external.get(
                "static_validated_row_count",
                context.get("external_static_validated_count", "0"),
            )
        )
        context["external_static_row_count"] = str(len(external.get("static_rows", [])))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        from .artifact_contract import validate_data_artifact_surface

        data_json = tuple(sorted(data_dir.glob("*.json")))
        context["artifact_receipt_count"] = str(len(data_json))
        context["artifact_receipt_error_count"] = str(
            len(validate_data_artifact_surface(output_dir.parent))
        )
    except (OSError, TypeError, AttributeError):
        pass
    try:
        program = json.loads(
            (data_dir / "scientific_program.json").read_text(encoding="utf-8")
        )
        context["scientific_program_milestone_count"] = str(
            len(program.get("milestones", []))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return context


def _figure_registry_rows(
    output_dir: Path, context: dict[str, str]
) -> list[dict[str, Any]]:
    """Build the complete figure metadata registry from the manifest contract."""

    from .reproducibility import MANIFEST_TARGETS

    target_by_filename = {
        Path(target.path).name: target
        for target in MANIFEST_TARGETS
        if target.artifact_type == "figure"
    }
    fallback_claims = {
        "artifact_hash_manifest": ("CLM-009",),
        "validation_summary": ("CLM-009",),
    }
    fallback_checks = {stem: (f"figure_{stem}",) for stem in FIGURE_CAPTIONS}
    population = {
        "assembly_cfg": "branch-bearing x86_guard fixture decoded with Capstone and bounded with Unicorn",
        "assembly_memory_timeline": "transparent x86_memory fixture and Unicorn memory-hook events",
        "generative_model": "source-controlled model constants",
        "package_provenance": "required and optional adapter probes",
        "scholarship_source_roles": "curated scholarship source ledger",
        "scholarship_evidence_matrix": "curated scholarship evidence tiers",
        "claim_evidence_map": "bounded claim register",
        "generative_adequacy": "same-length simulations of the fixed A/B/D HMM for each transparent fixture",
        "typed_observation_matrix": "label-blind typed records from Capstone detail and bounded Unicorn execution",
        "factorized_observation_comparison": "leave-one-fixture-out typed-model evaluation and field-family ablations",
        "corpus_coverage": "source-bound fixture-design inventory from Capstone and bounded Unicorn records",
        "action_sensing": "held-out transparent fixtures with typed field actions and training-only field likelihoods",
        "parameter_learning": "complete-fixture holdouts with observation-only ordered HMM EM, fixed-penalty Dirichlet MAP, supervised/fixed comparators, and label-free predictive checks",
        "representation_invariance_benchmark": "source-family-disjoint transparent x86-64 transformation families and negative-control witness rows",
        "intervention_power": "exact finite intervention-subset enumeration over the source-bound quotient outcome matrix",
        "trace_refinement": "executed instruction steps in the canonical bounded x86-64 fixture",
        "cross_architecture_benchmark": "real Capstone/Unicorn bounded fixtures for x86-64, AArch64, and RISC-V64",
        "formal_verification_receipt": "source-bound Lean theorem manifest for the closed Word64 accumulator fragment",
        "active_information_frontier": "exact finite-horizon dynamic-programming analysis of the declared label-blind sensor channels",
        "quotient_classifier": "source-family-disjoint transparent quotient-pair benchmark with three co-primary NumPy model families",
        "compiler_transfer": "repository-owned portable C sources compiled locally and statically decoded by target capability",
        "formal_correspondence": "transparent x86-64 fixtures compared step-by-step between the Python reference and bounded Unicorn",
        "sensor_costs": "finite typed-field sensor actions with deterministic vectors and local timing calibration",
        "replication_manifest": "frozen local public/gold/prediction/receipt bundle",
        "external_source_provenance": "pinned licensed-source manifest and static-only external-validation gate",
        "external_label_coverage": "source-bound external label-contract rows and frozen project/revision split allocation",
        "artifact_receipt_closure": "project-owned protocol-v2 receipt surface and manuscript/document boundary",
        "observation_semantics": "typed observation schema and public/protected replay boundary",
        "artifact_hash_manifest": "reproducibility manifest rows",
        "validation_summary": "generated validation checks",
        "counterexample_search": "declared candidate theorems over the bounded Word64 reference interpreter with width-aware sub-register writes, flag definedness, and memory/stack bounds",
        "sensing_stress": "declared label-blind typed-field sensor channels with exact dynamic-programming references and fixed-seed policy rollouts",
    }
    statistics = {
        "assembly_cfg": "Static direct-target block and edge inventory for the branch-bearing fixture; dynamic borders are address-membership annotations.",
        "assembly_memory_timeline": "Static memory-operand count and dynamic read/write event counts from memory_evidence.json; no inferential uncertainty is claimed.",
        "model_comparison": "Per-fixture and pooled decoding accuracy with descriptive Wilson score intervals.",
        "calibration_curve": "Equal-width confidence-bin accuracy and mean confidence; ECE source is evaluation_report.json.",
        "decoder_calibration": "Expected calibration error and Brier score by decoder.",
        "confusion_matrix": "Pooled phase-by-phase confusion counts for the temporal forward filter.",
        "ambiguity_ablation": "Accuracy, segmentation F1, and active-policy advantage across likelihood blends.",
        "segmentation_ribbons": "Pooled boundary F1 and per-fixture phase-boundary alignment.",
        "corpus_stability": "Leave-one-out pooled temporal accuracy range and identity-transition/chance references.",
        "phase_recovery": "Per-phase recall/F1 and macro F1.",
        "observation_informativeness": "Realized information gain by observation category.",
        "model_evidence": "Fixed-parameter pooled log marginal likelihood by model.",
        "transition_ablation": "Temporal accuracy while transition structure is blended toward uniform.",
        "generative_adequacy": "Observed transition rate and temporal recovery accuracy against central 95% simulation intervals.",
        "typed_observation_matrix": "Typed categorical fields and explicit static/dynamic provenance for the canonical fixture.",
        "factorized_observation_comparison": "Held-out accuracy, field-family ablation, and masked-memory sensitivity under the recorded missing-field policy.",
        "corpus_coverage": "Static-versus-dynamic instruction counts and binary evidence-regime flags per transparent fixture.",
        "action_sensing": "Censored action budget, final posterior confidence, confidence rate, and equal normalized action costs by policy.",
        "parameter_learning": "Held-out accuracy by fitting strategy, training log-evidence convergence, five-start initialization sensitivity, and receipt-level posterior-predictive checks; start dots are holdouts and bars are within-start means.",
        "representation_invariance_benchmark": "Family-level final-RAX consistency and near-miss separation with descriptive Wilson intervals.",
        "intervention_power": "Exact subset-level near-miss separation and singleton intervention power; no sampling uncertainty is applicable.",
        "trace_refinement": "Per-step symbolic/Unicorn RAX agreement and modeled EFLAGS bit comparisons.",
        "cross_architecture_benchmark": "Decoded instruction count, executed instruction count, and final accumulator agreement across three real architecture adapters.",
        "formal_verification_receipt": "Declared theorem count and locally verified Lean receipt status; no population uncertainty is claimed.",
        "active_information_frontier": "Expected information gain by horizon and cost weight from exact branch enumeration; coincident curves are explicitly marked when the finite contract yields the same frontier; the plotted action cost is a unit model-channel quantity, not wall-clock time.",
        "quotient_classifier": "Balanced accuracy and macro-F1 use source-family-clustered percentile intervals; Brier score and ten-bin ECE use held-out probabilities; paired baseline comparisons use source-family bootstrap and Holm adjustment.",
        "compiler_transfer": "Validated/unavailable status is a capability receipt; decoded instruction counts are static coverage diagnostics without execution.",
        "formal_correspondence": "All-step match status and reference-step counts are deterministic fixture checks, with a generated Lean boundary fixture; no population uncertainty is claimed.",
        "sensor_costs": "Deterministic vector counts are primary; local timing medians span five repetition values with IQR, p95, overhead correction, and regression residuals; policy certificates compare exact dynamic-program, fixed, greedy, seeded-random, and abstaining objectives.",
        "replication_manifest": "Public and gold counts plus protocol/hash separation are deterministic provenance checks for the local harness. A separately controlled replay receipt, when present, is reported as a distinct bounded result and does not imply an independent physical-machine replication or independent reimplementation.",
        "external_source_provenance": "Status, manifest cardinality, and five promotion gates are deterministic provenance checks; blocked or unavailable cells are retained as status values rather than plotted as performance zeros.",
        "external_label_coverage": "Accepted-row counts by source-defined role and frozen project/revision split; explicit exclusions are retained outside the accepted denominator and no inferential uncertainty is claimed.",
        "artifact_receipt_closure": "Deterministic count of project-owned JSON receipts, envelope errors, and roadmap status rows; no inferential uncertainty is applicable.",
        "observation_semantics": "Schema and provenance categories for typed values, explicit absence, unsupported/unavailable states, and protected labels; no inferential uncertainty is applicable.",
        "counterexample_search": "Per-candidate searched-domain size, minimal witness rank, and falsified/holds status from exhaustive smallest-first enumeration with independent witness re-verification; no population uncertainty is claimed.",
        "sensing_stress": "Per-policy maximum regret and exact-enumeration optimality gaps on the horizon x budget x cost-weight grid, budget-constrained value gaps against the unconstrained exact bound, and cross-repetition calibration coefficient drift over repeated local timing runs.",
    }
    rows: list[dict[str, Any]] = []
    for stem, raw_caption in FIGURE_CAPTIONS.items():
        filename = f"{stem}.png"
        path = output_dir / "figures" / filename
        target = target_by_filename.get(filename)
        caption = raw_caption.format(**context)
        first_sentence = caption.split(".", 1)[0].strip() + "."
        generator = target.generated_by if target else "src.figures"
        claims = (
            list(target.claim_ids)
            if target
            else list(fallback_claims.get(stem, ("CLM-009",)))
        )
        checks = (
            list(target.validation_checks) if target else list(fallback_checks[stem])
        )
        input_artifacts = [
            "output/data/trace_bundle.json",
            "output/data/evaluation_report.json",
        ]
        if stem == "assembly_cfg":
            input_artifacts = ["output/data/cfg_evidence.json"]
        if stem == "assembly_memory_timeline":
            input_artifacts = ["output/data/memory_evidence.json"]
        if stem == "rxinfer_categorical_free_energy":
            input_artifacts = [
                "output/rxinfer_categorical/categorical_free_energy_trace.jsonl",
                "output/rxinfer_categorical/categorical_diagnostics.json",
                "output/rxinfer_categorical/categorical_model_config.json",
            ]
        if "active" in stem or stem == "ambiguity_ablation":
            input_artifacts.append("output/data/active_selection.json")
        if "temporal" in stem or stem in {"segmentation_ribbons", "phase_recovery"}:
            input_artifacts.append("output/data/temporal_inference.json")
        if stem in {"scholarship_source_roles", "scholarship_evidence_matrix"}:
            input_artifacts = ["output/data/scholarship_sources.json"]
        elif stem == "claim_evidence_map":
            input_artifacts = ["output/data/claim_register.json"]
        elif stem == "generative_adequacy":
            input_artifacts = [
                "output/data/generative_adequacy.json",
                "output/data/evaluation_report.json",
            ]
        elif stem == "typed_observation_matrix":
            input_artifacts = ["output/data/typed_observations.json"]
        elif stem == "factorized_observation_comparison":
            input_artifacts = ["output/data/factorized_observation_model.json"]
        elif stem == "corpus_coverage":
            input_artifacts = ["output/data/corpus_coverage.json"]
        elif stem == "action_sensing":
            input_artifacts = [
                "output/data/action_sensing.json",
                "output/data/factorized_observation_model.json",
            ]
        elif stem == "parameter_learning":
            input_artifacts = ["output/data/parameter_learning.json"]
        elif stem == "representation_invariance_benchmark":
            input_artifacts = ["output/data/representation_invariance_benchmark.json"]
        elif stem == "intervention_power":
            input_artifacts = [
                "output/data/intervention_power.json",
                "output/data/representation_invariance_benchmark.json",
            ]
        elif stem == "cross_architecture_benchmark":
            input_artifacts = ["output/data/cross_architecture_fixtures.json"]
        elif stem == "formal_verification_receipt":
            input_artifacts = ["output/data/formal_verification.json"]
        elif stem == "active_information_frontier":
            input_artifacts = ["output/data/active_information_frontier.json"]
        elif stem == "quotient_classifier":
            input_artifacts = [
                "output/data/quotient_classifier.json",
                "output/data/research_protocol.json",
            ]
        elif stem == "compiler_transfer":
            input_artifacts = [
                "output/data/compiler_transfer.json",
                "output/data/research_protocol.json",
            ]
        elif stem == "formal_correspondence":
            input_artifacts = [
                "output/data/formal_trace_correspondence.json",
                "output/data/research_protocol.json",
            ]
        elif stem == "sensor_costs":
            input_artifacts = [
                "output/data/sensor_cost_receipt.json",
                "output/data/research_protocol.json",
            ]
        elif stem == "observation_semantics":
            input_artifacts = [
                "output/data/typed_observations.json",
                "output/data/replication_manifest.json",
                "output/data/research_protocol.json",
            ]
        elif stem == "replication_manifest":
            input_artifacts = [
                "output/data/replication_manifest.json",
                "output/replication/public/public_manifest.json",
            ]
        elif stem == "external_source_provenance":
            input_artifacts = [
                "output/data/external_source_validation.json",
                "configs/external_sources.yaml",
            ]
        elif stem == "external_label_coverage":
            input_artifacts = [
                "output/data/external_label_contract.json",
                "configs/external_label_contracts.yaml",
            ]
        elif stem == "artifact_receipt_closure":
            input_artifacts = [
                "output/data/research_protocol.json",
                "output/data/scientific_program.json",
            ]
        elif stem == "trace_refinement":
            input_artifacts = ["output/data/symbolic_execution.json"]
        elif stem == "artifact_hash_manifest":
            input_artifacts = ["output/data/reproducibility_manifest.json"]
        elif stem == "validation_summary":
            input_artifacts = ["output/data/validation_report.json"]
        elif stem == "counterexample_search":
            input_artifacts = [
                "output/data/counterexample_search.json",
                "output/data/formal_trace_correspondence.json",
            ]
        elif stem == "sensing_stress":
            input_artifacts = [
                "output/data/sensing_stress.json",
                "output/data/active_information_frontier.json",
            ]
        visual_qa: dict[str, Any]
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            variance = float(np.var(plt.imread(path)))
            visual_qa = {
                "status": "programmatic_nonblank" if variance > 0.00001 else "failed",
                "bytes": path.stat().st_size,
                "sha256": digest,
                "pixel_variance": variance,
            }
        else:
            visual_qa = {
                "status": "missing",
                "bytes": 0,
                "sha256": None,
                "pixel_variance": 0.0,
            }
        population_text = population.get(stem, "labelled toy corpus fixtures")
        statistics_text = statistics.get(
            stem, "Descriptive plot generated from the canonical analysis artifacts."
        )
        citation_keys = list(FIGURE_CITATIONS.get(stem, ()))
        rows.append(
            {
                "figure_id": stem,
                "filename": filename,
                "label": f"fig:{stem.replace('_', '-')}",
                "generator": generator,
                "input_artifacts": sorted(set(input_artifacts)),
                "analysis_population": population_text,
                "statistics": statistics_text,
                "citation_keys": citation_keys,
                "caption": caption,
                "alt_text": first_sentence,
                "long_description": (
                    f"{caption} Population: {population_text}. Statistics: {statistics_text}. "
                    f"Inputs: {', '.join(sorted(set(input_artifacts)))}. "
                    f"Scholarship context: {', '.join(citation_keys) if citation_keys else 'none required for this audit figure'}."
                ),
                "claim_ids": claims,
                "validation_checks": checks,
                "visual_qa": visual_qa,
            }
        )
    return rows


def write_figure_captions(output_dir: Path) -> Path:
    """Write the caption ledger and complete figure metadata registry."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    context = _caption_context(output_dir)
    registry_rows = _figure_registry_rows(output_dir, context)
    rows = [
        "# Figure Captions and Accessibility Registry",
        "",
        "The machine-readable companion is `output/data/figure_registry.json`.",
        "",
        "| Figure | Caption | Label | Generator | Inputs | Population | Statistics | Scholarship | Alt text | Long description | Claims | Checks | Visual QA |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in registry_rows:
        stem = entry["figure_id"]
        path = figure_dir / f"{stem}.png"
        status = "present" if path.exists() else "missing"
        rows.append(
            f"| `{stem}.png` ({status}) | {entry['caption']} | `{entry['label']}` | "
            f"`{entry['generator']}` | {', '.join(entry['input_artifacts'])} | "
            f"{entry['analysis_population']} | {entry['statistics']} | "
            f"{', '.join(entry['citation_keys']) or 'none'} | {entry['alt_text']} | "
            f"{entry['long_description']} | {', '.join(entry['claim_ids'])} | "
            f"{', '.join(entry['validation_checks'])} | `{entry['visual_qa']['status']}` |"
        )
    caption_path = figure_dir / "README.md"
    caption_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    registry_payload = {
        "schema_version": "1.1",
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").UTC)
        .isoformat(),
        "figure_count": len(registry_rows),
        "figures": registry_rows,
    }
    registry_path = output_dir / "data" / "figure_registry.json"
    write_json_artifact(
        registry_path,
        registry_payload,
        project_root=output_dir.parent,
        producer_path=Path(__file__),
    )

    # The sibling template consumes the format-neutral registry from the
    # figure directory. Keep the rich project registry canonical in
    # output/data, and emit a deliberately small boundary projection using the
    # template's documented ``alt`` field. This prevents renderer-contract
    # drift from silently falling back to Pandoc's visible caption as PDF alt.
    template_registry = {
        "schema_version": "figure-registry-v1",
        "figures": [
            {
                "label": entry["label"],
                "filename": entry["filename"],
                "alt": entry["alt_text"],
            }
            for entry in registry_rows
        ],
    }
    write_json_artifact(
        figure_dir / "figure_registry.json",
        template_registry,
        project_root=output_dir.parent,
        producer_path=Path(__file__),
    )
    return caption_path
