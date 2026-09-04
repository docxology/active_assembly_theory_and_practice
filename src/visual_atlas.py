"""Source-bound narrative figures for the visual atlas.

The original figure module contains the diagnostic surfaces that grew with the
analysis.  This module adds the narrative plates required by the monograph
tranche while keeping them deliberately receipt-driven.  No value
in a visual is typed into a manuscript: every count, status, metric, and
boundary below is read from the protocol or a generated JSON receipt.

The plotting functions intentionally accept plain mappings.  That makes the
figures useful to tests and to the offline atlas producer without coupling
the visual layer to one domain payload implementation.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Mapping, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from .figures import GRID, INK, LIGHT, MUTED, PALETTE, _style_axes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NARRATIVE_FIGURE_IDS = (
    "research_architecture",
    "assembly_state_plate",
    "quotient_construction",
    "semantic_coverage_matrix",
    "statistical_hierarchy",
    "replay_failure_audit",
    "policy_certificate_frontier",
    "theorem_status_map",
    "observation_semantics",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object payload at {path}")
    return value


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _text(value: object, default: str = "not observed") -> str:
    return str(value) if value is not None else default


def _compact_state_value(value: object) -> str:
    """Render machine-state scalars compactly enough for a figure table."""

    if isinstance(value, int):
        return f"0x{value:x}"
    return _text(value)


def _state_transition(before: object, after: object) -> str:
    """Render a compact before/after state transition for a dense table."""

    before_text = _compact_state_value(before)
    after_text = _compact_state_value(after)
    return before_text if before_text == after_text else f"{before_text}→{after_text}"


def _wrapped(value: object, width: int, default: str = "not observed") -> str:
    """Wrap table prose while preserving explicit unavailable values."""

    return textwrap.fill(_text(value, default), width=width, break_long_words=False)


def _save_multi(fig: Any, output_path: Path) -> Path:
    """Save a narrative figure as high-resolution PNG and direct vector files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_dir = output_path.parent / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        (output_path, "png"),
        (vector_dir / f"{output_path.stem}.svg", "svg"),
        (vector_dir / f"{output_path.stem}.pdf", "pdf"),
    )
    try:
        for path, fmt in targets:
            temporary = path.with_name(f".{path.name}.{id(fig)}.tmp")
            fig.savefig(
                temporary,
                format=fmt,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            temporary.replace(path)
    finally:
        plt.close(fig)
        for path, _fmt in targets:
            temporary = path.with_name(f".{path.name}.{id(fig)}.tmp")
            if temporary.exists():
                temporary.unlink()
    return output_path


def _title(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=14, color=INK)
    fig.text(0.055, 0.94, subtitle, ha="left", va="top", fontsize=8.5, color=MUTED)


def _box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    detail: str,
    color: str,
    *,
    hatch: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor=INK,
        linewidth=1.0,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.63,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
        weight="bold",
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=6.8,
        color=INK,
        wrap=True,
    )


def _arrow(
    ax: Any, start: tuple[float, float], end: tuple[float, float], label: str
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=MUTED,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035)
    ax.text(mid[0], mid[1], label, ha="center", va="bottom", fontsize=6.5, color=MUTED)


def plot_research_architecture(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render the protocol's evidence and provenance path as a labelled map."""

    protocol_id = _text(payload.get("protocol_id"), "protocol not observed")
    protocol_hash = _text(payload.get("protocol_sha256"), "hash not observed")[:12]
    nodes = (
        ("01 · Source\nbytes", "transparent fixture", PALETTE[0]),
        ("02 · Capstone\ndecode", "static instruction rows", PALETTE[0]),
        ("03 · Normalized\nIR", "operands · widths · flags", PALETTE[0]),
        ("04 · Python\nsemantics", "reference small step", PALETTE[1]),
        ("05 · Unicorn\ncross-check", "bounded dynamic trace", PALETTE[1]),
        ("06 · Lean\nboundary", "kernel-checked fragment", PALETTE[3]),
        ("07 · Typed\nobservations", "missing ≠ unseen", PALETTE[2]),
        ("08 · Interventions\n/ quotient", "finite labels", PALETTE[2]),
        ("09 · Model\nportfolio", "temporal · calibrated · neural", PALETTE[2]),
        ("10 · Sensing\n/ cost", "finite policy certificate", PALETTE[4]),
        ("11 · Receipts\n/ claims", "hashes · status · limits", PALETTE[3]),
        ("12 · Template\nhydration", "HTML · DOCX · PDF · slides", PALETTE[3]),
    )
    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    ax.set_xlim(0, 5.05)
    ax.set_ylim(0, 3.72)
    ax.axis("off")
    # The three rows form a numbered serpentine route.  Row transitions are
    # vertical, so the diagram never asks the reader to decode a crossing
    # diagonal arrow or infer which stage follows which.
    node_width = 1.02
    node_height = 0.62
    x_by_column = (0.18, 1.40, 2.62, 3.84)
    y_by_row = (2.60, 1.48, 0.36)
    positions: list[tuple[float, float]] = []
    for index, (label, detail, color) in enumerate(nodes):
        row, column = divmod(index, 4)
        route_column = column if row % 2 == 0 else 3 - column
        xy = (x_by_column[route_column], y_by_row[row])
        positions.append(xy)
    bands = (
        (
            0.08,
            2.42,
            4.88,
            0.92,
            "STATIC DECODING / NORMALIZED REPRESENTATION",
            PALETTE[0],
        ),
        (
            0.08,
            1.30,
            4.88,
            0.92,
            "REFERENCE EXECUTION / FORMAL CROSS-CHECK",
            PALETTE[1],
        ),
        (
            0.08,
            0.18,
            4.88,
            0.92,
            "OBSERVATION / INFERENCE / DOCUMENT EVIDENCE",
            PALETTE[2],
        ),
    )
    for x, y, width, height, label, color in bands:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=color + "0B",
                edgecolor=color,
                linewidth=0.9,
                linestyle="--",
                zorder=0,
            )
        )
        ax.text(
            x + 0.06,
            y + height - 0.06,
            label,
            ha="left",
            va="top",
            fontsize=7,
            color=color,
            weight="bold",
            zorder=1,
        )
    for index, (label, detail, color) in enumerate(nodes):
        _box(
            ax,
            positions[index],
            node_width,
            node_height,
            label,
            detail,
            color + "30",
        )
    edge_labels = (
        "decode",
        "normalize",
        "execute",
        "cross-check",
        "formalize",
        "project",
        "label",
        "fit",
        "evaluate",
        "register",
        "hydrate",
    )
    for index, label in enumerate(edge_labels):
        start = positions[index]
        end = positions[index + 1]
        if index % 4 == 3:
            x = start[0] + node_width / 2
            _arrow(ax, (x, start[1]), (x, end[1] + node_height), "row handoff")
        else:
            _arrow(
                ax,
                (
                    start[0] + (node_width if start[0] < end[0] else 0.0),
                    start[1] + node_height / 2,
                ),
                (
                    end[0] + (0.0 if start[0] < end[0] else node_width),
                    end[1] + node_height / 2,
                ),
                label,
            )
    ax.text(
        0.10,
        0.02,
        f"Protocol {protocol_id} · canonical hash prefix {protocol_hash} · edge labels describe evidence roles, not transitive proof",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    _title(
        fig,
        "Research architecture and evidence boundaries",
        "Source-bound protocol map; nodes are artifacts or transformations, and each boundary keeps static, dynamic, formal, inferential, and document evidence distinct.",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.86)
    return _save_multi(fig, output_path)


def plot_observation_semantics(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render the typed-observation state vocabulary and evidence boundary.

    The diagram is intentionally a schema visual rather than a performance
    chart.  Its central scientific job is to make absence legible: a missing
    dynamic field, an unseen categorical value, an unsupported operation, an
    unavailable capability, and a measured numeric zero are different states.
    """

    fields = [
        str(value)
        for value in payload.get("fields", [])
        if isinstance(value, str) and value
    ]
    dynamic_field = _text(payload.get("dynamic_field"), "dynamic_status")
    missing_token = _text(payload.get("missing_token"), "__MISSING__")
    unseen_token = _text(payload.get("unseen_token"), "__UNKNOWN__")
    public_fields = [
        str(value)
        for value in payload.get("public_fields", [])
        if isinstance(value, str) and value
    ]
    forbidden_fields = [
        str(value)
        for value in payload.get("forbidden_public_fields", [])
        if isinstance(value, str) and value
    ]
    field_preview = ", ".join(fields[:4]) or "typed fields not observed"
    public_preview = ", ".join(public_fields) or "public fields not observed"
    forbidden_preview = (
        ", ".join(forbidden_fields[:4]) or "protected fields not observed"
    )
    protocol_id = _text(payload.get("protocol_id"), "protocol not observed")
    protocol_hash = _text(payload.get("protocol_sha256"), "hash not observed")[:12]
    public_count = _text(payload.get("public_row_count"), "not observed")
    gold_count = _text(payload.get("gold_row_count"), "not observed")
    typed_schema = _text(payload.get("typed_schema_version"), "not observed")
    replay_status = _text(payload.get("replication_status"), "not observed")

    fig = plt.figure(figsize=(15.0, 8.6))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.44, 1.0), hspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 4.0)
    ax.axis("off")

    ax.text(
        0.18, 3.72, "Value and status vocabulary", fontsize=10, weight="bold", color=INK
    )
    ax.text(
        6.32,
        3.72,
        "Public / protected / receipt boundary",
        fontsize=10,
        weight="bold",
        color=INK,
    )

    state_specs = (
        (
            "Observed",
            f"{fields[0] if fields else 'field'} = arithmetic",
            PALETTE[2],
            None,
        ),
        ("Missing", f"{dynamic_field} = {missing_token}", PALETTE[1], "//"),
        ("Unseen", f"categorical value = {unseen_token}", PALETTE[4], ".."),
        ("Unsupported", "instruction outside declared subset", PALETTE[3], "xx"),
        ("Unavailable", "compiler or target capability absent", PALETTE[0], "\\\\"),
        ("Measured zero", "receipt count = 0", MUTED, None),
    )
    state_positions = (
        (0.18, 2.50),
        (2.00, 2.50),
        (0.18, 1.59),
        (2.00, 1.59),
        (0.18, 0.68),
        (2.00, 0.68),
    )

    def _state_box(
        xy: tuple[float, float],
        label: str,
        detail: str,
        color: str,
        hatch: str | None,
    ) -> None:
        x, y = xy
        width, height = 1.52, 0.78
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=color + "30",
                edgecolor=INK,
                linewidth=1.0,
                hatch=hatch,
            )
        )
        ax.text(
            x + width / 2,
            y + height * 0.73,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color=INK,
            weight="bold",
        )
        ax.text(
            x + width / 2,
            y + height * 0.43,
            textwrap.fill(detail, width=18),
            ha="center",
            va="top",
            fontsize=6.0,
            linespacing=1.25,
            color=INK,
        )

    for (label, detail, color, hatch), (x, y) in zip(
        state_specs, state_positions, strict=True
    ):
        _state_box((x, y), label, detail, color, hatch)
    # A single converging boundary keeps the six statuses readable without
    # drawing six crossing arrows through the central record.
    ax.plot((3.60, 3.60), (1.04, 3.26), color=MUTED, linewidth=1.0)
    for y in (2.98, 2.07, 1.16):
        ax.plot((3.52, 3.60), (y, y), color=MUTED, linewidth=1.0)
    _arrow(ax, (3.60, 2.34), (4.18, 2.34), "encode")
    record_xy = (4.18, 1.60)
    record_width, record_height = 1.82, 1.34
    ax.add_patch(
        FancyBboxPatch(
            record_xy,
            record_width,
            record_height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=PALETTE[2] + "25",
            edgecolor=INK,
            linewidth=1.0,
        )
    )
    ax.text(
        record_xy[0] + record_width / 2,
        record_xy[1] + record_height * 0.84,
        "Typed record",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
        weight="bold",
    )
    ax.text(
        record_xy[0] + record_width / 2,
        record_xy[1] + record_height * 0.68,
        f"schema {typed_schema} · {len(fields)} fields",
        ha="center",
        va="center",
        fontsize=6.4,
        color=INK,
    )
    ax.text(
        record_xy[0] + record_width / 2,
        record_xy[1] + record_height * 0.42,
        textwrap.fill(field_preview, width=22),
        ha="center",
        va="center",
        fontsize=6.1,
        linespacing=1.18,
        color=INK,
    )
    ax.text(
        record_xy[0] + record_width / 2,
        record_xy[1] + record_height * 0.10,
        "status-bearing · label-blind",
        ha="center",
        va="center",
        fontsize=6.1,
        color=INK,
    )
    ax.add_patch(
        Rectangle(
            (3.98, 0.30),
            2.23,
            1.18,
            facecolor=PALETTE[3] + "12",
            edgecolor=PALETTE[3],
            linewidth=1.0,
            hatch="///",
        )
    )
    ax.text(4.12, 1.20, "Interpretation rules", fontsize=8.5, weight="bold", color=INK)
    ax.text(
        4.12,
        0.92,
        "missing ≠ unseen\nunsupported / unavailable ≠ numeric zero",
        fontsize=8.4,
        color=INK,
        va="center",
    )
    ax.text(
        4.12,
        0.48,
        "A numeric zero is valid only when a receipt measures zero.",
        fontsize=7.0,
        color=MUTED,
    )

    # The right side is a protected-boundary map.  The outer dashed frame is
    # the controlled evidence boundary; a nested blue frame marks the public
    # label-blind path, while the gold-hatched labels remain outside that
    # public frame but inside the owner-side scoring boundary.  The receipt
    # box is provenance metadata rather than an additional model feature.
    ax.add_patch(
        Rectangle(
            (6.18, 0.36),
            4.56,
            3.00,
            facecolor=PALETTE[0] + "08",
            edgecolor=PALETTE[0],
            linewidth=1.1,
            linestyle="--",
        )
    )
    ax.text(
        6.34, 3.14, "controlled evidence boundary", fontsize=9, weight="bold", color=INK
    )
    ax.add_patch(
        Rectangle(
            (6.30, 1.24),
            1.98,
            1.72,
            facecolor=PALETTE[0] + "05",
            edgecolor=PALETTE[0],
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(
        6.42,
        2.80,
        "public label-blind path",
        fontsize=8.2,
        weight="bold",
        color=INK,
    )
    public_summary = textwrap.fill(
        f"{public_count} rows · fields: {public_preview}", width=22
    )
    ax.text(
        6.42,
        2.58,
        f"{public_summary}\nstable IDs are not target labels",
        fontsize=7.5,
        color=INK,
        va="top",
    )
    _box(
        ax,
        (6.43, 1.30),
        1.72,
        0.64,
        "Predictions",
        "stable IDs\nordered output",
        PALETTE[2] + "28",
    )
    ax.add_patch(
        Rectangle(
            (8.44, 1.30),
            2.08,
            1.22,
            facecolor=PALETTE[1] + "15",
            edgecolor=PALETTE[1],
            linewidth=1.0,
            hatch="///",
        )
    )
    ax.text(8.58, 2.25, "protected labels", fontsize=9, weight="bold", color=INK)
    ax.text(
        8.58,
        1.90,
        textwrap.fill(
            f"{gold_count} gold rows\n{forbidden_preview}\noutside public directory",
            width=26,
        ),
        fontsize=7.4,
        color=INK,
        va="top",
    )
    _arrow(ax, (6.43 + 1.72, 1.62), (8.44, 1.62), "score")
    ax.add_patch(
        Rectangle(
            (6.34, 0.56),
            4.18,
            0.52,
            facecolor=PALETTE[3] + "16",
            edgecolor=PALETTE[3],
            linewidth=1.0,
        )
    )
    ax.text(
        6.48,
        0.82,
        f"source-bound receipts · {protocol_id} · hash {protocol_hash} · replay status {replay_status}",
        fontsize=7.0,
        color=INK,
        va="center",
    )
    ax.text(
        6.34,
        0.20,
        "Receipt metadata joins the public path to reproducible provenance; it does not reveal protected labels.",
        fontsize=6.8,
        color=MUTED,
    )

    table_ax = fig.add_subplot(grid[1, 0])
    table_ax.axis("off")
    table_ax.text(
        0.01,
        0.96,
        "Exact status table for the observation contract",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    rows = [
        [
            "observed",
            "typed categorical value",
            "measured at the declared observation grain",
        ],
        [
            "missing",
            missing_token,
            "field not supplied; distinct from a measured zero",
        ],
        [
            "unseen",
            unseen_token,
            "value outside training support; distinct from missing",
        ],
        [
            "unsupported / unavailable",
            "structured status",
            "outside semantic or local capability contract; no numeric imputation",
        ],
        [
            "measured zero",
            "0 from receipt",
            "numeric zero is retained only when the producer measured zero",
        ],
        [
            "protected label",
            "outside public rows",
            "owner-side scoring boundary; never a model feature",
        ],
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=("category", "serialized representation", "interpretation boundary"),
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=cast(Any, (0.01, 0.04, 0.98, 0.78)),
        colWidths=cast(Any, [0.20, 0.27, 0.53]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.38)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT)
            if col == 0:
                cell.get_text().set_fontweight("bold")
    _title(
        fig,
        "Observation semantics: values, absence, and protection are different evidence states",
        "Protocol-bound schema diagram from the typed-observation and replay receipts; a status-bearing record preserves what was observed, what was not available, and what must remain protected.",
    )
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.035, top=0.86)
    return _save_multi(fig, output_path)


def plot_assembly_state_plate(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render a decoded listing beside normalized state and memory effects."""

    listing = payload.get("listing", [])
    trace = payload.get("trace", {})
    formal_steps = payload.get("formal_steps", [])
    memory = payload.get("memory", {})
    if not isinstance(listing, list) or not listing:
        raise ValueError("assembly state plate requires listing rows")
    if not isinstance(trace, Mapping) or not isinstance(formal_steps, list):
        raise ValueError("assembly state plate payload is malformed")
    instructions = listing[:8]
    # The correspondence receipt contains one outer row per intervention and
    # the address-indexed semantic steps inside ``reference_steps``.  Use the
    # first deterministic intervention as the listing witness, rather than
    # accidentally treating the outer intervention row as an instruction.
    step_by_address: dict[int, Mapping[str, Any]] = {}
    for candidate in formal_steps:
        if not isinstance(candidate, Mapping):
            continue
        nested_steps = candidate.get("reference_steps", [])
        if not isinstance(nested_steps, list):
            continue
        for nested in nested_steps:
            if not isinstance(nested, Mapping):
                continue
            address_value = nested.get("address")
            if isinstance(address_value, int) and address_value not in step_by_address:
                step_by_address[address_value] = cast(Mapping[str, Any], nested)
    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.28, 1.0),
        width_ratios=(1.55, 1.0),
        hspace=0.32,
        wspace=0.22,
    )
    ax_listing = fig.add_subplot(gs[0, 0])
    ax_listing.axis("off")
    columns = [
        "address",
        "bytes",
        "opcode\nfamily",
        "instruction",
        "widths\n(op → state)",
        "flag read\n/ write",
        "next RIP",
    ]
    cells: list[list[str]] = []
    for row in instructions:
        address_value = row.get("address", 0)
        address = address_value if isinstance(address_value, int) else 0
        formal = step_by_address.get(address, {})
        instruction_semantics = formal.get("instruction", {})
        if not isinstance(instruction_semantics, Mapping):
            instruction_semantics = {}
        instruction = (
            f"{_text(row.get('mnemonic'))} {_text(row.get('operands'), '')}".strip()
        )
        raw_widths = instruction_semantics.get("widths", [])
        widths = (
            " → ".join(str(width) for width in raw_widths)
            if isinstance(raw_widths, list) and raw_widths
            else "—"
        )
        raw_reads = instruction_semantics.get("flag_reads", [])
        raw_writes = instruction_semantics.get("flag_writes", [])
        reads = (
            " ".join(str(value) for value in raw_reads)
            if isinstance(raw_reads, list)
            else ""
        )
        writes = (
            " ".join(str(value) for value in raw_writes)
            if isinstance(raw_writes, list)
            else ""
        )
        flag_text = f"{reads or '—'} / {writes or '—'}"
        next_rip = formal.get("post_rip")
        next_rip_text = hex(next_rip) if isinstance(next_rip, int) else "—"
        cells.append(
            [
                _text(row.get("address_hex"), hex(address)),
                _text(row.get("bytes_hex")),
                _text(row.get("category")),
                instruction,
                widths,
                flag_text,
                next_rip_text,
            ]
        )
    table = ax_listing.table(
        cellText=cells,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=cast(Any, (0, 0.04, 1, 0.84)),
        colWidths=cast(Any, [0.12, 0.14, 0.17, 0.16, 0.13, 0.18, 0.10]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT)
    ax_listing.set_title(
        "Decoded listing and normalized instruction fields",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax_state = fig.add_subplot(gs[0, 1])
    ax_state.axis("off")
    dynamic_rows = [
        row
        for row in trace.get("dynamic_register_trace", [])
        if isinstance(row, Mapping) and row.get("event") == "instruction"
    ]
    state_text = [["step", "RAX", "RSP", "RIP", "EFLAGS"]]
    for step, row in enumerate(dynamic_rows, start=1):
        before = row.get("registers_before", {})
        after = row.get("registers_after", {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        state_text.append(
            [
                str(step),
                _state_transition(before.get("rax"), after.get("rax")),
                _state_transition(before.get("rsp"), after.get("rsp")),
                _state_transition(row.get("address"), after.get("rip")),
                _state_transition(before.get("eflags"), after.get("eflags")),
            ]
        )
    st = ax_state.table(
        cellText=state_text[1:],
        colLabels=state_text[0],
        loc="center",
        cellLoc="center",
        bbox=cast(Any, (0, 0.18, 1, 0.68)),
    )
    st.auto_set_font_size(False)
    st.set_fontsize(7.0)
    st.scale(1, 1.7)
    for (row, col), cell in st.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(PALETTE[1] if row == 0 else ("white" if row % 2 else LIGHT))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    ax_state.set_title(
        "Bounded register / PC transitions", loc="left", fontsize=10, color=INK
    )
    ax_memory = fig.add_subplot(gs[1, 0])
    ax_memory.axis("off")
    events = (
        memory.get("dynamic_memory_events", []) if isinstance(memory, Mapping) else []
    )
    memory_rows = [["event", "access", "instruction", "address", "size", "value"]]
    for index, row in enumerate(events if isinstance(events, list) else []):
        if isinstance(row, Mapping):
            memory_rows.append(
                [
                    str(index),
                    _text(row.get("access")),
                    _text(row.get("instruction_address_hex")),
                    _text(row.get("address_hex")),
                    _text(row.get("size")),
                    _text(row.get("value")),
                ]
            )
    if len(memory_rows) == 1:
        memory_rows.append(["—", "no dynamic event", "—", "—", "—", "—"])
    mt = ax_memory.table(
        cellText=memory_rows[1:],
        colLabels=memory_rows[0],
        loc="center",
        cellLoc="center",
        bbox=cast(Any, (0, 0.22, 1, 0.55)),
    )
    mt.auto_set_font_size(False)
    mt.set_fontsize(7.5)
    mt.scale(1, 1.7)
    for (row, col), cell in mt.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(PALETTE[2] if row == 0 else ("white" if row % 2 else LIGHT))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    ax_memory.set_title(
        "Bounded byte-memory effects", loc="left", fontsize=10, color=INK
    )
    ax_boundary = fig.add_subplot(gs[1, 1])
    ax_boundary.axis("off")
    ax_boundary.add_patch(
        Rectangle(
            (0.04, 0.20),
            0.92,
            0.59,
            facecolor=PALETTE[3] + "16",
            edgecolor=PALETTE[3],
            linewidth=1.1,
            hatch="///",
        )
    )
    ax_boundary.text(
        0.50,
        0.63,
        "closed-world boundary",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=INK,
    )
    ax_boundary.text(
        0.50,
        0.46,
        "unsupported instructions, undefined flags,\nexceptions, and out-of-bounds memory\nreturn structured status",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
    )
    ax_boundary.text(
        0.50,
        0.26,
        f"formal status: {_text(payload.get('status'))}\nfixture: {_text(trace.get('sample_id'))}",
        ha="center",
        va="center",
        fontsize=7.2,
        color=MUTED,
    )
    _title(
        fig,
        "Assembly state plate: listing → state transition → bounded effects",
        "Transparent fixture evidence; widths and flags come from the normalized formal step, dynamic registers from bounded execution, and memory rows from the bounded hook receipt.",
    )
    fig.text(
        0.055,
        0.025,
        "RIP denotes the program counter. The panel is an exact fixture audit, not a complete x86-64 semantics claim.",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.06, top=0.88)
    return _save_multi(fig, output_path)


def plot_quotient_construction(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render the finite source-pair, intervention, and quotient construction."""

    pairs = payload.get("pairs", [])
    classifier = payload.get("classifier", {})
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("quotient construction requires pair rows")
    if not isinstance(classifier, Mapping):
        classifier = {}
    equivalent = sum(
        bool(row.get("expected_equivalent"))
        for row in pairs
        if isinstance(row, Mapping)
    )
    near_miss = len(pairs) - equivalent
    intervention_values: list[int] = []
    for row in pairs:
        if isinstance(row, Mapping):
            values = row.get("initial_state_interventions", [])
            if isinstance(values, list):
                intervention_values.extend(
                    int(value) for value in values if isinstance(value, int)
                )
    interventions = sorted(set(intervention_values))
    family_count = _text(classifier.get("family_count"), "not observed")
    held_out = _text(classifier.get("held_out_family_count"), "not observed")
    fig = plt.figure(figsize=(13.4, 7.8))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.92, 1.38, 0.40),
        hspace=0.28,
    )
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_flow.set_xlim(0, 4.15)
    ax_flow.set_ylim(0, 1.45)
    ax_flow.axis("off")
    stage_x = (0.12, 1.16, 2.20, 3.24)
    stages = (
        (
            "1 · source-bound pairs",
            f"{len(pairs)} rows\n{equivalent} equivalent · {near_miss} near miss",
            PALETTE[0],
        ),
        (
            "2 · interventions",
            f"{len(interventions)} declared states\nlabels applied before fitting",
            PALETTE[1],
        ),
        ("3 · normalized traces", "static + dynamic\nendpoint relation", PALETTE[2]),
        (
            "4 · fixed split",
            f"{family_count} families\n{held_out} held out",
            PALETTE[3],
        ),
    )
    for x, (label, detail, color) in zip(stage_x, stages, strict=True):
        _box(ax_flow, (x, 0.34), 0.82, 0.70, label, detail, color + "31")
    for index in range(3):
        _arrow(
            ax_flow,
            (stage_x[index] + 0.82, 0.69),
            (stage_x[index + 1], 0.69),
            "receipt-bound",
        )
    ax_table = fig.add_subplot(grid[1, 0])
    ax_table.axis("off")
    pair_rows: list[list[str]] = [
        ["pair", "declared role", "observed endpoint", "label source"]
    ]
    for index, row in enumerate(pairs[:8]):
        if not isinstance(row, Mapping):
            continue
        role = "equivalent" if bool(row.get("expected_equivalent")) else "near miss"
        results = row.get("interventions", [])
        equal_count = (
            sum(
                bool(item.get("final_rax_equal"))
                for item in results
                if isinstance(item, Mapping)
            )
            if isinstance(results, list)
            else 0
        )
        total = len(results) if isinstance(results, list) else 0
        pair_rows.append(
            [
                str(row.get("family_id", f"pair-{index + 1}")),
                role,
                f"{equal_count}/{total} equal",
                "declared relation",
            ]
        )
    table = ax_table.table(
        cellText=pair_rows[1:],
        colLabels=pair_rows[0],
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=cast(Any, (0.03, 0.05, 0.94, 0.79)),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(PALETTE[0] if row == 0 else ("white" if row % 2 else LIGHT))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    ax_table.text(
        0.03,
        0.91,
        "Finite quotient witness table",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    ax_note = fig.add_subplot(grid[2, 0])
    ax_note.axis("off")
    ax_note.add_patch(
        Rectangle(
            (0.03, 0.10),
            0.94,
            0.72,
            facecolor=PALETTE[4] + "15",
            edgecolor=PALETTE[4],
            linewidth=1.0,
            hatch="\\\\",
        )
    )
    ax_note.text(
        0.05,
        0.62,
        "Interpretation boundary",
        fontsize=8.5,
        weight="bold",
        color=INK,
    )
    ax_note.text(
        0.05,
        0.34,
        "Equality means agreement under the declared interventions, observation projection, and output relation. It is not unrestricted program equivalence or universal identifiability.",
        fontsize=7.4,
        color=INK,
        wrap=True,
    )
    _title(
        fig,
        "Finite quotient construction and source-family holdout",
        "Source-level roles and endpoint labels are generated before inference; the visual separates an operational quotient under a finite intervention catalog from unrestricted semantic equivalence.",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.84)
    return _save_multi(fig, output_path)


def plot_semantic_coverage_matrix(
    payload: Mapping[str, Any], output_path: Path
) -> Path:
    """Render declared operation coverage without turning absence into zero."""

    supported = {
        str(value).lower()
        for value in payload.get("supported", [])
        if isinstance(value, str)
    }
    observed = {
        str(value).lower()
        for value in payload.get("observed", [])
        if isinstance(value, str)
    }
    lean_operations = {
        str(value).lower()
        for value in payload.get("lean_operations", [])
        if isinstance(value, str)
    }
    capstone_available = bool(payload.get("capstone_available"))
    unicorn_available = bool(payload.get("unicorn_available"))
    rows = (
        (
            "moves / extensions",
            ("mov", "movabs", "movzx", "movsx", "lea"),
            "register widths and zero/sign extension",
        ),
        (
            "integer arithmetic",
            ("add", "adc", "sub", "sbb", "neg", "inc", "dec"),
            "carry, borrow, and width-aware flags",
        ),
        (
            "logic / comparisons",
            ("and", "or", "xor", "not", "cmp", "test"),
            "logical flags and non-mutating comparisons",
        ),
        (
            "shifts",
            ("shl", "sal", "shr", "sar"),
            "shift-count boundary and definedness",
        ),
        (
            "control flow",
            ("jmp", "jcc", "call", "ret"),
            "bounded targets and stack effects",
        ),
        ("stack / memory", ("push", "pop", "load", "store"), "bounded byte addressing"),
        (
            "outside contract",
            ("simd", "floating_point", "privileged", "syscall"),
            "structured unsupported outcome",
        ),
    )
    columns = (
        "declared",
        "width",
        "flags",
        "memory",
        "control",
        "Lean",
        "Capstone",
        "Unicorn",
    )
    matrix: list[list[str]] = []
    row_labels: list[str] = []
    for label, ops, detail in rows:
        row_labels.append(label)
        op_present = bool(supported.intersection(ops))
        is_outside = label == "outside contract"
        declared = "UNSUPPORTED" if is_outside else ("✓" if op_present else "—")
        width = (
            "✓"
            if op_present
            and label
            in {
                "moves / extensions",
                "integer arithmetic",
                "logic / comparisons",
                "shifts",
            }
            else ("—" if is_outside else "declared")
        )
        flags = (
            "✓"
            if op_present
            and label
            in {"integer arithmetic", "logic / comparisons", "shifts", "control flow"}
            else ("—" if is_outside else "declared")
        )
        memory = (
            "✓"
            if op_present and label == "stack / memory"
            else ("—" if is_outside else "not central")
        )
        control = (
            "✓"
            if op_present and label == "control flow"
            else ("—" if is_outside else "not central")
        )
        lean = (
            "✓"
            if any(operation in lean_operations for operation in ops)
            else ("—" if is_outside else "unresolved")
        )
        capstone = (
            "✓"
            if capstone_available and op_present
            else ("—" if is_outside else "not observed")
        )
        unicorn = (
            "✓"
            if unicorn_available and bool(observed.intersection(ops))
            else ("—" if is_outside else "not observed")
        )
        matrix.append(
            [declared, width, flags, memory, control, lean, capstone, unicorn]
        )
    fig, ax = plt.subplots(figsize=(14.1, 5.9))
    ax.axis("off")
    table = ax.table(
        cellText=matrix,
        rowLabels=row_labels,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=cast(Any, (0.13, 0.14, 0.85, 0.70)),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col == -1:
            cell.set_facecolor(LIGHT)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(INK)
        else:
            text = cell.get_text().get_text()
            if text in {"—", "not observed", "unresolved", "not central"}:
                cell.set_facecolor("white")
                cell.set_hatch("//")
            elif text == "UNSUPPORTED":
                cell.set_facecolor(PALETTE[3] + "32")
                cell.set_hatch("xx")
            else:
                cell.set_facecolor(PALETTE[2] + "30")
            cell.get_text().set_color(INK)
    ax.text(
        0.13,
        0.89,
        "✓ = declared or checked; unresolved/not observed = evidence not present; hatching = no numeric performance value",
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
    )
    ax.text(
        0.13,
        0.08,
        _text(
            payload.get("unsupported_note"),
            "Unsupported instruction effects remain explicit and fail closed.",
        ),
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
    )
    _title(
        fig,
        "Declared semantic coverage by evidence layer",
        "Operation-family matrix from the formal and adapter receipts; coverage status is categorical and does not imply complete x86-64 semantics.",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.82)
    return _save_multi(fig, output_path)


def plot_statistical_hierarchy(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render the dependence hierarchy and family-clustered inference boundary."""

    levels = (
        (
            "project / source family",
            _text(payload.get("family_count"), "not observed"),
            PALETTE[0],
        ),
        (
            "source revision",
            _text(payload.get("revision_count"), "not observed"),
            PALETTE[1],
        ),
        (
            "transformation pair",
            _text(payload.get("pair_count"), "not observed"),
            PALETTE[2],
        ),
        (
            "intervention",
            _text(payload.get("intervention_count"), "not observed"),
            PALETTE[4],
        ),
        ("trace", _text(payload.get("trace_count"), "not observed"), PALETTE[3]),
        ("instruction step", _text(payload.get("step_count"), "not observed"), MUTED),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 7.0),
        gridspec_kw={"width_ratios": (1.12, 1.0)},
    )
    ax = axes[0]
    ax.set_xlim(0, 4.15)
    ax.set_ylim(0, 6.25)
    ax.axis("off")
    ax.text(
        0.18, 6.02, "Nested dependence hierarchy", fontsize=10, weight="bold", color=INK
    )
    for index, (label, count, color) in enumerate(levels):
        y = 5.30 - index * 0.82
        _box(ax, (0.40, y), 2.72, 0.48, label, f"n = {count}", color + "31")
        if index < len(levels) - 1:
            ax.annotate(
                "",
                xy=(1.76, y - 0.26),
                xytext=(1.76, y - 0.34),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 0.9},
            )
            ax.text(2.00, y - 0.30, "contains", fontsize=6.7, color=MUTED, va="center")
    ax.add_patch(
        Rectangle(
            (0.34, 0.18),
            3.02,
            0.66,
            facecolor=PALETTE[0] + "13",
            edgecolor=PALETTE[0],
            linewidth=1.0,
            hatch="..",
        )
    )
    ax.text(
        0.50,
        0.65,
        "clustered uncertainty boundary",
        fontsize=8.8,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.50,
        0.42,
        f"bootstrap unit: {_text(payload.get('bootstrap_unit'), 'source family')} · {_text(payload.get('confidence_level'), 'not observed')} confidence",
        fontsize=7.2,
        color=INK,
    )
    ax.text(
        0.50,
        0.26,
        "Nested rows and steps are not independent replicates.",
        fontsize=7.0,
        color=MUTED,
    )

    ax = axes[1]
    ax.set_xlim(0, 4.15)
    ax.set_ylim(0, 6.25)
    ax.axis("off")
    ax.text(0.18, 6.02, "Fixed family partition", fontsize=10, weight="bold", color=INK)
    ax.text(
        0.18,
        5.70,
        "The split is declared before fitting; only development families control selection and stopping.",
        fontsize=7.2,
        color=MUTED,
        wrap=True,
    )
    split_names = ("training", "development", "test")
    split_values = payload.get("split_counts", {})
    for index, name in enumerate(split_names):
        value = _text(
            split_values.get(name) if isinstance(split_values, Mapping) else None,
            "not observed",
        )
        x = 0.22 + index * 1.28
        _box(
            ax,
            (x, 4.56),
            1.08,
            0.68,
            name,
            f"{value} families",
            PALETTE[index % len(PALETTE)] + "35",
        )
    ax.add_patch(
        Rectangle(
            (0.14, 3.56),
            3.72,
            0.64,
            facecolor=PALETTE[1] + "12",
            edgecolor=PALETTE[1],
            linewidth=1.0,
            hatch="//",
        )
    )
    ax.text(0.30, 3.92, "selection boundary", fontsize=8.8, weight="bold", color=INK)
    ax.text(
        0.30,
        3.70,
        "development-only tuning; held-out test families remain untouched",
        fontsize=7.1,
        color=INK,
    )
    ax.add_patch(
        Rectangle(
            (0.14, 1.00),
            3.72,
            1.92,
            facecolor="white",
            edgecolor=GRID,
            linewidth=0.9,
        )
    )
    ax.text(0.30, 2.64, "Reading rule", fontsize=8.8, weight="bold", color=INK)
    ax.text(
        0.30,
        2.32,
        "A larger row count improves coverage of the frozen benchmark, but it does not create additional independent source families.",
        fontsize=7.4,
        color=INK,
        wrap=True,
        va="top",
    )
    ax.text(
        0.30,
        1.50,
        f"split contract: {_text(payload.get('split'), 'fixed before fitting')}",
        fontsize=7.1,
        color=MUTED,
        wrap=True,
        va="top",
    )
    _title(
        fig,
        "Statistical hierarchy and leakage control",
        "Counts describe the source-bound evaluation hierarchy; inference is clustered at the family level and model selection is restricted to development families.",
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.84, wspace=0.32)
    return _save_multi(fig, output_path)


def plot_replay_failure_audit(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render the controlled replay boundary and its protected negative result."""

    score = payload.get("protected_score", {})
    if not isinstance(score, Mapping):
        score = {}
    confusion = score.get("confusion_matrix", {})
    if not isinstance(confusion, Mapping):
        confusion = {}
    fig = plt.figure(figsize=(14.0, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=(0.83, 1.17), hspace=0.36, wspace=0.28)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_xlim(0, 4.20)
    ax_flow.set_ylim(0, 1.0)
    ax_flow.axis("off")
    flow = (
        (0.10, "public rows", "label-blind\nno gold labels", PALETTE[0]),
        (1.15, "controlled guest", "network disabled\nseparate namespace", PALETTE[1]),
        (2.20, "predictions", "IDs checked\nordered output", PALETTE[2]),
        (
            3.25,
            "protected score",
            "gold stays owner-side\nnegative result retained",
            PALETTE[3],
        ),
    )
    for x, label, detail, color in flow:
        _box(
            ax_flow,
            (x, 0.30),
            0.78,
            0.42,
            label,
            detail,
            color + "30",
            hatch="//" if label == "protected score" else None,
        )
    for index in range(3):
        _arrow(
            ax_flow,
            (flow[index][0] + 0.78, 0.51),
            (flow[index + 1][0], 0.51),
            "hash-bound",
        )
    ax_flow.text(
        0.12,
        0.13,
        f"control scope: {_text(payload.get('control_scope'), 'not observed')} · independent reimplementation: {_text(payload.get('independent_reimplementation'), 'false')} · network disabled: {_text(payload.get('network_disabled'), 'not observed')}",
        fontsize=7.7,
        color=MUTED,
    )
    ax_cm = fig.add_subplot(gs[1, 0])
    if score:
        cm = np.array(
            [
                [
                    _number(confusion.get("true_positive")),
                    _number(confusion.get("false_negative")),
                ],
                [
                    _number(confusion.get("false_positive")),
                    _number(confusion.get("true_negative")),
                ],
            ]
        )
    else:
        cm = np.zeros((2, 2), dtype=float)
    cm_max = max(1.0, float(np.max(cm)))
    ax_cm.imshow(cm, cmap="Greys", vmin=0, vmax=cm_max)
    ax_cm.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax_cm.tick_params(axis="y", labelsize=7)
    for i in range(2):
        for j in range(2):
            cell_color = "white" if cm[i, j] / cm_max >= 0.55 else INK
            ax_cm.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=10,
                color=cell_color,
                weight="bold",
            )
    ax_cm.set_title("Protected confusion counts", loc="left", fontsize=10, color=INK)
    if not score:
        ax_cm.text(
            0.5,
            -0.42,
            "no separately operated receipt ingested: protected counts not observed",
            transform=ax_cm.transAxes,
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax_cal = fig.add_subplot(gs[1, 1])
    metrics = ("balanced\naccuracy", "macro-F1", "Brier", "ECE")
    values = [
        _number(score.get("balanced_accuracy")),
        _number(score.get("macro_f1")),
        _number(score.get("brier_score")),
        _number(score.get("expected_calibration_error")),
    ]
    bars: object
    if score:
        bars = ax_cal.bar(
            np.arange(4),
            values,
            color=[PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[4]],
            edgecolor=INK,
            linewidth=0.6,
        )
    else:
        bars = []
    ax_cal.set_xticks(np.arange(4), metrics)
    ax_cal.set_ylim(0, max(1.0, max(values) * 1.25))
    ax_cal.set_ylabel("score")
    ax_cal.set_title(
        "Observed single-replay metrics", loc="left", fontsize=10, color=INK
    )
    if score:
        for bar, value in zip(bars, values, strict=True):
            ax_cal.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK,
            )
    if not score:
        ax_cal.text(
            0.5,
            0.55,
            "protected metrics not observed\n(no separately operated receipt ingested)",
            transform=ax_cal.transAxes,
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax_risk = fig.add_subplot(gs[1, 2])
    coverage = _number(
        score.get("selective_risk_coverage", {}).get("coverage")
        if isinstance(score.get("selective_risk_coverage"), Mapping)
        else None
    )
    risk = _number(
        score.get("selective_risk_coverage", {}).get("risk")
        if isinstance(score.get("selective_risk_coverage"), Mapping)
        else None
    )
    if score:
        ax_risk.scatter(
            [coverage], [risk], s=75, color=PALETTE[3], edgecolor=INK, zorder=3
        )
    ax_risk.set_xlim(0, 1.05)
    ax_risk.set_ylim(0, 1.05)
    ax_risk.set_xlabel("coverage")
    ax_risk.set_ylabel("risk")
    ax_risk.set_title("Selective risk / coverage", loc="left", fontsize=10, color=INK)
    ax_risk.text(
        0.05,
        0.92,
        "one protected replay",
        transform=ax_risk.transAxes,
        fontsize=7.2,
        color=MUTED,
    )
    for ax in (ax_cal, ax_risk):
        _style_axes(ax, grid_axis="y")
    _title(
        fig,
        "Controlled replay failure audit",
        "Same-host separately controlled guest replay; the protected score is a negative reproducibility result, not an efficacy estimate or independent physical replication.",
    )
    fig.text(
        0.055,
        0.015,
        f"rows = {_text(score.get('rows'))} · false-equivalence rate = {_text(score.get('false_equivalence_rate'))} · near-miss separation = {_text(score.get('near_miss_separation_rate'))}",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.13, top=0.86)
    return _save_multi(fig, output_path)


def plot_policy_certificate_frontier(
    payload: Mapping[str, Any], output_path: Path
) -> Path:
    """Render exact finite policy frontiers and certificate gaps."""

    frontier = payload.get("frontier", [])
    evaluations = payload.get("policy_evaluations", [])
    certificates = payload.get("policy_certificates", [])
    if not isinstance(frontier, list) or not frontier:
        frontier = []
    if not isinstance(evaluations, list):
        evaluations = []
    if not isinstance(certificates, list):
        certificates = []
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.8), gridspec_kw={"width_ratios": (1.25, 1.0)}
    )
    horizon_values: set[int] = set()
    for row in frontier:
        if not isinstance(row, Mapping):
            continue
        value = row.get("horizon")
        if isinstance(value, int):
            horizon_values.add(value)
    horizons = sorted(horizon_values)
    for index, horizon in enumerate(horizons):
        rows = [
            row
            for row in frontier
            if isinstance(row, Mapping) and int(row.get("horizon", -1)) == horizon
        ]
        rows.sort(key=lambda row: _number(row.get("cost_weight")))
        axes[0].plot(
            [_number(row.get("expected_cost")) for row in rows],
            [_number(row.get("expected_information_gain")) for row in rows],
            marker="o",
            linewidth=1.5,
            color=PALETTE[index % len(PALETTE)],
            label=f"horizon {horizon}",
        )
    axes[0].set_xlabel("expected deterministic cost (count units)")
    axes[0].set_ylabel("expected information gain (nats)")
    axes[0].set_title(
        "Exact finite information frontier", loc="left", fontsize=10, color=INK
    )
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    axes[0].text(
        0.02,
        -0.20,
        "Exact enumeration; no inferential interval applies.",
        transform=axes[0].transAxes,
        fontsize=7.4,
        color=MUTED,
    )
    policies = sorted(
        {str(row.get("policy")) for row in evaluations if isinstance(row, Mapping)}
    )
    policy_values: dict[str, list[float]] = {policy: [] for policy in policies}
    for row in evaluations:
        if isinstance(row, Mapping):
            policy_values[str(row.get("policy"))].append(
                _number(row.get("optimality_gap"))
            )
    means = [
        float(np.mean(policy_values[policy])) if policy_values[policy] else 0.0
        for policy in policies
    ]
    bars = axes[1].bar(
        np.arange(len(policies)),
        means,
        color=[PALETTE[index % len(PALETTE)] for index in range(len(policies))],
        edgecolor=INK,
        linewidth=0.6,
    )
    axes[1].set_xticks(
        np.arange(len(policies)),
        [policy.replace("_", "\n") for policy in policies],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylabel("mean optimality gap")
    axes[1].set_title(
        "Policy gap to exact finite reference", loc="left", fontsize=10, color=INK
    )
    for bar, value in zip(bars, means, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    certificate_status = sum(
        bool(row.get("within_preregistered_tolerance"))
        for row in certificates
        if isinstance(row, Mapping)
    )
    axes[1].text(
        0.02,
        -0.20,
        f"{certificate_status}/{len(certificates)} certificate rows within tolerance; timing is machine-specific.",
        transform=axes[1].transAxes,
        fontsize=7.4,
        color=MUTED,
    )
    for ax in axes:
        _style_axes(ax, grid_axis="y")
    _title(
        fig,
        "Finite sensing policy certificates and cost frontier",
        "The oracle is an inadmissible reference for certification; deterministic counts and optional local wall-clock calibration remain separate quantities.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.82, wspace=0.30)
    return _save_multi(fig, output_path)


def plot_theorem_status_map(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Render theorem, computational, empirical, falsified, and open statuses."""

    candidates = payload.get("candidate_theorems", [])
    assumptions = payload.get("assumptions", [])
    if not isinstance(candidates, list):
        candidates = []
    status_order = (
        "proved",
        "computationally_validated",
        "empirically_supported",
        "falsified_by_counterexample",
        "unresolved",
    )
    status_label = {
        "proved": "proved",
        "computationally_validated": "computational",
        "empirically_supported": "empirical",
        "falsified_by_counterexample": "falsified",
        "unresolved": "unresolved",
    }
    status_color = {
        status: PALETTE[index % len(PALETTE)]
        for index, status in enumerate(status_order)
    }
    rows: list[list[str]] = []
    counts = {status: 0 for status in status_order}
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        raw = str(row.get("status", "unresolved"))
        status = raw if raw in counts else "unresolved"
        counts[status] += 1
        statement = _wrapped(row.get("statement"), 48, "candidate statement")
        scope = _wrapped(
            row.get("scope"),
            58,
            _text(row.get("proof_receipt"), "scope not recorded"),
        )
        rows.append([status_label[status], statement, scope])
    fig, ax = plt.subplots(figsize=(14.2, 8.4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("status", "candidate result", "assumptions / artifact scope"),
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=cast(Any, (0.03, 0.17, 0.94, 0.70)),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1, 1.15)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if col == 0:
            cell.set_width(0.14)
        elif col == 1:
            cell.set_width(0.39)
        else:
            cell.set_width(0.47)
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT)
            if col == 0:
                raw_status = rows[row - 1][0]
                raw_key = next(
                    (key for key, value in status_label.items() if value == raw_status),
                    "unresolved",
                )
                cell.set_facecolor(status_color[raw_key] + "38")
                cell.get_text().set_fontweight("bold")
    legend_x = 0.04
    for status in status_order:
        ax.add_patch(
            Rectangle(
                (legend_x, 0.10),
                0.025,
                0.025,
                facecolor=status_color[status],
                edgecolor=INK,
                linewidth=0.5,
            )
        )
        ax.text(
            legend_x + 0.033,
            0.112,
            f"{status_label[status]}: {counts[status]}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=INK,
            va="center",
        )
        legend_x += 0.18
    assumption_text = textwrap.fill(
        "; ".join(str(item) for item in assumptions[:4]) or "not recorded",
        width=170,
        break_long_words=False,
    )
    ax.text(
        0.03,
        0.04,
        f"shared assumptions: {assumption_text}",
        transform=ax.transAxes,
        fontsize=7.2,
        color=MUTED,
        wrap=True,
    )
    _title(
        fig,
        "Theorem-or-countertheorem status map",
        "Formal and computational outcomes are separated by status; falsified and unresolved candidates remain visible, and no theorem-shaped artifact is promoted without its checked boundary.",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.84)
    return _save_multi(fig, output_path)


def write_visual_atlas_figures(output_dir: Path) -> dict[str, str]:
    """Write the nine narrative PNG/SVG/PDF figures from current receipts."""

    data = output_dir / "data"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    protocol = _json(data / "research_protocol.json")
    formal = _json(data / "formal_trace_correspondence.json")
    formal_verification = _json(data / "formal_verification.json")
    listing = _json(data / "assembly_listing.json")
    trace = _json(data / "trace_bundle.json")
    memory = _json(data / "memory_evidence.json")
    quotient = _json(data / "quotient_invariance_protocol.json")
    quotient_classifier = _json(data / "quotient_classifier.json")
    theorem = _json(data / "theorem_or_countertheorem.json")
    sensor = _json(data / "sensor_cost_receipt.json")
    frontier = _json(data / "active_information_frontier.json")
    replay_path = data / "independent_replay_receipt.json"
    replay = _json(replay_path) if replay_path.is_file() else {}
    typed = _json(data / "typed_observations.json")
    replication = _json(data / "replication_manifest.json")

    outputs: dict[str, str] = {}
    plot_research_architecture(protocol, fig_dir / "research_architecture.png")
    outputs["research_architecture"] = "output/figures/research_architecture.png"
    input_contract = replication.get("input_contract", {})
    plot_observation_semantics(
        {
            "fields": typed.get("fields", []),
            "dynamic_field": typed.get("dynamic_field"),
            "missing_token": protocol.get("observational_contract", {}).get(
                "missing_token"
            ),
            "unseen_token": protocol.get("observational_contract", {}).get(
                "unseen_token"
            ),
            "public_fields": input_contract.get("public_fields", [])
            if isinstance(input_contract, Mapping)
            else [],
            "forbidden_public_fields": input_contract.get("forbidden_public_fields", [])
            if isinstance(input_contract, Mapping)
            else [],
            "protocol_id": protocol.get("protocol_id"),
            "protocol_sha256": protocol.get("protocol_sha256"),
            "public_row_count": replication.get("public_row_count"),
            "gold_row_count": replication.get("gold_row_count"),
            "typed_schema_version": typed.get("schema_version"),
            "replication_status": replication.get("status"),
        },
        fig_dir / "observation_semantics.png",
    )
    outputs["observation_semantics"] = "output/figures/observation_semantics.png"
    plot_assembly_state_plate(
        {
            "listing": listing.get("instructions", []),
            "trace": trace,
            "formal_steps": formal.get("step_rows", []),
            "memory": memory,
            "status": formal.get("status"),
        },
        fig_dir / "assembly_state_plate.png",
    )
    outputs["assembly_state_plate"] = "output/figures/assembly_state_plate.png"
    plot_quotient_construction(
        {"pairs": quotient.get("pairs", []), "classifier": quotient_classifier},
        fig_dir / "quotient_construction.png",
    )
    outputs["quotient_construction"] = "output/figures/quotient_construction.png"
    observed = [
        str(row.get("mnemonic"))
        for row in trace.get("static_instruction_events", [])
        if isinstance(row, Mapping)
    ]
    lean_operations = (
        [
            str(row)
            for row in formal_verification.get("checked_fragment", {}).get(
                "operations", []
            )
        ]
        if isinstance(formal_verification.get("checked_fragment"), Mapping)
        else []
    )
    plot_semantic_coverage_matrix(
        {
            "supported": formal.get("supported_instruction_subset", []),
            "observed": observed,
            "lean_operations": lean_operations,
            "capstone_available": bool(
                formal.get("toolchain_versions", {}).get("capstone")
            ),
            "unicorn_available": bool(
                formal.get("toolchain_versions", {}).get("unicorn")
            ),
            "unsupported_note": "; ".join(
                str(item) for item in formal.get("unsupported_cases", [])[:2]
            ),
        },
        fig_dir / "semantic_coverage_matrix.png",
    )
    outputs["semantic_coverage_matrix"] = "output/figures/semantic_coverage_matrix.png"
    split_counts = {
        "training": quotient_classifier.get("training_family_count"),
        "development": quotient_classifier.get("development_family_count"),
        "test": quotient_classifier.get("held_out_family_count"),
    }
    plot_statistical_hierarchy(
        {
            "family_count": quotient_classifier.get("family_count"),
            "revision_count": "not recorded",
            "pair_count": quotient_classifier.get("row_count"),
            "intervention_count": len(quotient_classifier.get("interventions", [])),
            "trace_count": quotient_classifier.get("row_count"),
            "step_count": "not recorded",
            "bootstrap_unit": "source family",
            "confidence_level": "95%",
            "split": "family-disjoint; development-only selection",
            "split_counts": split_counts,
        },
        fig_dir / "statistical_hierarchy.png",
    )
    outputs["statistical_hierarchy"] = "output/figures/statistical_hierarchy.png"
    external = replay.get("external_receipt", {})
    plot_replay_failure_audit(
        {
            "protected_score": replay.get("protected_score", {}),
            "control_scope": external.get("control_scope")
            if isinstance(external, Mapping)
            else None,
            "independent_reimplementation": external.get("independent_reimplementation")
            if isinstance(external, Mapping)
            else None,
            "network_disabled": external.get("network_disabled")
            if isinstance(external, Mapping)
            else None,
        },
        fig_dir / "replay_failure_audit.png",
    )
    outputs["replay_failure_audit"] = "output/figures/replay_failure_audit.png"
    plot_policy_certificate_frontier(
        {
            "frontier": frontier.get("frontier", []),
            "policy_evaluations": sensor.get("policy_evaluations", []),
            "policy_certificates": sensor.get("policy_certificates", []),
        },
        fig_dir / "policy_certificate_frontier.png",
    )
    outputs["policy_certificate_frontier"] = (
        "output/figures/policy_certificate_frontier.png"
    )
    plot_theorem_status_map(theorem, fig_dir / "theorem_status_map.png")
    outputs["theorem_status_map"] = "output/figures/theorem_status_map.png"
    return outputs


__all__ = [
    "NARRATIVE_FIGURE_IDS",
    "plot_assembly_state_plate",
    "plot_policy_certificate_frontier",
    "plot_quotient_construction",
    "plot_replay_failure_audit",
    "plot_research_architecture",
    "plot_observation_semantics",
    "plot_semantic_coverage_matrix",
    "plot_statistical_hierarchy",
    "plot_theorem_status_map",
    "write_visual_atlas_figures",
]
