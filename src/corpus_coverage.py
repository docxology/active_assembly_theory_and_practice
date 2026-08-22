"""Source-bound coverage inventory for the transparent fixture corpus.

The Phase A typed-observation expansion is only meaningful if the fixtures
exercise more than one evidence regime.  This module computes a compact,
machine-readable design audit from the same Capstone, Unicorn, and typed
observation path used by inference.  It reports variation; it does not turn
hand-authored fixture labels into a population sample.
"""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

from .corpus import CorpusFixture, build_corpus
from .disassembly import disassemble_bytes
from .emulation import emulate_x86_64
from .factorized_model import build_typed_observation_corpus
from .typed_observations import (
    TYPED_OBSERVATION_FIELDS,
    TypedObservation,
    typed_field_value,
)


CORPUS_COVERAGE_SCHEMA_VERSION = "1.0.0"


class CorpusCoverageRow(TypedDict):
    """Typed contract for one source-bound fixture coverage row."""

    fixture: str
    static_instruction_count: int
    executed_instruction_count: int
    unexecuted_instruction_count: int
    phase_run_count: int
    repeated_phase_count: int
    memory_operand_count: int
    memory_addressing_modes: list[str]
    branch_target_classes: list[str]
    instruction_families: list[str]
    dynamic_path_fraction: float
    has_memory_variation: bool
    has_alternate_dynamic_path: bool
    phase_counts: dict[str, int]
    typed_field_cardinality: dict[str, int]


def _phase_run_count(labels: tuple[str, ...]) -> int:
    """Count contiguous authored phase runs in one labelled fixture."""

    if not labels:
        raise ValueError("phase labels must not be empty")
    return 1 + sum(left != right for left, right in zip(labels, labels[1:]))


def _fixture_row(
    fixture: CorpusFixture,
    typed_rows: tuple[TypedObservation, ...],
) -> CorpusCoverageRow:
    """Build one coverage row from real static and bounded dynamic records."""

    instructions = tuple(disassemble_bytes(fixture.code))
    trace = emulate_x86_64(fixture.code)
    if len(instructions) != len(fixture.phase_labels):
        raise ValueError(f"phase labels do not align for {fixture.name}")
    if len(typed_rows) != len(instructions):
        raise ValueError(f"typed observations do not align for {fixture.name}")

    memory_modes = sorted(
        {
            row.memory_addressing_mode
            for row in typed_rows
            if row.memory_addressing_mode != "none"
        }
    )
    branch_classes = sorted(
        {
            row.branch_target_class
            for row in typed_rows
            if row.branch_target_class != "none"
        }
    )
    phase_counts = dict(sorted(Counter(fixture.phase_labels).items()))
    executed_count = len(set(trace.addresses))
    static_count = len(instructions)
    return {
        "fixture": fixture.name,
        "static_instruction_count": static_count,
        "executed_instruction_count": executed_count,
        "unexecuted_instruction_count": static_count - executed_count,
        "phase_run_count": _phase_run_count(fixture.phase_labels),
        "repeated_phase_count": sum(count > 1 for count in phase_counts.values()),
        "memory_operand_count": sum(bool(row.memory_operands) for row in instructions),
        "memory_addressing_modes": memory_modes,
        "branch_target_classes": branch_classes,
        "instruction_families": sorted({row.instruction_family for row in typed_rows}),
        "dynamic_path_fraction": executed_count / static_count,
        "has_memory_variation": bool(memory_modes),
        "has_alternate_dynamic_path": executed_count < static_count,
        "phase_counts": phase_counts,
        "typed_field_cardinality": {
            field: len({typed_field_value(row, field) for row in typed_rows})
            for field in TYPED_OBSERVATION_FIELDS
        },
    }


def corpus_coverage_payload(
    corpus: tuple[CorpusFixture, ...] | None = None,
) -> dict[str, object]:
    """Return the deterministic design-coverage artifact for the corpus."""

    fixtures = build_corpus() if corpus is None else tuple(corpus)
    if not fixtures:
        raise ValueError("corpus must not be empty")
    typed = build_typed_observation_corpus(fixtures)
    rows = [_fixture_row(fixture, typed[fixture.name]) for fixture in fixtures]

    unique_families = sorted(
        {family for row in rows for family in row["instruction_families"]}
    )
    unique_memory_modes = sorted(
        {mode for row in rows for mode in row["memory_addressing_modes"]}
    )
    unique_branch_classes = sorted(
        {branch_class for row in rows for branch_class in row["branch_target_classes"]}
    )
    summary = {
        "fixture_count": len(rows),
        "total_static_instruction_count": sum(
            int(row["static_instruction_count"]) for row in rows
        ),
        "total_executed_instruction_count": sum(
            int(row["executed_instruction_count"]) for row in rows
        ),
        "unique_instruction_family_count": len(unique_families),
        "unique_memory_mode_count": len(unique_memory_modes),
        "unique_branch_target_class_count": len(unique_branch_classes),
        "memory_fixture_count": sum(bool(row["has_memory_variation"]) for row in rows),
        "alternate_path_fixture_count": sum(
            bool(row["has_alternate_dynamic_path"]) for row in rows
        ),
        "repeated_phase_fixture_count": sum(
            int(row["repeated_phase_count"]) > 0 for row in rows
        ),
        "total_phase_run_count": sum(int(row["phase_run_count"]) for row in rows),
    }
    return {
        "schema_version": CORPUS_COVERAGE_SCHEMA_VERSION,
        "method": "transparent_fixture_coverage_inventory",
        "scope": (
            "real Capstone static decoding and bounded Unicorn execution over "
            "hand-authored transparent x86-64 fixtures"
        ),
        "typed_fields": list(TYPED_OBSERVATION_FIELDS),
        "summary": summary,
        "unique_values": {
            "instruction_families": unique_families,
            "memory_addressing_modes": unique_memory_modes,
            "branch_target_classes": unique_branch_classes,
        },
        "per_fixture": rows,
        "interpretation": (
            "Coverage variation is a fixture-design diagnostic. It supports a "
            "more informative Phase A stress test but is not independent binary "
            "sampling, external validity, or evidence that typed fields encode "
            "intrinsic program semantics."
        ),
    }


def validate_corpus_coverage_payload(payload: dict[str, object]) -> tuple[bool, str]:
    """Validate the coverage contract without recomputing its source values."""

    if payload.get("schema_version") != CORPUS_COVERAGE_SCHEMA_VERSION:
        return False, "unsupported corpus coverage schema version"
    rows = payload.get("per_fixture")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict):
        return False, "coverage rows or summary missing"
    required_row_fields = {
        "fixture",
        "static_instruction_count",
        "executed_instruction_count",
        "phase_run_count",
        "memory_addressing_modes",
        "branch_target_classes",
        "instruction_families",
        "has_memory_variation",
        "has_alternate_dynamic_path",
    }
    if any(
        not isinstance(row, dict) or not required_row_fields.issubset(row)
        for row in rows
    ):
        return False, "coverage row fields incomplete"
    if int(summary.get("fixture_count", 0)) != len(rows):
        return False, "fixture count does not match rows"
    if int(summary.get("memory_fixture_count", -1)) != sum(
        bool(row["has_memory_variation"]) for row in rows
    ):
        return False, "memory fixture count does not match rows"
    if int(summary.get("alternate_path_fixture_count", -1)) != sum(
        bool(row["has_alternate_dynamic_path"]) for row in rows
    ):
        return False, "alternate-path count does not match rows"
    return True, f"fixtures={len(rows)}"
