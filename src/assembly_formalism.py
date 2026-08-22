"""Source-bound assembly listings for the manuscript and audit artifacts.

The project intentionally keeps the assembly example small.  This module does
not assemble or execute new code; it projects the real Capstone instruction
records and Unicorn address trace into a stable, human-readable representation.
That makes the notation in the manuscript an output of the same evidence bundle
used by the inference pipeline.
"""

from __future__ import annotations

from typing import Iterable

from .disassembly import InstructionEvent
from .emulation import EmulationTrace
from .features import AssemblyObservation

ASSEMBLY_LISTING_SCHEMA_VERSION = "1.0"


def build_assembly_listing(
    instructions: Iterable[InstructionEvent],
    observations: Iterable[AssemblyObservation],
    trace: EmulationTrace,
) -> list[dict[str, object]]:
    """Join static instruction, categorical, and dynamic evidence rows.

    The function requires one observation per decoded instruction and preserves
    the Capstone byte spelling.  ``dynamic_order`` is the first position at
    which Unicorn executed an address, or ``None`` for a statically decoded but
    dynamically unexecuted instruction.
    """

    instruction_rows = list(instructions)
    observation_rows = list(observations)
    if not instruction_rows:
        raise ValueError("instructions must not be empty")
    if len(instruction_rows) != len(observation_rows):
        raise ValueError("instructions and observations must have equal length")

    first_dynamic_order: dict[int, int] = {}
    for order, address in enumerate(trace.addresses, start=1):
        first_dynamic_order.setdefault(int(address), order)

    rows: list[dict[str, object]] = []
    for instruction, observation in zip(
        instruction_rows, observation_rows, strict=True
    ):
        if instruction.address != observation.address:
            raise ValueError(
                "instruction and observation addresses differ at index "
                f"{observation.index}: {instruction.address:#x} != {observation.address:#x}"
            )
        rows.append(
            {
                "index": int(observation.index),
                "address": int(instruction.address),
                "address_hex": f"0x{instruction.address:x}",
                "size": int(instruction.size),
                "bytes_hex": instruction.bytes_hex,
                "mnemonic": instruction.mnemonic,
                "operands": instruction.op_str,
                "groups": list(instruction.groups),
                "registers_read": list(instruction.registers_read),
                "registers_written": list(instruction.registers_written),
                "immediate_operands": list(instruction.immediate_operands),
                "memory_operands": [
                    dict(operand) for operand in instruction.memory_operands
                ],
                "category": observation.category,
                "observation_id": int(observation.observation_id),
                "executed": bool(observation.executed),
                "has_immediate_operand": bool(observation.has_immediate_operand),
                "uses_accumulator": bool(observation.uses_accumulator),
                "dynamic_order": first_dynamic_order.get(instruction.address),
            }
        )
    return rows


def assembly_listing_payload(
    sample_name: str,
    instructions: Iterable[InstructionEvent],
    observations: Iterable[AssemblyObservation],
    trace: EmulationTrace,
) -> dict[str, object]:
    """Return a JSON-safe assembly evidence artifact."""

    rows = build_assembly_listing(instructions, observations, trace)
    return {
        "schema_version": ASSEMBLY_LISTING_SCHEMA_VERSION,
        "sample_id": sample_name,
        "architecture": "x86_64",
        "base_address": int(trace.base_address),
        "instruction_count": len(rows),
        "executed_address_order": [int(address) for address in trace.addresses],
        "instructions": rows,
    }


def assembly_listing_markdown(payload: dict[str, object]) -> str:
    """Render the evidence artifact as a compact, syntax-highlighted listing."""

    sample_name = str(payload.get("sample_id", "unknown"))
    architecture = str(payload.get("architecture", "unknown"))
    rows = payload.get("instructions", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("assembly payload must contain instruction rows")
    lines = [
        f"; source fixture: {sample_name} | architecture: {architecture}",
        "; static bytes and mnemonics: Capstone; executed addresses: Unicorn",
        "; register access: Capstone detail; category and observation_id are model inputs",
    ]
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ValueError("assembly instruction rows must be objects")
        operands = str(raw_row.get("operands", ""))
        instruction = f"{raw_row['mnemonic']} {operands}".rstrip()
        executed = "yes" if bool(raw_row.get("executed")) else "no"
        reads = (
            ",".join(str(value) for value in raw_row.get("registers_read", [])) or "-"
        )
        writes = (
            ",".join(str(value) for value in raw_row.get("registers_written", []))
            or "-"
        )
        # Keep the source-bound listing legible in the manuscript's fixed-width
        # block.  The evidence annotation is deliberately wrapped onto a
        # continuation line rather than allowing a long category/register row
        # to run past the PDF text area and become visually clipped.
        lines.append(
            f"{str(raw_row['address_hex']):>8}  "
            f"{str(raw_row['bytes_hex']):<17}  "
            f"{instruction}"
        )
        lines.append(" " * 29 + f"; reads={reads} | writes={writes}")
        lines.append(
            " "
            * 29
            + f"; category={raw_row['category']} | obs_id={raw_row['observation_id']} | "
            f"executed={executed}"
        )
    return "\n".join(lines)


def validate_assembly_listing_payload(
    payload: dict[str, object], trace_payload: dict[str, object]
) -> tuple[bool, str]:
    """Check that the listing is a faithful projection of the trace bundle."""

    rows = payload.get("instructions")
    trace_rows = trace_payload.get("static_instruction_events")
    observations = trace_payload.get("observations")
    if (
        not isinstance(rows, list)
        or not isinstance(trace_rows, list)
        or not isinstance(observations, list)
    ):
        return False, "listing or trace instruction rows are missing"
    if payload.get("schema_version") != ASSEMBLY_LISTING_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    if len(rows) != len(trace_rows) or len(rows) != len(observations) or not rows:
        return (
            False,
            f"rows={len(rows)} trace={len(trace_rows)} observations={len(observations)}",
        )
    for index, (row, instruction, observation) in enumerate(
        zip(rows, trace_rows, observations, strict=True)
    ):
        if (
            row.get("index") != index
            or row.get("address") != instruction.get("address")
            or row.get("mnemonic") != instruction.get("mnemonic")
            or row.get("bytes_hex") != instruction.get("bytes_hex")
            or row.get("category") != observation.get("category")
            or row.get("observation_id") != observation.get("observation_id")
        ):
            return False, f"row {index} differs from trace bundle"
    return True, f"rows={len(rows)} sample={payload.get('sample_id')}"
