"""Capstone adapter with explicit package provenance."""

from __future__ import annotations

from importlib import metadata

from ..disassembly import InstructionEvent, disassemble_sample
from .schema import PackageProvenance


def capstone_provenance() -> PackageProvenance:
    """Return provenance for the Capstone static disassembly adapter."""

    try:
        version = metadata.version("capstone")
    except metadata.PackageNotFoundError:  # pragma: no cover - dependency gate covers this
        version = "unknown"
    return PackageProvenance(
        name="capstone",
        adapter="capstone_static_disassembly",
        version=version,
        optional=False,
        available=True,
        command="python-capstone",
        details={"role": "static x86-64 instruction decoding"},
    )


def disassemble_with_capstone(sample_name: str) -> tuple[list[InstructionEvent], PackageProvenance]:
    """Decode a sample fixture and attach Capstone provenance."""

    return disassemble_sample(sample_name), capstone_provenance()
