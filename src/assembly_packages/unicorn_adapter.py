"""Unicorn adapter with explicit package provenance."""

from __future__ import annotations

from importlib import metadata

from ..emulation import EmulationTrace, emulate_sample
from .schema import PackageProvenance


def unicorn_provenance() -> PackageProvenance:
    """Return provenance for the Unicorn bounded-emulation adapter."""

    try:
        version = metadata.version("unicorn")
    except metadata.PackageNotFoundError:  # pragma: no cover - dependency gate covers this
        version = "unknown"
    return PackageProvenance(
        name="unicorn",
        adapter="unicorn_bounded_emulation",
        version=version,
        optional=False,
        available=True,
        command="python-unicorn",
        details={"role": "bounded x86-64 dynamic execution"},
    )


def emulate_with_unicorn(sample_name: str) -> tuple[EmulationTrace, PackageProvenance]:
    """Run a sample fixture and attach Unicorn provenance."""

    return emulate_sample(sample_name), unicorn_provenance()
