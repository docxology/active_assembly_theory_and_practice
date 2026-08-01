"""Availability-probe tests for the optional assembly-analysis adapters.

These exercise the real probe logic (no mocks): an installed package (numpy)
resolves to its version, importable-but-metadata-less stdlib (os) resolves to
"unknown", an absent module resolves to None, and an absent command reports an
optional, unavailable probe.
"""

from __future__ import annotations

from src.assembly_packages.optional_adapters import (
    _command_probe,
    _python_package_version,
    probe_optional_packages,
)


def test_python_package_version_installed_resolves() -> None:
    version = _python_package_version("numpy")
    # numpy is a core dependency, so it must resolve to its installed version.
    assert version not in (None, "", "unknown")


def test_python_package_version_stdlib_is_unknown() -> None:
    # `os` is importable but ships no dist-info metadata, so the probe falls
    # back to "unknown" without raising.
    assert _python_package_version("os") == "unknown"


def test_python_package_version_absent_is_none() -> None:
    assert _python_package_version("no_such_module_hermes_zzz") is None


def test_command_probe_absent_is_optional_unavailable() -> None:
    probe = _command_probe(
        "sample",
        "sample_adapter",
        ("not-a-real-cmd-hermes-1", "not-a-real-cmd-hermes-2"),
    )
    assert probe.name == "sample"
    assert probe.optional is True
    assert probe.available is False
    assert probe.version == "unknown"
    assert probe.command == "not-a-real-cmd-hermes-1 | not-a-real-cmd-hermes-2"


def test_probe_optional_packages_returns_consistent_rows() -> None:
    rows = probe_optional_packages()
    names = [row.name for row in rows]
    # Every advertised optional package is present exactly once.
    assert len(names) == len(set(names))
    for row in rows:
        assert row.optional is True
        assert row.command.strip()
        # An available probe never reports an empty version (python-package
        # probes report the real version; command probes honestly say "unknown"
        # because we do not execute the binary to extract its version).
        if row.available:
            assert row.version not in ("",)
