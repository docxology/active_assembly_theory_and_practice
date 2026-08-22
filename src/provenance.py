"""Small source-bound provenance helpers shared by new roadmap producers."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
from typing import Iterable


def sha256_bytes(value: bytes) -> str:
    """Return a stable SHA-256 digest for an in-memory artifact."""

    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one source or generated file without normalizing its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_revision(source_path: Path) -> str:
    """Use the producer source digest as a revision for dirty worktrees."""

    return sha256_path(source_path)


def input_hashes(paths: Iterable[Path], project_root: Path) -> dict[str, str]:
    """Return reproducible project-relative hashes for existing input files."""

    result: dict[str, str] = {}
    for path in sorted(set(paths)):
        if path.is_file():
            try:
                key = str(path.relative_to(project_root))
            except ValueError:
                key = str(path)
            result[key] = sha256_path(path)
    return result


def toolchain_versions(package_names: Iterable[str]) -> dict[str, str]:
    """Record available Python package versions without treating absence as success."""

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in sorted(set(package_names)):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


__all__ = [
    "generator_revision",
    "input_hashes",
    "sha256_bytes",
    "sha256_path",
    "toolchain_versions",
]
