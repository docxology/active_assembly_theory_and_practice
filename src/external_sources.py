"""Pinned, static-only licensed-source acquisition and validation utilities.

This module implements the external-validation boundary without silently
turning a source manifest into permission to execute arbitrary artifacts.
Acquisition is opt-in, archive hashes and immutable revisions are mandatory,
and replay/manuscript generation never uses the network.  The optional build
helper accepts only an explicit compiler argument vector containing ``-c``;
it records the resulting object and disassembly hashes and never executes the
object or any source-produced program.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import yaml

from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_MANIFEST_SCHEMA_VERSION = "2.0.0"
EXTERNAL_MANIFEST_PATH = "configs/external_sources.yaml"
EXTERNAL_RECEIPT_ARTIFACT = "output/data/external_source_validation.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_COMPILERS = {"clang", "clang++", "gcc", "g++", "cc", "c++"}
_FORBIDDEN_ARG_TOKENS = {";", "&&", "||", "|", ">", "<", "`", "$"}


@dataclass(frozen=True)
class ExternalSourceSpec:
    """One immutable, license-cleared source acquisition contract."""

    source_id: str
    repository_url: str
    revision: str
    license: str
    license_file: str
    license_file_sha256: str
    source_archive_sha256: str
    source_family: str
    build_command: tuple[str, ...]
    compiler: str
    target_triple: str
    label_contract: str
    provenance_reviewer: str
    acquisition_timestamp: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest entry."""

        return {
            "source_id": self.source_id,
            "repository_url": self.repository_url,
            "revision": self.revision,
            "license": self.license,
            "license_file": self.license_file,
            "license_file_sha256": self.license_file_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "source_family": self.source_family,
            "build_command": list(self.build_command),
            "compiler": self.compiler,
            "target_triple": self.target_triple,
            "label_contract": self.label_contract,
            "provenance_reviewer": self.provenance_reviewer,
            "acquisition_timestamp": self.acquisition_timestamp,
        }


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _validate_relative_path(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{field} must be a relative path")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{field} may not escape the source root")


def _validate_build_command(command: object) -> tuple[str, ...]:
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ValueError("build_command must be a non-empty argument vector")
    if any(token in _FORBIDDEN_ARG_TOKENS for token in command):
        raise ValueError("build_command may not contain shell syntax")
    executable = Path(command[0]).name
    if executable not in _ALLOWED_COMPILERS:
        raise ValueError("build_command must begin with an allow-listed compiler")
    if "-c" not in command:
        raise ValueError("build_command must contain -c for static compilation")
    return tuple(command)


def _spec_from_row(row: object, index: int) -> ExternalSourceSpec:
    if not isinstance(row, dict):
        raise ValueError(f"external source row {index} is not an object")
    required = {
        "source_id",
        "repository_url",
        "revision",
        "license",
        "license_file",
        "license_file_sha256",
        "source_archive_sha256",
        "source_family",
        "build_command",
        "compiler",
        "target_triple",
        "label_contract",
        "provenance_reviewer",
        "acquisition_timestamp",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"external source row {index} missing {missing}")
    source_id = row["source_id"]
    url = row["repository_url"]
    revision = row["revision"]
    if not isinstance(source_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", source_id
    ):
        raise ValueError(f"external source row {index} has an invalid source_id")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("repository_url must use https")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be a 40-character immutable commit hash")
    if not isinstance(row["license"], str) or not row["license"].strip():
        raise ValueError("license must be non-empty")
    _validate_relative_path(row["license_file"], field="license_file")
    if not _is_hash(row["license_file_sha256"]):
        raise ValueError("license_file_sha256 must be a lowercase SHA-256 digest")
    if not _is_hash(row["source_archive_sha256"]):
        raise ValueError("source_archive_sha256 must be a lowercase SHA-256 digest")
    command = _validate_build_command(row["build_command"])
    compiler = row["compiler"]
    if not isinstance(compiler, str) or Path(compiler).name not in _ALLOWED_COMPILERS:
        raise ValueError("compiler must be an allow-listed compiler")
    if Path(command[0]).name != Path(compiler).name:
        raise ValueError("build_command compiler does not match compiler field")
    for field_name in (
        "source_family",
        "target_triple",
        "label_contract",
        "provenance_reviewer",
        "acquisition_timestamp",
    ):
        if not isinstance(row[field_name], str) or not row[field_name].strip():
            raise ValueError(f"{field_name} must be non-empty")
    return ExternalSourceSpec(
        source_id=source_id,
        repository_url=url,
        revision=revision,
        license=str(row["license"]),
        license_file=str(row["license_file"]),
        license_file_sha256=str(row["license_file_sha256"]),
        source_archive_sha256=str(row["source_archive_sha256"]),
        source_family=str(row["source_family"]),
        build_command=command,
        compiler=str(compiler),
        target_triple=str(row["target_triple"]),
        label_contract=str(row["label_contract"]),
        provenance_reviewer=str(row["provenance_reviewer"]),
        acquisition_timestamp=str(row["acquisition_timestamp"]),
    )


def load_external_manifest(
    path: Path = PROJECT_ROOT / EXTERNAL_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load and validate the pinned YAML manifest."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"external source manifest unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("external source manifest must be an object")
    if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("external source manifest schema mismatch")
    if payload.get("network_policy") != "explicit_acquisition_only":
        raise ValueError("external source network policy is not explicit-only")
    if (
        not isinstance(payload.get("manifest_id"), str)
        or not payload["manifest_id"].strip()
    ):
        raise ValueError("external source manifest_id must be non-empty")
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("external source manifest sources must be a list")
    specs = [_spec_from_row(row, index) for index, row in enumerate(rows)]
    if len({spec.source_id for spec in specs}) != len(specs):
        raise ValueError("external source ids must be unique")
    if len({spec.source_family for spec in specs}) != len(specs):
        raise ValueError("external source families must be unique")
    return {
        "schema_version": EXTERNAL_MANIFEST_SCHEMA_VERSION,
        "manifest_id": str(payload.get("manifest_id", "")),
        "network_policy": "explicit_acquisition_only",
        "static_only": payload.get("static_only") is True,
        "sources": [spec.to_dict() for spec in specs],
    }


def validate_external_manifest(payload: object) -> tuple[bool, str]:
    """Validate a parsed manifest without acquiring or compiling anything."""

    if not isinstance(payload, dict):
        return False, "external source manifest must be an object"
    try:
        if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
            return False, "external source manifest schema mismatch"
        if payload.get("network_policy") != "explicit_acquisition_only":
            return False, "external source network policy drift"
        if (
            not isinstance(payload.get("manifest_id"), str)
            or not payload["manifest_id"].strip()
        ):
            return False, "external source manifest_id is missing"
        if payload.get("static_only") is not True:
            return False, "external source static_only must be true"
        rows = payload.get("sources")
        if not isinstance(rows, list):
            return False, "external source rows must be a list"
        for index, row in enumerate(rows):
            _spec_from_row(row, index)
        specs = [_spec_from_row(row, index) for index, row in enumerate(rows)]
        if len({spec.source_id for spec in specs}) != len(specs):
            return False, "external source ids must be unique"
        if len({spec.source_family for spec in specs}) != len(specs):
            return False, "external source families must be unique"
    except ValueError as exc:
        return False, str(exc)
    return True, f"sources={len(cast(list[object], payload['sources']))}"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a tar/zip archive while rejecting path traversal."""

    destination = destination.resolve()
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for member in members:
                if member.issym() or member.islnk():
                    raise ValueError("archive links are not allowed")
                target = (destination / member.name).resolve()
                if target != destination and destination not in target.parents:
                    raise ValueError("archive member escapes extraction directory")
            for member in members:
                handle.extract(member, destination, filter="data")
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ValueError("archive links are not allowed")
                target = (destination / info.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise ValueError("archive member escapes extraction directory")
            handle.extractall(destination)
        return
    raise ValueError("source archive is neither a supported tar nor zip archive")


def acquire_external_source(
    spec: ExternalSourceSpec,
    destination: Path,
    *,
    allow_network: bool = False,
) -> dict[str, object]:
    """Acquire one pinned archive; default behavior is an explicit block."""

    if not allow_network:
        return {
            "source_id": spec.source_id,
            "status": "blocked_external",
            "reason": "network acquisition requires --allow-network",
            "network_used": False,
        }
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{spec.source_id}-{spec.revision}.archive"
    request = Request(
        spec.repository_url, headers={"User-Agent": "active-assembly-research"}
    )
    try:
        with (
            urlopen(request, timeout=30) as response,
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    except (OSError, URLError) as exc:
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": f"acquisition_failed:{exc}",
            "network_used": True,
        }
    digest = _sha256_path(archive_path)
    if digest != spec.source_archive_sha256:
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": "source archive hash mismatch",
            "network_used": True,
            "archive_sha256": digest,
        }
    extraction = destination / spec.source_id
    extraction.mkdir(parents=True, exist_ok=True)
    try:
        _safe_extract(archive_path, extraction)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": f"safe_extraction_failed:{exc}",
            "network_used": True,
        }
    license_path = extraction / spec.license_file
    license_digest = _sha256_path(license_path) if license_path.is_file() else None
    if license_digest != spec.license_file_sha256:
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": "license file hash mismatch or missing",
            "network_used": True,
            "license_file_sha256": license_digest,
        }
    return {
        "source_id": spec.source_id,
        "status": "validated",
        "network_used": True,
        "archive_path": str(archive_path),
        "archive_sha256": digest,
        "license_file_sha256": license_digest,
        "extraction_root": str(extraction),
    }


def acquire_external_manifest(
    manifest_path: Path,
    destination: Path,
    *,
    allow_network: bool = False,
) -> dict[str, object]:
    """Acquire every pinned manifest row and aggregate failures fail-closed."""

    manifest = load_external_manifest(manifest_path)
    rows = cast(list[dict[str, object]], manifest["sources"])
    if not allow_network:
        return {
            "status": "blocked_external",
            "reason": "network acquisition requires --allow-network",
            "source_count": len(rows),
            "network_used": False,
            "rows": [],
        }
    results = [
        acquire_external_source(
            _spec_from_row(row, index), destination, allow_network=True
        )
        for index, row in enumerate(rows)
    ]
    if not results:
        status = "blocked_external"
    elif all(result.get("status") == "validated" for result in results):
        status = "validated"
    else:
        status = "failed"
    return {
        "status": status,
        "source_count": len(rows),
        "network_used": True,
        "rows": results,
    }


def static_compile_external_source(
    spec: ExternalSourceSpec,
    source_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Compile one pinned source translation unit without executing output."""

    command = list(spec.build_command)
    if Path(command[0]).name not in _ALLOWED_COMPILERS or "-c" not in command:
        raise ValueError("external build command is not an allow-listed static compile")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{spec.source_id}.o"
    if "-o" not in command:
        command.extend(["-o", str(output_path)])
    else:
        output_index = command.index("-o") + 1
        if output_index >= len(command):
            raise ValueError("external build command has no -o destination")
        output_path = (source_root / command[output_index]).resolve()
        if output_directory.resolve() not in output_path.parents:
            raise ValueError("external build output escapes the output directory")
    completed = subprocess.run(
        command,
        cwd=source_root,
        capture_output=True,
        text=True,
        shell=False,
        timeout=120,
        check=False,
    )
    result: dict[str, object] = {
        "source_id": spec.source_id,
        "status": "validated"
        if completed.returncode == 0 and output_path.is_file()
        else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "execution_performed": False,
        "static_only": True,
    }
    if output_path.is_file():
        result["object_sha256"] = _sha256_path(output_path)
        result["object_path"] = str(output_path)
    return result


def external_source_validation_payload(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Return a receipt for the pinned manifest without network access."""

    manifest_path = project_root / EXTERNAL_MANIFEST_PATH
    manifest = load_external_manifest(manifest_path)
    protocol = scientific_protocol_payload()
    rows = cast(list[dict[str, object]], manifest["sources"])
    status = "blocked_external" if not rows else "not_yet_observed"
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            EXTERNAL_MANIFEST_PATH: sha256_path(manifest_path),
            "src/external_sources.py": sha256_path(Path(__file__)),
        },
        toolchain_versions=toolchain_versions(("pyyaml",)),
        status=status,
        limitations=(
            "The canonical replay path is offline and does not acquire sources.",
            "Application-facing effectiveness claims require the full project/revision/example and independent-replay gates.",
        ),
        unsupported_cases=(
            "Arbitrary external binaries are never executed.",
            "Build commands are restricted to allow-listed compiler argument vectors.",
        ),
        underpowered_cases=(
            "At least five independent projects, two revisions where available, and 100 labelable examples are not yet present.",
        ),
    )
    return {
        **envelope,
        "schema_version": EXTERNAL_MANIFEST_SCHEMA_VERSION,
        "artifact": EXTERNAL_RECEIPT_ARTIFACT,
        "manifest": manifest,
        "network_used": False,
        "execution_performed": False,
        "static_only": True,
        "project_count": len({row["source_family"] for row in rows}),
        "labelable_example_count": 0,
        "rows": [],
        "promotion_gate": {
            "minimum_projects": 5,
            "minimum_revisions_per_project": 2,
            "minimum_labelable_examples": 100,
            "status": "blocked_external",
        },
    }


def validate_external_source_validation_payload(payload: object) -> tuple[bool, str]:
    """Validate the offline external-source receipt and safety boundary."""

    if not isinstance(payload, dict):
        return False, "external source receipt must be an object"
    protocol = scientific_protocol_payload()
    if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
        return False, "external source receipt schema mismatch"
    valid, detail = validate_receipt_envelope(payload, expected_protocol=protocol)
    if not valid:
        return False, f"external source receipt: {detail}"
    if payload.get("network_used") is not False:
        return False, "offline external-source receipt used the network"
    if payload.get("execution_performed") is not False:
        return False, "external validation executed an artifact"
    if payload.get("static_only") is not True:
        return False, "external validation is not static-only"
    manifest_valid, manifest_detail = validate_external_manifest(
        payload.get("manifest")
    )
    if not manifest_valid:
        return False, manifest_detail
    return True, "ok"


def write_external_source_validation_payload(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the offline external-source manifest receipt."""

    payload = external_source_validation_payload(project_root)
    valid, detail = validate_external_source_validation_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / EXTERNAL_RECEIPT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "EXTERNAL_MANIFEST_PATH",
    "EXTERNAL_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_RECEIPT_ARTIFACT",
    "ExternalSourceSpec",
    "acquire_external_manifest",
    "acquire_external_source",
    "external_source_validation_payload",
    "load_external_manifest",
    "static_compile_external_source",
    "validate_external_manifest",
    "validate_external_source_validation_payload",
    "write_external_source_validation_payload",
]
