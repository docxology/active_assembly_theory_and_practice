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
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import yaml

from .artifact_contract import write_json_artifact
from .compiler_transfer import _decode_bytes, _extract_text_bytes
from .external_corpus import (
    LABEL_CONTRACT_PATH,
    label_contract_summary,
    load_label_contracts,
    validate_label_contracts,
)
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_MANIFEST_SCHEMA_VERSION = "2.0.0"
EXTERNAL_MANIFEST_PATH = "configs/external_sources.yaml"
EXTERNAL_RECEIPT_ARTIFACT = "output/data/external_source_validation.json"
EXTERNAL_STATIC_RECEIPT_ARTIFACT = "output/data/external_static_validation.json"
EXTERNAL_CAPABILITY_ARTIFACT = "output/data/external_capabilities.json"
EXTERNAL_ACQUISITION_ARTIFACT = "output/data/external_acquisition.json"
EXTERNAL_LABEL_CONTRACT_PATH = "configs/external_label_contracts.yaml"
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
    project_id: str = ""
    revision_label: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest entry."""

        return {
            "source_id": self.source_id,
            "project_id": self.project_id or self.source_id,
            "revision_label": self.revision_label or self.revision[:12],
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
        project_id=str(row.get("project_id") or source_id),
        revision_label=str(row.get("revision_label") or revision[:12]),
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
    if len({(spec.project_id, spec.revision_label) for spec in specs}) != len(specs):
        raise ValueError("external source project/revision pairs must be unique")
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
        if len({(spec.project_id, spec.revision_label) for spec in specs}) != len(
            specs
        ):
            return False, "external source project/revision pairs must be unique"
    except ValueError as exc:
        return False, str(exc)
    return True, f"sources={len(cast(list[object], payload['sources']))}"


def prepare_external_summary(root: Path) -> tuple[dict[str, Any], bool]:
    """Validate the pinned manifest and label contracts without executing anything.

    Returns the read-only summary payload and whether both validation
    surfaces passed, which maps directly onto the CLI exit code.
    """

    manifest = load_external_manifest(root / EXTERNAL_MANIFEST_PATH)
    contracts = load_label_contracts(root / LABEL_CONTRACT_PATH)
    manifest_valid, manifest_detail = validate_external_manifest(manifest)
    contracts_valid, contracts_detail = validate_label_contracts(contracts)
    payload = {
        "status": "validated" if manifest_valid and contracts_valid else "failed",
        "manifest": manifest_detail,
        "label_contracts": contracts_detail,
        "label_summary": label_contract_summary(root),
        "network_used": False,
        "execution_performed": False,
    }
    return payload, manifest_valid and contracts_valid


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

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{spec.source_id}-{spec.revision}.archive"
    # A wrong cached archive is never overwritten automatically.  This is a
    # deliberate fail-closed rule: silently replacing a file that may be part
    # of a prior receipt would make the provenance of a later extraction
    # ambiguous.  The operator must remove or quarantine the bad cache after
    # reviewing it.
    if archive_path.is_file():
        try:
            cached_digest = _sha256_path(archive_path)
        except OSError as exc:
            return {
                "source_id": spec.source_id,
                "status": "failed",
                "reason": f"cached_archive_unreadable:{exc}",
                "network_used": False,
                "acquisition_mode": "cache",
            }
        if cached_digest != spec.source_archive_sha256:
            return {
                "source_id": spec.source_id,
                "status": "failed",
                "reason": "cached source archive digest mismatch",
                "network_used": False,
                "acquisition_mode": "cache",
                "archive_sha256": cached_digest,
                "expected_archive_sha256": spec.source_archive_sha256,
            }
        digest = cached_digest
        acquisition_mode = "cache"
        network_used = False
    elif not allow_network:
        return {
            "source_id": spec.source_id,
            "status": "blocked_external",
            "reason": "network acquisition requires --allow-network",
            "network_used": False,
            "acquisition_mode": "absent",
        }
    else:
        # Download into a same-directory temporary file, verify its digest,
        # and publish it with os.replace only after verification.  A network
        # failure or hash mismatch therefore cannot truncate a previously
        # valid archive or leave a partially downloaded archive at the
        # canonical path.
        request = Request(
            spec.repository_url, headers={"User-Agent": "active-assembly-research"}
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{spec.source_id}-",
                suffix=".download",
                dir=destination,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with urlopen(request, timeout=30) as response:  # noqa: S310
                    shutil.copyfileobj(response, temporary)
            digest = _sha256_path(temporary_path)
            if digest != spec.source_archive_sha256:
                return {
                    "source_id": spec.source_id,
                    "status": "failed",
                    "reason": "source archive hash mismatch",
                    "network_used": True,
                    "acquisition_mode": "network",
                    "archive_sha256": digest,
                    "expected_archive_sha256": spec.source_archive_sha256,
                }
            os.replace(temporary_path, archive_path)
            temporary_path = None
        except (OSError, URLError) as exc:
            return {
                "source_id": spec.source_id,
                "status": "failed",
                "reason": f"acquisition_failed:{exc}",
                "network_used": True,
                "acquisition_mode": "network",
            }
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        acquisition_mode = "network"
        network_used = True

    extraction = destination / spec.source_id
    if extraction.is_symlink():
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": "extraction destination may not be a symbolic link",
            "network_used": network_used,
            "acquisition_mode": acquisition_mode,
            "archive_path": str(archive_path),
            "archive_sha256": digest,
        }
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.source_id}-", dir=destination))
    try:
        _safe_extract(archive_path, staging)
        license_path = staging / spec.license_file
        license_digest = _sha256_path(license_path) if license_path.is_file() else None
        if license_digest != spec.license_file_sha256:
            return {
                "source_id": spec.source_id,
                "status": "failed",
                "reason": "license file hash mismatch or missing",
                "network_used": network_used,
                "acquisition_mode": acquisition_mode,
                "archive_path": str(archive_path),
                "archive_sha256": digest,
                "license_file_sha256": license_digest,
            }
        if extraction.exists():
            shutil.rmtree(extraction)
        os.replace(staging, extraction)
        staging = Path()
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return {
            "source_id": spec.source_id,
            "status": "failed",
            "reason": f"safe_extraction_failed:{exc}",
            "network_used": network_used,
            "acquisition_mode": acquisition_mode,
            "archive_path": str(archive_path),
            "archive_sha256": digest,
        }
    finally:
        # ``Path()`` is used only as a local sentinel after a successful
        # os.replace; no filesystem operation is attempted for it.
        if staging != Path():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "source_id": spec.source_id,
        "status": "validated",
        "network_used": network_used,
        "acquisition_mode": acquisition_mode,
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
        "network_used": any(result.get("network_used") is True for result in results),
        "rows": results,
    }


def external_acquisition_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    manifest_path: Path | None = None,
    destination: Path | None = None,
    allow_network: bool = False,
) -> dict[str, object]:
    """Return a source-bound receipt for one explicit acquisition attempt.

    The acquisition receipt is deliberately separate from static validation:
    an archive can be pinned and safely extracted without implying that an
    emitted object is executable or that a source-level label is valid.  Cache
    reuse is recorded as non-network evidence, while an absent cache remains a
    visible ``blocked_external`` result when network opt-in is omitted.
    """

    resolved_manifest = manifest_path or project_root / EXTERNAL_MANIFEST_PATH
    resolved_destination = destination or project_root / "output" / "external_sources"
    manifest = load_external_manifest(resolved_manifest)
    result = acquire_external_manifest(
        resolved_manifest, resolved_destination, allow_network=allow_network
    )
    protocol = scientific_protocol_payload()
    input_hashes: dict[str, str] = {
        EXTERNAL_MANIFEST_PATH: sha256_path(resolved_manifest),
        "src/external_sources.py": sha256_path(Path(__file__)),
    }
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=input_hashes,
        toolchain_versions=toolchain_versions(("pyyaml",)),
        status=str(result["status"]),
        limitations=(
            "Acquisition is a pinned source-ingress receipt; static compilation and source-level labels are separate evidence surfaces.",
            "Cache reuse is valid only when the cached archive digest equals the manifest digest.",
        ),
        unsupported_cases=(
            "Arbitrary URLs, mutable revisions, unsafe archive links, and path traversal are rejected.",
        ),
        underpowered_cases=(
            "Acquisition and safe extraction do not establish behavioral equivalence or application effectiveness.",
        ),
    )
    return {
        **envelope,
        "schema_version": EXTERNAL_MANIFEST_SCHEMA_VERSION,
        "artifact": EXTERNAL_ACQUISITION_ARTIFACT,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_path(resolved_manifest),
        "destination": str(resolved_destination),
        "allow_network": allow_network,
        "acquisition": result,
        "network_used": result.get("network_used") is True,
        "execution_performed": False,
        "static_only": True,
    }


def validate_external_acquisition_payload(payload: object) -> tuple[bool, str]:
    """Validate acquisition receipt structure and the no-execution boundary."""

    if not isinstance(payload, dict):
        return False, "external acquisition receipt must be an object"
    protocol = scientific_protocol_payload()
    valid, detail = validate_receipt_envelope(payload, expected_protocol=protocol)
    if not valid:
        return False, f"external acquisition receipt: {detail}"
    if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
        return False, "external acquisition receipt schema mismatch"
    if payload.get("execution_performed") is not False:
        return False, "external acquisition receipt records execution"
    if payload.get("static_only") is not True:
        return False, "external acquisition receipt is not static-only"
    if (
        not isinstance(payload.get("manifest_id"), str)
        or not str(payload["manifest_id"]).strip()
    ):
        return False, "external acquisition manifest_id is missing"
    for key in ("manifest_sha256",):
        value = payload.get(key)
        if not isinstance(value, str) or not _is_hash(value):
            return False, f"external acquisition {key} is invalid"
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, dict):
        return False, "external acquisition result is missing"
    if acquisition.get("status") not in {
        "validated",
        "blocked_external",
        "failed",
    }:
        return False, "external acquisition result has an unknown status"
    if acquisition.get("execution_performed") is True:
        return False, "external acquisition attempted execution"
    rows = acquisition.get("rows")
    if not isinstance(rows, list):
        return False, "external acquisition rows must be a list"
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in {
            "validated",
            "failed",
            "blocked_external",
        }:
            return False, "external acquisition row has an unknown status"
        if row.get("network_used") not in {True, False}:
            return False, "external acquisition row network flag is invalid"
    if payload.get("status") == "validated" and (
        not rows or any(row.get("status") != "validated" for row in rows)
    ):
        return False, "validated acquisition receipt contains a nonvalidated row"
    if payload.get("status") == "blocked_external" and payload.get("network_used"):
        return False, "blocked acquisition receipt cannot report network use"
    return True, f"rows={len(rows)}"


def validate_external_acquisition_freshness(
    payload: object, project_root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    """Validate an acquisition receipt against local pinned cache state.

    This check is intentionally offline.  It never retries a URL and never
    treats a source archive's presence as evidence that its emitted object was
    executed or that a source-level label is correct.  It only confirms that a
    receipt still names the current manifest/producer and that every row
    recorded as acquired remains backed by the expected archive and license
    bytes.
    """

    valid, detail = validate_external_acquisition_payload(payload)
    if not valid:
        return False, detail
    if not isinstance(payload, dict):
        return False, "external acquisition receipt must be an object"
    manifest_path = project_root / EXTERNAL_MANIFEST_PATH
    source_path = project_root / "src" / "external_sources.py"
    if not manifest_path.is_file():
        return False, "pinned external manifest is missing"
    if not source_path.is_file():
        return False, "external acquisition producer is missing"
    if payload.get("manifest_sha256") != _sha256_path(manifest_path):
        return False, "external acquisition manifest hash is stale"
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        return False, "external acquisition input hashes are missing"
    if input_hashes.get(EXTERNAL_MANIFEST_PATH) != _sha256_path(manifest_path):
        return False, "external acquisition manifest input hash is stale"
    if input_hashes.get("src/external_sources.py") != _sha256_path(source_path):
        return False, "external acquisition producer input hash is stale"
    manifest = load_external_manifest(manifest_path)
    specs = {
        spec.source_id: spec
        for index, row in enumerate(cast(list[dict[str, object]], manifest["sources"]))
        for spec in (_spec_from_row(row, index),)
    }
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, dict):
        return False, "external acquisition result is missing"
    rows = acquisition.get("rows")
    if not isinstance(rows, list):
        return False, "external acquisition rows are missing"
    if acquisition.get("status") == "validated" and len(rows) != len(specs):
        return False, "validated acquisition receipt does not cover the manifest"
    destination = project_root / "output" / "external_sources"
    for row in rows:
        if not isinstance(row, dict):
            return False, "external acquisition row is not an object"
        source_id = row.get("source_id")
        spec = specs.get(str(source_id))
        if spec is None:
            return False, f"external acquisition row names unknown source {source_id!r}"
        if row.get("status") != "validated":
            continue
        archive = destination / f"{spec.source_id}-{spec.revision}.archive"
        if not archive.is_file() or _sha256_path(archive) != spec.source_archive_sha256:
            return False, f"acquired archive is stale or missing for {spec.source_id}"
        extraction = destination / spec.source_id
        license_path = extraction / spec.license_file
        if (
            not license_path.is_file()
            or _sha256_path(license_path) != spec.license_file_sha256
        ):
            return (
                False,
                f"acquired license evidence is stale or missing for {spec.source_id}",
            )
    return True, f"offline cache freshness verified for {len(rows)} row(s)"


def write_external_acquisition_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    manifest_path: Path | None = None,
    destination: Path | None = None,
    allow_network: bool = False,
) -> Path:
    """Write the explicit source-acquisition receipt atomically."""

    payload = external_acquisition_payload(
        project_root,
        manifest_path=manifest_path,
        destination=destination,
        allow_network=allow_network,
    )
    valid, detail = validate_external_acquisition_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / EXTERNAL_ACQUISITION_ARTIFACT
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        status=str(payload["status"]),
        limitations=tuple(
            str(item) for item in cast(list[object], payload["limitations"])
        ),
        unsupported_cases=tuple(
            str(item) for item in cast(list[object], payload["unsupported_cases"])
        ),
        underpowered_cases=tuple(
            str(item) for item in cast(list[object], payload["underpowered_cases"])
        ),
    )
    return path


def static_compile_external_source(
    spec: ExternalSourceSpec,
    source_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Compile one pinned source translation unit without executing output."""

    command = list(spec.build_command)
    if Path(command[0]).name not in _ALLOWED_COMPILERS or "-c" not in command:
        raise ValueError("external build command is not an allow-listed static compile")
    # Resolve the exact executable before invoking it. A missing compiler is
    # an unavailable capability, not a failed compilation and never a reason
    # to silently fall back to another compiler on PATH. Absolute paths are
    # checked directly so a pinned path cannot drift through PATH lookup.
    compiler_candidate = Path(command[0])
    compiler_path = (
        str(compiler_candidate)
        if compiler_candidate.is_absolute() and compiler_candidate.is_file()
        else shutil.which(command[0])
        if not compiler_candidate.is_absolute()
        else None
    )
    if compiler_path is None:
        return {
            "source_id": spec.source_id,
            "project_id": spec.project_id or spec.source_id,
            "revision_label": spec.revision_label or spec.revision[:12],
            "source_family": spec.source_family,
            "target_triple": spec.target_triple,
            "status": "unavailable",
            "reason": "compiler_unavailable",
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "execution_performed": False,
            "static_only": True,
            "compiler": spec.compiler,
            "generated_files": {},
        }
    command[0] = compiler_path
    if "-target" not in command and "--target" not in command:
        command[1:1] = ["-target", spec.target_triple]
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{spec.source_id}.o"
    generated_files: dict[str, str] = {}
    if any(token.endswith("/png.c") or token == "png.c" for token in command):
        prebuilt_candidates = list(source_root.rglob("scripts/pnglibconf.h.prebuilt"))
        if len(prebuilt_candidates) == 1:
            generated_header = prebuilt_candidates[0].parent.parent / "pnglibconf.h"
            shutil.copyfile(prebuilt_candidates[0], generated_header)
            generated_files[str(generated_header.relative_to(source_root))] = (
                _sha256_path(generated_header)
            )
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
        "project_id": spec.project_id or spec.source_id,
        "revision_label": spec.revision_label or spec.revision[:12],
        "source_family": spec.source_family,
        "target_triple": spec.target_triple,
        "status": "validated"
        if completed.returncode == 0 and output_path.is_file()
        else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "execution_performed": False,
        "static_only": True,
        "compiler": spec.compiler,
        "generated_files": generated_files,
    }
    if output_path.is_file():
        result["object_sha256"] = _sha256_path(output_path)
        result["object_path"] = str(output_path)
        disassembler = shutil.which("llvm-objdump") or shutil.which("objdump")
        if disassembler is None:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "static_disassembler_unavailable",
                }
            )
            return result
        architecture = (
            "aarch64"
            if spec.target_triple.startswith(("aarch64", "arm64"))
            else "x86_64"
        )
        disassembly = subprocess.run(
            [disassembler, "-d", str(output_path)],
            capture_output=True,
            text=True,
            shell=False,
            timeout=120,
            check=False,
        )
        result["disassembler"] = disassembler
        result["disassembly_returncode"] = disassembly.returncode
        result["disassembly_output_sha256"] = hashlib.sha256(
            disassembly.stdout.encode("utf-8")
        ).hexdigest()
        if disassembly.returncode != 0:
            result.update(
                {
                    "status": "failed",
                    "reason": "static_disassembler_failed",
                    "disassembly_stderr": disassembly.stderr,
                }
            )
            return result
        try:
            text_bytes = _extract_text_bytes(disassembly.stdout, architecture)
            decoded = _decode_bytes(text_bytes, architecture)
        except (ValueError, RuntimeError) as exc:
            result.update(
                {
                    "status": "unsupported",
                    "reason": f"static_capstone_decode_failed:{exc}",
                }
            )
            return result
        result.update(
            {
                "status": "validated",
                "architecture": architecture,
                "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "disassembly_sha256": hashlib.sha256(
                    json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "decoded_instruction_count": len(decoded),
                "decoded_instructions": decoded,
            }
        )
    return result


def _compiler_probe(compiler: str) -> dict[str, object]:
    """Probe one compiler and verify targets with a temporary static object."""

    executable = shutil.which(compiler)
    if executable is None:
        return {
            "compiler": compiler,
            "status": "unavailable",
            "executable": None,
            "targets": {},
        }
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    target_probe = subprocess.run(
        [executable, "-print-targets"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    advertised = set(
        re.findall(r"^\s+([A-Za-z0-9_+-]+)\s+-", target_probe.stdout, re.MULTILINE)
    )
    targets = {
        "x86_64-apple-darwin": "x86-64" in advertised or "x86" in advertised,
        "aarch64-apple-darwin": "aarch64" in advertised or "arm64" in advertised,
        "riscv64-unknown-elf": "riscv64" in advertised,
    }
    with tempfile.TemporaryDirectory(prefix="active-assembly-capability-") as directory:
        source = Path(directory) / "probe.c"
        source.write_text(
            "unsigned long probe(unsigned long x) { return x + 1; }\n", encoding="utf-8"
        )
        for target in targets:
            object_path = Path(directory) / (target.replace("-", "_") + ".o")
            completed = subprocess.run(
                [
                    executable,
                    "-target",
                    target,
                    "-c",
                    "-x",
                    "c",
                    "-o",
                    str(object_path),
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            targets[target] = bool(completed.returncode == 0 and object_path.is_file())
    return {
        "compiler": compiler,
        "status": "validated" if version.returncode == 0 else "failed",
        "executable": executable,
        "version": version.stdout.splitlines()[0].strip()
        if version.stdout
        else version.stderr.strip(),
        "advertised_targets": sorted(advertised),
        "targets": targets,
        "stderr": target_probe.stderr.strip(),
    }


def compiler_capabilities_payload() -> dict[str, object]:
    """Return a deterministic local compiler/target capability receipt."""

    rows = [_compiler_probe(name) for name in ("clang", "gcc")]
    return {
        "schema_version": EXTERNAL_MANIFEST_SCHEMA_VERSION,
        "artifact": EXTERNAL_CAPABILITY_ARTIFACT,
        "status": "validated"
        if any(row["status"] == "validated" for row in rows)
        else "unavailable",
        "platform": platform.platform(),
        "rows": rows,
        "limitations": [
            "A target is counted only after a real static object was produced.",
            "Capability probing does not execute the object or establish compiler correctness.",
        ],
    }


def write_compiler_capabilities(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the local capability receipt without network or external execution."""

    from .artifact_contract import write_json_artifact

    path = project_root / EXTERNAL_CAPABILITY_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = compiler_capabilities_payload()
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        status=str(payload["status"]),
        limitations=tuple(
            str(item) for item in cast(list[object], payload["limitations"])
        ),
    )
    return path


def _load_static_rows(project_root: Path) -> list[dict[str, object]]:
    """Read the separately generated local static-validation rows."""

    path = project_root / EXTERNAL_STATIC_RECEIPT_ARTIFACT
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    return (
        [cast(dict[str, object], row) for row in rows if isinstance(row, dict)]
        if isinstance(rows, list)
        else []
    )


def validate_external_static_sources(
    project_root: Path = PROJECT_ROOT,
    *,
    destination: Path | None = None,
) -> dict[str, object]:
    """Compile acquired source rows statically and retain every failure reason."""

    manifest = load_external_manifest(project_root / EXTERNAL_MANIFEST_PATH)
    source_destination = destination or project_root / "output" / "external_sources"
    object_destination = project_root / "output" / "external_objects"
    rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(cast(list[dict[str, object]], manifest["sources"])):
        spec = _spec_from_row(raw_row, index)
        source_root = source_destination / spec.source_id
        if not source_root.is_dir():
            rows.append(
                {
                    "source_id": spec.source_id,
                    "project_id": spec.project_id,
                    "revision_label": spec.revision_label,
                    "source_family": spec.source_family,
                    "status": "unavailable",
                    "reason": "source_not_acquired",
                    "static_only": True,
                    "execution_performed": False,
                }
            )
            continue
        try:
            result = static_compile_external_source(
                spec, source_root, object_destination
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result = {
                "source_id": spec.source_id,
                "project_id": spec.project_id,
                "revision_label": spec.revision_label,
                "source_family": spec.source_family,
                "status": "failed",
                "reason": f"static_compile_failed:{exc}",
                "static_only": True,
                "execution_performed": False,
            }
        rows.append(result)
    return {
        "schema_version": EXTERNAL_MANIFEST_SCHEMA_VERSION,
        "artifact": EXTERNAL_STATIC_RECEIPT_ARTIFACT,
        "status": "validated"
        if rows and all(row.get("status") == "validated" for row in rows)
        else "underpowered",
        "static_only": True,
        "execution_performed": False,
        "rows": rows,
        "limitations": [
            "Static rows establish source/build/disassembly coverage only; they do not create equivalence labels.",
            "External emitted objects are never executed.",
        ],
    }


def write_external_static_validation(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the static-only external compilation receipt."""

    from .artifact_contract import write_json_artifact

    payload = validate_external_static_sources(project_root)
    path = project_root / EXTERNAL_STATIC_RECEIPT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        status=str(payload["status"]),
        limitations=tuple(
            str(item) for item in cast(list[object], payload["limitations"])
        ),
        unsupported_cases=("No external emitted object is executed.",),
        underpowered_cases=(
            "Source-bound equivalence labels are maintained separately from static coverage.",
        ),
    )
    return path


def validate_external_static_validation_payload(payload: object) -> tuple[bool, str]:
    """Validate the static-only compilation receipt independently."""

    if not isinstance(payload, dict):
        return False, "external static receipt must be an object"
    if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
        return False, "external static receipt schema mismatch"
    if payload.get("static_only") is not True:
        return False, "external static receipt is not static-only"
    if payload.get("execution_performed") is not False:
        return False, "external static receipt records execution"
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False, "external static receipt rows must be a list"
    for row in rows:
        if not isinstance(row, dict):
            return False, "external static receipt row is not an object"
        if row.get("status") not in {
            "validated",
            "unavailable",
            "failed",
            "unsupported",
        }:
            return False, "external static receipt row has an unknown status"
        if row.get("execution_performed") is not False:
            return False, "external static receipt row records execution"
    return True, f"rows={len(rows)}"


def validate_compiler_capabilities_payload(payload: object) -> tuple[bool, str]:
    """Validate the local compiler capability receipt."""

    if not isinstance(payload, dict):
        return False, "external capability receipt must be an object"
    if payload.get("schema_version") != EXTERNAL_MANIFEST_SCHEMA_VERSION:
        return False, "external capability receipt schema mismatch"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return False, "external capability receipt rows are empty"
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("targets"), dict):
            return False, "external capability row is malformed"
        if any(not isinstance(value, bool) for value in row["targets"].values()):
            return False, "external capability target values must be boolean"
    return True, f"compilers={len(rows)}"


def external_source_validation_payload(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Return a receipt for the pinned manifest without network access."""

    manifest_path = project_root / EXTERNAL_MANIFEST_PATH
    manifest = load_external_manifest(manifest_path)
    protocol = scientific_protocol_payload()
    rows = cast(list[dict[str, object]], manifest["sources"])
    static_rows = _load_static_rows(project_root)
    label_summary = label_contract_summary(project_root)

    def _row_text(row: dict[str, object], key: str, fallback: str = "") -> str:
        value = row.get(key)
        return value if isinstance(value, str) else fallback

    project_ids = sorted(
        {_row_text(row, "project_id", _row_text(row, "source_id")) for row in rows}
    )
    revision_counts: dict[str, int] = {
        project: len(
            {
                _row_text(
                    row,
                    "revision_label",
                    _row_text(row, "revision")[:12],
                )
                for row in rows
                if _row_text(row, "project_id", _row_text(row, "source_id")) == project
            }
        )
        for project in project_ids
    }
    static_validated = [row for row in static_rows if row.get("status") == "validated"]
    labelable_value = label_summary.get("labelable_example_count", 0)
    labelable_count = (
        int(labelable_value)
        if isinstance(labelable_value, (int, float))
        and not isinstance(labelable_value, bool)
        else 0
    )
    gate = {
        "minimum_projects": 5,
        "minimum_revisions_per_project": 2,
        "minimum_labelable_examples": 100,
        "project_count_pass": len(project_ids) >= 5,
        "revision_count_pass": bool(revision_counts)
        and all(count >= 2 for count in revision_counts.values()),
        "labelable_example_count_pass": labelable_count >= 100,
        # This is the scientifically meaningful gate: every test project must
        # be absent from both fitting partitions.  The legacy
        # ``project_disjoint_test`` field is retained only for compatibility
        # with older receipts and is intentionally not used here.
        "project_held_out_pass": bool(label_summary.get("project_held_out_test")),
        "license_provenance_pass": bool(rows)
        and all(
            row.get("license_file_sha256") and row.get("source_archive_sha256")
            for row in rows
        ),
        "reproducible_static_build_pass": bool(static_validated)
        and len(static_validated) == len(rows),
        "static_only_pass": all(
            row.get("static_only") is True and row.get("execution_performed") is False
            for row in static_rows
        ),
        "independent_replay_pass": False,
    }
    status = (
        "validated"
        if all(gate[key] for key in gate if key.endswith("_pass"))
        else "underpowered"
        if static_rows
        else "blocked_external"
    )
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            EXTERNAL_MANIFEST_PATH: sha256_path(manifest_path),
            LABEL_CONTRACT_PATH: sha256_path(project_root / LABEL_CONTRACT_PATH),
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
            "Static coverage and source-level label contracts are separate evidence surfaces; no label is inferred from a compiled object.",
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
        "project_count": len(project_ids),
        "projects": project_ids,
        "revision_counts_by_project": revision_counts,
        "labelable_example_count": labelable_count,
        "label_contract_summary": label_summary,
        "static_validated_row_count": len(static_validated),
        "static_rows": static_rows,
        "promotion_gate": {**gate, "status": status},
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
    for key in (
        "project_count",
        "labelable_example_count",
        "static_validated_row_count",
    ):
        if not isinstance(payload.get(key), int) or int(payload[key]) < 0:
            return False, f"external source receipt {key} is invalid"
    if not isinstance(payload.get("static_rows"), list):
        return False, "external source receipt static_rows must be a list"
    gate = payload.get("promotion_gate")
    if not isinstance(gate, dict) or gate.get("status") != payload.get("status"):
        return False, "external source promotion gate status drift"
    label_summary = payload.get("label_contract_summary")
    if not isinstance(label_summary, dict):
        return False, "external source label-contract summary is missing"
    return True, "ok"


def write_external_source_validation_payload(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the offline external-source manifest receipt."""

    payload = external_source_validation_payload(project_root)
    valid, detail = validate_external_source_validation_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / EXTERNAL_RECEIPT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path


__all__ = [
    "EXTERNAL_MANIFEST_PATH",
    "EXTERNAL_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_RECEIPT_ARTIFACT",
    "EXTERNAL_STATIC_RECEIPT_ARTIFACT",
    "EXTERNAL_CAPABILITY_ARTIFACT",
    "EXTERNAL_ACQUISITION_ARTIFACT",
    "EXTERNAL_LABEL_CONTRACT_PATH",
    "ExternalSourceSpec",
    "acquire_external_manifest",
    "acquire_external_source",
    "external_acquisition_payload",
    "validate_external_acquisition_payload",
    "validate_external_acquisition_freshness",
    "write_external_acquisition_payload",
    "external_source_validation_payload",
    "compiler_capabilities_payload",
    "validate_compiler_capabilities_payload",
    "validate_external_static_validation_payload",
    "validate_external_static_sources",
    "write_compiler_capabilities",
    "write_external_static_validation",
    "load_external_manifest",
    "static_compile_external_source",
    "validate_external_manifest",
    "validate_external_source_validation_payload",
    "write_external_source_validation_payload",
]
