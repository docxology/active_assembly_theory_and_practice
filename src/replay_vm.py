"""Reproducible Lima VM orchestration for the label-blind replay surface.

The VM is deliberately a separate control boundary, not a new scientific
sample or an independent reimplementation. Provisioning may use the network
only when explicitly requested. The replay itself receives a clean source
archive and the four public bundle files, never the private gold directory,
and runs in a guest network namespace with no non-loopback interface.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Sequence
from urllib.request import Request, urlopen

import yaml

from .artifact_contract import write_json_atomic
from .provenance import sha256_path
from .replication import (
    PUBLIC_INSTRUCTIONS,
    PUBLIC_MANIFEST,
    PUBLIC_PROTOCOL,
    PUBLIC_TOOLCHAIN,
    _public_bundle_digest,
    ingest_independent_replay,
    validate_public_replay_bundle,
    validate_replay_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLAY_VM_CONFIG = "configs/replay_vm.yaml"
VM_SCHEMA_VERSION = "1.0.0"
VM_NAME = "aapta-replay"
VM_ROOT_ENV = "AAPTA_VM_ROOT"
DEFAULT_VM_ROOT = Path("/Volumes/blue/aapta-replay-qemu")
LIMA_VERSION = "2.2.0"
QEMU_BINARY = "qemu-system-aarch64"
QEMU_VERSION = "11.1.1"
UBUNTU_IMAGE_URL = (
    "https://cloud-images.ubuntu.com/releases/noble/release-20260705/"
    "ubuntu-24.04-server-cloudimg-arm64.img"
)
UBUNTU_IMAGE_SHA256 = "7df0201546f75b8bcc1044594c806c35749421ad3c9bc1be2a3ab806cfae39cc"
UV_VERSION = "0.10.7"
UV_ARCHIVE_URL = (
    "https://github.com/astral-sh/uv/releases/download/0.10.7/"
    "uv-aarch64-unknown-linux-gnu.tar.gz"
)
UV_ARCHIVE_SHA256 = "20efc27d946860093650bcf26096a016b10fdaf03b13c33b75fbde02962beea9"
GUEST_ROOT = "/opt/aapta-replay"
GUEST_REPOSITORY = f"{GUEST_ROOT}/repository"
GUEST_TOOLCHAIN = f"{GUEST_ROOT}/toolchain"
GUEST_REPLAY_OUTPUT = f"{GUEST_ROOT}/replay-output"
PUBLIC_BUNDLE_FILES = (
    PUBLIC_MANIFEST,
    PUBLIC_PROTOCOL,
    PUBLIC_TOOLCHAIN,
    PUBLIC_INSTRUCTIONS,
)
# The guest does not need the research data, private source contracts, or
# manuscript surfaces to replay the public CLI.  Keeping the transfer archive
# to this runtime allow-list makes the public/private boundary inspectable and
# prevents tracked generated data from crossing it accidentally.
SOURCE_ARCHIVE_INCLUDED_PATHS = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "src/",
    "scripts/",
)
SOURCE_ARCHIVE_EXCLUDED_PATHS = (
    "AGENTS.md",
    "configs/",
    "data/",
    "docs/",
    "figures/",
    "formal/",
    "julia/",
    "manuscript/",
    "tests/",
)


def vm_root() -> Path:
    """Return the required blue-backed VM root.

    AAPTA_VM_ROOT is provided for explicit test/CI overrides, but every
    operational path must still be below /Volumes/blue. This prevents a
    large VM disk or receipt transfer from being silently created on the
    system volume.
    """

    configured = os.environ.get(VM_ROOT_ENV)
    candidate = Path(configured) if configured else DEFAULT_VM_ROOT
    if not candidate.is_absolute():
        raise ValueError(f"{VM_ROOT_ENV} must be an absolute path")
    resolved = candidate.resolve()
    blue_root = Path("/Volumes/blue").resolve()
    try:
        resolved.relative_to(blue_root)
    except ValueError as exc:
        raise ValueError(f"VM root must be below {blue_root}") from exc
    if resolved == blue_root:
        raise ValueError("VM root may not be the blue volume root itself")
    return resolved


def vm_paths(root: Path | None = None) -> dict[str, Path]:
    """Return explicit host-side VM storage paths."""

    vm_storage_root = vm_root() if root is None else root.resolve()
    return {
        "root": vm_storage_root,
        "lima_home": vm_storage_root / "lima-home",
        "toolchain": vm_storage_root / "toolchain",
        "transfer": vm_storage_root / "transfer",
        "receipts": vm_storage_root / "receipts",
        "vm_manifest": vm_storage_root / "receipts" / "vm_manifest.json",
        "provision_receipt": vm_storage_root / "receipts" / "provision_receipt.json",
        "replay_receipt": vm_storage_root / "receipts" / "vm_replay_receipt.json",
        "pending_receipt": vm_storage_root
        / "receipts"
        / "external-replay-receipt.json",
        "predictions": vm_storage_root
        / "receipts"
        / "calibrated_interpretable_predictions.json",
        "attested_receipt": vm_storage_root
        / "receipts"
        / "vm-attested-replay-receipt.json",
        "network_proof": vm_storage_root / "receipts" / "network-proof.txt",
        "replay_stdout": vm_storage_root / "receipts" / "replay.stdout.txt",
        "replay_stderr": vm_storage_root / "receipts" / "replay.stderr.txt",
    }


def _sha256(path: Path) -> str:
    return sha256_path(path)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_replay_vm_config(
    path: Path = PROJECT_ROOT / REPLAY_VM_CONFIG,
) -> dict[str, Any]:
    """Load and validate the checked-in Lima configuration."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"replay VM config is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("replay VM config must be an object")
    expected = {
        "minimumLimaVersion": LIMA_VERSION,
        "vmType": "qemu",
        "arch": "aarch64",
        "os": "Linux",
        "cpus": 4,
        "memory": "8GiB",
        "disk": "64GiB",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"replay VM config field {key} is not pinned to {value!r}")
    images = payload.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("replay VM config must contain exactly one pinned image")
    image = images[0]
    if not isinstance(image, dict):
        raise ValueError("replay VM image entry must be an object")
    if image.get("location") != UBUNTU_IMAGE_URL:
        raise ValueError("replay VM image URL drifted")
    if image.get("arch") != "aarch64" or image.get("variant") != "server":
        raise ValueError("replay VM image architecture or variant drifted")
    if image.get("digest") != f"sha256:{UBUNTU_IMAGE_SHA256}":
        raise ValueError("replay VM image digest drifted")
    if payload.get("mounts") != []:
        raise ValueError("replay VM must not mount the host checkout")
    containerd = payload.get("containerd")
    if (
        not isinstance(containerd, dict)
        or containerd.get("system") is not False
        or containerd.get("user") is not False
    ):
        raise ValueError("replay VM containerd must be disabled")
    if payload.get("networks") != []:
        raise ValueError("replay VM additional networks must remain empty")
    if payload.get("propagateProxyEnv") is not False:
        raise ValueError("replay VM proxy propagation must be disabled")
    return payload


def _limactl_path() -> str:
    path = shutil.which("limactl")
    if path is None:
        raise RuntimeError(
            "limactl is unavailable; install the pinned Lima formula before VM setup"
        )
    return path


def _qemu_version() -> str:
    """Return the exact pinned ARM64 QEMU version used by Lima."""

    path = shutil.which(QEMU_BINARY)
    if path is None:
        raise RuntimeError(
            f"{QEMU_BINARY} is unavailable; install the pinned QEMU formula before VM setup"
        )
    result = subprocess.run(
        [path, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect {QEMU_BINARY}: {result.stderr.strip()}")
    version_line = result.stdout.splitlines()[0] if result.stdout else ""
    if f"QEMU emulator version {QEMU_VERSION}" not in version_line:
        raise RuntimeError(
            f"{QEMU_BINARY} version drifted: expected {QEMU_VERSION}, got {version_line!r}"
        )
    return version_line.strip()


def _lima_environment(paths: dict[str, Path]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["LIMA_HOME"] = str(paths["lima_home"])
    return environment


def _run_lima(
    arguments: Sequence[str],
    paths: dict[str, Path],
    *,
    input_text: str | None = None,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [_limactl_path(), *arguments]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_lima_environment(paths),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Lima command failed ({result.returncode}): {' '.join(command)}\n{detail}"
        )
    return result


def _instance_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    result = _run_lima(["list", "--format", "json"], paths, check=False, timeout=30.0)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "No instance found" in detail:
            return []
        raise RuntimeError(f"could not inspect Lima instances: {detail}")
    if not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Lima instance listing was not JSON") from exc
    # Lima 2.2.0 emits one JSON object when the configured LIMA_HOME contains
    # one instance, while some versions/commands emit a JSON list.  Accept
    # both representations but keep the shape check strict so a malformed
    # response cannot be interpreted as an absent or healthy VM.
    if isinstance(payload, dict):
        if not payload.get("name"):
            raise RuntimeError("Lima instance listing was not a named JSON object")
        return [payload]
    if not isinstance(payload, list):
        raise RuntimeError("Lima instance listing was not a JSON list or object")
    if not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("Lima instance listing contained a non-object row")
    return payload


def vm_status(paths: dict[str, Path]) -> dict[str, Any] | None:
    """Return the current named-instance status, or None if absent."""

    for row in _instance_rows(paths):
        if row.get("name") == VM_NAME:
            return row
    return None


def _blue_usage() -> dict[str, int]:
    usage = shutil.disk_usage("/Volumes/blue")
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def write_vm_manifest(
    project_root: Path = PROJECT_ROOT,
    *,
    status: str = "implemented_local",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write an auditable host/VM configuration receipt on blue."""

    config_path = project_root / REPLAY_VM_CONFIG
    load_replay_vm_config(config_path)
    paths = vm_paths()
    paths["receipts"].mkdir(parents=True, exist_ok=True)
    lima_version = _run_lima(["--version"], paths, timeout=30.0).stdout.strip()
    instance = vm_status(paths)
    payload: dict[str, Any] = {
        "schema_version": VM_SCHEMA_VERSION,
        "receipt_kind": "blue_backed_lima_vm_manifest",
        "status": status,
        "created_at": _iso_now(),
        "vm_name": VM_NAME,
        "vm_root": str(paths["root"]),
        "lima_home": str(paths["lima_home"]),
        "config_path": REPLAY_VM_CONFIG,
        "config_sha256": _sha256(config_path),
        "lima_version": lima_version,
        "qemu_version": _qemu_version(),
        "host": {
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "blue_volume": _blue_usage(),
        "instance": instance,
        "image": {
            "location": UBUNTU_IMAGE_URL,
            "sha256": UBUNTU_IMAGE_SHA256,
            "architecture": "aarch64",
        },
        "virtualization": {
            "backend": "qemu",
            "binary": QEMU_BINARY,
            "version": QEMU_VERSION,
        },
        "storage_contract": {
            "lima_home_on_blue": True,
            "host_mounts": False,
            "containerd": False,
            "disk_gib": 64,
        },
        "replay_contract": {
            "public_bundle_only": True,
            "private_gold_transferred": False,
            "network_namespace_required": True,
            "external_binaries_executed": False,
            "control_scope": "same_host_separately_controlled_guest",
        },
        "limitations": [
            "This is a separately controlled guest on the same physical host, not an independent machine.",
            "It is replay evidence for the frozen interface, not independent reimplementation or new data collection.",
        ],
    }
    if extra:
        payload.update(deepcopy(extra))
    write_json_atomic(paths["vm_manifest"], payload)
    return paths["vm_manifest"]


def create_replay_vm(
    project_root: Path = PROJECT_ROOT,
    *,
    allow_network: bool = False,
) -> Path:
    """Create/start the pinned VM, requiring explicit network permission initially."""

    config_path = project_root / REPLAY_VM_CONFIG
    load_replay_vm_config(config_path)
    paths = vm_paths()
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["lima_home"].mkdir(parents=True, exist_ok=True)
    paths["receipts"].mkdir(parents=True, exist_ok=True)
    existing_manifest = paths["vm_manifest"]
    if existing_manifest.is_file():
        try:
            prior = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing VM manifest is unreadable: {exc}") from exc
        if prior.get("config_sha256") != _sha256(config_path):
            raise ValueError(
                "VM configuration changed; use a new blue-backed VM root instead of mutating the instance"
            )
    instance = vm_status(paths)
    if instance is None:
        if not allow_network:
            raise RuntimeError(
                "initial VM image acquisition requires explicit --allow-network"
            )
        result = _run_lima(
            [
                "start",
                "--yes",
                "--name",
                VM_NAME,
                "--timeout",
                "10m",
                str(config_path),
            ],
            paths,
            timeout=900.0,
        )
        (paths["receipts"] / "create.stdout.txt").write_text(
            result.stdout, encoding="utf-8"
        )
        (paths["receipts"] / "create.stderr.txt").write_text(
            result.stderr, encoding="utf-8"
        )
    elif str(instance.get("status", "")).lower() != "running":
        _run_lima(["start", "--yes", VM_NAME], paths, timeout=300.0)
    return write_vm_manifest(project_root, status="implemented_local")


def stop_replay_vm(project_root: Path = PROJECT_ROOT) -> Path:
    """Stop the VM without deleting its disk or receipts."""

    paths = vm_paths()
    instance = vm_status(paths)
    if instance is not None and str(instance.get("status", "")).lower() == "running":
        _run_lima(["stop", VM_NAME], paths, timeout=300.0)
    return write_vm_manifest(project_root, status="implemented_local")


def _download_pinned_uv(paths: dict[str, Path], *, allow_network: bool) -> Path:
    paths["toolchain"].mkdir(parents=True, exist_ok=True)
    archive_path = paths["toolchain"] / f"uv-{UV_VERSION}-aarch64-linux.tar.gz"
    if archive_path.is_file():
        if _sha256(archive_path) != UV_ARCHIVE_SHA256:
            raise ValueError(
                "cached uv archive digest does not match the pinned release"
            )
        return archive_path
    if not allow_network:
        raise RuntimeError(
            "pinned uv archive is absent; rerun provisioning with explicit --allow-network"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="uv-", suffix=".tmp", dir=paths["toolchain"], delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(
                UV_ARCHIVE_URL,
                headers={"User-Agent": "active-assembly-replay-vm/1.0"},
            )
            with urlopen(request, timeout=120.0) as response:  # noqa: S310
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
        if _sha256(temporary_path) != UV_ARCHIVE_SHA256:
            raise ValueError(
                "downloaded uv archive digest does not match the pinned release"
            )
        os.replace(temporary_path, archive_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return archive_path


def _git_output(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _deterministic_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))


def _runtime_source_archive(
    project_root: Path, source_archive: Path, source_revision: str
) -> None:
    """Create a deterministic runtime-only archive from the committed tree.

    A full ``git archive`` is intentionally not transferred: this repository
    tracks generated data and source-bound label contracts that are useful to
    the owner-side scientific pipeline but are outside the public replay
    surface.  The archive is filtered by an explicit allow-list and checked
    for unsafe members before it is repacked.
    """

    raw_archive = source_archive.with_suffix(".git-archive.tar")
    try:
        with raw_archive.open("wb") as handle:
            result = subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    "--prefix=repository/",
                    source_revision,
                ],
                cwd=project_root,
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace")
            raise RuntimeError(f"could not create source archive: {detail}")
        members: dict[str, bytes] = {}
        with tarfile.open(raw_archive, mode="r:") as archive:
            for member in archive.getmembers():
                name = member.name
                if name == "repository":
                    continue
                if not name.startswith("repository/"):
                    raise ValueError(f"source archive member escapes prefix: {name!r}")
                relative = name.removeprefix("repository/")
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"source archive member is unsafe: {name!r}")
                if member.issym() or member.islnk():
                    raise ValueError(f"source archive contains a link: {name!r}")
                if member.isdir():
                    continue
                included = any(
                    relative == prefix or relative.startswith(prefix)
                    for prefix in SOURCE_ARCHIVE_INCLUDED_PATHS
                )
                if not included:
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"source archive member is not a regular file: {name!r}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"source archive member cannot be read: {name!r}")
                members[name] = extracted.read()
        required = {
            "repository/README.md",
            "repository/pyproject.toml",
            "repository/uv.lock",
        }
        missing = sorted(required - members.keys())
        if missing:
            raise ValueError(
                f"runtime source archive is missing required files: {missing}"
            )
        _deterministic_tar(source_archive, members)
    finally:
        raw_archive.unlink(missing_ok=True)


def prepare_public_guest_bundle(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create clean, label-blind transfer archives for the guest."""

    public_valid, public_detail = validate_public_replay_bundle(project_root)
    if not public_valid:
        raise ValueError(public_detail)
    dirty = _git_output(project_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(
            "VM source transfer requires a clean committed checkout; commit or isolate all source changes first"
        )
    source_revision = _git_output(project_root, "rev-parse", "HEAD")
    paths = vm_paths()
    paths["transfer"].mkdir(parents=True, exist_ok=True)
    source_archive = paths["transfer"] / f"source-{source_revision}.tar"
    if not source_archive.is_file():
        try:
            _runtime_source_archive(project_root, source_archive, source_revision)
        except (OSError, RuntimeError, ValueError):
            source_archive.unlink(missing_ok=True)
            raise
    public_members: dict[str, bytes] = {}
    for relative_path in PUBLIC_BUNDLE_FILES:
        source_path = project_root / relative_path
        public_members[f"repository/{relative_path}"] = source_path.read_bytes()
    public_archive = paths["transfer"] / "public-bundle.tar"
    _deterministic_tar(public_archive, public_members)
    metadata_path = paths["transfer"] / "transfer_manifest.json"
    transfer = {
        "schema_version": VM_SCHEMA_VERSION,
        "source_revision": source_revision,
        "source_archive_sha256": _sha256(source_archive),
        "source_archive_included_paths": list(SOURCE_ARCHIVE_INCLUDED_PATHS),
        "source_archive_excluded_paths": list(SOURCE_ARCHIVE_EXCLUDED_PATHS),
        "source_archive_filter": "runtime_source_allowlist_v1",
        "public_archive_sha256": _sha256(public_archive),
        "public_bundle_sha256": _public_bundle_digest(project_root),
        "public_files": list(PUBLIC_BUNDLE_FILES),
        "private_gold_transferred": False,
        "source_bound_label_contract_transferred": False,
        "network_required_for_replay": False,
    }
    write_json_atomic(metadata_path, transfer)
    return {
        **transfer,
        "source_archive": source_archive,
        "public_archive": public_archive,
        "metadata_path": metadata_path,
    }


def _copy_to_guest(paths: dict[str, Path], source: Path, destination: str) -> None:
    _run_lima(
        ["copy", "--backend=scp", str(source), f"{VM_NAME}:{destination}"],
        paths,
        timeout=300.0,
    )


def _copy_from_guest(paths: dict[str, Path], source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_lima(
        ["copy", "--backend=scp", f"{VM_NAME}:{source}", str(destination)],
        paths,
        timeout=300.0,
    )


def _guest_shell(
    paths: dict[str, Path], script: str, *, timeout: float = 1200.0
) -> subprocess.CompletedProcess[str]:
    return _run_lima(
        ["shell", "--shell", "/bin/bash", VM_NAME, "bash", "-s"],
        paths,
        input_text=script,
        timeout=timeout,
    )


def provision_replay_vm(
    project_root: Path = PROJECT_ROOT,
    *,
    allow_network: bool = False,
) -> Path:
    """Provision the guest using the locked project environment."""

    if not allow_network:
        raise RuntimeError(
            "VM provisioning requires explicit --allow-network for image and dependency acquisition"
        )
    create_replay_vm(project_root, allow_network=True)
    paths = vm_paths()
    transfer = prepare_public_guest_bundle(project_root)
    uv_archive = _download_pinned_uv(paths, allow_network=True)
    _copy_to_guest(paths, uv_archive, "/tmp/aapta-uv.tar.gz")
    _copy_to_guest(paths, transfer["source_archive"], "/tmp/aapta-source.tar")
    _copy_to_guest(paths, transfer["public_archive"], "/tmp/aapta-public.tar")
    bootstrap = f"""#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates python3 python3-venv util-linux iproute2 build-essential
sudo install -d -m 0755 {GUEST_TOOLCHAIN} {GUEST_ROOT}
sudo install -m 0644 /tmp/aapta-uv.tar.gz {GUEST_ROOT}/uv.tar.gz
printf '%s  %s\\n' '{UV_ARCHIVE_SHA256}' '{GUEST_ROOT}/uv.tar.gz' | sudo sha256sum -c -
sudo tar --extract --gzip --file {GUEST_ROOT}/uv.tar.gz --directory {GUEST_TOOLCHAIN} --strip-components=1 --no-same-owner --no-same-permissions
test "$(sudo {GUEST_TOOLCHAIN}/uv --version)" = 'uv {UV_VERSION}'
if [ -e {GUEST_REPOSITORY} ]; then
  sudo find {GUEST_REPOSITORY} -xdev -depth -delete
fi
sudo install -d -m 0755 {GUEST_REPOSITORY}
sudo tar --extract --file /tmp/aapta-source.tar --directory {GUEST_ROOT} --no-same-owner --no-same-permissions
sudo tar --extract --file /tmp/aapta-public.tar --directory {GUEST_ROOT} --no-same-owner --no-same-permissions
sudo chown -R "$(id -u):$(id -g)" {GUEST_REPOSITORY}
test ! -e {GUEST_REPOSITORY}/output/replication/private
test ! -e {GUEST_REPOSITORY}/configs/external_label_contracts.yaml
test ! -e {GUEST_REPOSITORY}/data
{GUEST_TOOLCHAIN}/uv sync --locked --no-dev --python-preference only-system --directory {GUEST_REPOSITORY}
test -x {GUEST_REPOSITORY}/.venv/bin/python
{GUEST_REPOSITORY}/.venv/bin/python --version
{GUEST_TOOLCHAIN}/uv --version
"""
    result = _guest_shell(paths, bootstrap, timeout=1800.0)
    (paths["receipts"] / "provision.stdout.txt").write_text(
        result.stdout, encoding="utf-8"
    )
    (paths["receipts"] / "provision.stderr.txt").write_text(
        result.stderr, encoding="utf-8"
    )
    receipt = {
        "schema_version": VM_SCHEMA_VERSION,
        "receipt_kind": "blue_backed_lima_vm_provision",
        "status": "validated",
        "created_at": _iso_now(),
        "vm_name": VM_NAME,
        "vm_manifest_sha256": _sha256(paths["vm_manifest"]),
        "transfer": {
            key: value
            for key, value in transfer.items()
            if key not in {"source_archive", "public_archive", "metadata_path"}
        },
        "uv": {
            "version": UV_VERSION,
            "archive_url": UV_ARCHIVE_URL,
            "archive_sha256": UV_ARCHIVE_SHA256,
            "host_archive_sha256": _sha256(uv_archive),
        },
        "guest": {
            "source_path": GUEST_REPOSITORY,
            "runtime_tree_reset_before_extraction": True,
            "public_bundle_only": True,
            "private_gold_transferred": False,
            "network_used_for_provisioning": True,
            "replay_network_policy": "guest_network_namespace_only_loopback",
        },
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "limitations": [
            "Provisioning uses current Ubuntu archive packages and records the guest toolchain; replay is the offline evidence boundary.",
            "The guest is same-host separately controlled and is not an independent physical machine.",
        ],
    }
    write_json_atomic(paths["provision_receipt"], receipt)
    write_vm_manifest(project_root, status="validated", extra={"provisioned": True})
    return paths["provision_receipt"]


def replay_in_vm(
    project_root: Path = PROJECT_ROOT,
    *,
    model: str = "calibrated_interpretable",
    seed: int = 0,
) -> Path:
    """Run the public-only replay inside a guest network namespace."""

    if model not in {
        "calibrated_interpretable",
        "temporal_probabilistic",
        "neural_sequence",
    }:
        raise ValueError(f"unsupported VM replay model {model!r}")
    paths = vm_paths()
    instance = vm_status(paths)
    if instance is None or str(instance.get("status", "")).lower() != "running":
        raise RuntimeError(
            "VM must be provisioned and running before offline replay; start it explicitly with vm-create"
        )
    if not paths["provision_receipt"].is_file():
        raise RuntimeError("VM provision receipt is missing; provision before replay")
    guest_python = f"{GUEST_REPOSITORY}/.venv/bin/python"
    prediction_path = f"{GUEST_REPLAY_OUTPUT}/predictions.json"
    receipt_path = f"{GUEST_REPLAY_OUTPUT}/external-replay-receipt.json"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {GUEST_REPOSITORY}
test ! -e output/replication/private
test ! -e output/data/replication_manifest.json
test -x {guest_python}
command -v ip >/dev/null
sudo install -d -m 0755 {GUEST_REPLAY_OUTPUT}
sudo chown "$(id -u):$(id -g)" {GUEST_REPLAY_OUTPUT}
sudo unshare --net --fork --pid --mount-proc /bin/bash -s <<'INNER'
set -euo pipefail
cd {GUEST_REPOSITORY}
interfaces="$(ip -o link show | awk -F': ' '{{print $2}}' | cut -d@ -f1 | tr '\\n' ' ' | sed 's/[[:space:]]*$//')"
routes="$(ip route show)"
test "$interfaces" = "lo"
test -z "$routes"
printf 'interfaces=%s\\nroutes=%s\\n' "$interfaces" "$routes" > {GUEST_REPLAY_OUTPUT}/network-proof.txt
started_at="$(date -u --iso-8601=seconds)"
{guest_python} scripts/run_scientific_program.py verify-public
{guest_python} scripts/run_scientific_program.py replay --offline
{guest_python} scripts/run_scientific_program.py run-model --model {model} --output {prediction_path}
{guest_python} scripts/run_scientific_program.py verify-separation
completed_at="$(date -u --iso-8601=seconds)"
{guest_python} scripts/run_scientific_program.py write-replay-receipt \\
  --public-path output/replication/public/public_manifest.json \\
  --predictions {prediction_path} \\
  --output {receipt_path} \\
  --operator-id aapta-lima-vm \\
  --machine-id {VM_NAME}-same-host \\
  --bundle-sha256 current \\
  --model {model} \\
  --seed {seed} \\
  --started-at "$started_at" \\
  --completed-at "$completed_at" \\
  --command replay --offline
chmod -R a+rX {GUEST_REPLAY_OUTPUT}
INNER
"""
    result = _guest_shell(paths, script, timeout=900.0)
    paths["replay_stdout"].write_text(result.stdout, encoding="utf-8")
    paths["replay_stderr"].write_text(result.stderr, encoding="utf-8")
    _copy_from_guest(paths, prediction_path, paths["predictions"])
    _copy_from_guest(paths, receipt_path, paths["pending_receipt"])
    _copy_from_guest(
        paths,
        f"{GUEST_REPLAY_OUTPUT}/network-proof.txt",
        paths["network_proof"],
    )
    try:
        pending = json.loads(paths["pending_receipt"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"VM replay receipt is unreadable: {exc}") from exc
    valid, detail = validate_replay_receipt(
        pending, project_root=project_root, predictions_path=paths["predictions"]
    )
    if not valid:
        raise ValueError(f"VM replay receipt failed host validation: {detail}")
    replay_receipt = {
        "schema_version": VM_SCHEMA_VERSION,
        "receipt_kind": "blue_backed_lima_vm_replay",
        "status": "validated",
        "created_at": _iso_now(),
        "vm_name": VM_NAME,
        "vm_manifest_sha256": _sha256(paths["vm_manifest"]),
        "pending_receipt_sha256": _sha256(paths["pending_receipt"]),
        "prediction_sha256": _sha256(paths["predictions"]),
        "public_bundle_sha256": _public_bundle_digest(project_root),
        "network_proof_sha256": _sha256(paths["network_proof"]),
        "network_namespace_verified": True,
        "interfaces": "lo",
        "routes": "",
        "private_gold_transferred": False,
        "source_bound_label_contract_transferred": False,
        "control_scope": "same_host_separately_controlled_guest",
        "independent_reimplementation": False,
        "limitations": [
            "The replay was run in a Lima guest on the same host; this is not an independent machine or independent implementation.",
            "Protected scoring remains on the owner host and is performed only after receipt validation.",
        ],
    }
    write_json_atomic(paths["replay_receipt"], replay_receipt)
    return paths["replay_receipt"]


def attest_and_ingest_vm_replay(project_root: Path = PROJECT_ROOT) -> Path:
    """Attest the completed same-host VM boundary and invoke protected scoring."""

    paths = vm_paths()
    for key in ("pending_receipt", "predictions", "replay_receipt", "network_proof"):
        if not paths[key].is_file():
            raise ValueError(f"VM replay evidence is missing: {paths[key]}")
    pending = json.loads(paths["pending_receipt"].read_text(encoding="utf-8"))
    valid, detail = validate_replay_receipt(
        pending, project_root=project_root, predictions_path=paths["predictions"]
    )
    if not valid:
        raise ValueError(f"pending VM receipt failed validation: {detail}")
    replay_summary = json.loads(paths["replay_receipt"].read_text(encoding="utf-8"))
    if replay_summary.get("network_namespace_verified") is not True:
        raise ValueError("VM replay did not verify its isolated network namespace")
    if replay_summary.get("private_gold_transferred") is not False:
        raise ValueError("VM replay evidence reports private gold transfer")
    attested = deepcopy(pending)
    attestation = dict(attested.get("attestation", {}))
    attestation.update(
        {
            "separate_controlled_run": True,
            "external_operator_confirmation": (
                "User-authorized same-host Lima VM replay: clean source archive and public bundle only; "
                "guest network namespace verified as loopback-only; protected gold remained on the owner host."
            ),
            "control_scope": "same_host_separately_controlled_guest",
            "network_namespace_verified": True,
            "private_gold_transferred": False,
            "vm_manifest_sha256": _sha256(paths["vm_manifest"]),
            "network_proof_sha256": _sha256(paths["network_proof"]),
        }
    )
    attested["attestation"] = attestation
    input_hashes = dict(attested.get("input_hashes", {}))
    input_hashes["vm_manifest"] = _sha256(paths["vm_manifest"])
    input_hashes["network_proof"] = _sha256(paths["network_proof"])
    attested["input_hashes"] = input_hashes
    attested["vm_manifest_sha256"] = _sha256(paths["vm_manifest"])
    attested["control_scope"] = "same_host_separately_controlled_guest"
    write_json_atomic(paths["attested_receipt"], attested)
    valid, detail = validate_replay_receipt(
        attested, project_root=project_root, predictions_path=paths["predictions"]
    )
    if not valid:
        raise ValueError(f"attested VM receipt failed validation: {detail}")
    return ingest_independent_replay(
        paths["attested_receipt"], paths["predictions"], project_root
    )


__all__ = [
    "DEFAULT_VM_ROOT",
    "LIMA_VERSION",
    "QEMU_BINARY",
    "QEMU_VERSION",
    "REPLAY_VM_CONFIG",
    "UBUNTU_IMAGE_SHA256",
    "UBUNTU_IMAGE_URL",
    "UV_ARCHIVE_SHA256",
    "UV_ARCHIVE_URL",
    "UV_VERSION",
    "VM_NAME",
    "VM_SCHEMA_VERSION",
    "attest_and_ingest_vm_replay",
    "create_replay_vm",
    "load_replay_vm_config",
    "prepare_public_guest_bundle",
    "provision_replay_vm",
    "replay_in_vm",
    "stop_replay_vm",
    "vm_paths",
    "vm_root",
    "vm_status",
    "write_vm_manifest",
]
