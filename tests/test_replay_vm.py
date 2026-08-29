"""Regression tests for the blue-backed replay VM boundary.

The tests exercise deterministic validation and receipt logic without creating
or mutating a real VM. The real Lima, guest namespace, and native toolchain
APIs are exercised by the operator commands during the controlled replay.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest
import yaml

import src.replay_vm as replay_vm
from src.artifact_contract import write_json_atomic


def test_checked_in_vm_config_is_pinned() -> None:
    payload = replay_vm.load_replay_vm_config()
    assert payload["vmType"] == "qemu"
    assert payload["arch"] == "aarch64"
    assert payload["mounts"] == []
    assert payload["networks"] == []
    assert payload["images"][0]["digest"] == (f"sha256:{replay_vm.UBUNTU_IMAGE_SHA256}")


def test_vm_config_drift_is_rejected(tmp_path: Path) -> None:
    payload = replay_vm.load_replay_vm_config()
    payload["vmType"] = "vz"
    config_path = tmp_path / "replay_vm.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="vmType"):
        replay_vm.load_replay_vm_config(config_path)


def test_vm_root_rejects_non_blue_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(replay_vm.VM_ROOT_ENV, "/tmp/aapta-replay-vm")
    with pytest.raises(ValueError, match="below /Volumes/blue"):
        replay_vm.vm_root()


def test_vm_root_rejects_relative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(replay_vm.VM_ROOT_ENV, "relative/replay-vm")
    with pytest.raises(ValueError, match="absolute path"):
        replay_vm.vm_root()


def test_vm_root_rejects_volume_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(replay_vm.VM_ROOT_ENV, "/Volumes/blue")
    with pytest.raises(ValueError, match="volume root"):
        replay_vm.vm_root()


def test_vm_root_accepts_a_blue_backed_child(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = "/Volumes/blue/aapta-test/replay-vm"
    monkeypatch.setenv(replay_vm.VM_ROOT_ENV, candidate)
    assert replay_vm.vm_root() == Path(candidate).resolve()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not: [valid", "unreadable"),
        ("[]", "object"),
        ("images: []", "exactly one pinned image"),
        ("images: [text]", "image entry"),
    ],
)
def test_vm_config_rejects_malformed_structure(
    tmp_path: Path, content: str, message: str
) -> None:
    config_path = tmp_path / "replay_vm.yaml"
    if content.startswith("images:"):
        payload = replay_vm.load_replay_vm_config()
        payload["images"] = [] if content == "images: []" else ["text"]
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    else:
        config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        replay_vm.load_replay_vm_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("location", "https://example.invalid/image", "URL drifted"),
        ("arch", "x86_64", "architecture"),
        ("variant", "cloud", "variant"),
        ("digest", "sha256:" + "0" * 64, "digest drifted"),
    ],
)
def test_vm_config_rejects_image_drift(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = replay_vm.load_replay_vm_config()
    payload["images"][0][field] = value
    config_path = tmp_path / "replay_vm.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        replay_vm.load_replay_vm_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mounts", ["/tmp:/tmp"], "mount"),
        ("containerd", {}, "containerd"),
        ("networks", ["vzNAT"], "networks"),
        ("propagateProxyEnv", True, "proxy"),
    ],
)
def test_vm_config_rejects_host_boundary_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = replay_vm.load_replay_vm_config()
    payload[field] = value
    config_path = tmp_path / "replay_vm.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        replay_vm.load_replay_vm_config(config_path)


def test_vm_paths_are_explicit_and_nested(tmp_path: Path) -> None:
    paths = replay_vm.vm_paths(tmp_path / "vm")
    assert paths["root"] == (tmp_path / "vm").resolve()
    assert paths["lima_home"].parent == paths["root"]
    assert paths["pending_receipt"].name == "external-replay-receipt.json"
    assert paths["predictions"].parent == paths["receipts"]


def test_deterministic_tar_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    members = {
        "repository/z.txt": b"z",
        "repository/a.txt": b"a",
    }
    replay_vm._deterministic_tar(first, members)
    replay_vm._deterministic_tar(second, members)
    assert first.read_bytes() == second.read_bytes()
    with replay_vm.tarfile.open(first) as archive:
        assert archive.getnames() == ["repository/a.txt", "repository/z.txt"]
        assert archive.getmember("repository/a.txt").mtime == 0


def test_uv_download_requires_explicit_network(tmp_path: Path) -> None:
    paths = replay_vm.vm_paths(tmp_path / "vm")
    with pytest.raises(RuntimeError, match="explicit --allow-network"):
        replay_vm._download_pinned_uv(paths, allow_network=False)


def test_uv_download_is_digest_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"pinned uv archive fixture"
    digest = sha256(content).hexdigest()
    monkeypatch.setattr(replay_vm, "UV_ARCHIVE_SHA256", digest)
    monkeypatch.setattr(replay_vm, "UV_ARCHIVE_URL", "https://example.invalid/uv.tgz")

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            value, self.value = self.value, b""
            return value

        value = content

    monkeypatch.setattr(replay_vm, "urlopen", lambda *_args, **_kwargs: Response())
    paths = replay_vm.vm_paths(tmp_path / "vm")
    downloaded = replay_vm._download_pinned_uv(paths, allow_network=True)
    assert downloaded.read_bytes() == content
    assert replay_vm._download_pinned_uv(paths, allow_network=False) == downloaded


def test_uv_cached_digest_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(replay_vm, "UV_ARCHIVE_SHA256", "0" * 64)
    paths = replay_vm.vm_paths(tmp_path / "vm")
    paths["toolchain"].mkdir(parents=True)
    cached = paths["toolchain"] / f"uv-{replay_vm.UV_VERSION}-aarch64-linux.tar.gz"
    cached.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="cached uv archive digest"):
        replay_vm._download_pinned_uv(paths, allow_network=False)


def test_lima_path_and_environment_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        replay_vm.shutil, "which", lambda _name: "/usr/local/bin/limactl"
    )
    assert replay_vm._limactl_path() == "/usr/local/bin/limactl"
    paths = replay_vm.vm_paths(tmp_path / "vm")
    assert replay_vm._lima_environment(paths)["LIMA_HOME"] == str(paths["lima_home"])


def test_lima_path_missing_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay_vm.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="limactl is unavailable"):
        replay_vm._limactl_path()


def test_qemu_version_is_pinned_and_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay_vm.shutil, "which", lambda _name: "/usr/local/bin/qemu-system-aarch64"
    )
    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "QEMU emulator version 11.1.1\n", ""
        ),
    )
    assert replay_vm._qemu_version() == "QEMU emulator version 11.1.1"

    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "QEMU emulator version 10.0.0\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match="version drifted"):
        replay_vm._qemu_version()


def test_qemu_version_missing_or_unreadable_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_vm.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        replay_vm._qemu_version()

    monkeypatch.setattr(
        replay_vm.shutil, "which", lambda _name: "/usr/local/bin/qemu-system-aarch64"
    )
    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "qemu error"),
    )
    with pytest.raises(RuntimeError, match="could not inspect"):
        replay_vm._qemu_version()


def test_run_lima_constructs_command_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = replay_vm.vm_paths(tmp_path / "vm")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(replay_vm, "_limactl_path", lambda: "/bin/limactl")
    monkeypatch.setattr(replay_vm.subprocess, "run", fake_run)
    result = replay_vm._run_lima(["status"], paths, input_text="x")
    assert result.stdout == "ok"
    assert calls[0][0] == ["/bin/limactl", "status"]
    assert calls[0][1]["input"] == "x"

    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "bad lima"
        ),
    )
    with pytest.raises(RuntimeError, match="Lima command failed"):
        replay_vm._run_lima(["status"], paths)


def test_lima_instance_listing_handles_absence_and_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = replay_vm.vm_paths(tmp_path / "vm")
    absent = subprocess.CompletedProcess(
        ["limactl"], 1, stdout="", stderr="No instance found"
    )
    monkeypatch.setattr(replay_vm, "_run_lima", lambda *_args, **_kwargs: absent)
    assert replay_vm._instance_rows(paths) == []

    listed = subprocess.CompletedProcess(
        ["limactl"],
        0,
        stdout=json.dumps([{"name": replay_vm.VM_NAME, "status": "Running"}]),
        stderr="",
    )
    monkeypatch.setattr(replay_vm, "_run_lima", lambda *_args, **_kwargs: listed)
    assert replay_vm.vm_status(paths) == {
        "name": replay_vm.VM_NAME,
        "status": "Running",
    }

    singleton = subprocess.CompletedProcess(
        ["limactl"],
        0,
        stdout=json.dumps({"name": replay_vm.VM_NAME, "status": "Running"}),
        stderr="",
    )
    monkeypatch.setattr(replay_vm, "_run_lima", lambda *_args, **_kwargs: singleton)
    assert replay_vm.vm_status(paths) == {
        "name": replay_vm.VM_NAME,
        "status": "Running",
    }


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (subprocess.CompletedProcess(["limactl"], 0, "", ""), ""),
        (subprocess.CompletedProcess(["limactl"], 0, "not-json", ""), "not JSON"),
        (
            subprocess.CompletedProcess(["limactl"], 0, "{}", ""),
            "named JSON object",
        ),
        (
            subprocess.CompletedProcess(["limactl"], 0, "5", ""),
            "JSON list or object",
        ),
        (
            subprocess.CompletedProcess(["limactl"], 0, "[1]", ""),
            "non-object row",
        ),
        (
            subprocess.CompletedProcess(["limactl"], 1, "", "unexpected failure"),
            "could not inspect",
        ),
    ],
)
def test_lima_instance_listing_rejects_invalid_responses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    paths = replay_vm.vm_paths(tmp_path / "vm")
    monkeypatch.setattr(replay_vm, "_run_lima", lambda *_args, **_kwargs: result)
    if result.returncode != 0:
        with pytest.raises(RuntimeError, match=message):
            replay_vm._instance_rows(paths)
    elif not result.stdout:
        assert replay_vm._instance_rows(paths) == []
    else:
        with pytest.raises(RuntimeError, match=message):
            replay_vm._instance_rows(paths)


def test_vm_status_returns_none_for_other_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        replay_vm,
        "_instance_rows",
        lambda _paths: [{"name": "other", "status": "Running"}],
    )
    assert replay_vm.vm_status(replay_vm.vm_paths(tmp_path / "vm")) is None


def test_write_vm_manifest_records_storage_and_replay_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)

    def fake_lima(
        arguments: object, *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if list(arguments) == ["--version"]:
            return subprocess.CompletedProcess([], 0, "limactl version 2.2.0\n", "")
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps([{"name": replay_vm.VM_NAME, "status": "Stopped"}]),
            "",
        )

    monkeypatch.setattr(replay_vm, "_run_lima", fake_lima)
    monkeypatch.setattr(
        replay_vm, "_qemu_version", lambda: "QEMU emulator version 11.1.1"
    )
    manifest = replay_vm.write_vm_manifest(
        status="implemented_local", extra={"test_marker": True}
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["storage_contract"]["lima_home_on_blue"] is True
    assert payload["replay_contract"]["private_gold_transferred"] is False
    assert (
        payload["replay_contract"]["control_scope"]
        == "same_host_separately_controlled_guest"
    )
    assert payload["test_marker"] is True
    replay_vm.write_vm_manifest(status="implemented_local")


def test_runtime_source_archive_rejects_missing_required_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(
        _command: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        handle = kwargs["stdout"]
        with replay_vm.tarfile.open(fileobj=handle, mode="w") as archive:
            info = replay_vm.tarfile.TarInfo("repository/README.md")
            info.size = 7
            archive.addfile(info, replay_vm.io.BytesIO(b"readme\n"))
        return subprocess.CompletedProcess([], 0, b"", b"")

    monkeypatch.setattr(replay_vm.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="missing required files"):
        replay_vm._runtime_source_archive(
            tmp_path / "repo", tmp_path / "source.tar", "revision"
        )


@pytest.mark.parametrize(
    ("member_name", "message"),
    [
        ("outside/file", "escapes prefix"),
        ("repository/../escape", "unsafe"),
    ],
)
def test_runtime_source_archive_rejects_unsafe_members(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    member_name: str,
    message: str,
) -> None:
    def fake_run(
        _command: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        handle = kwargs["stdout"]
        with replay_vm.tarfile.open(fileobj=handle, mode="w") as archive:
            info = replay_vm.tarfile.TarInfo(member_name)
            info.size = 1
            archive.addfile(info, replay_vm.io.BytesIO(b"x"))
        return subprocess.CompletedProcess([], 0, b"", b"")

    monkeypatch.setattr(replay_vm.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=message):
        replay_vm._runtime_source_archive(
            tmp_path / "repo", tmp_path / "source.tar", "revision"
        )


def test_create_vm_requires_network_for_first_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(replay_vm, "vm_status", lambda _paths: None)
    with pytest.raises(RuntimeError, match="explicit --allow-network"):
        replay_vm.create_replay_vm(allow_network=False)


def test_create_vm_rejects_unreadable_or_stale_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["vm_manifest"].parent.mkdir(parents=True)
    paths["vm_manifest"].write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        replay_vm.create_replay_vm(allow_network=False)

    paths["vm_manifest"].write_text(
        json.dumps({"config_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="configuration changed"):
        replay_vm.create_replay_vm(allow_network=False)


def test_create_vm_records_first_creation_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(replay_vm, "vm_status", lambda _paths: None)
    calls: list[list[str]] = []

    def fake_lima(
        arguments: object, *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(arguments))
        return subprocess.CompletedProcess([], 0, "created", "")

    monkeypatch.setattr(replay_vm, "_run_lima", fake_lima)
    manifest = root / "receipts" / "vm_manifest.json"
    monkeypatch.setattr(
        replay_vm, "write_vm_manifest", lambda *_args, **_kwargs: manifest
    )
    result = replay_vm.create_replay_vm(allow_network=True)
    assert result == manifest
    paths = replay_vm.vm_paths(root)
    assert paths["receipts"].joinpath("create.stdout.txt").read_text() == "created"
    assert calls[0][0] == "start"


def test_create_and_stop_vm_preserve_existing_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["vm_manifest"].parent.mkdir(parents=True)
    write_json_atomic(
        paths["vm_manifest"],
        {"config_sha256": replay_vm._sha256(Path("configs/replay_vm.yaml"))},
    )
    calls: list[list[str]] = []

    def fake_lima(
        arguments: object, *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(arguments))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(replay_vm, "_run_lima", fake_lima)
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Stopped"},
    )
    monkeypatch.setattr(
        replay_vm, "write_vm_manifest", lambda *_args, **_kwargs: paths["vm_manifest"]
    )
    replay_vm.create_replay_vm(allow_network=False)
    assert ["start", "--yes", replay_vm.VM_NAME] in calls

    calls.clear()
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    replay_vm.create_replay_vm(allow_network=False)
    assert calls == []

    calls.clear()
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    replay_vm.stop_replay_vm()
    assert ["stop", replay_vm.VM_NAME] in calls

    calls.clear()
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Stopped"},
    )
    replay_vm.stop_replay_vm()
    assert calls == []


def test_prepare_public_guest_bundle_is_clean_and_label_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    for relative in replay_vm.PUBLIC_BUNDLE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative in ("README.md", "pyproject.toml", "uv.lock"):
        (repo / relative).write_text(relative, encoding="utf-8")
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(
        replay_vm, "validate_public_replay_bundle", lambda _root: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_public_bundle_digest", lambda _root: "bundle")
    transfer = replay_vm.prepare_public_guest_bundle(repo)
    assert transfer["private_gold_transferred"] is False
    assert transfer["source_archive"].is_file()
    assert transfer["public_archive"].is_file()
    assert transfer["public_bundle_sha256"] == "bundle"


def test_prepare_public_guest_bundle_filters_owner_only_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    for relative in replay_vm.PUBLIC_BUNDLE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative, content in {
        "README.md": "runtime",
        "pyproject.toml": "[project]\nname='runtime'\n",
        "uv.lock": "version = 1\n",
        "src/runtime.py": "runtime = True\n",
        "scripts/runtime.py": "print('runtime')\n",
        "configs/external_label_contracts.yaml": "secret: true\n",
        "data/owner-results.json": '{"label": 1}\n',
        "tests/owner_test.py": "secret = True\n",
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(
        replay_vm, "validate_public_replay_bundle", lambda _root: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_public_bundle_digest", lambda _root: "bundle")

    transfer = replay_vm.prepare_public_guest_bundle(repo)
    with replay_vm.tarfile.open(transfer["source_archive"]) as archive:
        names = set(archive.getnames())
    assert "repository/README.md" in names
    assert "repository/src/runtime.py" in names
    assert "repository/scripts/runtime.py" in names
    assert "repository/configs/external_label_contracts.yaml" not in names
    assert "repository/data/owner-results.json" not in names
    assert "repository/tests/owner_test.py" not in names
    assert transfer["source_archive_filter"] == "runtime_source_allowlist_v1"
    assert transfer["source_archive_excluded_paths"] == list(
        replay_vm.SOURCE_ARCHIVE_EXCLUDED_PATHS
    )


def test_prepare_public_guest_bundle_rejects_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        replay_vm, "validate_public_replay_bundle", lambda _root: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_git_output", lambda _root, *_args: " M dirty.py")
    with pytest.raises(ValueError, match="clean committed checkout"):
        replay_vm.prepare_public_guest_bundle(repo)


def test_prepare_public_guest_bundle_rejects_invalid_public_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        replay_vm,
        "validate_public_replay_bundle",
        lambda _root: (False, "public drift"),
    )
    with pytest.raises(ValueError, match="public drift"):
        replay_vm.prepare_public_guest_bundle(tmp_path)


def test_prepare_public_guest_bundle_reuses_existing_source_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    for relative in replay_vm.PUBLIC_BUNDLE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(
        replay_vm, "validate_public_replay_bundle", lambda _root: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_public_bundle_digest", lambda _root: "bundle")
    monkeypatch.setattr(
        replay_vm,
        "_git_output",
        lambda _root, *arguments: "rev" if arguments == ("rev-parse", "HEAD") else "",
    )
    source_archive = root / "transfer" / "source-rev.tar"
    source_archive.parent.mkdir(parents=True)
    source_archive.write_bytes(b"already archived")
    transfer = replay_vm.prepare_public_guest_bundle(repo)
    assert transfer["source_archive"] == source_archive


def test_prepare_public_guest_bundle_records_archive_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    for relative in replay_vm.PUBLIC_BUNDLE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(
        replay_vm, "validate_public_replay_bundle", lambda _root: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_public_bundle_digest", lambda _root: "bundle")
    monkeypatch.setattr(
        replay_vm,
        "_git_output",
        lambda _root, *arguments: "rev" if arguments == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, b"", b"archive failed"
        ),
    )
    with pytest.raises(RuntimeError, match="could not create source archive"):
        replay_vm.prepare_public_guest_bundle(repo)


def test_git_and_guest_wrappers_use_explicit_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "git output\n", ""
        ),
    )
    assert replay_vm._git_output(tmp_path, "rev-parse", "HEAD") == "git output"
    paths = replay_vm.vm_paths(tmp_path / "vm")
    calls: list[list[str]] = []

    def fake_lima(
        arguments: object, *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(arguments))
        return subprocess.CompletedProcess([], 0, "guest", "")

    monkeypatch.setattr(replay_vm, "_run_lima", fake_lima)
    source = tmp_path / "archive.tar"
    source.write_bytes(b"archive")
    replay_vm._copy_to_guest(paths, source, "/tmp/archive.tar")
    replay_vm._copy_from_guest(paths, "/tmp/archive.tar", tmp_path / "copied.tar")
    result = replay_vm._guest_shell(paths, "true")
    assert result.stdout == "guest"
    assert calls[0][0] == "copy"
    assert calls[1][0] == "copy"
    assert calls[2][:3] == ["shell", "--shell", "/bin/bash"]


def test_git_wrapper_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        replay_vm.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "bad git"
        ),
    )
    with pytest.raises(RuntimeError, match="git command failed"):
        replay_vm._git_output(tmp_path, "status")


def test_provision_requires_network_opt_in(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="explicit --allow-network"):
        replay_vm.provision_replay_vm(tmp_path, allow_network=False)


def test_uv_download_cleans_mismatched_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(replay_vm, "UV_ARCHIVE_SHA256", "0" * 64)

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            value, self.value = self.value, b""
            return value

        value = b"wrong archive"

    monkeypatch.setattr(replay_vm, "urlopen", lambda *_args, **_kwargs: Response())
    paths = replay_vm.vm_paths(tmp_path / "vm")
    with pytest.raises(ValueError, match="downloaded uv archive digest"):
        replay_vm._download_pinned_uv(paths, allow_network=True)
    assert not list(paths["toolchain"].glob("*.tmp"))


def test_provision_records_transfer_and_guest_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    paths["vm_manifest"].write_text("manifest", encoding="utf-8")
    transfer_dir = tmp_path / "transfer"
    transfer_dir.mkdir()
    source_archive = transfer_dir / "source.tar"
    public_archive = transfer_dir / "public.tar"
    metadata_path = transfer_dir / "manifest.json"
    source_archive.write_bytes(b"source")
    public_archive.write_bytes(b"public")
    metadata_path.write_text("{}", encoding="utf-8")
    uv_archive = transfer_dir / "uv.tar.gz"
    uv_archive.write_bytes(b"uv")
    transfer = {
        "schema_version": replay_vm.VM_SCHEMA_VERSION,
        "source_revision": "revision",
        "source_archive_sha256": replay_vm._sha256(source_archive),
        "public_archive_sha256": replay_vm._sha256(public_archive),
        "public_bundle_sha256": "b" * 64,
        "public_files": list(replay_vm.PUBLIC_BUNDLE_FILES),
        "private_gold_transferred": False,
        "source_bound_label_contract_transferred": False,
        "network_required_for_replay": False,
        "source_archive": source_archive,
        "public_archive": public_archive,
        "metadata_path": metadata_path,
    }
    monkeypatch.setattr(
        replay_vm, "create_replay_vm", lambda *_args, **_kwargs: paths["vm_manifest"]
    )
    monkeypatch.setattr(
        replay_vm, "prepare_public_guest_bundle", lambda _root: transfer
    )
    monkeypatch.setattr(
        replay_vm, "_download_pinned_uv", lambda *_args, **_kwargs: uv_archive
    )
    copied: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        replay_vm,
        "_copy_to_guest",
        lambda _paths, source, destination: copied.append((source, destination)),
    )
    monkeypatch.setattr(
        replay_vm,
        "_guest_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "guest stdout", "guest stderr"
        ),
    )
    monkeypatch.setattr(
        replay_vm,
        "write_vm_manifest",
        lambda *_args, **_kwargs: paths["vm_manifest"],
    )
    receipt_path = replay_vm.provision_replay_vm(tmp_path, allow_network=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "validated"
    assert receipt["guest"]["private_gold_transferred"] is False
    assert len(copied) == 3
    assert (
        paths["receipts"].joinpath("provision.stdout.txt").read_text() == "guest stdout"
    )


def test_replay_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported VM replay model"):
        replay_vm.replay_in_vm(tmp_path, model="unknown")


def test_replay_requires_running_provisioned_vm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    monkeypatch.setattr(replay_vm, "vm_status", lambda _paths: None)
    with pytest.raises(RuntimeError, match="running"):
        replay_vm.replay_in_vm(tmp_path)

    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    with pytest.raises(RuntimeError, match="provision receipt"):
        replay_vm.replay_in_vm(tmp_path)


def test_replay_rejects_unreadable_or_invalid_guest_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    paths["provision_receipt"].write_text("{}", encoding="utf-8")
    paths["vm_manifest"].write_text("manifest", encoding="utf-8")
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    monkeypatch.setattr(
        replay_vm,
        "_guest_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    def bad_copy(_paths: dict[str, Path], source: str, destination: Path) -> None:
        if source.endswith("external-replay-receipt.json"):
            destination.write_text("not-json", encoding="utf-8")
        elif source.endswith("predictions.json"):
            destination.write_text("[]", encoding="utf-8")
        else:
            destination.write_text("interfaces=lo\nroutes=\n", encoding="utf-8")

    monkeypatch.setattr(replay_vm, "_copy_from_guest", bad_copy)
    with pytest.raises(ValueError, match="unreadable"):
        replay_vm.replay_in_vm(tmp_path)


def test_replay_rejects_guest_receipt_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    paths["provision_receipt"].write_text("{}", encoding="utf-8")
    paths["vm_manifest"].write_text("manifest", encoding="utf-8")
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    monkeypatch.setattr(
        replay_vm,
        "_guest_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    def invalid_copy(_paths: dict[str, Path], source: str, destination: Path) -> None:
        if source.endswith("external-replay-receipt.json"):
            write_json_atomic(destination, {"pending": True})
        elif source.endswith("predictions.json"):
            write_json_atomic(destination, {"predictions": []})
        else:
            destination.write_text("interfaces=lo\nroutes=\n", encoding="utf-8")

    monkeypatch.setattr(replay_vm, "_copy_from_guest", invalid_copy)
    monkeypatch.setattr(
        replay_vm,
        "validate_replay_receipt",
        lambda *_args, **_kwargs: (False, "receipt drift"),
    )
    with pytest.raises(ValueError, match="receipt drift"):
        replay_vm.replay_in_vm(tmp_path)


def test_replay_and_attestation_preserve_control_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    for key in ("vm_manifest", "provision_receipt"):
        paths[key].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        replay_vm,
        "vm_status",
        lambda _paths: {"name": replay_vm.VM_NAME, "status": "Running"},
    )
    guest_scripts: list[str] = []

    def capture_guest_script(
        _paths: dict[str, Path], script: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        guest_scripts.append(script)
        return subprocess.CompletedProcess([], 0, "guest", "")

    monkeypatch.setattr(
        replay_vm,
        "_guest_shell",
        capture_guest_script,
    )

    def fake_copy(_paths: dict[str, Path], source: str, destination: Path) -> None:
        if source.endswith("predictions.json"):
            write_json_atomic(
                destination,
                {
                    "model": "calibrated_interpretable",
                    "label_blind": True,
                    "predictions": [],
                },
            )
        elif source.endswith("external-replay-receipt.json"):
            write_json_atomic(destination, {"pending": True})
        else:
            destination.write_text("interfaces=lo\nroutes=\n", encoding="utf-8")

    monkeypatch.setattr(replay_vm, "_copy_from_guest", fake_copy)
    monkeypatch.setattr(
        replay_vm, "validate_replay_receipt", lambda *_args, **_kwargs: (True, "ok")
    )
    monkeypatch.setattr(replay_vm, "_public_bundle_digest", lambda _root: "b" * 64)
    replay_path = replay_vm.replay_in_vm(tmp_path, model="calibrated_interpretable")
    assert guest_scripts
    assert (
        f"sudo install -d -m 0755 {replay_vm.GUEST_REPLAY_OUTPUT}" in guest_scripts[0]
    )
    summary = json.loads(replay_path.read_text(encoding="utf-8"))
    assert summary["private_gold_transferred"] is False
    assert summary["independent_reimplementation"] is False

    paths["replay_receipt"].write_text(
        replay_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    result_path = tmp_path / "ingested.json"
    monkeypatch.setattr(
        replay_vm, "ingest_independent_replay", lambda *_args: result_path
    )
    ingested = replay_vm.attest_and_ingest_vm_replay(tmp_path)
    assert ingested == result_path
    attested = json.loads(paths["attested_receipt"].read_text(encoding="utf-8"))
    assert attested["attestation"]["separate_controlled_run"] is True
    assert attested["control_scope"] == "same_host_separately_controlled_guest"


@pytest.mark.parametrize(
    ("summary_field", "summary_value", "message"),
    [
        ("network_namespace_verified", False, "isolated network namespace"),
        ("private_gold_transferred", True, "private gold transfer"),
    ],
)
def test_attestation_rejects_unsafe_vm_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    summary_field: str,
    summary_value: object,
    message: str,
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    for key in ("pending_receipt", "predictions", "vm_manifest", "network_proof"):
        paths[key].write_text("{}", encoding="utf-8")
    summary = {
        "network_namespace_verified": True,
        "private_gold_transferred": False,
    }
    summary[summary_field] = summary_value
    paths["replay_receipt"].write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        replay_vm, "validate_replay_receipt", lambda *_args, **_kwargs: (True, "ok")
    )
    with pytest.raises(ValueError, match=message):
        replay_vm.attest_and_ingest_vm_replay(tmp_path)


def test_attestation_requires_all_vm_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    with pytest.raises(ValueError, match="VM replay evidence is missing"):
        replay_vm.attest_and_ingest_vm_replay(tmp_path)


def test_attestation_rejects_invalid_pending_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    for key in ("pending_receipt", "predictions", "vm_manifest", "network_proof"):
        paths[key].write_text("{}", encoding="utf-8")
    paths["replay_receipt"].write_text(
        json.dumps(
            {"network_namespace_verified": True, "private_gold_transferred": False}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        replay_vm,
        "validate_replay_receipt",
        lambda *_args, **_kwargs: (False, "pending drift"),
    )
    with pytest.raises(ValueError, match="pending drift"):
        replay_vm.attest_and_ingest_vm_replay(tmp_path)


def test_attestation_rejects_invalid_attested_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "vm"
    monkeypatch.setattr(replay_vm, "vm_root", lambda: root)
    paths = replay_vm.vm_paths(root)
    paths["receipts"].mkdir(parents=True)
    for key in ("pending_receipt", "predictions", "vm_manifest", "network_proof"):
        paths[key].write_text("{}", encoding="utf-8")
    paths["replay_receipt"].write_text(
        json.dumps(
            {"network_namespace_verified": True, "private_gold_transferred": False}
        ),
        encoding="utf-8",
    )
    results = iter(((True, "ok"), (False, "attested drift")))
    monkeypatch.setattr(
        replay_vm, "validate_replay_receipt", lambda *_args, **_kwargs: next(results)
    )
    with pytest.raises(ValueError, match="attested drift"):
        replay_vm.attest_and_ingest_vm_replay(tmp_path)
