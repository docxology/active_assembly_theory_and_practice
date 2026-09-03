"""Safety and provenance tests for the pinned external-source boundary."""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest
import src.external_sources as external_module

from src.external_sources import (
    ExternalSourceSpec,
    _safe_extract,
    _spec_from_row,
    _validate_build_command,
    _validate_relative_path,
    acquire_external_manifest,
    acquire_external_source,
    compiler_capabilities_payload,
    external_acquisition_payload,
    external_source_validation_payload,
    load_external_manifest,
    static_compile_external_source,
    validate_external_acquisition_freshness,
    validate_external_acquisition_payload,
    write_external_source_validation_payload,
    write_external_acquisition_payload,
    validate_compiler_capabilities_payload,
    validate_external_manifest,
    validate_external_static_validation_payload,
    validate_external_source_validation_payload,
    write_compiler_capabilities,
)


def _valid_row() -> dict[str, object]:
    digest = "a" * 64
    return {
        "source_id": "project_a",
        "repository_url": "https://example.invalid/project_a.tar.gz",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "license": "BSD-3-Clause",
        "license_file": "LICENSE",
        "license_file_sha256": digest,
        "source_archive_sha256": digest,
        "source_family": "project_a_revision_1",
        "build_command": ["clang", "-c", "fixture.c", "-o", "fixture.o"],
        "compiler": "clang",
        "target_triple": "x86_64-apple-darwin",
        "label_contract": "recorded_source_bound_equivalence_contract",
        "provenance_reviewer": "research-software-review",
        "acquisition_timestamp": "2026-08-23T00:00:00Z",
    }


def test_pinned_repository_manifest_is_underpowered_without_labels() -> None:
    payload = external_source_validation_payload()
    valid, detail = validate_external_source_validation_payload(payload)
    assert valid, detail
    assert payload["status"] == "underpowered"
    assert payload["network_used"] is False
    assert payload["execution_performed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(repository_url="http://example.invalid/a.tar.gz"),
        lambda row: row.update(revision="not-an-immutable-commit"),
        lambda row: row.update(license_file="../LICENSE"),
        lambda row: row.update(
            build_command=["clang", "-c", "fixture.c", "&&", "true"]
        ),
        lambda row: row.update(build_command=["sh", "-c", "clang -c fixture.c"]),
        lambda row: row.update(source_archive_sha256="short"),
    ],
)
def test_manifest_rejects_unpinned_or_unsafe_rows(mutation) -> None:
    row = _valid_row()
    mutation(row)
    payload = {
        "schema_version": "2.0.0",
        "manifest_id": "test",
        "network_policy": "explicit_acquisition_only",
        "static_only": True,
        "sources": [row],
    }
    valid, _ = validate_external_manifest(payload)
    assert not valid


def test_static_compile_records_object_without_executing_it(tmp_path: Path) -> None:
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("clang unavailable")
    source_root = tmp_path / "source"
    output = tmp_path / "objects"
    source_root.mkdir()
    source = source_root / "fixture.c"
    source.write_text("unsigned long fixture(unsigned long x) { return x + 1; }\n")
    spec = ExternalSourceSpec(
        source_id="fixture",
        repository_url="https://example.invalid/fixture.tar.gz",
        revision="0123456789abcdef0123456789abcdef01234567",
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256="a" * 64,
        source_archive_sha256="b" * 64,
        source_family="fixture_family",
        build_command=(compiler, "-c", "fixture.c"),
        compiler=compiler,
        target_triple="x86_64-apple-darwin",
        label_contract="fixture-only",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )
    result = static_compile_external_source(spec, source_root, output)
    assert result["status"] == "validated"
    assert result["execution_performed"] is False
    assert result["static_only"] is True
    assert len(result["object_sha256"]) == 64
    assert len(result["disassembly_sha256"]) == 64
    assert int(result["decoded_instruction_count"]) > 0
    assert result["target_triple"] == "x86_64-apple-darwin"
    assert (output / "fixture.o").is_file()


def test_static_compile_marks_missing_compiler_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    spec = ExternalSourceSpec(
        source_id="missing-compiler",
        repository_url="https://example.invalid/fixture.tar.gz",
        revision="0123456789abcdef0123456789abcdef01234567",
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256="a" * 64,
        source_archive_sha256="b" * 64,
        source_family="fixture_family",
        build_command=("clang", "-c", "fixture.c"),
        compiler="clang",
        target_triple="x86_64-apple-darwin",
        label_contract="fixture-only",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )
    monkeypatch.setattr(external_module.shutil, "which", lambda _: None)
    result = static_compile_external_source(spec, source_root, tmp_path / "objects")
    assert result["status"] == "unavailable"
    assert result["reason"] == "compiler_unavailable"
    assert result["execution_performed"] is False
    assert result["static_only"] is True


def test_manifest_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable"):
        load_external_manifest(tmp_path / "missing.yaml")


def test_manifest_and_command_validation_rejects_structural_drift() -> None:
    row = _valid_row()
    spec = ExternalSourceSpec(
        source_id=str(row["source_id"]),
        repository_url=str(row["repository_url"]),
        revision=str(row["revision"]),
        license=str(row["license"]),
        license_file=str(row["license_file"]),
        license_file_sha256=str(row["license_file_sha256"]),
        source_archive_sha256=str(row["source_archive_sha256"]),
        source_family=str(row["source_family"]),
        build_command=tuple(row["build_command"]),
        compiler=str(row["compiler"]),
        target_triple=str(row["target_triple"]),
        label_contract=str(row["label_contract"]),
        provenance_reviewer=str(row["provenance_reviewer"]),
        acquisition_timestamp=str(row["acquisition_timestamp"]),
    )
    assert spec.to_dict()["source_id"] == "project_a"
    with pytest.raises(ValueError, match="relative"):
        _validate_relative_path("/absolute/LICENSE", field="license_file")
    with pytest.raises(ValueError, match="relative"):
        _validate_relative_path("", field="license_file")
    with pytest.raises(ValueError, match="non-empty"):
        _validate_build_command([])
    with pytest.raises(ValueError, match="allow-listed"):
        _validate_build_command(["sh", "-c", "clang -c fixture.c"])
    with pytest.raises(ValueError, match="-c"):
        _validate_build_command(["clang", "fixture.c"])
    with pytest.raises(ValueError, match="shell syntax"):
        _validate_build_command(["clang", "-c", "fixture.c", ";"])
    mismatch = {**row, "compiler": "gcc"}
    assert (
        validate_external_manifest(
            {
                "schema_version": "2.0.0",
                "manifest_id": "test",
                "network_policy": "explicit_acquisition_only",
                "static_only": True,
                "sources": [mismatch],
            }
        )[0]
        is False
    )

    duplicate_id = {
        "schema_version": "2.0.0",
        "manifest_id": "test",
        "network_policy": "explicit_acquisition_only",
        "static_only": True,
        "sources": [_valid_row(), _valid_row()],
    }
    assert validate_external_manifest(duplicate_id)[0] is False
    duplicate_family = {
        **duplicate_id,
        "sources": [
            _valid_row(),
            {
                **_valid_row(),
                "source_id": "project_b",
                "project_id": "project_a",
                "revision_label": "0123456789ab",
            },
        ],
    }
    assert validate_external_manifest(duplicate_family)[0] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("source_id"), "missing"),
        (lambda row: row.update(source_id="bad id"), "invalid source_id"),
        (lambda row: row.update(license=""), "license must be non-empty"),
        (lambda row: row.update(license_file_sha256="bad"), "license_file_sha256"),
        (lambda row: row.update(compiler="gcc"), "does not match compiler"),
        (lambda row: row.update(target_triple=""), "target_triple must be non-empty"),
    ],
)
def test_source_spec_rejects_required_field_and_provenance_drift(
    mutation, message: str
) -> None:
    row = _valid_row()
    mutation(row)
    with pytest.raises(ValueError, match=message):
        _spec_from_row(row, 0)


def test_safe_extract_rejects_tar_traversal_and_symbolic_links(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.tar"
    with tarfile.open(traversal, "w") as archive:
        member = tarfile.TarInfo("../escape.txt")
        member.size = 6
        archive.addfile(member, io.BytesIO(b"unsafe"))
    with pytest.raises(ValueError, match="escapes"):
        _safe_extract(traversal, tmp_path / "tar-traversal")

    linked = tmp_path / "linked.tar"
    with tarfile.open(linked, "w") as archive:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "target"
        archive.addfile(member)
    with pytest.raises(ValueError, match="links"):
        _safe_extract(linked, tmp_path / "tar-link")

    zip_link = tmp_path / "linked.zip"
    with zipfile.ZipFile(zip_link, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16) | 0xA000
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="links"):
        _safe_extract(zip_link, tmp_path / "zip-link")


def test_safe_archive_extraction_accepts_local_files_and_rejects_links_and_traversal(
    tmp_path: Path,
) -> None:
    tar_path = tmp_path / "safe.tar"
    source = tmp_path / "source.txt"
    source.write_text("safe", encoding="utf-8")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname="nested/source.txt")
    extracted = tmp_path / "tar_out"
    extracted.mkdir()
    _safe_extract(tar_path, extracted)
    assert (extracted / "nested/source.txt").read_text(encoding="utf-8") == "safe"

    zip_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/source.txt", "safe zip")
    zip_out = tmp_path / "zip_out"
    zip_out.mkdir()
    _safe_extract(zip_path, zip_out)
    assert (zip_out / "nested/source.txt").read_text(encoding="utf-8") == "safe zip"

    traversal_zip = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal_zip, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="escapes"):
        _safe_extract(traversal_zip, tmp_path / "traversal_out")

    unsupported = tmp_path / "unsupported.bin"
    unsupported.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="neither"):
        _safe_extract(unsupported, tmp_path / "unsupported_out")


def test_acquisition_is_blocked_without_explicit_network_opt_in(tmp_path: Path) -> None:
    row = _valid_row()
    spec = ExternalSourceSpec(
        source_id=str(row["source_id"]),
        repository_url=str(row["repository_url"]),
        revision=str(row["revision"]),
        license=str(row["license"]),
        license_file=str(row["license_file"]),
        license_file_sha256=str(row["license_file_sha256"]),
        source_archive_sha256=str(row["source_archive_sha256"]),
        source_family=str(row["source_family"]),
        build_command=tuple(row["build_command"]),
        compiler=str(row["compiler"]),
        target_triple=str(row["target_triple"]),
        label_contract=str(row["label_contract"]),
        provenance_reviewer=str(row["provenance_reviewer"]),
        acquisition_timestamp=str(row["acquisition_timestamp"]),
    )
    result = acquire_external_source(spec, tmp_path)
    assert result["status"] == "blocked_external"
    assert result["network_used"] is False


def test_manifest_acquisition_aggregates_failures_and_never_reports_false_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = _valid_row()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "schema_version: '2.0.0'\n"
        "manifest_id: test\n"
        "network_policy: explicit_acquisition_only\n"
        "static_only: true\n"
        "sources:\n"
        "  - source_id: project_a\n"
        "    repository_url: https://example.invalid/project_a.tar.gz\n"
        f"    revision: {row['revision']}\n"
        "    license: BSD-3-Clause\n"
        "    license_file: LICENSE\n"
        f"    license_file_sha256: {'a' * 64}\n"
        f"    source_archive_sha256: {'a' * 64}\n"
        "    source_family: project_a_revision_1\n"
        "    build_command: [clang, -c, fixture.c]\n"
        "    compiler: clang\n"
        "    target_triple: x86_64-apple-darwin\n"
        "    label_contract: fixture-only\n"
        "    provenance_reviewer: test\n"
        "    acquisition_timestamp: '2026-08-23T00:00:00Z'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        external_module,
        "acquire_external_source",
        lambda spec, destination, *, allow_network: {
            "source_id": spec.source_id,
            "status": "failed",
            "network_used": allow_network,
        },
    )
    result = acquire_external_manifest(
        manifest_path, tmp_path / "out", allow_network=True
    )
    assert result["status"] == "failed"
    assert result["rows"][0]["status"] == "failed"


def test_manifest_acquisition_requires_explicit_network_opt_in(tmp_path: Path) -> None:
    manifest_path = tmp_path / "empty.yaml"
    manifest_path.write_text(
        "schema_version: '2.0.0'\n"
        "manifest_id: empty\n"
        "network_policy: explicit_acquisition_only\n"
        "static_only: true\n"
        "sources: []\n",
        encoding="utf-8",
    )

    result = acquire_external_manifest(
        manifest_path, tmp_path / "out", allow_network=False
    )

    assert result == {
        "status": "blocked_external",
        "reason": "network acquisition requires --allow-network",
        "source_count": 0,
        "network_used": False,
        "rows": [],
    }


def test_acquisition_receipt_preserves_blocked_network_status(
    tmp_path: Path,
) -> None:
    row = _valid_row()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "schema_version: '2.0.0'\n"
        "manifest_id: test\n"
        "network_policy: explicit_acquisition_only\n"
        "static_only: true\n"
        "sources:\n"
        "  - "
        + "\n    ".join(
            f"{key}: {json.dumps(value)}"
            for key, value in row.items()
            if key != "build_command"
        )
        + "\n    build_command: [clang, -c, fixture.c]\n",
        encoding="utf-8",
    )
    payload = external_acquisition_payload(
        tmp_path,
        manifest_path=manifest_path,
        destination=tmp_path / "archives",
        allow_network=False,
    )
    valid, detail = validate_external_acquisition_payload(payload)
    assert valid, detail
    assert payload["status"] == "blocked_external"
    assert payload["network_used"] is False
    assert payload["acquisition"]["rows"] == []

    path = write_external_acquisition_payload(
        tmp_path,
        manifest_path=manifest_path,
        destination=tmp_path / "archives",
        allow_network=False,
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["artifact"] == "output/data/external_acquisition.json"
    assert stored["status"] == "blocked_external"


def test_acquisition_receipt_rejects_validated_status_with_failed_row() -> None:
    payload = {
        "schema_version": "2.0.0",
        "protocol_id": "active-assembly-open-roadmap-v2",
        "protocol_sha256": "0" * 64,
        "generator_revision": "x",
        "input_hashes": {},
        "toolchain_versions": {},
        "status": "validated",
        "limitations": ["test"],
        "unsupported_cases": ["test"],
        "underpowered_cases": ["test"],
        "manifest_id": "test",
        "manifest_sha256": "a" * 64,
        "acquisition": {
            "status": "validated",
            "rows": [{"status": "failed", "network_used": True}],
        },
        "network_used": True,
        "execution_performed": False,
        "static_only": True,
    }
    valid, detail = validate_external_acquisition_payload(payload)
    assert not valid
    assert "nonvalidated row" in detail or "receipt" in detail


def _spec_for_archive(archive_digest: str, license_digest: str) -> ExternalSourceSpec:
    row = _valid_row()
    return ExternalSourceSpec(
        source_id="project_a",
        repository_url="https://example.invalid/project_a.tar.gz",
        revision=str(row["revision"]),
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256=license_digest,
        source_archive_sha256=archive_digest,
        source_family="project_a_revision_1",
        build_command=("clang", "-c", "fixture.c"),
        compiler="clang",
        target_triple="x86_64-apple-darwin",
        label_contract="recorded_source_bound_equivalence_contract",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )


def _archive_bytes(license_text: str = "BSD-3-Clause\n") -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in (
            ("LICENSE", license_text.encode()),
            ("fixture.c", b"unsigned long f(unsigned long x) { return x + 1; }\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def test_acquisition_freshness_is_offline_and_detects_producer_drift(
    tmp_path: Path,
) -> None:
    archive = _archive_bytes()
    row = _valid_row()
    row["license_file_sha256"] = sha256(b"BSD-3-Clause\n").hexdigest()
    row["source_archive_sha256"] = sha256(archive).hexdigest()
    manifest_path = tmp_path / "configs" / "external_sources.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "schema_version: '2.0.0'\n"
        "manifest_id: test\n"
        "network_policy: explicit_acquisition_only\n"
        "static_only: true\n"
        "sources:\n"
        "  - "
        + "\n    ".join(
            f"{key}: {json.dumps(value)}"
            for key, value in row.items()
            if key != "build_command"
        )
        + "\n    build_command: [clang, -c, fixture.c, -o, fixture.o]\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "src" / "external_sources.py"
    source_path.parent.mkdir()
    shutil.copy(Path(__file__).parents[1] / "src" / "external_sources.py", source_path)
    destination = tmp_path / "output" / "external_sources"
    destination.mkdir(parents=True)
    archive_path = destination / f"{row['source_id']}-{row['revision']}.archive"
    archive_path.write_bytes(archive)

    payload = external_acquisition_payload(
        tmp_path,
        manifest_path=manifest_path,
        destination=destination,
        allow_network=True,
    )
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert valid, detail

    source_path.write_text("# producer drift\n", encoding="utf-8")
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert not valid
    assert "producer input hash" in detail


def test_acquisition_freshness_rejects_manifest_cache_and_row_drift(
    tmp_path: Path,
) -> None:
    archive = _archive_bytes()
    row = _valid_row()
    row["license_file_sha256"] = sha256(b"BSD-3-Clause\n").hexdigest()
    row["source_archive_sha256"] = sha256(archive).hexdigest()
    manifest_path = tmp_path / "configs" / "external_sources.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_text = (
        "schema_version: '2.0.0'\n"
        "manifest_id: test\n"
        "network_policy: explicit_acquisition_only\n"
        "static_only: true\n"
        "sources:\n  - "
        + "\n    ".join(
            f"{key}: {json.dumps(value)}"
            for key, value in row.items()
            if key != "build_command"
        )
        + "\n    build_command: [clang, -c, fixture.c, -o, fixture.o]\n"
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    source_path = tmp_path / "src" / "external_sources.py"
    source_path.parent.mkdir()
    shutil.copy(Path(__file__).parents[1] / "src" / "external_sources.py", source_path)
    destination = tmp_path / "output" / "external_sources"
    destination.mkdir(parents=True)
    archive_path = destination / f"{row['source_id']}-{row['revision']}.archive"
    archive_path.write_bytes(archive)
    (destination / str(row["source_id"])).mkdir()
    (destination / str(row["source_id"]) / "LICENSE").write_text(
        "BSD-3-Clause\n", encoding="utf-8"
    )
    payload = external_acquisition_payload(
        tmp_path,
        manifest_path=manifest_path,
        destination=destination,
        allow_network=False,
    )
    payload["acquisition"] = {
        "status": "validated",
        "rows": [
            {
                "source_id": row["source_id"],
                "status": "validated",
                "network_used": False,
            }
        ],
    }
    payload["status"] = "validated"
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert valid, detail

    manifest_path.unlink()
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert not valid
    assert "manifest is missing" in detail
    manifest_path.write_text(manifest_text, encoding="utf-8")

    source_path.unlink()
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert not valid
    assert "producer is missing" in detail
    shutil.copy(Path(__file__).parents[1] / "src" / "external_sources.py", source_path)

    stale = json.loads(json.dumps(payload))
    stale["manifest_sha256"] = "0" * 64
    valid, detail = validate_external_acquisition_freshness(stale, tmp_path)
    assert not valid
    assert "manifest hash is stale" in detail
    stale = json.loads(json.dumps(payload))
    stale["input_hashes"][external_module.EXTERNAL_MANIFEST_PATH] = "0" * 64
    valid, detail = validate_external_acquisition_freshness(stale, tmp_path)
    assert not valid
    assert "manifest input hash is stale" in detail
    stale = json.loads(json.dumps(payload))
    stale["input_hashes"]["src/external_sources.py"] = "0" * 64
    valid, detail = validate_external_acquisition_freshness(stale, tmp_path)
    assert not valid
    assert "producer input hash is stale" in detail

    fewer_rows = json.loads(json.dumps(payload))
    fewer_rows["status"] = "blocked_external"
    fewer_rows["acquisition"]["rows"] = []
    valid, detail = validate_external_acquisition_freshness(fewer_rows, tmp_path)
    assert not valid
    assert "does not cover" in detail
    unknown_row = json.loads(json.dumps(payload))
    unknown_row["status"] = "blocked_external"
    unknown_row["acquisition"]["status"] = "blocked_external"
    unknown_row["acquisition"]["rows"] = [
        {"source_id": "unknown", "status": "blocked_external", "network_used": False}
    ]
    valid, detail = validate_external_acquisition_freshness(unknown_row, tmp_path)
    assert not valid
    assert "unknown source" in detail

    archive_path.unlink()
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert not valid
    assert "archive is stale or missing" in detail
    archive_path.write_bytes(archive)
    (destination / str(row["source_id"]) / "LICENSE").write_text(
        "wrong\n", encoding="utf-8"
    )
    valid, detail = validate_external_acquisition_freshness(payload, tmp_path)
    assert not valid
    assert "license evidence is stale or missing" in detail


class _BytesResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self._content)

    def __exit__(self, *args: object) -> None:
        return None


def test_controlled_acquisition_validates_archive_and_license_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _archive_bytes()
    license_digest = sha256(b"BSD-3-Clause\n").hexdigest()
    spec = _spec_for_archive(sha256(archive).hexdigest(), license_digest)
    monkeypatch.setattr(
        external_module,
        "urlopen",
        lambda request, timeout: _BytesResponse(archive),
    )
    result = external_module.acquire_external_source(spec, tmp_path, allow_network=True)
    assert result["status"] == "validated"
    assert result["network_used"] is True
    assert result["license_file_sha256"] == license_digest
    assert "execution_performed" not in result


def test_acquisition_reuses_verified_cache_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _archive_bytes()
    license_digest = sha256(b"BSD-3-Clause\n").hexdigest()
    spec = _spec_for_archive(sha256(archive).hexdigest(), license_digest)
    archive_path = tmp_path / f"{spec.source_id}-{spec.revision}.archive"
    archive_path.write_bytes(archive)

    def network_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("verified cache should not require network access")

    monkeypatch.setattr(external_module, "urlopen", network_must_not_run)
    result = external_module.acquire_external_source(
        spec, tmp_path, allow_network=False
    )
    assert result["status"] == "validated"
    assert result["network_used"] is False
    assert result["acquisition_mode"] == "cache"
    assert result["archive_sha256"] == sha256(archive).hexdigest()
    assert (tmp_path / spec.source_id / "LICENSE").read_text() == "BSD-3-Clause\n"


def test_acquisition_rejects_bad_cache_without_overwriting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _archive_bytes()
    license_digest = sha256(b"BSD-3-Clause\n").hexdigest()
    spec = _spec_for_archive(sha256(archive).hexdigest(), license_digest)
    archive_path = tmp_path / f"{spec.source_id}-{spec.revision}.archive"
    original = b"operator-review-required"
    archive_path.write_bytes(original)

    monkeypatch.setattr(
        external_module,
        "urlopen",
        lambda *_args, **_kwargs: _BytesResponse(archive),
    )
    result = external_module.acquire_external_source(spec, tmp_path, allow_network=True)
    assert result["status"] == "failed"
    assert result["reason"] == "cached source archive digest mismatch"
    assert archive_path.read_bytes() == original


def test_network_hash_mismatch_does_not_leave_a_partial_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _archive_bytes()
    spec = _spec_for_archive("a" * 64, sha256(b"BSD-3-Clause\n").hexdigest())
    monkeypatch.setattr(
        external_module,
        "urlopen",
        lambda *_args, **_kwargs: _BytesResponse(archive),
    )
    result = external_module.acquire_external_source(spec, tmp_path, allow_network=True)
    assert result["status"] == "failed"
    assert result["reason"] == "source archive hash mismatch"
    assert not (tmp_path / f"{spec.source_id}-{spec.revision}.archive").exists()
    assert not list(tmp_path.glob(".*.download"))


def test_controlled_acquisition_fails_closed_on_network_hash_license_and_archive_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _archive_bytes()
    license_digest = sha256(b"BSD-3-Clause\n").hexdigest()

    def raise_network(request, timeout):
        raise OSError("offline test")

    monkeypatch.setattr(external_module, "urlopen", raise_network)
    network_spec = _spec_for_archive("a" * 64, license_digest)
    network_result = external_module.acquire_external_source(
        network_spec, tmp_path / "network", allow_network=True
    )
    assert network_result["status"] == "failed"
    assert "acquisition_failed" in str(network_result["reason"])

    monkeypatch.setattr(
        external_module,
        "urlopen",
        lambda request, timeout: _BytesResponse(archive),
    )
    hash_result = external_module.acquire_external_source(
        _spec_for_archive("b" * 64, license_digest),
        tmp_path / "hash",
        allow_network=True,
    )
    assert hash_result["status"] == "failed"
    assert hash_result["reason"] == "source archive hash mismatch"

    license_result = external_module.acquire_external_source(
        _spec_for_archive(sha256(archive).hexdigest(), "c" * 64),
        tmp_path / "license",
        allow_network=True,
    )
    assert license_result["status"] == "failed"
    assert "license file hash mismatch" in str(license_result["reason"])

    invalid_archive = b"not a supported archive"
    invalid_spec = _spec_for_archive(
        sha256(invalid_archive).hexdigest(), license_digest
    )
    monkeypatch.setattr(
        external_module,
        "urlopen",
        lambda request, timeout: _BytesResponse(invalid_archive),
    )
    extraction_result = external_module.acquire_external_source(
        invalid_spec, tmp_path / "extraction", allow_network=True
    )
    assert extraction_result["status"] == "failed"
    assert "safe_extraction_failed" in str(extraction_result["reason"])


def test_static_compile_rejects_output_escape_and_records_compile_failure(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("clang unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "broken.c").write_text("this is not valid C;\n")
    output = tmp_path / "objects"
    bad_spec = ExternalSourceSpec(
        source_id="broken",
        repository_url="https://example.invalid/broken.tar.gz",
        revision="0123456789abcdef0123456789abcdef01234567",
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256="a" * 64,
        source_archive_sha256="b" * 64,
        source_family="broken_family",
        build_command=(compiler, "-c", "broken.c", "-o", "../escape.o"),
        compiler=compiler,
        target_triple="x86_64-apple-darwin",
        label_contract="fixture-only",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )
    with pytest.raises(ValueError, match="escapes"):
        external_module.static_compile_external_source(bad_spec, source_root, output)

    failure_spec = ExternalSourceSpec(
        **{**bad_spec.__dict__, "build_command": (compiler, "-c", "broken.c")}
    )
    result = external_module.static_compile_external_source(
        failure_spec, source_root, output
    )
    assert result["status"] == "failed"
    assert result["execution_performed"] is False


def test_capability_receipt_and_static_receipt_validators_are_fail_closed(
    tmp_path: Path,
) -> None:
    missing = external_module._compiler_probe("compiler-that-does-not-exist")
    assert missing["status"] == "unavailable"
    assert missing["targets"] == {}

    payload = compiler_capabilities_payload()
    valid, detail = validate_compiler_capabilities_payload(payload)
    assert valid, detail
    capability_path = write_compiler_capabilities(tmp_path)
    stored = json.loads(capability_path.read_text(encoding="utf-8"))
    assert stored["artifact"] == "output/data/external_capabilities.json"

    for candidate in (
        None,
        {"schema_version": "0.0.0", "rows": []},
        {"schema_version": "2.0.0", "rows": []},
        {
            "schema_version": "2.0.0",
            "rows": [{"targets": {"x86_64-apple-darwin": "yes"}}],
        },
    ):
        assert not validate_compiler_capabilities_payload(candidate)[0]

    static_base = {
        "schema_version": "2.0.0",
        "static_only": True,
        "execution_performed": False,
        "rows": [],
    }
    assert validate_external_static_validation_payload(static_base)[0]
    for candidate in (
        None,
        {**static_base, "schema_version": "0.0.0"},
        {**static_base, "static_only": False},
        {**static_base, "execution_performed": True},
        {**static_base, "rows": [{}]},
        {
            **static_base,
            "rows": [{"status": "unknown", "execution_performed": False}],
        },
        {
            **static_base,
            "rows": [{"status": "validated", "execution_performed": True}],
        },
    ):
        assert not validate_external_static_validation_payload(candidate)[0]

    external = external_source_validation_payload()
    valid, detail = validate_external_source_validation_payload(external)
    assert valid, detail
    for candidate in (
        None,
        {**external, "schema_version": "0.0.0"},
        {**external, "execution_performed": True},
        {**external, "static_only": False},
        {**external, "manifest": {}},
    ):
        assert not validate_external_source_validation_payload(candidate)[0]

    envelope_drift = dict(external)
    envelope_drift["receipt_schema_version"] = "0.0.0"
    valid, detail = validate_external_source_validation_payload(envelope_drift)
    assert not valid
    assert "external source receipt" in detail

    manifest_drift = dict(external)
    manifest_drift["manifest"] = {}
    assert not validate_external_source_validation_payload(manifest_drift)[0]


def test_external_receipt_writer_and_static_row_loader_keep_missing_evidence_visible(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copy(
        external_module.PROJECT_ROOT / "configs/external_sources.yaml",
        tmp_path / "configs/external_sources.yaml",
    )
    shutil.copy(
        external_module.PROJECT_ROOT / "configs/external_label_contracts.yaml",
        tmp_path / "configs/external_label_contracts.yaml",
    )
    assert external_module._load_static_rows(tmp_path) == []
    malformed = tmp_path / "output/data/external_static_validation.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    assert external_module._load_static_rows(tmp_path) == []

    receipt_path = write_external_source_validation_payload(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_external"
    assert payload["execution_performed"] is False


def test_static_compile_records_disassembler_unavailability_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("clang unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "fixture.c").write_text(
        "unsigned long f(unsigned long x) { return x + 1; }\n", encoding="utf-8"
    )
    spec = ExternalSourceSpec(
        source_id="fixture-no-disassembler",
        repository_url="https://example.invalid/fixture.tar.gz",
        revision="0123456789abcdef0123456789abcdef01234567",
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256="a" * 64,
        source_archive_sha256="b" * 64,
        source_family="fixture_family",
        build_command=(compiler, "-c", "fixture.c"),
        compiler=compiler,
        target_triple="x86_64-apple-darwin",
        label_contract="fixture-only",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )
    monkeypatch.setattr(external_module.shutil, "which", lambda _: None)
    result = static_compile_external_source(spec, source_root, tmp_path / "objects")
    assert result["status"] == "unavailable"
    assert result["reason"] == "static_disassembler_unavailable"
    assert result["execution_performed"] is False


def test_manifest_loader_fails_closed_for_structural_variants(
    tmp_path: Path,
) -> None:
    cases = [
        [],
        {
            "schema_version": "0.0.0",
            "manifest_id": "test",
            "network_policy": "explicit_acquisition_only",
            "static_only": True,
            "sources": [],
        },
        {
            "schema_version": "2.0.0",
            "manifest_id": "test",
            "network_policy": "network_allowed",
            "static_only": True,
            "sources": [],
        },
        {
            "schema_version": "2.0.0",
            "manifest_id": "",
            "network_policy": "explicit_acquisition_only",
            "static_only": True,
            "sources": [],
        },
        {
            "schema_version": "2.0.0",
            "manifest_id": "test",
            "network_policy": "explicit_acquisition_only",
            "static_only": True,
            "sources": {},
        },
        {
            "schema_version": "2.0.0",
            "manifest_id": "test",
            "network_policy": "explicit_acquisition_only",
            "static_only": True,
            "sources": [[]],
        },
    ]
    path = tmp_path / "manifest.json"
    for case in cases:
        path.write_text(json.dumps(case), encoding="utf-8")
        with pytest.raises(ValueError):
            load_external_manifest(path)

    duplicate = _valid_row()
    duplicate_payload = {
        "schema_version": "2.0.0",
        "manifest_id": "duplicate",
        "network_policy": "explicit_acquisition_only",
        "static_only": True,
        "sources": [duplicate, duplicate.copy()],
    }
    path.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source ids"):
        load_external_manifest(path)
    duplicate_revision = {
        **duplicate_payload,
        "sources": [
            duplicate,
            {
                **duplicate,
                "source_id": "project_b",
                "project_id": "project_a",
                "revision_label": duplicate["revision"][:12],
            },
        ],
    }
    path.write_text(json.dumps(duplicate_revision), encoding="utf-8")
    with pytest.raises(ValueError, match="project/revision"):
        load_external_manifest(path)
    invalid_compiler = {**duplicate, "compiler": "sh"}
    with pytest.raises(ValueError, match="allow-listed"):
        _spec_from_row(invalid_compiler, 0)

    for candidate in (
        None,
        {**duplicate_payload, "schema_version": "0.0.0"},
        {**duplicate_payload, "network_policy": "network_allowed"},
        {**duplicate_payload, "manifest_id": ""},
        {**duplicate_payload, "static_only": False},
        {**duplicate_payload, "sources": {}},
        {**duplicate_payload, "sources": [[]]},
    ):
        assert not validate_external_manifest(candidate)[0]


def test_acquisition_handles_cache_edges_empty_manifests_and_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = _archive_bytes()
    license_digest = sha256(b"BSD-3-Clause\n").hexdigest()
    spec = _spec_for_archive(sha256(archive).hexdigest(), license_digest)
    archive_path = tmp_path / f"{spec.source_id}-{spec.revision}.archive"
    archive_path.write_bytes(archive)
    extraction = tmp_path / spec.source_id
    extraction.mkdir()
    (extraction / "stale.txt").write_text("replace me", encoding="utf-8")
    cached = acquire_external_source(spec, tmp_path, allow_network=False)
    assert cached["status"] == "validated"
    assert not (extraction / "stale.txt").exists()

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    symlink_archive = symlink_root / f"{spec.source_id}-{spec.revision}.archive"
    symlink_archive.write_bytes(archive)
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    (symlink_root / spec.source_id).symlink_to(symlink_target, target_is_directory=True)
    symlink_result = acquire_external_source(spec, symlink_root, allow_network=False)
    assert symlink_result["status"] == "failed"
    assert "symbolic link" in str(symlink_result["reason"])

    unreadable_archive = tmp_path / "unreadable"
    unreadable_archive.mkdir()
    unreadable_path = unreadable_archive / f"{spec.source_id}-{spec.revision}.archive"
    unreadable_path.write_bytes(archive)
    original_hash = external_module._sha256_path

    def raise_for_unreadable(path: Path) -> str:
        if path == unreadable_path:
            raise OSError("permission denied")
        return original_hash(path)

    monkeypatch.setattr(external_module, "_sha256_path", raise_for_unreadable)
    unreadable = acquire_external_source(spec, unreadable_archive, allow_network=True)
    assert unreadable["status"] == "failed"
    assert "cached_archive_unreadable" in str(unreadable["reason"])
    monkeypatch.setattr(external_module, "_sha256_path", original_hash)

    empty_manifest = tmp_path / "empty.json"
    empty_manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "manifest_id": "empty",
                "network_policy": "explicit_acquisition_only",
                "static_only": True,
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    empty_result = acquire_external_manifest(
        empty_manifest, tmp_path / "empty-output", allow_network=True
    )
    assert empty_result["status"] == "blocked_external"

    row = _valid_row()
    one_manifest = tmp_path / "one.json"
    one_manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "manifest_id": "one",
                "network_policy": "explicit_acquisition_only",
                "static_only": True,
                "sources": [row],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        external_module,
        "acquire_external_source",
        lambda spec, destination, *, allow_network: {
            "source_id": spec.source_id,
            "status": "validated",
            "network_used": False,
        },
    )
    successful = acquire_external_manifest(
        one_manifest, tmp_path / "one-output", allow_network=True
    )
    assert successful["status"] == "validated"
    assert successful["network_used"] is False


def test_acquisition_validator_rejects_each_boundary_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = external_acquisition_payload()
    mutations = [
        ("schema_version", "0.0.0"),
        ("execution_performed", True),
        ("static_only", False),
        ("manifest_id", ""),
        ("manifest_sha256", "bad"),
        ("acquisition", None),
    ]
    for key, value in mutations:
        candidate = json.loads(json.dumps(baseline))
        candidate[key] = value
        assert not validate_external_acquisition_payload(candidate)[0]

    for mutation in (
        lambda acquisition: acquisition.update(status="unknown"),
        lambda acquisition: acquisition.update(execution_performed=True),
        lambda acquisition: acquisition.update(rows={}),
        lambda acquisition: acquisition.update(
            rows=[{"status": "unknown", "network_used": False}]
        ),
        lambda acquisition: acquisition.update(
            rows=[{"status": "failed", "network_used": "yes"}]
        ),
    ):
        candidate = json.loads(json.dumps(baseline))
        mutation(candidate["acquisition"])
        assert not validate_external_acquisition_payload(candidate)[0]

    blocked_with_network = json.loads(json.dumps(baseline))
    blocked_with_network["status"] = "blocked_external"
    blocked_with_network["network_used"] = True
    assert not validate_external_acquisition_payload(blocked_with_network)[0]

    validated_with_failure = json.loads(json.dumps(baseline))
    validated_with_failure["status"] = "validated"
    validated_with_failure["acquisition"] = {
        "status": "validated",
        "rows": [{"status": "failed", "network_used": False}],
    }
    assert not validate_external_acquisition_payload(validated_with_failure)[0]
    assert not validate_external_acquisition_payload(None)[0]
    monkeypatch.setattr(
        external_module, "external_acquisition_payload", lambda *args, **kwargs: {}
    )
    with pytest.raises(ValueError, match="receipt"):
        write_external_acquisition_payload(tmp_path)


def test_static_validation_retains_missing_and_compilation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(
        external_module.PROJECT_ROOT / "configs/external_sources.yaml",
        configs / "external_sources.yaml",
    )
    shutil.copy(
        external_module.PROJECT_ROOT / "configs/external_label_contracts.yaml",
        configs / "external_label_contracts.yaml",
    )
    destination = tmp_path / "sources"
    missing = external_module.validate_external_static_sources(
        tmp_path, destination=destination
    )
    assert missing["status"] == "underpowered"
    assert all(row["status"] == "unavailable" for row in missing["rows"])
    receipt_path = external_module.write_external_static_validation(tmp_path)
    assert receipt_path.is_file()

    manifest = load_external_manifest(configs / "external_sources.yaml")
    first = manifest["sources"][0]
    source_root = destination / str(first["source_id"])
    source_root.mkdir(parents=True)
    monkeypatch.setattr(
        external_module,
        "static_compile_external_source",
        lambda spec, source_root, output_directory: (_ for _ in ()).throw(
            OSError("compiler test failure")
        ),
    )
    failed = external_module.validate_external_static_sources(
        tmp_path, destination=destination
    )
    first_row = failed["rows"][0]
    assert first_row["status"] == "failed"
    assert "static_compile_failed" in str(first_row["reason"])


def test_static_receipt_validator_and_compile_command_fail_closed(
    tmp_path: Path,
) -> None:
    static_base = {
        "schema_version": "2.0.0",
        "static_only": True,
        "execution_performed": False,
        "rows": [],
    }
    assert not validate_external_static_validation_payload({**static_base, "rows": {}})[
        0
    ]
    assert not validate_external_static_validation_payload(
        {**static_base, "rows": [[]]}
    )[0]
    assert not validate_external_static_validation_payload(
        {**static_base, "rows": [{"status": "validated"}]}
    )[0]
    assert not validate_compiler_capabilities_payload(
        {"schema_version": "2.0.0", "rows": [{}]}
    )[0]

    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("clang unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "fixture.c").write_text(
        "unsigned long f(unsigned long x) { return x + 1; }\n",
        encoding="utf-8",
    )
    base = ExternalSourceSpec(
        source_id="command-boundary",
        repository_url="https://example.invalid/fixture.tar.gz",
        revision="0123456789abcdef0123456789abcdef01234567",
        license="BSD-3-Clause",
        license_file="LICENSE",
        license_file_sha256="a" * 64,
        source_archive_sha256="b" * 64,
        source_family="fixture_family",
        build_command=(compiler, "-c", "fixture.c"),
        compiler=compiler,
        target_triple="x86_64-apple-darwin",
        label_contract="fixture-only",
        provenance_reviewer="test",
        acquisition_timestamp="2026-08-23T00:00:00Z",
    )
    with pytest.raises(ValueError, match="allow-listed"):
        static_compile_external_source(
            ExternalSourceSpec(**{**base.__dict__, "build_command": ("sh", "-c")}),
            source_root,
            tmp_path / "objects",
        )
    with pytest.raises(ValueError, match="no -o"):
        static_compile_external_source(
            ExternalSourceSpec(
                **{
                    **base.__dict__,
                    "build_command": (compiler, "-c", "fixture.c", "-o"),
                }
            ),
            source_root,
            tmp_path / "objects",
        )


def test_external_receipt_validators_reject_numeric_gate_and_status_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = external_source_validation_payload()
    for key in (
        "project_count",
        "labelable_example_count",
        "static_validated_row_count",
    ):
        candidate = json.loads(json.dumps(payload))
        candidate[key] = -1
        assert not validate_external_source_validation_payload(candidate)[0]
    candidate = json.loads(json.dumps(payload))
    candidate["static_rows"] = {}
    assert not validate_external_source_validation_payload(candidate)[0]
    candidate = json.loads(json.dumps(payload))
    candidate["promotion_gate"]["status"] = "validated"
    assert not validate_external_source_validation_payload(candidate)[0]
    candidate = json.loads(json.dumps(payload))
    candidate["label_contract_summary"] = []
    assert not validate_external_source_validation_payload(candidate)[0]
    candidate = json.loads(json.dumps(payload))
    candidate["network_used"] = True
    assert not validate_external_source_validation_payload(candidate)[0]
    monkeypatch.setattr(
        external_module,
        "external_source_validation_payload",
        lambda project_root: {},
    )
    with pytest.raises(ValueError, match="receipt"):
        write_external_source_validation_payload(tmp_path)
