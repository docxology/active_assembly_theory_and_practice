"""Safety and provenance tests for the pinned external-source boundary."""

from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest
import src.external_sources as external_module

from src.external_sources import (
    ExternalSourceSpec,
    _safe_extract,
    _validate_build_command,
    _validate_relative_path,
    acquire_external_manifest,
    acquire_external_source,
    external_source_validation_payload,
    load_external_manifest,
    static_compile_external_source,
    validate_external_manifest,
    validate_external_source_validation_payload,
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


def test_empty_repository_manifest_is_explicitly_blocked_and_valid() -> None:
    payload = external_source_validation_payload()
    valid, detail = validate_external_source_validation_payload(payload)
    assert valid, detail
    assert payload["status"] == "blocked_external"
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
    assert (output / "fixture.o").is_file()


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
            {**_valid_row(), "source_id": "project_b"},
        ],
    }
    assert validate_external_manifest(duplicate_family)[0] is False


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
