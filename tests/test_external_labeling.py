"""Tests for deterministic source-bound external pair generation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

import src.external_labeling as labeling
from src.external_corpus import (
    exclusion_accounting_summary,
    validate_label_contracts,
)
from src.external_sources import _spec_from_row
from src.external_labeling import (
    external_label_contract_freshness,
    generate_label_contract,
    load_label_design,
    validate_external_label_contract_payload,
    write_external_label_contract,
)


def _write_fixture_inputs(project_root: Path) -> None:
    digest = "a" * 64
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "output" / "external_sources" / "fixture-source" / "repo").mkdir(
        parents=True, exist_ok=True
    )
    (
        project_root
        / "output"
        / "external_sources"
        / "fixture-source"
        / "repo"
        / "fixture.c"
    ).write_text(
        "unsigned long fixture(unsigned long x) {\n"
        "  unsigned long y = x + 1;\n"
        "  if (y > 3) return 1;\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "2.0.0",
        "manifest_id": "fixture-manifest",
        "network_policy": "explicit_acquisition_only",
        "static_only": True,
        "sources": [
            {
                "source_id": "fixture-source",
                "project_id": "fixture-project",
                "revision_label": "r1",
                "repository_url": "https://example.invalid/fixture.tar.gz",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "license": "MIT",
                "license_file": "LICENSE",
                "license_file_sha256": digest,
                "source_archive_sha256": digest,
                "source_family": "fixture-family",
                "build_command": ["clang", "-c", "fixture.c"],
                "compiler": "clang",
                "target_triple": "x86_64-apple-darwin",
                "label_contract": "fixture-source-contract",
                "provenance_reviewer": "test",
                "acquisition_timestamp": "2026-08-27T00:00:00Z",
            }
        ],
    }
    (project_root / "configs" / "external_sources.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    design = {
        "schema_version": "1.0.0",
        "design_id": "fixture-label-design",
        "minimum_labelable_examples": 1,
        "source_designs": [
            {
                "source_id": "fixture-source",
                "source_path": "fixture.c",
                "function_id": "fixture",
                "include_dirs": ["."],
                "compile_flags": [],
                "split": "test",
                "near_miss": [
                    {
                        "id": "addend",
                        "needle": "x + 1",
                        "replacement": "x + 2",
                        "preconditions": "A nonzero input reaches the addition.",
                        "argument": "The changed addend changes the source contract.",
                    },
                    {
                        "id": "branch_boundary",
                        "needle": "y > 3",
                        "replacement": "y >= 3",
                        "preconditions": "The boundary value is observed.",
                        "argument": "The strictness change changes the branch contract.",
                    },
                    {
                        "id": "then_value",
                        "needle": "return 1;",
                        "replacement": "return 2;",
                        "preconditions": "The then branch is taken.",
                        "argument": "The returned branch value changes.",
                    },
                    {
                        "id": "else_value",
                        "needle": "return 0;",
                        "replacement": "return 3;",
                        "preconditions": "The else branch is taken.",
                        "argument": "The returned branch value changes.",
                    },
                ],
            }
        ],
    }
    (project_root / "configs" / "external_label_design.yaml").write_text(
        yaml.safe_dump(design, sort_keys=False), encoding="utf-8"
    )


def test_fixture_label_producer_writes_static_source_bound_receipt(
    tmp_path: Path,
) -> None:
    _write_fixture_inputs(tmp_path)
    receipt_path = write_external_label_contract(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))

    # This fixture intentionally has only a test partition; the tightened
    # project-holdout gate correctly keeps it underpowered because it has no
    # fitting project to hold out from.
    assert payload["status"] == "underpowered"
    assert len(payload["contracts"]) == 10
    assert payload["exclusions"] == []
    assert payload["summary"]["project_held_out_test"] is False
    assert payload["summary"]["source_family_split_disjoint"] is True
    valid, detail = validate_external_label_contract_payload(payload)
    assert valid, detail
    fresh, detail = external_label_contract_freshness(payload, tmp_path)
    assert fresh, detail
    first = payload["contracts"][0]
    assert first["static_only"] is True
    assert first["execution_performed"] is False
    assert first["base_object_sha256"] != ""
    assert first["pair_disassembly_sha256"] != ""
    assert (tmp_path / "configs" / "external_label_contracts.yaml").is_file()


def test_label_receipt_writer_fails_closed_if_post_write_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)

    def reject_receipt(_: object) -> tuple[bool, str]:
        return False, "forced post-write receipt failure"

    monkeypatch.setattr(
        labeling, "validate_external_label_contract_payload", reject_receipt
    )
    with pytest.raises(ValueError, match="forced post-write receipt failure"):
        write_external_label_contract(tmp_path)


def test_generator_retains_missing_source_as_exclusion(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    shutil.rmtree(tmp_path / "output" / "external_sources")
    payload, _ = generate_label_contract(tmp_path)
    assert payload["contracts"] == []
    assert len(payload["exclusions"]) == 1
    assert payload["exclusions"][0]["status"] == "excluded"


def test_v2_validator_rejects_project_split_leakage(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    payload, _ = generate_label_contract(tmp_path)
    row = dict(payload["contracts"][0])
    leaked = dict(row)
    leaked["example_id"] = "leaked-row"
    leaked["split"] = "train"
    valid, detail = validate_label_contracts(
        {
            "schema_version": "2.0.0",
            "contracts": [row, leaked],
        }
    )
    assert not valid
    assert "split leakage" in detail


def test_v2_validator_rejects_execution_and_unsafe_artifact(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    payload, _ = generate_label_contract(tmp_path)
    row = dict(payload["contracts"][0])
    row["execution_performed"] = True
    valid, detail = validate_label_contracts(
        {"schema_version": "2.0.0", "contracts": [row]}
    )
    assert not valid
    assert "execution_performed" in detail
    row["execution_performed"] = False
    row["pair_object_artifact"] = "../outside.o"
    valid, detail = validate_external_label_contract_payload(
        {
            "schema_version": "2.0.0",
            "contracts": [row],
            "exclusions": [],
        }
    )
    assert not valid
    assert "safe relative path" in detail or "escapes project root" in detail


def test_label_design_rejects_nonpositive_occurrence(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    design["source_designs"][0]["near_miss"][0]["occurrence"] = 0
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")
    with pytest.raises(ValueError, match="occurrence"):
        load_label_design(design_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.clear(), "schema mismatch"),
        (
            lambda payload: payload.__setitem__("schema_version", "0.0.0"),
            "schema mismatch",
        ),
        (
            lambda payload: payload.__setitem__("source_designs", []),
            "must be non-empty",
        ),
        (
            lambda payload: payload.__setitem__("source_designs", ["bad"]),
            "is not an object",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__("source_id", ""),
            "no source_id",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__(
                "source_path", "/tmp/escape.c"
            ),
            "relative path",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__(
                "source_path", "../escape.c"
            ),
            "escape the source root",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__(
                "include_dirs", [""]
            ),
            "include_dirs malformed",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__(
                "compile_flags", [1]
            ),
            "compile_flags malformed",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__("near_miss", []),
            "needs near_miss",
        ),
    ],
)
def test_label_design_loader_rejects_structural_drift(
    tmp_path: Path, mutation: object, message: str
) -> None:
    _write_fixture_inputs(tmp_path)
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    payload = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert callable(mutation)
    mutation(payload)
    design_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_label_design(design_path)


def test_label_design_loader_rejects_non_object_and_unreadable_inputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "design.yaml"
    path.write_text("- not-an-object\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_label_design(path)
    path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_label_design(path)
    with pytest.raises(ValueError, match="unreadable"):
        load_label_design(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["source_designs"].append(
                deepcopy(payload["source_designs"][0])
            ),
            "source_id duplicated",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__("function_id", ""),
            "field function_id is empty",
        ),
        (
            lambda payload: payload["source_designs"][0].__setitem__(
                "near_miss", ["bad"]
            ),
            "transformation .* is not an object",
        ),
        (
            lambda payload: payload["source_designs"][0]["near_miss"][0].pop(
                "argument"
            ),
            "missing .*argument",
        ),
        (
            lambda payload: payload["source_designs"][0]["near_miss"][0].__setitem__(
                "id", ""
            ),
            "transformation id duplicated",
        ),
        (
            lambda payload: payload["source_designs"][0]["near_miss"][0].__setitem__(
                "replacement", "x + 1"
            ),
            "identity edit",
        ),
    ],
)
def test_label_design_loader_rejects_transformation_drift(
    tmp_path: Path, mutation: object, message: str
) -> None:
    _write_fixture_inputs(tmp_path)
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    payload = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert callable(mutation)
    mutation(payload)
    design_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_label_design(design_path)


def _fixture_spec_and_repo(tmp_path: Path):
    _write_fixture_inputs(tmp_path)
    manifest = yaml.safe_load(
        (tmp_path / "configs" / "external_sources.yaml").read_text(encoding="utf-8")
    )
    spec = _spec_from_row(manifest["sources"][0], 0)
    return spec, tmp_path / "output" / "external_sources" / "fixture-source" / "repo"


def test_static_variant_fail_closed_for_escape_unavailable_and_compile_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, repository_root = _fixture_spec_and_repo(tmp_path)
    escaped = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / "escape",
        include_dirs=["../outside"],
        compile_flags=[],
    )
    assert escaped["status"] == "failed"
    assert escaped["reason"] == "include_directory_escapes_source_root"

    monkeypatch.setattr(labeling.shutil, "which", lambda _: None)
    unavailable = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / "unavailable",
        include_dirs=[],
        compile_flags=[],
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "compiler_unavailable"

    monkeypatch.setattr(labeling.shutil, "which", lambda _: "/usr/bin/clang")
    monkeypatch.setattr(
        labeling.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("compiler missing")),
    )
    failed = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / "exception",
        include_dirs=[],
        compile_flags=[],
    )
    assert failed["status"] == "failed"
    assert "static_compile_exception" in str(failed["reason"])


@pytest.mark.parametrize("failure_kind", ["compile", "disassembler", "decode"])
def test_static_variant_records_compile_disassembly_and_decode_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    spec, repository_root = _fixture_spec_and_repo(tmp_path)

    def fake_which(name: str) -> str | None:
        if name == "clang":
            return "/usr/bin/clang"
        return "/usr/bin/objdump"

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[0].endswith("clang"):
            if failure_kind == "compile":
                return SimpleNamespace(returncode=1, stdout="", stderr="bad source")
            object_path = Path(command[command.index("-o") + 1])
            object_path.write_bytes(b"fixture-object")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if failure_kind == "disassembler":
            return SimpleNamespace(returncode=1, stdout="", stderr="bad objdump")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(labeling.shutil, "which", fake_which)
    monkeypatch.setattr(labeling.subprocess, "run", fake_run)
    if failure_kind == "decode":
        monkeypatch.setattr(
            labeling,
            "_extract_text_bytes",
            lambda *_: (_ for _ in ()).throw(ValueError("bad text")),
        )
    result = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / failure_kind,
        include_dirs=[],
        compile_flags=[],
    )
    assert (
        result["status"]
        == {
            "compile": "failed",
            "disassembler": "failed",
            "decode": "unsupported",
        }[failure_kind]
    )


def test_label_receipt_validation_and_freshness_reject_drift(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    receipt_path = write_external_label_contract(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))

    invalid_payloads = [
        None,
        {**payload, "schema_version": "0.0.0"},
        {**payload, "contracts": {}},
        {**payload, "exclusions": {}},
    ]
    for candidate in invalid_payloads:
        valid, _ = validate_external_label_contract_payload(candidate)
        assert not valid

    bad_digest = deepcopy(payload)
    bad_digest["contracts"][0]["base_object_sha256"] = "bad-digest"
    valid, detail = validate_external_label_contract_payload(bad_digest)
    assert not valid
    assert "hash mismatch" not in detail
    assert "digest" in detail

    missing_input = deepcopy(payload)
    (tmp_path / "configs" / "external_label_design.yaml").unlink()
    fresh, detail = external_label_contract_freshness(missing_input, tmp_path)
    assert not fresh
    assert "inputs are missing" in detail


def test_label_generator_retains_manifest_and_base_build_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    design["source_designs"][0]["source_id"] = "not-in-manifest"
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")
    payload, _ = generate_label_contract(tmp_path)
    assert payload["contracts"] == []
    assert payload["exclusions"][0]["reason"] == (
        "source_id_missing_from_pinned_manifest"
    )

    # A failed base compilation excludes every intended role explicitly; it
    # must not silently reduce the denominator.
    design["source_designs"][0]["source_id"] = "fixture-source"
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")
    monkeypatch.setattr(
        labeling,
        "_prepare_compiled_variant",
        lambda *args, **kwargs: {
            "status": "failed",
            "reason": "forced_base_failure",
        },
    )
    payload, _ = generate_label_contract(tmp_path)
    assert payload["contracts"] == []
    assert len(payload["exclusions"]) == 10
    assert all(
        "base_static_validation" in row["reason"] for row in payload["exclusions"]
    )


def test_label_generator_retains_pair_build_exclusions_and_missing_needles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)
    calls = 0

    def fake_prepare(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            cache_directory = Path(str(kwargs["cache_directory"]))
            cache_directory.mkdir(parents=True, exist_ok=True)
            (cache_directory / "source.c").write_text(
                "int fixture(void) { return 0; }\n", encoding="utf-8"
            )
            (cache_directory / "object.o").write_bytes(b"fixture-object")
            (cache_directory / "disassembly.json").write_text("[]", encoding="utf-8")
            return {
                "status": "validated",
                "object_sha256": "a" * 64,
                "disassembly_sha256": "b" * 64,
            }
        return {"status": "failed", "reason": "forced_pair_failure"}

    monkeypatch.setattr(labeling, "_prepare_compiled_variant", fake_prepare)
    payload, _ = generate_label_contract(tmp_path)
    assert len(payload["contracts"]) == 2
    assert {row["role"] for row in payload["contracts"]} == {"negative_control"}
    assert len(payload["exclusions"]) == 8
    assert {row["role"] for row in payload["exclusions"]} == {
        "equivalent",
        "near_miss",
    }

    # Restore the real compiler path and make one declared source edit
    # unavailable.  The remaining rows still proceed, while the unavailable
    # contract remains visible as an exclusion.
    monkeypatch.undo()
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    design["source_designs"][0]["near_miss"][0]["needle"] = "never-present"
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")
    payload, _ = generate_label_contract(tmp_path)
    assert payload["contracts"]
    assert any(
        "needle_occurrence_count=0" in row["reason"] for row in payload["exclusions"]
    )


def test_static_variant_records_disassembler_unavailability_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, repository_root = _fixture_spec_and_repo(tmp_path)

    def fake_which(name: str) -> str | None:
        return "/usr/bin/clang" if name == "clang" else None

    def fake_compile(command: list[str], **_: object) -> SimpleNamespace:
        object_path = Path(command[command.index("-o") + 1])
        object_path.write_bytes(b"fixture-object")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(labeling.shutil, "which", fake_which)
    monkeypatch.setattr(labeling.subprocess, "run", fake_compile)
    unavailable = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / "no-disassembler",
        include_dirs=[],
        compile_flags=[],
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "disassembler_unavailable"

    def raising_disassembler(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0].endswith("clang"):
            return fake_compile(command, **kwargs)
        raise subprocess.SubprocessError("objdump failed to start")

    monkeypatch.setattr(
        labeling.shutil,
        "which",
        lambda name: "/usr/bin/clang" if name == "clang" else "/usr/bin/objdump",
    )
    monkeypatch.setattr(labeling.subprocess, "run", raising_disassembler)
    failed = labeling._compile_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        artifact_directory=tmp_path / "disassembler-exception",
        include_dirs=[],
        compile_flags=[],
    )
    assert failed["status"] == "failed"
    assert "disassembly_exception" in str(failed["reason"])


def test_source_locator_rejects_ambiguous_and_escaped_matches(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    (root / "one").mkdir(parents=True)
    (root / "two").mkdir(parents=True)
    (root / "one" / "fixture.c").write_text("int a;\n", encoding="utf-8")
    (root / "two" / "fixture.c").write_text("int b;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resolved to 2 files"):
        labeling._find_source_file(root, "fixture.c")

    outside = tmp_path / "outside.c"
    outside.write_text("int outside;\n", encoding="utf-8")
    escaped_root = tmp_path / "escaped"
    escaped_root.mkdir()
    (escaped_root / "fixture.c").symlink_to(outside)
    with pytest.raises(ValueError, match="escaped the acquired source root"):
        labeling._find_source_file(escaped_root, "fixture.c")


def test_label_generator_retains_inconsistent_occurrence_lookup_as_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)
    design_path = tmp_path / "configs" / "external_label_design.yaml"
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    design["source_designs"][0]["near_miss"][0]["needle"] = "x + 1"
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")

    class InconsistentText(str):
        def count(self, substring: str, *args: object) -> int:
            return 1 if substring == "x + 1" else super().count(substring, *args)

        def find(self, substring: str, *args: object) -> int:
            return -1 if substring == "x + 1" else super().find(substring, *args)

    class SourceProxy:
        def read_text(self, **_: object) -> InconsistentText:
            return InconsistentText(
                "unsigned long fixture(unsigned long x) { return x + 1; }\n"
            )

    repository_root = (
        tmp_path / "output" / "external_sources" / "fixture-source" / "repo"
    )
    monkeypatch.setattr(
        labeling,
        "_find_source_file",
        lambda *_: (SourceProxy(), repository_root),
    )
    payload, _ = generate_label_contract(tmp_path)
    assert any("could not be located" in row["reason"] for row in payload["exclusions"])


def test_label_validation_matrix_and_freshness_matrix_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)
    receipt_path = write_external_label_contract(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))

    # Bypass the row-schema validator only for this matrix so the receipt
    # boundary's own static-safety, digest, and path checks are each tested.
    monkeypatch.setattr(labeling, "validate_label_contracts", lambda _: (True, "ok"))
    row = deepcopy(payload["contracts"][0])
    for candidate, expected in (
        ({**payload, "receipt_schema_version": "0.0.0"}, "receipt"),
        ({**payload, "contracts": [object()]}, "row is not an object"),
        (
            {**payload, "contracts": [{**row, "execution_performed": True}]},
            "static-only safety",
        ),
        (
            {**payload, "contracts": [{**row, "pair_object_sha256": "bad"}]},
            "is not a digest",
        ),
        (
            {
                **payload,
                "contracts": [{**row, "pair_object_artifact": "../outside.o"}],
            },
            "escapes project root",
        ),
    ):
        valid, detail = validate_external_label_contract_payload(candidate)
        assert not valid
        assert expected in detail

    monkeypatch.undo()
    stale_design = deepcopy(payload)
    stale_design["design_sha256"] = "0" * 64
    valid, detail = external_label_contract_freshness(stale_design, tmp_path)
    assert not valid and "design hash" in detail
    stale_manifest = deepcopy(payload)
    stale_manifest["manifest_sha256"] = "0" * 64
    valid, detail = external_label_contract_freshness(stale_manifest, tmp_path)
    assert not valid and "manifest hash" in detail
    stale_summary = deepcopy(payload)
    stale_summary["summary"] = {}
    valid, detail = external_label_contract_freshness(stale_summary, tmp_path)
    assert not valid and "summary" in detail

    missing_artifact = deepcopy(payload)
    missing_artifact["contracts"][0]["pair_source_artifact"] = "missing.c"
    valid, detail = external_label_contract_freshness(missing_artifact, tmp_path)
    assert not valid and "artifact is missing" in detail
    unreadable = deepcopy(payload)
    unreadable["contracts"][0]["pair_disassembly_artifact"] = unreadable["contracts"][
        0
    ]["pair_source_artifact"]
    valid, detail = external_label_contract_freshness(unreadable, tmp_path)
    assert not valid and "disassembly artifact is unreadable" in detail
    mismatched = deepcopy(payload)
    mismatched["contracts"][0]["pair_source_sha256"] = "0" * 64
    valid, detail = external_label_contract_freshness(mismatched, tmp_path)
    assert not valid and "artifact hash mismatch" in detail


def test_label_receipt_rejects_summary_and_config_row_drift(tmp_path: Path) -> None:
    _write_fixture_inputs(tmp_path)
    receipt_path = write_external_label_contract(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))

    missing_summary = deepcopy(payload)
    missing_summary.pop("summary")
    valid, detail = validate_external_label_contract_payload(missing_summary)
    assert not valid and "receipt fields missing" in detail

    stale_summary = deepcopy(payload)
    stale_summary["summary"]["contract_count"] += 1
    valid, detail = validate_external_label_contract_payload(stale_summary)
    assert not valid and "summary count drift" in detail

    config_path = tmp_path / "configs" / "external_label_contracts.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["contracts"][0]["equivalence_argument"] = "edited after generation"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    fresh, detail = external_label_contract_freshness(payload, tmp_path)
    assert not fresh and "configuration rows are stale" in detail


def test_label_payload_helper_and_prepare_failure_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_inputs(tmp_path)
    payload = labeling.external_label_contract_payload(tmp_path)
    assert payload["contracts"]
    spec, repository_root = _fixture_spec_and_repo(tmp_path)
    monkeypatch.setattr(
        labeling,
        "_compile_variant",
        lambda *args, **kwargs: {"status": "failed", "reason": "forced"},
    )
    prepared = labeling._prepare_compiled_variant(
        "int fixture(void) { return 0; }",
        spec=spec,
        repository_root=repository_root,
        cache_directory=tmp_path / "prepared-failure",
        include_dirs=[],
        compile_flags=[],
    )
    assert prepared["status"] == "failed"


def test_exclusion_accounting_aggregates_step_reason_role_counts() -> None:
    rows = [
        {
            "source_id": "fixture-a",
            "role": "all",
            "transformation_id": "source_lookup",
            "status": "excluded",
            "reason": "source_unavailable",
        },
        {
            "source_id": "fixture-a",
            "role": "equivalent",
            "status": "excluded",
            "reason": "pair_static_validation:compile_failed",
        },
        {
            "source_id": "fixture-a",
            "role": "equivalent",
            "status": "excluded",
            "reason": "pair_static_validation:compile_failed",
        },
        {
            "step": "archive_digest_verify",
            "role": "all",
            "reason": "cached_source_archive_digest_mismatch",
        },
    ]
    summary = exclusion_accounting_summary(rows)
    assert summary["total"] == 4
    # Label-contract exclusions carry no step and default to label_contract.
    assert summary["by_step"] == {
        "archive_digest_verify": 1,
        "label_contract": 3,
    }
    assert summary["by_reason"]["pair_static_validation:compile_failed"] == 2
    assert summary["by_role"] == {"all": 2, "equivalent": 2}
    assert {
        "step": "label_contract",
        "reason": "source_unavailable",
        "role": "all",
        "count": 1,
    } in summary["groups"]
    assert {
        "step": "archive_digest_verify",
        "reason": "cached_source_archive_digest_mismatch",
        "role": "all",
        "count": 1,
    } in summary["groups"]
    # An empty exclusion set is recorded explicitly, never dropped.
    assert exclusion_accounting_summary([]) == {
        "total": 0,
        "by_step": {},
        "by_reason": {},
        "by_role": {},
        "groups": [],
    }
    assert exclusion_accounting_summary(None)["total"] == 0
