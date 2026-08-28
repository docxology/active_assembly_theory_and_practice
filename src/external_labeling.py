"""Produce source-bound labels for the pinned external-source track.

The external corpus is deliberately built from source text and a written
contract.  This module never labels a row from a model prediction, compiled
bytes, a filename, or a transformation name alone.  Equivalent rows are
lexical witnesses (a trailing C comment is appended after the source),
near-miss rows are explicit contract-directed source edits, and identity rows
are retained as non-separating controls.  Every accepted row must compile both
the base and pair source through an explicit ``-c`` command and decode the
resulting object with Capstone.  Emitted objects are never executed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, cast

import yaml

from .artifact_contract import write_json_artifact, write_json_atomic
from .compiler_transfer import _decode_bytes, _extract_text_bytes
from .external_corpus import (
    LABEL_CONTRACT_PATH,
    LABEL_CONTRACT_SCHEMA_VERSION,
    label_contract_summary,
    load_label_contracts,
    validate_label_contracts,
)
from .external_sources import (
    EXTERNAL_MANIFEST_PATH,
    ExternalSourceSpec,
    _spec_from_row,
    load_external_manifest,
)
from .provenance import (
    generator_revision,
    sha256_bytes,
    sha256_path,
    toolchain_versions,
)
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DESIGN_PATH = "configs/external_label_design.yaml"
LABEL_PAIR_OUTPUT = "output/external_label_pairs"
LABEL_CONTRACT_ARTIFACT = "output/data/external_label_contract.json"
LABEL_DESIGN_SCHEMA_VERSION = "1.0.0"
LABEL_RECEIPT_SCHEMA_VERSION = "2.0.0"
_SAFE_RELATIVE = re.compile(r"^[^/].*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "artifact",
        "design_id",
        "design_schema_version",
        "design_sha256",
        "manifest_sha256",
        "status",
        "minimum_labelable_examples",
        "summary",
        "exclusions",
        "static_only",
        "execution_performed",
        "limitations",
        "unsupported_cases",
        "underpowered_cases",
    }
)


def _digest(value: bytes) -> str:
    """Return the canonical digest used for source and object evidence."""

    return hashlib.sha256(value).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _validate_relative_source_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{field} must be a relative path")
    candidate = Path(value)
    if ".." in candidate.parts or not _SAFE_RELATIVE.fullmatch(value):
        raise ValueError(f"{field} may not escape the source root")
    return value


def load_label_design(
    path: Path = PROJECT_ROOT / LABEL_DESIGN_PATH,
) -> dict[str, Any]:
    """Load and structurally validate the deterministic label design."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"external label design unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("external label design must be an object")
    if payload.get("schema_version") != LABEL_DESIGN_SCHEMA_VERSION:
        raise ValueError("external label design schema mismatch")
    designs = payload.get("source_designs")
    if not isinstance(designs, list) or not designs:
        raise ValueError("external label design source_designs must be non-empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw_design in enumerate(designs):
        if not isinstance(raw_design, dict):
            raise ValueError(f"external label design row {index} is not an object")
        source_id = raw_design.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"external label design row {index} has no source_id")
        if source_id in seen:
            raise ValueError(f"external label design source_id duplicated: {source_id}")
        seen.add(source_id)
        for field in ("source_path", "function_id", "split"):
            if field not in raw_design or not str(raw_design[field]).strip():
                raise ValueError(
                    f"external label design row {index} field {field} is empty"
                )
        source_path = _validate_relative_source_path(
            raw_design["source_path"], field="source_path"
        )
        include_dirs = raw_design.get("include_dirs", [])
        compile_flags = raw_design.get("compile_flags", [])
        if not isinstance(include_dirs, list) or any(
            not isinstance(item, str) or not item.strip() for item in include_dirs
        ):
            raise ValueError(
                f"external label design row {index} include_dirs malformed"
            )
        if not isinstance(compile_flags, list) or any(
            not isinstance(item, str) or not item for item in compile_flags
        ):
            raise ValueError(
                f"external label design row {index} compile_flags malformed"
            )
        near_miss = raw_design.get("near_miss")
        if not isinstance(near_miss, list) or not near_miss:
            raise ValueError(
                f"external label design row {index} needs near_miss transformations"
            )
        transformation_ids: set[str] = set()
        normalized_near_miss: list[dict[str, str]] = []
        for transformation_index, raw_transformation in enumerate(near_miss):
            if not isinstance(raw_transformation, dict):
                raise ValueError(
                    f"external label design transformation {index}:{transformation_index} is not an object"
                )
            required = ("id", "needle", "replacement", "preconditions", "argument")
            missing = [field for field in required if field not in raw_transformation]
            if missing:
                raise ValueError(
                    f"external label design transformation {index}:{transformation_index} missing {missing}"
                )
            transformation_id = str(raw_transformation["id"])
            if not transformation_id or transformation_id in transformation_ids:
                raise ValueError(
                    f"external label design transformation id duplicated: {transformation_id}"
                )
            transformation_ids.add(transformation_id)
            if raw_transformation["needle"] == raw_transformation["replacement"]:
                raise ValueError(
                    f"external label design transformation {transformation_id} is an identity edit"
                )
            occurrence = raw_transformation.get("occurrence", 1)
            if not isinstance(occurrence, int) or occurrence < 1:
                raise ValueError(
                    f"external label design transformation {transformation_id} occurrence must be a positive integer"
                )
            normalized_near_miss.append(
                {
                    **{field: str(raw_transformation[field]) for field in required},
                    "occurrence": str(occurrence),
                }
            )
        normalized.append(
            {
                "source_id": source_id,
                "source_path": source_path,
                "function_id": str(raw_design["function_id"]),
                "include_dirs": [str(item) for item in include_dirs],
                "compile_flags": [str(item) for item in compile_flags],
                "split": str(raw_design["split"]),
                "near_miss": normalized_near_miss,
            }
        )
    result = dict(payload)
    result["source_designs"] = normalized
    result["design_id"] = str(payload.get("design_id", ""))
    return result


def _find_source_file(source_root: Path, relative_path: str) -> tuple[Path, Path]:
    """Find exactly one source file and its extracted repository root."""

    path = _validate_relative_source_path(relative_path, field="source_path")
    candidates = sorted(
        candidate
        for candidate in source_root.rglob(path)
        if candidate.is_file() and ".cache" not in candidate.parts
    )
    if len(candidates) != 1:
        raise ValueError(
            f"source path {path} resolved to {len(candidates)} files under {source_root}"
        )
    source_file = candidates[0].resolve()
    source_parts = len(Path(path).parts)
    repository_root = source_file.parents[source_parts - 1]
    root_resolved = source_root.resolve()
    if root_resolved not in source_file.parents and source_file != root_resolved:
        raise ValueError("source file escaped the acquired source root")
    if (
        repository_root != root_resolved
        and root_resolved not in repository_root.parents
    ):
        raise ValueError("repository root escaped the acquired source root")
    return source_file, repository_root


def _write_yaml_atomic(path: Path, payload: object) -> None:
    """Write a generated YAML contract without exposing partial rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _compile_variant(
    source: str,
    *,
    spec: ExternalSourceSpec,
    repository_root: Path,
    artifact_directory: Path,
    include_dirs: list[str],
    compile_flags: list[str],
) -> dict[str, Any]:
    """Compile and decode one source variant through the static-only boundary."""

    artifact_directory.mkdir(parents=True, exist_ok=True)
    source_path = artifact_directory / "source.c"
    object_path = artifact_directory / "object.o"
    disassembly_path = artifact_directory / "disassembly.json"
    source_path.write_text(source, encoding="utf-8")
    compiler = shutil.which(spec.compiler)
    command: list[str] = [
        compiler or spec.compiler,
        "-target",
        spec.target_triple,
        "-c",
        "-O0",
        "-x",
        "c",
        *compile_flags,
    ]
    for include_dir in include_dirs:
        include_path = (repository_root / include_dir).resolve()
        if (
            repository_root.resolve() not in include_path.parents
            and include_path != repository_root.resolve()
        ):
            return {
                "status": "failed",
                "reason": "include_directory_escapes_source_root",
                "command": command,
                "static_only": True,
                "execution_performed": False,
            }
        command.extend(["-I", str(include_path)])
    command.extend(["-o", str(object_path), "-"])
    recorded_command = ["<compiler>", *command[1:-2], "<object>", "<source-stdin>"]
    if compiler is None:
        return {
            "status": "unavailable",
            "reason": "compiler_unavailable",
            "command": recorded_command,
            "static_only": True,
            "execution_performed": False,
        }
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            input=source,
            text=True,
            capture_output=True,
            shell=False,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "failed",
            "reason": f"static_compile_exception:{exc}",
            "command": recorded_command,
            "static_only": True,
            "execution_performed": False,
        }
    result: dict[str, Any] = {
        "status": "failed",
        "command": recorded_command,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4000:],
        "static_only": True,
        "execution_performed": False,
    }
    if completed.returncode != 0 or not object_path.is_file():
        result["reason"] = "static_compile_failed"
        return result
    object_bytes = object_path.read_bytes()
    result["object_sha256"] = _digest(object_bytes)
    disassembler = shutil.which("llvm-objdump") or shutil.which("objdump")
    if disassembler is None:
        result.update({"status": "unavailable", "reason": "disassembler_unavailable"})
        return result
    try:
        disassembly = subprocess.run(
            [disassembler, "-d", str(object_path)],
            capture_output=True,
            text=True,
            shell=False,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result.update({"status": "failed", "reason": f"disassembly_exception:{exc}"})
        return result
    if disassembly.returncode != 0:
        result.update(
            {
                "status": "failed",
                "reason": "disassembler_failed",
                "disassembly_stderr": disassembly.stderr[-4000:],
            }
        )
        return result
    try:
        text_bytes = _extract_text_bytes(disassembly.stdout, "x86_64")
        decoded = _decode_bytes(text_bytes, "x86_64")
    except (ValueError, RuntimeError) as exc:
        result.update(
            {"status": "unsupported", "reason": f"capstone_decode_failed:{exc}"}
        )
        return result
    write_json_atomic(disassembly_path, decoded)
    result.update(
        {
            "status": "validated",
            "disassembler": disassembler,
            "text_sha256": _digest(text_bytes),
            "disassembly_sha256": _digest(
                json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
            "decoded_instruction_count": len(decoded),
            "compiler": spec.compiler,
            "target_triple": spec.target_triple,
            "optimization": "-O0",
        }
    )
    return result


def _copy_variant_artifacts(
    compiled: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    """Copy cache artifacts into the row-specific audit directory."""

    source = Path(str(compiled["source_path"]))
    object_path = Path(str(compiled["object_path"]))
    disassembly_path = Path(str(compiled["disassembly_path"]))
    destination.mkdir(parents=True, exist_ok=True)
    copied = {
        "source": destination / "source.c",
        "object": destination / "object.o",
        "disassembly": destination / "disassembly.json",
    }
    shutil.copyfile(source, copied["source"])
    shutil.copyfile(object_path, copied["object"])
    shutil.copyfile(disassembly_path, copied["disassembly"])
    return {key: str(value) for key, value in copied.items()}


def _prepare_compiled_variant(
    source: str,
    *,
    spec: ExternalSourceSpec,
    repository_root: Path,
    cache_directory: Path,
    include_dirs: list[str],
    compile_flags: list[str],
) -> dict[str, Any]:
    result = _compile_variant(
        source,
        spec=spec,
        repository_root=repository_root,
        artifact_directory=cache_directory,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )
    if result.get("status") == "validated":
        result.update(
            {
                "source_path": str(cache_directory / "source.c"),
                "object_path": str(cache_directory / "object.o"),
                "disassembly_path": str(cache_directory / "disassembly.json"),
            }
        )
    return result


def _row(
    *,
    example_id: str,
    spec: ExternalSourceSpec,
    design: dict[str, Any],
    role: str,
    transformation_family: str,
    preconditions: str,
    argument: str,
    split: str,
    base_source: str,
    pair_source: str,
    base: dict[str, Any],
    pair: dict[str, Any],
    project_root: Path,
    base_artifacts: dict[str, str],
    pair_artifacts: dict[str, str],
) -> dict[str, object]:
    source_family = f"{spec.project_id}:{spec.source_family}:{design['function_id']}"
    base_source_digest = sha256_bytes(base_source.encode("utf-8"))
    pair_source_digest = sha256_bytes(pair_source.encode("utf-8"))
    base_object_digest = str(base["object_sha256"])
    pair_object_digest = str(pair["object_sha256"])
    base_disassembly_digest = str(base["disassembly_sha256"])
    pair_disassembly_digest = str(pair["disassembly_sha256"])
    return {
        "example_id": example_id,
        "project_id": spec.project_id,
        "revision_label": spec.revision_label,
        "source_id": spec.source_id,
        "source_family": source_family,
        "source_path": design["source_path"],
        "function_id": design["function_id"],
        "role": role,
        "transformation_family": transformation_family,
        "preconditions": preconditions,
        "equivalence_argument": argument,
        "contract_basis": "source_text_mechanical_contract",
        "source_sha256": base_source_digest,
        "pair_source_sha256": pair_source_digest,
        "base_object_sha256": base_object_digest,
        "pair_object_sha256": pair_object_digest,
        "base_disassembly_sha256": base_disassembly_digest,
        "pair_disassembly_sha256": pair_disassembly_digest,
        # Compatibility aliases intentionally point to the pair artifact.
        "object_sha256": pair_object_digest,
        "disassembly_sha256": pair_disassembly_digest,
        "base_source_artifact": _relative_path(
            Path(base_artifacts["source"]), project_root
        ),
        "pair_source_artifact": _relative_path(
            Path(pair_artifacts["source"]), project_root
        ),
        "base_object_artifact": _relative_path(
            Path(base_artifacts["object"]), project_root
        ),
        "pair_object_artifact": _relative_path(
            Path(pair_artifacts["object"]), project_root
        ),
        "base_disassembly_artifact": _relative_path(
            Path(base_artifacts["disassembly"]), project_root
        ),
        "pair_disassembly_artifact": _relative_path(
            Path(pair_artifacts["disassembly"]), project_root
        ),
        "compiler": spec.compiler,
        "target_triple": spec.target_triple,
        "optimization": "-O0",
        "static_only": True,
        "execution_performed": False,
        "labelability_status": "labelable",
        "split": split,
    }


def _exclusion(
    *, source_id: str, role: str, transformation_id: str, reason: str
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "role": role,
        "transformation_id": transformation_id,
        "status": "excluded",
        "reason": reason,
    }


def generate_label_contract(
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], list[Path]]:
    """Generate the source-bound YAML payload and its input-file list."""

    design_payload = load_label_design(project_root / LABEL_DESIGN_PATH)
    manifest = load_external_manifest(project_root / EXTERNAL_MANIFEST_PATH)
    manifest_rows = cast(list[dict[str, object]], manifest["sources"])
    specs = {
        spec.source_id: spec
        for index, row in enumerate(manifest_rows)
        for spec in [_spec_from_row(row, index)]
    }
    designs = cast(list[dict[str, Any]], design_payload["source_designs"])
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    input_paths = [
        project_root / LABEL_DESIGN_PATH,
        project_root / EXTERNAL_MANIFEST_PATH,
    ]
    pair_root = project_root / LABEL_PAIR_OUTPUT
    for design in designs:
        source_id = str(design["source_id"])
        spec = specs.get(source_id)
        if spec is None:
            exclusions.append(
                _exclusion(
                    source_id=source_id,
                    role="all",
                    transformation_id="source_manifest_lookup",
                    reason="source_id_missing_from_pinned_manifest",
                )
            )
            continue
        source_root = project_root / "output" / "external_sources" / source_id
        try:
            source_file, repository_root = _find_source_file(
                source_root, str(design["source_path"])
            )
            base_source = source_file.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            exclusions.append(
                _exclusion(
                    source_id=source_id,
                    role="all",
                    transformation_id="source_lookup",
                    reason=f"source_unavailable:{exc}",
                )
            )
            continue
        input_paths.append(source_file)
        cache_root = pair_root / ".cache" / source_id
        base = _prepare_compiled_variant(
            base_source,
            spec=spec,
            repository_root=repository_root,
            cache_directory=cache_root / "base",
            include_dirs=cast(list[str], design["include_dirs"]),
            compile_flags=cast(list[str], design["compile_flags"]),
        )
        if base.get("status") != "validated":
            reason = str(base.get("reason", "base_static_validation_failed"))
            for role, transformation_id in (
                [
                    ("equivalent", f"lexical_comment_{index:02d}")
                    for index in range(1, 5)
                ]
                + [("near_miss", str(item["id"])) for item in design["near_miss"]]
                + [
                    ("negative_control", f"identity_{index:02d}")
                    for index in range(1, 3)
                ]
            ):
                exclusions.append(
                    _exclusion(
                        source_id=source_id,
                        role=role,
                        transformation_id=transformation_id,
                        reason=f"base_static_validation:{reason}",
                    )
                )
            continue
        base["source_path"] = str(cache_root / "base" / "source.c")
        base["object_path"] = str(cache_root / "base" / "object.o")
        base["disassembly_path"] = str(cache_root / "base" / "disassembly.json")
        for variant_index in range(1, 5):
            example_id = f"ext-{source_id}-equiv-lexical-comment-{variant_index:02d}"
            pair_source = (
                base_source
                + f"\n/* AAP lexical-equivalence witness {source_id}-{variant_index:02d}: comments do not change preprocessing tokens. */\n"
            )
            pair = _prepare_compiled_variant(
                pair_source,
                spec=spec,
                repository_root=repository_root,
                cache_directory=cache_root / f"equiv-{variant_index:02d}",
                include_dirs=cast(list[str], design["include_dirs"]),
                compile_flags=cast(list[str], design["compile_flags"]),
            )
            if pair.get("status") != "validated":
                exclusions.append(
                    _exclusion(
                        source_id=source_id,
                        role="equivalent",
                        transformation_id=f"lexical_comment_{variant_index:02d}",
                        reason=f"pair_static_validation:{pair.get('reason', 'unknown')}",
                    )
                )
                continue
            row_directory = pair_root / example_id
            base_artifacts = _copy_variant_artifacts(base, row_directory / "base")
            pair_artifacts = _copy_variant_artifacts(pair, row_directory / "pair")
            rows.append(
                _row(
                    example_id=example_id,
                    spec=spec,
                    design=design,
                    role="equivalent",
                    transformation_family="lexical_comment_insertion",
                    preconditions="The source is compiled under the pinned static C translation-unit contract.",
                    argument="Appending a C comment after the translation unit preserves preprocessing tokens; this is a lexical witness, not a universal semantic proof.",
                    split=str(design["split"]),
                    base_source=base_source,
                    pair_source=pair_source,
                    base=base,
                    pair=pair,
                    project_root=project_root,
                    base_artifacts=base_artifacts,
                    pair_artifacts=pair_artifacts,
                )
            )
        for transformation in cast(list[dict[str, str]], design["near_miss"]):
            transformation_id = transformation["id"]
            needle = transformation["needle"]
            occurrences = base_source.count(needle)
            occurrence = int(transformation.get("occurrence", "1"))
            if occurrences < occurrence:
                exclusions.append(
                    _exclusion(
                        source_id=source_id,
                        role="near_miss",
                        transformation_id=transformation_id,
                        reason=f"needle_occurrence_count={occurrences}; requested occurrence={occurrence} is unavailable",
                    )
                )
                continue
            start = -1
            for _ in range(occurrence):
                start = base_source.find(needle, start + 1)
            if start < 0:
                exclusions.append(
                    _exclusion(
                        source_id=source_id,
                        role="near_miss",
                        transformation_id=transformation_id,
                        reason=f"needle_occurrence={occurrence} could not be located",
                    )
                )
                continue
            pair_source = (
                base_source[:start]
                + transformation["replacement"]
                + base_source[start + len(needle) :]
            )
            example_id = f"ext-{source_id}-near-miss-{transformation_id}"
            pair = _prepare_compiled_variant(
                pair_source,
                spec=spec,
                repository_root=repository_root,
                cache_directory=cache_root / f"near-{transformation_id}",
                include_dirs=cast(list[str], design["include_dirs"]),
                compile_flags=cast(list[str], design["compile_flags"]),
            )
            if pair.get("status") != "validated":
                exclusions.append(
                    _exclusion(
                        source_id=source_id,
                        role="near_miss",
                        transformation_id=transformation_id,
                        reason=f"pair_static_validation:{pair.get('reason', 'unknown')}",
                    )
                )
                continue
            row_directory = pair_root / example_id
            base_artifacts = _copy_variant_artifacts(base, row_directory / "base")
            pair_artifacts = _copy_variant_artifacts(pair, row_directory / "pair")
            rows.append(
                _row(
                    example_id=example_id,
                    spec=spec,
                    design=design,
                    role="near_miss",
                    transformation_family=f"source_contract_near_miss:{transformation_id}",
                    preconditions=transformation["preconditions"],
                    argument=transformation["argument"],
                    split=str(design["split"]),
                    base_source=base_source,
                    pair_source=pair_source,
                    base=base,
                    pair=pair,
                    project_root=project_root,
                    base_artifacts=base_artifacts,
                    pair_artifacts=pair_artifacts,
                )
            )
        for control_index in range(1, 3):
            example_id = (
                f"ext-{source_id}-negative-control-identity-{control_index:02d}"
            )
            pair_source = base_source
            row_directory = pair_root / example_id
            base_artifacts = _copy_variant_artifacts(base, row_directory / "base")
            pair_artifacts = _copy_variant_artifacts(base, row_directory / "pair")
            rows.append(
                _row(
                    example_id=example_id,
                    spec=spec,
                    design=design,
                    role="negative_control",
                    transformation_family="null_identity_control",
                    preconditions="No source transformation is applied; the pair is intentionally non-separating.",
                    argument="Byte-identical pairs test whether a scoring system invents a distinction; they are not semantic-difference labels.",
                    split=str(design["split"]),
                    base_source=base_source,
                    pair_source=pair_source,
                    base=base,
                    pair=base,
                    project_root=project_root,
                    base_artifacts=base_artifacts,
                    pair_artifacts=pair_artifacts,
                )
            )
    rows.sort(key=lambda item: str(item["example_id"]))
    valid, detail = validate_label_contracts(
        {
            "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
            "contracts": rows,
        }
    )
    if not valid and rows:
        raise ValueError(f"generated external label contract is invalid: {detail}")
    minimum = int(design_payload.get("minimum_labelable_examples", 100))
    # The proper project-held-out gate is computed by the corpus module after
    # the YAML is written; this conservative pre-write value is only used for
    # the empty/underpowered case.
    status = "validated" if len(rows) >= minimum else "underpowered"
    payload: dict[str, Any] = {
        "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
        "design_id": str(design_payload["design_id"]),
        "design_schema_version": LABEL_DESIGN_SCHEMA_VERSION,
        "status": status,
        "minimum_labelable_examples": minimum,
        "generated_by": "src.external_labeling.generate_label_contract",
        "design_sha256": sha256_path(project_root / LABEL_DESIGN_PATH),
        "manifest_sha256": sha256_path(project_root / EXTERNAL_MANIFEST_PATH),
        "contracts": rows,
        "exclusions": exclusions,
        "limitations": [
            "Equivalent rows are lexical comment-insertion witnesses and do not prove arbitrary semantic equivalence.",
            "Near-miss rows are source-contract controls with explicit preconditions; static compilation and disassembly do not execute or prove their behavior.",
            "Negative controls are byte-identical pairs retained to detect invented distinctions.",
            "Rows are labelable only when both source variants compile statically and decode with Capstone.",
        ],
    }
    return payload, input_paths


def external_label_contract_payload(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the generated receipt payload without writing the YAML manifest."""

    payload, _ = generate_label_contract(project_root)
    return payload


def validate_external_label_contract_payload(payload: object) -> tuple[bool, str]:
    """Validate a generated label receipt and its static-only boundary."""

    if not isinstance(payload, dict):
        return False, "external label contract receipt must be an object"
    if payload.get("schema_version") != LABEL_RECEIPT_SCHEMA_VERSION:
        return False, "external label contract schema mismatch"
    if "receipt_schema_version" in payload:
        valid, detail = validate_receipt_envelope(payload)
        if not valid:
            return False, f"external label contract receipt: {detail}"
    rows = payload.get("contracts")
    if not isinstance(rows, list):
        return False, "external label contract receipt contracts must be a list"
    valid, detail = validate_label_contracts(
        {"schema_version": LABEL_CONTRACT_SCHEMA_VERSION, "contracts": rows}
    )
    if not valid:
        return False, detail
    exclusions = payload.get("exclusions", [])
    if not isinstance(exclusions, list):
        return False, "external label contract exclusions must be a list"
    for row in rows:
        if not isinstance(row, dict):
            return False, "external label contract row is not an object"
        if (
            row.get("static_only") is not True
            or row.get("execution_performed") is not False
        ):
            return False, "external label contract row violates static-only safety"
        for field in (
            "base_object_sha256",
            "pair_object_sha256",
            "base_disassembly_sha256",
            "pair_disassembly_sha256",
        ):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or not re.fullmatch(r"[0-9a-f]{64}", value)
            ):
                return False, f"external label contract row {field} is not a digest"
        for field in (
            "base_source_artifact",
            "pair_source_artifact",
            "base_object_artifact",
            "pair_object_artifact",
            "base_disassembly_artifact",
            "pair_disassembly_artifact",
        ):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                return (
                    False,
                    f"external label contract row {field} escapes project root",
                )
    missing = sorted(_LABEL_RECEIPT_REQUIRED_FIELDS - set(payload))
    if missing:
        return False, f"external label contract receipt fields missing: {missing}"
    if payload.get("artifact") != LABEL_CONTRACT_ARTIFACT:
        return False, "external label contract receipt artifact path drifted"
    if payload.get("design_schema_version") != LABEL_DESIGN_SCHEMA_VERSION:
        return False, "external label contract design schema mismatch"
    if (
        not isinstance(payload.get("design_id"), str)
        or not payload["design_id"].strip()
    ):
        return False, "external label contract receipt design_id is empty"
    for field in ("design_sha256", "manifest_sha256"):
        if not _SHA256_RE.fullmatch(str(payload.get(field, ""))):
            return False, f"external label contract receipt {field} is not a digest"
    minimum = payload.get("minimum_labelable_examples")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        return False, "external label contract minimum_labelable_examples is invalid"
    if payload.get("static_only") is not True:
        return False, "external label contract receipt is not static-only"
    if payload.get("execution_performed") is not False:
        return False, "external label contract receipt records execution"
    status = payload.get("status")
    if status not in {"validated", "underpowered"}:
        return False, f"external label contract receipt status is invalid: {status!r}"
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False, "external label contract receipt summary must be an object"
    if summary.get("contract_count") != len(rows):
        return False, "external label contract receipt summary count drift"
    if summary.get("labelable_example_count") != len(rows):
        return False, "external label contract receipt summary labelable count drift"
    if summary.get("minimum_labelable_examples") != minimum:
        return False, "external label contract receipt summary minimum drift"
    projects = {str(row["project_id"]) for row in rows}
    test_projects = {str(row["project_id"]) for row in rows if row["split"] == "test"}
    fit_projects = {
        str(row["project_id"])
        for row in rows
        if row["split"] in {"train", "development"}
    }
    expected_project_holdout = (
        bool(test_projects)
        and bool(fit_projects)
        and test_projects.isdisjoint(fit_projects)
    )
    if summary.get("project_count") != len(projects):
        return False, "external label contract receipt summary project count drift"
    if summary.get("project_held_out_test") is not expected_project_holdout:
        return False, "external label contract receipt summary holdout drift"
    role_counts = {
        role: sum(row.get("role") == role for row in rows)
        for role in sorted({"equivalent", "near_miss", "negative_control"})
    }
    if summary.get("roles") != role_counts:
        return False, "external label contract receipt summary role counts drift"
    split_counts = {
        split: sum(row.get("split") == split for row in rows)
        for split in ("train", "development", "test")
    }
    if summary.get("split_counts") != split_counts:
        return False, "external label contract receipt summary split counts drift"
    expected_status = (
        "validated"
        if rows and len(rows) >= minimum and expected_project_holdout
        else "underpowered"
    )
    if status != expected_status:
        return False, "external label contract receipt status does not match summary"
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list) or any(
        not isinstance(row, dict) for row in exclusions
    ):
        return False, "external label contract exclusions must contain objects"
    return True, f"contracts={len(rows)} exclusions={len(exclusions)}"


def external_label_contract_freshness(
    payload: object,
    project_root: Path = PROJECT_ROOT,
) -> tuple[bool, str]:
    """Recompute source, object, and disassembly digests without mutation."""

    valid, detail = validate_external_label_contract_payload(payload)
    if not valid or not isinstance(payload, dict):
        return False, detail
    design_path = project_root / LABEL_DESIGN_PATH
    manifest_path = project_root / EXTERNAL_MANIFEST_PATH
    if not design_path.is_file() or not manifest_path.is_file():
        return False, "label freshness inputs are missing"
    if payload.get("design_sha256") != sha256_path(design_path):
        return False, "label design hash is stale"
    if payload.get("manifest_sha256") != sha256_path(manifest_path):
        return False, "label manifest hash is stale"
    try:
        current_summary = label_contract_summary(project_root)
    except (OSError, ValueError) as exc:
        return False, f"label contract configuration is unreadable: {exc}"
    if payload.get("summary") != current_summary:
        return False, "label contract summary is stale"
    rows = cast(list[dict[str, object]], payload["contracts"])
    for row in rows:
        for digest_field, artifact_field in (
            ("source_sha256", "base_source_artifact"),
            ("pair_source_sha256", "pair_source_artifact"),
            ("base_object_sha256", "base_object_artifact"),
            ("pair_object_sha256", "pair_object_artifact"),
            ("base_disassembly_sha256", "base_disassembly_artifact"),
            ("pair_disassembly_sha256", "pair_disassembly_artifact"),
        ):
            artifact_value = row.get(artifact_field)
            if not isinstance(artifact_value, str):
                return False, f"label artifact path is missing: {artifact_field}"
            artifact_path = (project_root / artifact_value).resolve()
            if project_root.resolve() not in artifact_path.parents:
                return False, f"label artifact escapes project root: {artifact_field}"
            if not artifact_path.is_file():
                return False, f"label artifact is missing: {artifact_value}"
            if "disassembly" in artifact_field:
                try:
                    decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    return False, f"label disassembly artifact is unreadable: {exc}"
                digest = _digest(
                    json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
            else:
                digest = sha256_path(artifact_path)
            if digest != row.get(digest_field):
                return False, f"label artifact hash mismatch: {artifact_value}"
    try:
        configured = load_label_contracts(project_root / LABEL_CONTRACT_PATH)
    except (OSError, ValueError) as exc:
        return False, f"label contract configuration is unreadable: {exc}"
    if configured.get("status") != payload.get("status"):
        return False, "label contract configuration status is stale"
    if configured.get("contracts") != rows:
        return False, "label contract configuration rows are stale"
    if configured.get("exclusions", []) != payload.get("exclusions"):
        return False, "label contract configuration exclusions are stale"
    return True, f"rows={len(rows)} source/object/disassembly digests match"


def write_external_label_contract(project_root: Path = PROJECT_ROOT) -> Path:
    """Generate source-bound pair artifacts, YAML contracts, and a receipt."""

    config_payload, input_paths = generate_label_contract(project_root)
    config_path = project_root / LABEL_CONTRACT_PATH
    _write_yaml_atomic(config_path, config_payload)
    summary = label_contract_summary(project_root)
    config_payload["summary"] = summary
    minimum = int(config_payload["minimum_labelable_examples"])
    final_status = (
        "validated"
        if len(cast(list[object], config_payload["contracts"])) >= minimum
        and bool(summary.get("project_held_out_test"))
        else "underpowered"
    )
    config_payload["status"] = final_status
    _write_yaml_atomic(config_path, config_payload)
    envelope = build_receipt_envelope(
        protocol=scientific_protocol_payload(),
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={
            LABEL_DESIGN_PATH: sha256_path(project_root / LABEL_DESIGN_PATH),
            EXTERNAL_MANIFEST_PATH: sha256_path(project_root / EXTERNAL_MANIFEST_PATH),
            **{
                # Include the complete project-relative path.  Two pinned
                # revisions deliberately contain files with the same basename
                # (for example ``png.c``); hashing by basename would silently
                # overwrite one input in the receipt and weaken freshness.
                f"source/{_relative_path(path, project_root)}": sha256_path(path)
                for path in input_paths
                if path.is_file()
                and path.name
                not in {"external_sources.yaml", "external_label_design.yaml"}
            },
        },
        toolchain_versions=toolchain_versions(("pyyaml", "capstone", "numpy")),
        status=final_status,
        limitations=cast(list[str], config_payload["limitations"]),
        unsupported_cases=(
            "No emitted external object is executed; all rows are static-only.",
            "Rows involving unsupported source/build conditions remain exclusions.",
        ),
        underpowered_cases=(
            "Source-level contracts do not establish universal semantic equivalence or application effectiveness.",
            "Independent replay remains a separate evidence gate.",
        ),
    )
    receipt: dict[str, Any] = {
        **envelope,
        "schema_version": LABEL_RECEIPT_SCHEMA_VERSION,
        "artifact": LABEL_CONTRACT_ARTIFACT,
        "design_id": config_payload["design_id"],
        "design_schema_version": config_payload["design_schema_version"],
        "design_sha256": config_payload["design_sha256"],
        "manifest_sha256": config_payload["manifest_sha256"],
        "status": final_status,
        "minimum_labelable_examples": minimum,
        "summary": summary,
        "contracts": config_payload["contracts"],
        "exclusions": config_payload["exclusions"],
        "static_only": True,
        "execution_performed": False,
    }
    path = project_root / LABEL_CONTRACT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    written = write_json_artifact(
        path,
        receipt,
        project_root=project_root,
        producer_path=Path(__file__),
        status=final_status,
        input_paths=input_paths,
    )
    valid, detail = validate_external_label_contract_payload(written)
    if not valid:
        raise ValueError(detail)
    return path


__all__ = [
    "LABEL_CONTRACT_ARTIFACT",
    "LABEL_DESIGN_PATH",
    "LABEL_DESIGN_SCHEMA_VERSION",
    "LABEL_PAIR_OUTPUT",
    "LABEL_RECEIPT_SCHEMA_VERSION",
    "external_label_contract_payload",
    "external_label_contract_freshness",
    "generate_label_contract",
    "load_label_design",
    "validate_external_label_contract_payload",
    "write_external_label_contract",
]
