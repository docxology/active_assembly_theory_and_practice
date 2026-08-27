"""Compiler/source/architecture transfer receipts with static-only decoding.

The producer uses the locally available Clang and object disassembler to build
small source families.  It never executes the resulting objects.  Every target
is probed independently; an unavailable target is retained as ``unavailable``
instead of being silently replaced by the host architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from capstone import CS_ARCH_ARM64, CS_ARCH_X86, CS_MODE_64, Cs

from .artifact_contract import write_json_artifact
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILER_TRANSFER_SCHEMA_VERSION = "2.0.0"
COMPILER_TRANSFER_ARTIFACT = "output/data/compiler_transfer.json"
_HEX_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+((?:[0-9a-fA-F]{2}\s+)+)")
_WORD_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+")

SOURCE_FAMILIES = {
    "straight_line_arithmetic": (
        "unsigned long aa_add(unsigned long x) { return x + 1; }\n"
        "unsigned long aa_mix(unsigned long x) { return (x + 3) * 2; }\n"
    ),
    "flag_sensitive_branch": (
        "unsigned long aa_branch(unsigned long x) { return x > 7 ? x + 2 : x - 2; }\n"
        "unsigned long aa_zero(unsigned long x) { return x == 0 ? 17 : x; }\n"
    ),
    "bounded_memory": (
        "unsigned long aa_memory(unsigned long x) {\n"
        "  unsigned long values[2] = {x, x + 1};\n"
        "  return values[0] + values[1];\n"
        "}\n"
    ),
    "stack_local": (
        "unsigned long aa_stack(unsigned long x) {\n"
        "  unsigned long local = x ^ 0x55UL;\n"
        "  return local + 9;\n"
        "}\n"
    ),
    "call_return": (
        "static unsigned long aa_helper(unsigned long x) { return x + 4; }\n"
        "unsigned long aa_call(unsigned long x) { return aa_helper(x) * 2; }\n"
    ),
    "bitwise_rewrite": (
        "unsigned long aa_bits(unsigned long x) {\n"
        "  return ((x << 1) | (x >> 3)) ^ 0xA5UL;\n"
        "}\n"
    ),
}
OPTIMIZATION_LEVELS = ("-O0", "-O2")
TARGETS = {
    "x86_64": "x86_64-apple-darwin",
    "aarch64": "aarch64-apple-darwin",
    "riscv64": "riscv64-unknown-elf",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(command: list[str], *, input_text: str | None = None) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _compiler_version() -> str:
    code, stdout, stderr = _run(["clang", "--version"])
    if code != 0:
        return f"unavailable:{stderr.strip()}"
    return stdout.splitlines()[0].strip() if stdout.splitlines() else "unknown"


def _extract_text_bytes(output: str, architecture: str) -> bytes:
    chunks: list[bytes] = []
    for line in output.splitlines():
        if architecture == "aarch64":
            match = _WORD_LINE.match(line)
            if match:
                chunks.append(bytes.fromhex(match.group(1))[::-1])
            continue
        match = _HEX_LINE.match(line)
        if match:
            chunks.append(bytes.fromhex("".join(match.group(1).split())))
    return b"".join(chunks)


def _decode_bytes(code: bytes, architecture: str) -> list[dict[str, Any]]:
    if not code:
        raise ValueError("static disassembly produced no text bytes")
    if architecture == "x86_64":
        decoder = Cs(CS_ARCH_X86, CS_MODE_64)
    elif architecture == "aarch64":
        decoder = Cs(CS_ARCH_ARM64, 0)
    else:
        raise ValueError(f"unsupported decoder architecture {architecture}")
    decoder.detail = True
    rows = [
        {
            "address": int(instruction.address),
            "mnemonic": str(instruction.mnemonic),
            "op_str": str(instruction.op_str),
            "size": int(instruction.size),
            "bytes_hex": bytes(instruction.bytes).hex(" "),
        }
        for instruction in decoder.disasm(code, 0)
    ]
    if not rows:
        raise ValueError("Capstone decoded zero compiler-emitted instructions")
    return rows


def _compile_one(
    source_id: str,
    source: str,
    target_id: str,
    target: str,
    optimization: str = "-O0",
) -> dict[str, Any]:
    if optimization not in OPTIMIZATION_LEVELS:
        raise ValueError(f"unsupported optimization level {optimization!r}")
    source_bytes = source.encode("utf-8")
    base: dict[str, Any] = {
        "source_id": source_id,
        "target_id": target_id,
        "target_triple": target,
        "source_sha256": _sha256_bytes(source_bytes),
        "compiler": "clang",
        "compiler_version": _compiler_version(),
        "optimization": optimization,
        "execution_performed": False,
        "analysis_mode": "static_object_disassembly_only",
    }
    with tempfile.TemporaryDirectory(prefix="active-assembly-compiler-") as directory:
        object_path = Path(directory) / f"{source_id}-{target_id}.o"
        command = [
            "clang",
            "-target",
            target,
            "-c",
            optimization,
            "-x",
            "c",
            "-o",
            str(object_path),
            "-",
        ]
        code, stdout, stderr = _run(command, input_text=source)
        base["command"] = command[:-2] + ["<source-stdin>"]
        base["compiler_stdout"] = stdout.strip()
        base["compiler_stderr"] = stderr.strip()
        if code != 0 or not object_path.is_file():
            base.update(
                {
                    "status": "unavailable",
                    "reason": "compiler_target_unavailable_or_compile_failed",
                    "returncode": code,
                }
            )
            return base
        object_bytes = object_path.read_bytes()
        base["object_sha256"] = _sha256_bytes(object_bytes)
        disassembly_code, disassembly_stdout, disassembly_stderr = _run(
            ["objdump", "-d", str(object_path)]
        )
        base["objdump_stderr"] = disassembly_stderr.strip()
        if disassembly_code != 0:
            base.update(
                {
                    "status": "unavailable",
                    "reason": "object_disassembler_failed",
                    "returncode": disassembly_code,
                }
            )
            return base
        text_bytes = _extract_text_bytes(disassembly_stdout, target_id)
        decoded = _decode_bytes(text_bytes, target_id)
        base.update(
            {
                "status": "validated_static_object",
                "text_sha256": _sha256_bytes(text_bytes),
                "disassembly_sha256": _sha256_bytes(
                    json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
                ),
                "decoded_instruction_count": len(decoded),
                "decoded_instructions": decoded,
            }
        )
        return base


def compiler_transfer_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compile the portable source families and record exact capabilities."""

    rows = [
        _compile_one(source_id, source, target_id, target, optimization)
        for source_id, source in SOURCE_FAMILIES.items()
        for target_id, target in TARGETS.items()
        for optimization in OPTIMIZATION_LEVELS
    ]
    available = [row for row in rows if row["status"] == "validated_static_object"]
    protocol = scientific_protocol_payload()
    limitations = [
        "local compiler capability is not a cross-version compiler study",
        "static object disassembly does not establish semantic equivalence",
        "RISC-V remains explicit unavailable when local Clang lacks the target",
    ]
    unsupported_cases = [
        "No emitted object is executed; external validation is static-only.",
        "Instruction effects outside the declared formal subset remain unsupported for semantic scoring.",
    ]
    underpowered_cases = [
        "The local matrix is a capability receipt, not a 64-family transfer study.",
        "The licensed-source external track is separately receipted and remains underpowered until source-bound labels and held-out evaluation pass.",
    ]
    source_hashes = {
        f"source:{source_id}": _sha256_bytes(source.encode("utf-8"))
        for source_id, source in SOURCE_FAMILIES.items()
    }
    source_hashes["src/compiler_transfer.py"] = sha256_path(Path(__file__))
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=source_hashes,
        toolchain_versions={
            **toolchain_versions(("capstone",)),
            "clang": _compiler_version(),
            "objdump": "local executable",
        },
        status="implemented_local",
        limitations=limitations,
        unsupported_cases=unsupported_cases,
        underpowered_cases=underpowered_cases,
    )
    return {
        **envelope,
        "schema_version": COMPILER_TRANSFER_SCHEMA_VERSION,
        "artifact": COMPILER_TRANSFER_ARTIFACT,
        "endpoint_status": "local_capability_receipt",
        "source_family_disjoint": True,
        "compiler_matrix_claim": "only rows with validated_static_object status",
        "architecture_matrix_claim": "only targets with validated_static_object status",
        "optimization_matrix_claim": "only declared optimization levels with validated_static_object status",
        "execution_performed": False,
        "source_families": [
            {
                "source_id": source_id,
                "source_sha256": _sha256_bytes(source.encode("utf-8")),
                "license": "repository-owned transparent fixture source",
            }
            for source_id, source in SOURCE_FAMILIES.items()
        ],
        "capability_summary": {
            "row_count": len(rows),
            "validated_static_objects": len(available),
            "unavailable_rows": len(rows) - len(available),
            "architectures_validated": sorted({row["target_id"] for row in available}),
            "optimization_levels_validated": sorted(
                {row["optimization"] for row in available}
            ),
            "source_family_count": len(SOURCE_FAMILIES),
            "compiler": "clang",
        },
        "external_source_manifest": {
            "status": "underpowered",
            "reason": "licensed external evidence is tracked by the separate pinned-source receipt and lacks source-bound labels",
            "network_required_for_replay": False,
            "arbitrary_external_execution": False,
        },
        "rows": rows,
        "limitations": limitations,
        "unsupported_cases": unsupported_cases,
        "underpowered_cases": underpowered_cases,
    }


def validate_compiler_transfer_payload(payload: object) -> tuple[bool, str]:
    """Validate capability, provenance, and static-only safety contracts."""

    if not isinstance(payload, dict):
        return False, "compiler transfer payload must be an object"
    if payload.get("schema_version") != COMPILER_TRANSFER_SCHEMA_VERSION:
        return False, "compiler transfer schema mismatch"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"compiler transfer receipt: {receipt_detail}"
    if payload.get("status") != "implemented_local":
        return False, "compiler transfer evidence status is invalid"
    if (
        not isinstance(payload.get("generator_revision"), str)
        or not isinstance(payload.get("input_hashes"), dict)
        or not isinstance(payload.get("toolchain_versions"), dict)
    ):
        return False, "compiler transfer provenance is incomplete"
    if payload.get("execution_performed") is not False:
        return False, "compiler transfer executed an external artifact"
    if payload.get("source_family_disjoint") is not True:
        return False, "compiler transfer source-family split is not frozen"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return False, "compiler transfer rows are missing"
    for row in rows:
        if not isinstance(row, dict):
            return False, "compiler transfer row is not an object"
        if row.get("status") not in {"validated_static_object", "unavailable"}:
            return False, "compiler transfer row has unknown status"
        if row.get("optimization") not in OPTIMIZATION_LEVELS:
            return False, "compiler transfer row has unknown optimization level"
        if row.get("execution_performed") is not False:
            return False, "compiler transfer row execution flag drift"
        if row.get("status") == "validated_static_object":
            digest = row.get("object_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                return False, "validated compiler row lacks object hash"
    external = payload.get("external_source_manifest")
    if not isinstance(external, dict) or external.get("arbitrary_external_execution"):
        return False, "external-source safety contract drift"
    return True, "ok"


def write_compiler_transfer(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the local compiler/source/architecture capability receipt."""

    payload = compiler_transfer_payload(project_root)
    valid, detail = validate_compiler_transfer_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / COMPILER_TRANSFER_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path
