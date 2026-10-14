"""Compiler/source/architecture transfer receipts with static-only decoding.

The producer uses the locally available toolchains (Clang, NASM, rustc) and an
object disassembler to build small source families.  It never executes the
resulting objects.  The capability matrix is frozen in module constants: every
(toolchain, source family, target, object format, optimization) cell is either
probed and validated locally or retained as an explicit ``unavailable`` row
with the requested triple.  Absent toolchains are never downloaded, installed,
or substituted by the host architecture.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, cast

from capstone import CS_ARCH_ARM64, CS_ARCH_X86, CS_MODE_64, Cs

from .artifact_contract import write_json_artifact
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import (
    RECEIPT_SCHEMA_VERSION,
    build_receipt_envelope,
    validate_receipt_envelope,
)
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILER_TRANSFER_SCHEMA_VERSION = "2.1.0"
COMPILER_TRANSFER_ARTIFACT = "output/data/compiler_transfer.json"
_HEX_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+((?:[0-9a-fA-F]{2}\s+)+)")
_WORD_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+")

# ---------------------------------------------------------------------------
# Frozen capability matrix.
#
# The matrix is declared, not discovered: toolchains, source families, target
# triples, object formats, and optimization vocabularies are fixed here.  A
# missing toolchain still occupies every one of its declared cells as an
# explicit ``unavailable`` row with reason ``toolchain_not_installed``.
# ---------------------------------------------------------------------------

CLANG_FIXTURE_FAMILIES: dict[str, str] = {
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

NASM_FIXTURE_FAMILIES: dict[str, str] = {
    "nasm_x86_arithmetic": (
        "; x86-64 arithmetic/bitwise fixture assembled for the static matrix.\n"
        "global aa_add, aa_mix, aa_bits\n"
        "\n"
        "section .text\n"
        "\n"
        "aa_add:\n"
        "  lea rax, [rdi + 1]\n"
        "  ret\n"
        "\n"
        "aa_mix:\n"
        "  lea rax, [rdi + 3]\n"
        "  add rax, rax\n"
        "  ret\n"
        "\n"
        "aa_bits:\n"
        "  mov rax, rdi\n"
        "  shl rax, 1\n"
        "  mov rcx, rdi\n"
        "  shr rcx, 3\n"
        "  or rax, rcx\n"
        "  xor rax, 0xA5\n"
        "  ret\n"
    ),
}

RUSTC_FIXTURE_FAMILIES: dict[str, str] = {
    "rustc_no_std_staticlib": (
        "#![no_std]\n"
        "#![no_builtins]\n"
        "use core::panic::PanicInfo;\n"
        "\n"
        "#[panic_handler]\n"
        "fn panic(_info: &PanicInfo) -> ! {\n"
        "    loop {}\n"
        "}\n"
        "\n"
        "#[no_mangle]\n"
        'pub extern "C" fn aa_add(x: u64) -> u64 {\n'
        "    x.wrapping_add(1)\n"
        "}\n"
        "\n"
        "#[no_mangle]\n"
        'pub extern "C" fn aa_mix(x: u64) -> u64 {\n'
        "    x.wrapping_add(3).wrapping_mul(2)\n"
        "}\n"
        "\n"
        "#[no_mangle]\n"
        'pub extern "C" fn aa_bits(x: u64) -> u64 {\n'
        "    ((x << 1) | (x >> 3)) ^ 0xA5\n"
        "}\n"
    ),
}

# Declared for absent toolchains: their cells still exist in the frozen matrix
# and are emitted as explicit ``toolchain_not_installed`` rows.  No source body
# is embedded because the cell can never be compiled on this machine.
GO_FIXTURE_FAMILIES: dict[str, str | None] = {"go_capability_fixture": None}
ZIG_FIXTURE_FAMILIES: dict[str, str | None] = {"zig_capability_fixture": None}

CLANG_OPTIMIZATIONS = ("-O0", "-O1", "-O2", "-O3", "-Oz")
NASM_OPTIMIZATIONS = ("none",)
RUSTC_OPTIMIZATIONS = (
    "-C opt-level=0",
    "-C opt-level=1",
    "-C opt-level=2",
    "-C opt-level=3",
    "-C opt-level=s",
    "-C opt-level=z",
)
GO_OPTIMIZATIONS = ("none",)
ZIG_OPTIMIZATIONS = ("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall")

CLANG_TARGETS = {
    "x86_64": {
        "triple": "x86_64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "x86_64",
    },
    "aarch64": {
        "triple": "aarch64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "aarch64",
    },
    "x86_64_linux": {
        "triple": "x86_64-unknown-linux-gnu",
        "object_format": "elf",
        "decoder_arch": "x86_64",
    },
    "aarch64_linux": {
        "triple": "aarch64-unknown-linux-gnu",
        "object_format": "elf",
        "decoder_arch": "aarch64",
    },
    "riscv64": {"triple": "riscv64-unknown-elf", "object_format": "elf"},
}
NASM_TARGETS = {
    "x86_64": {
        "triple": "x86_64-apple-darwin",
        "object_format": "macho",
        "format": "macho64",
        "decoder_arch": "x86_64",
    },
    "x86_64_linux": {
        "triple": "x86_64-unknown-linux-gnu",
        "object_format": "elf",
        "format": "elf64",
        "decoder_arch": "x86_64",
    },
}
RUSTC_TARGETS = {
    "x86_64": {
        "triple": "x86_64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "x86_64",
    },
    "aarch64": {
        "triple": "aarch64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "aarch64",
    },
}
GO_TARGETS = {
    "x86_64": {
        "triple": "x86_64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "x86_64",
        "requested_env": {"GOOS": "darwin", "GOARCH": "amd64"},
    },
    "aarch64": {
        "triple": "aarch64-apple-darwin",
        "object_format": "macho",
        "decoder_arch": "aarch64",
        "requested_env": {"GOOS": "darwin", "GOARCH": "arm64"},
    },
    "x86_64_linux": {
        "triple": "x86_64-unknown-linux-gnu",
        "object_format": "elf",
        "decoder_arch": "x86_64",
        "requested_env": {"GOOS": "linux", "GOARCH": "amd64"},
    },
    "aarch64_linux": {
        "triple": "aarch64-unknown-linux-gnu",
        "object_format": "elf",
        "decoder_arch": "aarch64",
        "requested_env": {"GOOS": "linux", "GOARCH": "arm64"},
    },
}
ZIG_TARGETS = {
    "x86_64": {
        "triple": "x86_64-apple-darwin",
        "object_format": "macho",
        "zig_target": "x86_64-macos",
    },
    "aarch64": {
        "triple": "aarch64-apple-darwin",
        "object_format": "macho",
        "zig_target": "aarch64-macos",
    },
    "x86_64_linux": {
        "triple": "x86_64-unknown-linux-gnu",
        "object_format": "elf",
        "zig_target": "x86_64-linux",
    },
    "aarch64_linux": {
        "triple": "aarch64-unknown-linux-gnu",
        "object_format": "elf",
        "zig_target": "aarch64-linux",
    },
}


def _clang_requested_command(target: dict[str, Any], optimization: str) -> list[str]:
    command = ["clang", "-target", target["triple"]]
    if target.get("march"):
        command.append(f"-march={target['march']}")
    if target.get("mcpu"):
        command.append(f"-mcpu={target['mcpu']}")
    command += ["-c", optimization, "-x", "c", "-o", "<object>", "-"]
    return command


def _nasm_requested_command(target: dict[str, Any], optimization: str) -> list[str]:
    # NASM has no optimization flag; ``none`` is recorded without an argv entry.
    return ["nasm", "-f", target["format"], "-o", "<object>", "<source>"]


def _rustc_requested_command(target: dict[str, Any], optimization: str) -> list[str]:
    return [
        "rustc",
        "--edition",
        "2021",
        "--crate-name",
        "aa_transfer",
        "--crate-type",
        "staticlib",
        "--emit=obj",
        "-C",
        "panic=abort",
        *optimization.split(),
        "--target",
        target["triple"],
        "-o",
        "<object>",
        "<source>",
    ]


def _go_requested_command(target: dict[str, Any], optimization: str) -> list[str]:
    return ["go", "tool", "compile", "-p", "aa_fixture", "-o", "<object>", "<source>"]


def _zig_requested_command(target: dict[str, Any], optimization: str) -> list[str]:
    return [
        "zig",
        "build-obj",
        "-O",
        optimization,
        "-target",
        target["zig_target"],
        "-femit-bin=<object>",
        "<source>",
    ]


COMPILER_SPECS: dict[str, dict[str, Any]] = {
    "clang": {
        "language": "c",
        "fixture_families": CLANG_FIXTURE_FAMILIES,
        "targets": CLANG_TARGETS,
        "optimizations": CLANG_OPTIMIZATIONS,
        "executable": "clang",
        "version_flag": "--version",
        "stdin_source": True,
        "source_extension": ".c",
        "requested_command": _clang_requested_command,
    },
    "nasm": {
        "language": "x86_assembly",
        "fixture_families": NASM_FIXTURE_FAMILIES,
        "targets": NASM_TARGETS,
        "optimizations": NASM_OPTIMIZATIONS,
        "executable": "nasm",
        "version_flag": "-v",
        "stdin_source": False,
        "source_extension": ".asm",
        "requested_command": _nasm_requested_command,
    },
    "rustc": {
        "language": "rust",
        "fixture_families": RUSTC_FIXTURE_FAMILIES,
        "targets": RUSTC_TARGETS,
        "optimizations": RUSTC_OPTIMIZATIONS,
        "executable": "rustc",
        "version_flag": "--version",
        "stdin_source": False,
        "source_extension": ".rs",
        "requested_command": _rustc_requested_command,
    },
    "go": {
        "language": "go",
        "fixture_families": GO_FIXTURE_FAMILIES,
        "targets": GO_TARGETS,
        "optimizations": GO_OPTIMIZATIONS,
        "executable": "go",
        "version_flag": "version",
        "stdin_source": False,
        "source_extension": ".go",
        "requested_command": _go_requested_command,
    },
    "zig": {
        "language": "zig",
        "fixture_families": ZIG_FIXTURE_FAMILIES,
        "targets": ZIG_TARGETS,
        "optimizations": ZIG_OPTIMIZATIONS,
        "executable": "zig",
        "version_flag": "version",
        "stdin_source": False,
        "source_extension": ".zig",
        "requested_command": _zig_requested_command,
    },
}

# Probe-only toolchain: on Apple hosts ``gcc`` is typically an alias for Clang,
# so it is recorded (and labeled as an alias) without expanding the matrix.
PROBE_ONLY_TOOLCHAINS = ("gcc",)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(
    command: list[str], *, input_text: str | None = None, cwd: str | None = None
) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
    )
    return completed.returncode, completed.stdout, completed.stderr


@functools.lru_cache(maxsize=None)
def _probe_toolchain(executable: str, version_flag: str) -> dict[str, Any]:
    """Probe one toolchain's availability and version (mirrors the capability
    probe in ``src/external_sources.py``).  Never installs anything."""

    path = shutil.which(executable)
    if path is None:
        return {
            "available": False,
            "executable": None,
            "version": None,
            "reason": "toolchain_not_installed",
        }
    code, stdout, stderr = _run([path, version_flag])
    version = stdout.splitlines()[0].strip() if stdout.splitlines() else ""
    if code != 0 or not version:
        return {
            "available": False,
            "executable": path,
            "version": None,
            "reason": "toolchain_version_probe_failed",
        }
    return {"available": True, "executable": path, "version": version, "reason": None}


def _probe_record(compiler: str) -> dict[str, Any]:
    spec = COMPILER_SPECS[compiler]
    probe = _probe_toolchain(spec["executable"], spec["version_flag"])
    return {
        "compiler": compiler,
        "role": "matrix",
        "available": probe["available"],
        "executable": probe["executable"],
        "version": probe["version"],
        "reason": probe["reason"],
    }


def _gcc_probe_record() -> dict[str, Any]:
    probe = _probe_toolchain("gcc", "--version")
    version = probe["version"]
    alias = bool(version and "clang" in version.lower())
    return {
        "compiler": "gcc",
        "role": "probe_only",
        "available": probe["available"],
        "executable": probe["executable"],
        "version": version,
        "reason": probe["reason"],
        "gcc_clang_alias": alias,
        "alias_label": "gcc (Apple clang alias)" if alias else "gcc",
    }


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
    compiler: str,
    source_id: str,
    source: str | None,
    target_id: str,
    optimization: str,
) -> dict[str, Any]:
    """Run one frozen-matrix cell or record it as explicitly unavailable."""

    spec = COMPILER_SPECS.get(compiler)
    if spec is None:
        raise ValueError(f"unknown compiler {compiler!r}")
    if optimization not in spec["optimizations"]:
        raise ValueError(
            f"unsupported optimization level {optimization!r} for {compiler!r}"
        )
    target = spec["targets"].get(target_id)
    if target is None:
        raise ValueError(f"unknown target id {target_id!r} for {compiler!r}")
    requested_triple = target["triple"]
    decoder_arch = target.get("decoder_arch", target_id)
    probe = _probe_toolchain(spec["executable"], spec["version_flag"])
    row: dict[str, Any] = {
        "source_id": source_id,
        "target_id": target_id,
        "target_triple": requested_triple,
        "requested_triple": requested_triple,
        "object_format": target["object_format"],
        "march": target.get("march"),
        "mcpu": target.get("mcpu"),
        "decoder_arch": decoder_arch,
        "source_sha256": (
            _sha256_bytes(source.encode("utf-8")) if source is not None else None
        ),
        "compiler": compiler,
        "compiler_version": probe["version"],
        "optimization": optimization,
        "command": spec["requested_command"](target, optimization),
        "disassembler_argv": ["objdump", "-d", "<object>"],
        "requested_env": dict(target["requested_env"])
        if target.get("requested_env")
        else None,
        "execution_performed": False,
        "analysis_mode": "static_object_disassembly_only",
        "returncode": None,
        "compiler_stdout": "",
        "compiler_stderr": "",
        "objdump_stderr": "",
        "status": "unavailable",
        "reason": probe["reason"],
    }
    if not probe["available"]:
        return row
    if source is None:
        row["reason"] = "fixture_source_not_embedded"
        return row
    with tempfile.TemporaryDirectory(prefix="active-assembly-compiler-") as directory:
        object_path = Path(directory) / f"{source_id}-{compiler}-{target_id}.o"
        template = spec["requested_command"](target, optimization)
        source_path: Path | None = None
        if not spec["stdin_source"]:
            source_path = Path(directory) / f"{source_id}{spec['source_extension']}"
            source_path.write_text(source, encoding="utf-8")
        # Assemble relative to the temp directory: NASM embeds the source path
        # in the ELF symbol table, so absolute temp paths would make the object
        # hash non-deterministic across runs.  Clang reads the source from
        # stdin and keeps an absolute object path.
        object_ref = str(object_path) if spec["stdin_source"] else object_path.name
        source_ref = source_path.name if source_path is not None else ""
        command = [
            object_ref
            if part == "<object>"
            else source_ref
            if part == "<source>"
            else part
            for part in template
        ]
        code, stdout, stderr = _run(
            command,
            input_text=source if spec["stdin_source"] else None,
            cwd=None if spec["stdin_source"] else directory,
        )
        row["compiler_stdout"] = stdout.strip()
        row["compiler_stderr"] = stderr.strip()
        if code != 0 or not object_path.is_file():
            row["reason"] = "compiler_target_unavailable_or_compile_failed"
            row["returncode"] = code
            return row
        object_bytes = object_path.read_bytes()
        row["object_sha256"] = _sha256_bytes(object_bytes)
        disassembly_code, disassembly_stdout, disassembly_stderr = _run(
            ["objdump", "-d", str(object_path)]
        )
        row["objdump_stderr"] = disassembly_stderr.strip()
        if disassembly_code != 0:
            row["reason"] = "object_disassembler_failed"
            row["returncode"] = disassembly_code
            return row
        try:
            text_bytes = _extract_text_bytes(disassembly_stdout, decoder_arch)
            decoded = _decode_bytes(text_bytes, decoder_arch)
        except ValueError:
            row["reason"] = "object_decoding_failed"
            return row
        row.update(
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
        return row


def compiler_transfer_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compile the frozen source-family matrix and record exact capabilities."""

    rows = [
        _compile_one(compiler, source_id, source, target_id, optimization)
        for compiler, spec in COMPILER_SPECS.items()
        for source_id, source in spec["fixture_families"].items()
        for target_id in spec["targets"]
        for optimization in spec["optimizations"]
    ]
    available = [row for row in rows if row["status"] == "validated_static_object"]
    probes = [_probe_record(compiler) for compiler in COMPILER_SPECS]
    gcc_probe = _gcc_probe_record()
    protocol = scientific_protocol_payload()
    limitations = [
        "local compiler capability is not a cross-version compiler study",
        "static object disassembly does not establish semantic equivalence",
        "absent toolchains, unsupported targets, and missing Rust cross-target "
        "standard libraries remain explicit unavailable rows; nothing is "
        "downloaded, installed, or host-substituted on demand",
    ]
    unsupported_cases = [
        "No emitted object is executed; external validation is static-only.",
        "Instruction effects outside the declared formal subset remain unsupported for semantic scoring.",
    ]
    underpowered_cases = [
        "The local matrix is a capability receipt, not a 64-family transfer study.",
        "The licensed-source external track is separately receipted and remains underpowered until source-bound labels and held-out evaluation pass.",
    ]
    source_families = [
        {
            "source_id": source_id,
            "compiler": compiler,
            "source_sha256": _sha256_bytes(source.encode("utf-8")),
            "license": "repository-owned transparent fixture source",
        }
        for compiler, spec in COMPILER_SPECS.items()
        for source_id, source in spec["fixture_families"].items()
        if source is not None
    ]
    source_hashes = {
        f"source:{source_id}": _sha256_bytes(source.encode("utf-8"))
        for spec in COMPILER_SPECS.values()
        for source_id, source in spec["fixture_families"].items()
        if source is not None
    }
    source_hashes["src/compiler_transfer.py"] = sha256_path(Path(__file__))
    toolchain_version_records: dict[str, str] = {}
    for probe in probes:
        toolchain_version_records[probe["compiler"]] = (
            probe["version"] if probe["available"] else "toolchain_not_installed"
        )
    toolchain_version_records["gcc"] = (
        f"{gcc_probe['version']} (clang alias)"
        if gcc_probe["available"] and gcc_probe["gcc_clang_alias"]
        else gcc_probe["version"] or "toolchain_not_installed"
    )
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=source_hashes,
        toolchain_versions={
            **toolchain_versions(("capstone",)),
            **toolchain_version_records,
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
        "compiler_matrix_claim": "only (compiler, target, object format, optimization) cells with validated_static_object status",
        "architecture_matrix_claim": "only declared target triples with validated_static_object status",
        "object_format_matrix_claim": "only Mach-O/ELF object formats with validated_static_object status",
        "optimization_matrix_claim": "only per-compiler declared optimization levels with validated_static_object status",
        "execution_performed": False,
        "source_families": source_families,
        "toolchain_probes": [*probes, gcc_probe],
        "capability_summary": {
            "row_count": len(rows),
            "validated_static_objects": len(available),
            "unavailable_rows": len(rows) - len(available),
            "architectures_validated": sorted({row["target_id"] for row in available}),
            "optimization_levels_validated": sorted(
                {row["optimization"] for row in available}
            ),
            "source_family_count": len(source_families),
            "compiler": "clang",
            "compilers": sorted(COMPILER_SPECS),
            "compilers_validated": sorted({row["compiler"] for row in available}),
            "object_formats_declared": sorted(
                {
                    target["object_format"]
                    for spec in COMPILER_SPECS.values()
                    for target in spec["targets"].values()
                }
            ),
            "object_formats_validated": sorted(
                {row["object_format"] for row in available}
            ),
            "toolchains_probed": sorted(
                {record["compiler"] for record in [*probes, gcc_probe]}
            ),
            "gcc_clang_alias": gcc_probe["available"] and gcc_probe["gcc_clang_alias"],
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
    # The shared envelope contract pins the receipt layer, not the domain
    # schema; validate against a view carrying the shared receipt version so
    # the domain schema may advance independently (mirrors artifact_contract).
    envelope_view = dict(payload)
    envelope_view["schema_version"] = RECEIPT_SCHEMA_VERSION
    receipt_valid, receipt_detail = validate_receipt_envelope(
        envelope_view, expected_protocol=protocol
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
    expected_cells: set[tuple[str, str, str, str]] = set()
    for compiler, spec in COMPILER_SPECS.items():
        for source_id in spec["fixture_families"]:
            for target_id in spec["targets"]:
                for optimization in spec["optimizations"]:
                    expected_cells.add((compiler, source_id, target_id, optimization))
    seen_cells: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False, "compiler transfer row is not an object"
        row_compiler = row.get("compiler")
        row_spec = (
            COMPILER_SPECS.get(row_compiler) if isinstance(row_compiler, str) else None
        )
        if row_spec is None:
            return False, "compiler transfer row names an unknown compiler"
        cell = (
            row_compiler,
            row.get("source_id"),
            row.get("target_id"),
            row.get("optimization"),
        )
        if not all(isinstance(part, str) for part in cell):
            return False, "compiler transfer row matrix key is incomplete"
        if cell in seen_cells:
            return False, "compiler transfer matrix contains duplicate cells"
        seen_cells.add(cast(tuple[str, str, str, str], cell))
        target = row_spec["targets"].get(row.get("target_id"))
        if target is None:
            return False, "compiler transfer row names an unknown target id"
        if row.get("optimization") not in row_spec["optimizations"]:
            return False, "compiler transfer row has unknown optimization level"
        if row.get("object_format") != target["object_format"]:
            return False, "compiler transfer row object-format drift"
        requested_triple = row.get("requested_triple")
        if not isinstance(requested_triple, str) or not requested_triple:
            return False, "unavailable compiler row lacks the requested triple"
        if requested_triple != target["triple"]:
            return False, "compiler transfer row host-substituted the requested triple"
        if row.get("target_triple") != requested_triple:
            return False, "compiler transfer row target-triple drift"
        if row.get("execution_performed") is not False:
            return False, "compiler transfer row execution flag drift"
        status = row.get("status")
        if status not in {"validated_static_object", "unavailable"}:
            return False, "compiler transfer row has unknown status"
        if status == "validated_static_object":
            for field in (
                "source_sha256",
                "object_sha256",
                "text_sha256",
                "disassembly_sha256",
            ):
                digest = row.get(field)
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    return False, f"validated compiler row lacks {field}"
            count = row.get("decoded_instruction_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                return False, "validated compiler row lacks a positive decoded count"
            instructions = row.get("decoded_instructions")
            if not isinstance(instructions, list) or not instructions:
                return False, "validated compiler row lacks decoded instructions"
        else:
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason:
                return False, "unavailable compiler row lacks a reason"
    if seen_cells != expected_cells:
        return False, "compiler transfer matrix cell inventory drift"
    summary = payload.get("capability_summary")
    if not isinstance(summary, dict):
        return False, "compiler transfer capability summary is missing"
    validated_rows = [row for row in rows if row["status"] == "validated_static_object"]
    if (
        summary.get("row_count") != len(rows)
        or summary.get("validated_static_objects") != len(validated_rows)
        or summary.get("unavailable_rows") != len(rows) - len(validated_rows)
    ):
        return False, "compiler transfer capability summary row counts drift"
    if summary.get("compilers_validated") != sorted(
        {row["compiler"] for row in validated_rows}
    ):
        return False, "compiler transfer capability summary compiler set drift"
    if summary.get("object_formats_validated") != sorted(
        {row["object_format"] for row in validated_rows}
    ):
        return False, "compiler transfer capability summary object-format set drift"
    source_families = payload.get("source_families")
    if not isinstance(source_families, list) or summary.get(
        "source_family_count"
    ) != len(source_families):
        return False, "compiler transfer capability summary source-family count drift"
    for claim_field in (
        "compiler_matrix_claim",
        "architecture_matrix_claim",
        "object_format_matrix_claim",
        "optimization_matrix_claim",
    ):
        claim = payload.get(claim_field)
        if not isinstance(claim, str) or not claim:
            return False, f"compiler transfer {claim_field} is missing"
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
