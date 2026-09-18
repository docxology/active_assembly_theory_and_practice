"""Run and receipt the kernel-checked formal subset.

The formal project is intentionally small and closed-world. This module makes
the build an executable research artifact: it records the pinned Lean
toolchain, exact source hashes, theorem names, and a digest of compiler
output. A successful receipt means that the checked subset compiled locally;
it does not certify the Python lifter, Capstone, Unicorn, or the whole x86 ISA.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .artifact_contract import write_json_artifact
from .provenance import generator_revision, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload
from .lean_bridge import (
    BRIDGE_FIXTURE_THEOREMS,
    BRIDGE_MACHINE_STATE_THEOREMS,
    run_generated_lean_fixture,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_PROJECT_DIR = PROJECT_ROOT / "formal"
FORMAL_ARTIFACT = "output/data/formal_verification.json"
FORMAL_SCHEMA_VERSION = "2.0.0"
FORMAL_TOOLCHAIN = "leanprover/lean4:v4.28.0"
FORMAL_THEOREMS = (
    "ActiveAssembly.inc_inc_eq_add_two",
    "ActiveAssembly.constant_add_equivalence",
    "ActiveAssembly.neutral_nop_equivalence",
    "ActiveAssembly.changed_subtraction_is_not_equivalent",
    "ActiveAssembly.run_nop_append",
    "ActiveAssembly.write32_zero_extends",
    "ActiveAssembly.inc_preserves_cf",
    "ActiveAssembly.adc_zero_carry",
    "ActiveAssembly.unsupported_is_explicit",
    "ActiveAssembly.explicit_unsupported_is_not_supported",
    "ActiveAssembly.declared_opcode_has_decidable_status",
    # MED-02 expansion: exact flag definedness on the bounded machine state.
    "ActiveAssembly.arithmetic_flags_fully_defined",
    "ActiveAssembly.logic_flags_af_undefined",
    "ActiveAssembly.logic_flags_cf_defined_and_clear",
    "ActiveAssembly.add64N_flags",
    "ActiveAssembly.sub64N_flags",
    "ActiveAssembly.cmp64N_preserves_registers_and_memory",
    "ActiveAssembly.sub64N_result_val",
    "ActiveAssembly.word64n_add_sub_inverse_val",
    "ActiveAssembly.word64n_add_sub_inverse",
    "ActiveAssembly.add_sub_inverse_gprs",
    "ActiveAssembly.add_sub_inverse",
    "ActiveAssembly.cmp_matches_sub_flags",
    "ActiveAssembly.arithmetic_flags_defined_after_add_sub",
    "ActiveAssembly.flag_definedness_logic_ops",
    # Width-aware move semantics.
    "ActiveAssembly.movLowN_val",
    "ActiveAssembly.mov8N_val",
    "ActiveAssembly.mov16N_val",
    "ActiveAssembly.mov8N_preserves_upper_bits",
    "ActiveAssembly.mov16N_preserves_upper_bits",
    "ActiveAssembly.movzx32N_zero_extension",
    "ActiveAssembly.movsx32N_sign_extension",
    "ActiveAssembly.movzx32_to_64_zero_extension",
    # Bounded little-endian stack: push/pop.
    "ActiveAssembly.readLittleEndianN_congr",
    "ActiveAssembly.mulAddMod256_pow2",
    "ActiveAssembly.read_write_little_endian_roundtrip",
    "ActiveAssembly.push64N_out_of_bounds_is_explicit",
    "ActiveAssembly.pop64N_out_of_bounds_is_explicit",
    "ActiveAssembly.push64N_pop64N_roundtrip",
    "ActiveAssembly.push_pop_roundtrip",
    # Conditional branches over the definedness map.
    "ActiveAssembly.jcc_zero_flag_branch_distinct",
    "ActiveAssembly.jcc_undefined_flag_is_exceptional",
    "ActiveAssembly.jcc_e_taken_sets_rip",
    "ActiveAssembly.jcc_e_not_taken_sets_rip",
    "ActiveAssembly.jcc_taken_vs_not_taken_distinct",
    "ActiveAssembly.generated_trace_matches",
    *BRIDGE_MACHINE_STATE_THEOREMS,
    *BRIDGE_FIXTURE_THEOREMS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_source_hashes(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash formal sources and lock/configuration files, excluding build products."""

    formal_root = project_root / "formal"
    return {
        str(path.relative_to(project_root)): _sha256(path)
        for path in sorted(formal_root.rglob("*"))
        if path.is_file()
        if ".lake" not in path.parts
    }


def _axiom_report(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Scan Lean sources for unfinished proofs or undocumented escape hatches."""

    findings: dict[str, list[str]] = {
        "sorry": [],
        "admit": [],
        "unsafe": [],
        "axiom": [],
    }
    for path in sorted((project_root / "formal").rglob("*.lean")):
        if ".lake" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(project_root))
        for key, pattern in (
            ("sorry", r"\bsorry\b"),
            ("admit", r"\badmit\b"),
            ("unsafe", r"\bunsafe\b"),
            ("axiom", r"^\s*axiom\s+"),
        ):
            if re.search(pattern, text, flags=re.MULTILINE):
                findings[key].append(relative)
    return {
        "sorries": findings["sorry"],
        "admits": findings["admit"],
        "unsafe_declarations": findings["unsafe"],
        "axiom_declarations": findings["axiom"],
        "kernel_only_proof_boundary": True,
        "clean": not any(findings.values()),
    }


def _elan_executable() -> str:
    """Resolve the explicit elan executable used by the formal gate."""

    executable = shutil.which("elan")
    if executable is None:
        raise RuntimeError("elan is required for the formal verification gate")
    return executable


def build_formal_project(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compile the formal project with the pinned installed Lean toolchain."""

    formal_dir = project_root / "formal"
    if not formal_dir.is_dir():
        raise FileNotFoundError(f"missing formal project: {formal_dir}")
    command = (
        _elan_executable(),
        "run",
        FORMAL_TOOLCHAIN,
        "lake",
        "build",
    )
    completed = subprocess.run(
        command,
        cwd=formal_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "formal Lean build failed with exit code "
            f"{completed.returncode}: {output[-4000:]}"
        )
    bridge = run_generated_lean_fixture(project_root)
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "build_completed": True,
        "generated_bridge": bridge,
        "axiom_report": _axiom_report(project_root),
    }


def formal_verification_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build the deterministic receipt for the locally checked formal subset."""

    build = build_formal_project(project_root)
    protocol = scientific_protocol_payload()
    source_hashes = _formal_source_hashes(project_root)
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=source_hashes,
        toolchain_versions=toolchain_versions(("numpy",)),
        status="validated",
        limitations=(
            "The Lean project checks a bounded normalized-state fragment and does not formalize complete x86-64 semantics.",
            "Cross-language Python/Unicorn agreement is recorded separately as computational evidence.",
        ),
        unsupported_cases=(
            "Complete instruction decoding, exceptions, SIMD, floating point, privileged state, and unbounded memory are outside this proof.",
        ),
        underpowered_cases=(
            "A mechanized theorem connecting every Python/Capstone/Unicorn trace row to Lean remains unresolved.",
        ),
    )
    return {
        **envelope,
        "schema_version": FORMAL_SCHEMA_VERSION,
        "endpoint_status": "locally_verified",
        "toolchain": FORMAL_TOOLCHAIN,
        "formal_project": "formal",
        "source_hashes": source_hashes,
        "theorems": list(FORMAL_THEOREMS),
        "build": build,
        "trusted_base": (
            "Lean kernel and imported Lean core/standard library; no external "
            "solver proof certificate"
        ),
        "checked_fragment": {
            "state": "explicit 16-register normalized bounded machine-state schema",
            "operations": [
                "mov immediate",
                "add immediate",
                "sub immediate",
                "inc",
                "cmp (exact sub flags with no writeback)",
                "and/or/xor logic flags with AF undefined-on-logic",
                "width-aware moves (8/16-bit preserve upper bits, 32-bit "
                "zero-extends, movzx/movsx extension)",
                "push/pop over the bounded little-endian stack window with "
                "explicit boundExceeded failure",
                "jcc conditional branches over the flag definedness map with "
                "fail-closed exceptional status on undefined flag reads",
                "width-aware register writes",
                "explicit flag definedness and carry preservation",
                "bounded memory-state fields",
                "nop",
            ],
            "relation": (
                "universal final accumulator equality over the formal Word64 domain"
            ),
        },
        "scope": (
            "formal algebraic correspondence prerequisite for the transparent "
            "quotient witnesses; not a complete x86 semantics, lifter, compiler "
            "correctness theorem, or external-validity result"
        ),
        "limitations": list(envelope["limitations"]),
        "unsupported_cases": list(envelope["unsupported_cases"]),
        "underpowered_cases": list(envelope["underpowered_cases"]),
        "cross_language_bridge": build["generated_bridge"],
        "axiom_report": build["axiom_report"],
    }


def validate_formal_verification_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    """Fail closed on a missing, stale-looking, or scope-inflated receipt."""

    if payload.get("schema_version") != FORMAL_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"formal receipt: {receipt_detail}"
    if payload.get("status") != "validated":
        return False, "formal receipt is not validated"
    if payload.get("endpoint_status") != "locally_verified":
        return False, "formal receipt is not locally verified"
    if payload.get("toolchain") != FORMAL_TOOLCHAIN:
        return False, "formal toolchain drift"
    if set(payload.get("theorems", [])) != set(FORMAL_THEOREMS):
        return False, "formal theorem manifest drift"
    bridge = payload.get("cross_language_bridge")
    if not isinstance(bridge, dict) or bridge.get("kernel_checked") is not True:
        return False, "generated Python-Lean bridge is not kernel checked"
    if bridge.get("python_reference_final_rax") != bridge.get(
        "lean_expected_final_rax"
    ):
        return False, "Python and Lean bridge values disagree"
    if bridge.get("machine_state_schema") != (
        "explicit_16_gpr_rip_flags_memory_stack_step_status"
    ):
        return False, "generated Lean bridge machine-state schema is incomplete"
    if bridge.get("machine_state_register_names") != [
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbp",
        "rsp",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    ]:
        return False, "generated Lean bridge omits an architectural register"
    if bridge.get("lean_machine_state_theorems") != list(BRIDGE_MACHINE_STATE_THEOREMS):
        return False, "generated Lean bridge machine-state theorem manifest drift"
    initial_state = bridge.get("python_initial_machine_state")
    if not isinstance(initial_state, dict) or set(
        initial_state.get("registers", {})
    ) != set(bridge["machine_state_register_names"]):
        return False, "generated Lean bridge Python state is incomplete"
    if not isinstance(bridge.get("python_initial_machine_state_sha256"), str):
        return False, "generated Lean bridge Python state digest is missing"
    axiom_report = payload.get("axiom_report")
    if not isinstance(axiom_report, dict) or axiom_report.get("clean") is not True:
        return False, "formal source contains unfinished proof or escape-hatch findings"
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        return False, "formal source hashes are missing"
    scope = str(payload.get("scope", "")).lower()
    if "complete" in scope and "not a complete" not in scope:
        return False, "formal scope overclaims completeness"
    return True, f"theorems={len(FORMAL_THEOREMS)} sources={len(source_hashes)}"


def write_formal_verification(project_root: Path = PROJECT_ROOT) -> str:
    """Compile the formal project and write its source-bound receipt."""

    payload = formal_verification_payload(project_root)
    valid, detail = validate_formal_verification_payload(payload)
    if not valid:
        raise ValueError(f"formal verification receipt is invalid: {detail}")
    path = project_root / FORMAL_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return FORMAL_ARTIFACT


__all__ = [
    "FORMAL_ARTIFACT",
    "FORMAL_SCHEMA_VERSION",
    "FORMAL_THEOREMS",
    "FORMAL_TOOLCHAIN",
    "build_formal_project",
    "formal_verification_payload",
    "validate_formal_verification_payload",
    "write_formal_verification",
]
