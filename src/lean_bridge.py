"""Generated Python-to-Lean correspondence fixtures.

The bridge pins the Python reference machine semantics to kernel-checked Lean
theorems.  It carries the original accumulator/increment fixture plus the
MED-02 expansion families: push/pop over the bounded little-endian stack,
``cmp`` + conditional branch with exact arithmetic flags, and width-aware
moves.  Every fixture is verified twice from the Python side: once against a
real Unicorn execution and once against the normalized reference interpreter,
and any mismatch raises (fail-closed).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_TOOLCHAIN = "leanprover/lean4:v4.35.0"
GENERATED_RELATIVE_PATH = Path("formal/ActiveAssembly/GeneratedCorrespondence.lean")
BRIDGE_CODE = bytes.fromhex("48 83 c0 02 48 ff c0")
BRIDGE_INITIAL_RAX = (1 << 64) - 1
BRIDGE_EXPECTED_RAX = 2
BRIDGE_REGISTER_NAMES = (
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
)
BRIDGE_INITIAL_REGISTER_VALUES = (
    BRIDGE_INITIAL_RAX,
    0x101,
    0x102,
    0x103,
    0x104,
    0x105,
    0x106,
    0x101000,
    0x108,
    0x109,
    0x10A,
    0x10B,
    0x10C,
    0x10D,
    0x10E,
    0x10F,
)
BRIDGE_INITIAL_RIP = 0x1000
BRIDGE_INITIAL_EFLAGS = 0x2 | 0x1
BRIDGE_STACK_BOUNDS = (0x100000, 0x102000)
BRIDGE_MEMORY_BOUNDS = (0x1000, 0x200000)
BRIDGE_MACHINE_STATE_THEOREMS = (
    "ActiveAssembly.generated_machine_state_rax_increment",
    "ActiveAssembly.generated_machine_state_other_registers_preserved",
    "ActiveAssembly.generated_machine_state_preserves_cf",
    "ActiveAssembly.generated_machine_state_increments_step",
    "ActiveAssembly.generated_machine_state_preserves_bounds",
)

BRIDGE_DEFAULT_STACK_BOUNDS = (0x100000, 0x102000)
BRIDGE_DEFAULT_MEMORY_BOUNDS = (0x1000, 0x200000)
BRIDGE_DEFAULT_RIP = 0x1000
BRIDGE_DEFAULT_EFLAGS = 0x2 | 0x1
BRIDGE_EXPANSION_EFLAGS = 0x2 | (1 << 9)

_BRIDGE_STACK_MEMORY_BYTES = (0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11)
_BRIDGE_PUSH_ADDRESS = 0x100FF8

BRIDGE_PUSH_POP_FIXTURE: dict[str, Any] = {
    "bridge_id": "push_pop",
    "code_hex": "50 5b",
    "lean_program": (".push raxIndexN", ".pop rbxIndexN"),
    "instruction_count": 2,
    "initial": {
        "registers": {
            "rax": 0x1122334455667788,
            "rbx": 0x101,
            "rbp": 0x106000,
            "rsp": 0x101000,
        },
        "rip": BRIDGE_DEFAULT_RIP,
        "eflags": BRIDGE_EXPANSION_EFLAGS,
        "memory": {},
        "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
        "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
        "flags_defined": True,
        "step_index": 0,
        "status": "ok",
    },
    "expected_boundary": {
        "registers": {
            "rax": 0x1122334455667788,
            "rbx": 0x1122334455667788,
            "rbp": 0x106000,
            "rsp": 0x101000,
        },
        "rip": 0x1002,
        "eflags_masked": 0x0,
        "memory": {
            _BRIDGE_PUSH_ADDRESS + offset: byte
            for offset, byte in enumerate(_BRIDGE_STACK_MEMORY_BYTES)
        },
    },
    "expected_final": {
        "registers": {
            "rax": 0x1122334455667788,
            "rbx": 0x1122334455667788,
            "rbp": 0x106000,
            "rsp": 0x101000,
        },
        "rip": 0x1002,
        "eflags_masked": 0x0,
        "memory": {
            _BRIDGE_PUSH_ADDRESS + offset: byte
            for offset, byte in enumerate(_BRIDGE_STACK_MEMORY_BYTES)
        },
    },
    "theorems": (
        "ActiveAssembly.bridge_push_pop_rsp_restored",
        "ActiveAssembly.bridge_push_pop_rbx_loaded",
        "ActiveAssembly.bridge_push_pop_other_registers_preserved",
        "ActiveAssembly.bridge_push_pop_memory_little_endian",
        "ActiveAssembly.bridge_push_pop_flags_preserved",
        "ActiveAssembly.bridge_push_pop_status_running",
    ),
}

BRIDGE_CMP_JCC_CODE_HEX = "48 83 f8 05 7e 03 83 c0 01 83 e8 00"
BRIDGE_CMP_JCC_LEAN_PROGRAM = (
    ".cmp (word64n 5)",
    ".jcc JccConditionN.le 4105 4102",
)

BRIDGE_CMP_JCC_TAKEN_FIXTURE: dict[str, Any] = {
    "bridge_id": "cmp_jcc_taken",
    "code_hex": BRIDGE_CMP_JCC_CODE_HEX,
    "lean_program": BRIDGE_CMP_JCC_LEAN_PROGRAM,
    "instruction_count": 2,
    "initial": {
        "registers": {"rax": 3, "rsp": 0x101000},
        "rip": BRIDGE_DEFAULT_RIP,
        "eflags": BRIDGE_EXPANSION_EFLAGS,
        "memory": {},
        "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
        "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
        "flags_defined": True,
        "step_index": 0,
        "status": "ok",
    },
    "expected_boundary": {
        "registers": {"rax": 3, "rsp": 0x101000},
        "rip": 0x1009,
        "eflags_masked": 0x1 | 0x10 | 0x80,
        "memory": {},
    },
    "expected_final": {
        "registers": {"rax": 3, "rsp": 0x101000},
        "rip": 0x100C,
        "eflags_masked": 0x4,
        "memory": {},
    },
    "theorems": (
        "ActiveAssembly.bridge_cmp_jcc_taken_branch_target",
        "ActiveAssembly.bridge_cmp_jcc_taken_rax_boundary",
        "ActiveAssembly.bridge_cmp_jcc_taken_flags_exact",
        "ActiveAssembly.bridge_cmp_jcc_taken_status_running",
    ),
}

BRIDGE_CMP_JCC_NOT_TAKEN_FIXTURE: dict[str, Any] = {
    "bridge_id": "cmp_jcc_not_taken",
    "code_hex": BRIDGE_CMP_JCC_CODE_HEX,
    "lean_program": BRIDGE_CMP_JCC_LEAN_PROGRAM,
    "instruction_count": 2,
    "initial": {
        "registers": {"rax": 9, "rsp": 0x101000},
        "rip": BRIDGE_DEFAULT_RIP,
        "eflags": BRIDGE_EXPANSION_EFLAGS,
        "memory": {},
        "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
        "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
        "flags_defined": True,
        "step_index": 0,
        "status": "ok",
    },
    "expected_boundary": {
        "registers": {"rax": 9, "rsp": 0x101000},
        "rip": 0x1006,
        "eflags_masked": 0x0,
        "memory": {},
    },
    "expected_final": {
        "registers": {"rax": 10, "rsp": 0x101000},
        "rip": 0x100C,
        "eflags_masked": 0x4,
        "memory": {},
    },
    "theorems": (
        "ActiveAssembly.bridge_cmp_jcc_not_taken_fallthrough",
        "ActiveAssembly.bridge_cmp_jcc_not_taken_rax_boundary",
        "ActiveAssembly.bridge_cmp_jcc_not_taken_flags_exact",
        "ActiveAssembly.bridge_cmp_jcc_not_taken_status_running",
    ),
}

BRIDGE_MOV_WIDTH_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "bridge_id": "mov_width_32",
        "code_hex": "b8 ef be 00 00",
        "lean_program": (".movImm 32 48879",),
        "instruction_count": 1,
        "initial": {
            "registers": {"rax": 0x1122334455667788, "rsp": 0x101000},
            "rip": BRIDGE_DEFAULT_RIP,
            "eflags": BRIDGE_EXPANSION_EFLAGS,
            "memory": {},
            "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
            "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
            "flags_defined": True,
            "step_index": 0,
            "status": "ok",
        },
        "expected_boundary": {
            "registers": {"rax": 0xBEEF, "rsp": 0x101000},
            "rip": 0x1005,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "expected_final": {
            "registers": {"rax": 0xBEEF, "rsp": 0x101000},
            "rip": 0x1005,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "theorems": (
            "ActiveAssembly.bridge_mov_width_32_zero_extends",
            "ActiveAssembly.bridge_mov_width_32_flags_preserved",
        ),
    },
    {
        "bridge_id": "mov_width_64",
        "code_hex": "48 b8 ef be 00 00 00 00 00 00",
        "lean_program": (".mov (word64 48879)",),
        "instruction_count": 1,
        "initial": {
            "registers": {"rax": 0x1122334455667788, "rsp": 0x101000},
            "rip": BRIDGE_DEFAULT_RIP,
            "eflags": BRIDGE_EXPANSION_EFLAGS,
            "memory": {},
            "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
            "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
            "flags_defined": True,
            "step_index": 0,
            "status": "ok",
        },
        "expected_boundary": {
            "registers": {"rax": 0xBEEF, "rsp": 0x101000},
            "rip": 0x100A,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "expected_final": {
            "registers": {"rax": 0xBEEF, "rsp": 0x101000},
            "rip": 0x100A,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "theorems": (
            "ActiveAssembly.bridge_mov_width_64_full_replace",
            "ActiveAssembly.bridge_mov_width_64_flags_preserved",
        ),
    },
    {
        "bridge_id": "mov_width_16",
        "code_hex": "66 b8 ef be",
        "lean_program": (".movImm 16 48879",),
        "instruction_count": 1,
        "initial": {
            "registers": {"rax": 0x1122334455667788, "rsp": 0x101000},
            "rip": BRIDGE_DEFAULT_RIP,
            "eflags": BRIDGE_EXPANSION_EFLAGS,
            "memory": {},
            "stack_bounds": BRIDGE_DEFAULT_STACK_BOUNDS,
            "memory_bounds": BRIDGE_DEFAULT_MEMORY_BOUNDS,
            "flags_defined": True,
            "step_index": 0,
            "status": "ok",
        },
        "expected_boundary": {
            "registers": {"rax": 0x112233445566BEEF, "rsp": 0x101000},
            "rip": 0x1004,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "expected_final": {
            "registers": {"rax": 0x112233445566BEEF, "rsp": 0x101000},
            "rip": 0x1004,
            "eflags_masked": 0x0,
            "memory": {},
        },
        "theorems": (
            "ActiveAssembly.bridge_mov_width_16_preserves_upper_bits",
            "ActiveAssembly.bridge_mov_width_16_flags_preserved",
        ),
    },
)

BRIDGE_FIXTURES: tuple[dict[str, Any], ...] = (
    BRIDGE_PUSH_POP_FIXTURE,
    BRIDGE_CMP_JCC_TAKEN_FIXTURE,
    BRIDGE_CMP_JCC_NOT_TAKEN_FIXTURE,
    *BRIDGE_MOV_WIDTH_FIXTURES,
)

BRIDGE_FIXTURE_THEOREMS: tuple[str, ...] = tuple(
    theorem for fixture in BRIDGE_FIXTURES for theorem in fixture["theorems"]
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _python_machine_state_payload() -> dict[str, Any]:
    """Return the complete Python-side state used by the generated witness."""

    registers = dict(zip(BRIDGE_REGISTER_NAMES, BRIDGE_INITIAL_REGISTER_VALUES))
    return {
        "registers": registers,
        "rip": BRIDGE_INITIAL_RIP,
        "eflags": BRIDGE_INITIAL_EFLAGS,
        "flags_defined": {name: True for name in ("CF", "PF", "AF", "ZF", "SF", "OF")},
        "memory": {},
        "stack_bounds": list(BRIDGE_STACK_BOUNDS),
        "memory_bounds": list(BRIDGE_MEMORY_BOUNDS),
        "step_index": 0,
        "status": "ok",
    }


def _fixture_python_state(fixture: dict[str, Any]) -> dict[str, Any]:
    """Return the complete Python-side initial state of an expansion fixture."""

    registers = {
        name: int(fixture["initial"]["registers"].get(name, 0))
        for name in BRIDGE_REGISTER_NAMES
    }
    return {
        "registers": registers,
        "rip": int(fixture["initial"]["rip"]),
        "eflags": int(fixture["initial"]["eflags"]),
        "flags_defined": {
            name: bool(fixture["initial"]["flags_defined"])
            for name in ("CF", "PF", "AF", "ZF", "SF", "OF")
        },
        "memory": {
            int(address): int(byte)
            for address, byte in fixture["initial"]["memory"].items()
        },
        "stack_bounds": list(fixture["initial"]["stack_bounds"]),
        "memory_bounds": list(fixture["initial"]["memory_bounds"]),
        "step_index": int(fixture["initial"]["step_index"]),
        "status": str(fixture["initial"]["status"]),
    }


def _lean_register_definition() -> str:
    """Serialize every architectural register into a Lean function."""

    lines = ["def generatedGpr (index : Fin 16) : Word64N :=", "  match index.val with"]
    for index, value in enumerate(BRIDGE_INITIAL_REGISTER_VALUES):
        lines.append(f"  | {index} => word64n {value}")
    lines.append("  | _ => word64n 0")
    return "\n".join(lines)


def _fixture_lean_name(fixture: dict[str, Any]) -> str:
    return "".join(part.capitalize() for part in str(fixture["bridge_id"]).split("_"))


def _fixture_lean_registers(fixture: dict[str, Any]) -> str:
    camel = _fixture_lean_name(fixture)
    lines = [
        f"def bridgeGpr{camel} (index : Fin 16) : Word64N :=",
        "  match index.val with",
    ]
    for index, name in enumerate(BRIDGE_REGISTER_NAMES):
        lines.append(
            f"  | {index} => word64n {fixture['initial']['registers'].get(name, 0)}"
        )
    lines.append("  | _ => word64n 0")
    return "\n".join(lines)


def _lean_flags_literal(eflags: int) -> str:
    cf = "true" if eflags & (1 << 0) else "false"
    pf = "true" if eflags & (1 << 2) else "false"
    af = "true" if eflags & (1 << 4) else "false"
    zf = "true" if eflags & (1 << 6) else "false"
    sf = "true" if eflags & (1 << 7) else "false"
    of = "true" if eflags & (1 << 11) else "false"
    return f"FlagsN.mk {cf} {pf} {af} {zf} {sf} {of} (fun _ => true)"


def _fixture_lean_source(fixture: dict[str, Any]) -> str:
    camel = _fixture_lean_name(fixture)
    bridge_id = fixture["bridge_id"]
    initial = fixture["initial"]
    program = ", ".join(fixture["lean_program"])
    header = f"""
/-! ### Bridge fixture ``{bridge_id}``

Byte code ``{fixture["code_hex"]}`` executed by both the Lean machine
transformers (over the modeled prefix of {fixture["instruction_count"]}
instruction(s)) and, from the Python side, a real Unicorn run plus the
normalized reference interpreter. -/

{_fixture_lean_registers(fixture)}

def bridge{camel}Initial : MachineStateN :=
  {{ gprs := bridgeGpr{camel}
    rip := word64n {initial["rip"]}
    flags := {_lean_flags_literal(int(initial["eflags"]))}
    memory := fun _ => 0
    stackLower := {initial["stack_bounds"][0]}
    stackUpper := {initial["stack_bounds"][1]}
    memoryLower := {initial["memory_bounds"][0]}
    memoryUpper := {initial["memory_bounds"][1]}
    stepIndex := {initial["step_index"]}
    status := MachineStatusN.running }}

def bridge{camel}Final : MachineStateN :=
  runN [{program}] bridge{camel}Initial
"""

    if bridge_id == "push_pop":
        expected = fixture["expected_boundary"]
        push_address = int(initial["registers"]["rsp"]) - 8
        memory_clauses = "\n    ∧ ".join(
            f"bridge{camel}Final.memory {address} = {byte}"
            for address, byte in sorted(expected["memory"].items())
        )
        return (
            header
            + f"""
theorem bridge_{bridge_id}_rsp_restored :
    bridge{camel}Final.gprs rspIndexN = word64n {expected["registers"]["rsp"]} := by
  apply Fin.ext
  decide

theorem bridge_{bridge_id}_rbx_loaded :
    bridge{camel}Final.gprs rbxIndexN = word64n {expected["registers"]["rbx"]} := by
  apply Fin.ext
  decide

theorem bridge_{bridge_id}_other_registers_preserved :
    ∀ index : Fin 16, index ≠ rspIndexN → index ≠ rbxIndexN →
      bridge{camel}Final.gprs index = bridge{camel}Initial.gprs index := by
  intro i h1 h2
  show (if i = rspIndexN then
      word64n (((pushInBoundsN bridge{camel}Initial raxIndexN {push_address}).gprs rspIndexN).val + 8)
    else if i = rbxIndexN then
      word64n (readLittleEndianN 8 {push_address}
        (pushInBoundsN bridge{camel}Initial raxIndexN {push_address}).memory)
    else (pushInBoundsN bridge{camel}Initial raxIndexN {push_address}).gprs i)
    = bridge{camel}Initial.gprs i
  rw [if_neg h1, if_neg h2]
  show (if i = rspIndexN then word64n {push_address} else bridge{camel}Initial.gprs i)
    = bridge{camel}Initial.gprs i
  rw [if_neg h1]

theorem bridge_{bridge_id}_memory_little_endian :
    {memory_clauses} := by
  decide

theorem bridge_{bridge_id}_flags_preserved :
    bridge{camel}Final.flags.cf = false ∧
    bridge{camel}Final.flags.pf = false ∧
    bridge{camel}Final.flags.af = false ∧
    bridge{camel}Final.flags.zf = false ∧
    bridge{camel}Final.flags.sf = false ∧
    bridge{camel}Final.flags.of = false ∧
    bridge{camel}Final.flags.defined cfBitN = true ∧
    bridge{camel}Final.flags.defined pfBitN = true ∧
    bridge{camel}Final.flags.defined afBitN = true ∧
    bridge{camel}Final.flags.defined zfBitN = true ∧
    bridge{camel}Final.flags.defined sfBitN = true ∧
    bridge{camel}Final.flags.defined ofBitN = true := by
  decide

theorem bridge_{bridge_id}_status_running :
    bridge{camel}Final.status = MachineStatusN.running := by
  decide
"""
        )

    if bridge_id.startswith("cmp_jcc"):
        expected = fixture["expected_boundary"]
        if bridge_id == "cmp_jcc_taken":
            flag_values = ("true", "false", "true", "false", "true", "false")
        else:
            flag_values = ("false", "false", "false", "false", "false", "false")
        flag_names = ("cf", "pf", "af", "zf", "sf", "of")
        flag_clauses = "\n    ∧ ".join(
            f"bridge{camel}Final.flags.{name} = {value}"
            for name, value in zip(flag_names, flag_values)
        )
        defined_clauses = "\n    ∧ ".join(
            f"bridge{camel}Final.flags.defined {name}BitN = true" for name in flag_names
        )
        if bridge_id == "cmp_jcc_taken":
            outcome = f"""theorem bridge_{bridge_id}_branch_target :
    bridge{camel}Final.rip = word64n {expected["rip"]} := by
  apply Fin.ext
  decide"""
        else:
            outcome = f"""theorem bridge_{bridge_id}_fallthrough :
    bridge{camel}Final.rip = word64n {expected["rip"]} := by
  apply Fin.ext
  decide"""
        return (
            header
            + f"""
{outcome}

theorem bridge_{bridge_id}_rax_boundary :
    bridge{camel}Final.gprs raxIndexN = word64n {expected["registers"]["rax"]} := by
  apply Fin.ext
  decide

theorem bridge_{bridge_id}_flags_exact :
    {flag_clauses} ∧
    {defined_clauses} := by
  decide

theorem bridge_{bridge_id}_status_running :
    bridge{camel}Final.status = MachineStatusN.running := by
  decide
"""
        )

    expected = fixture["expected_boundary"]
    primary_theorem = fixture["theorems"][0].removeprefix("ActiveAssembly.")
    flags_theorem = fixture["theorems"][1].removeprefix("ActiveAssembly.")
    return (
        header
        + f"""
theorem {primary_theorem} :
    bridge{camel}Final.gprs raxIndexN = word64n {expected["registers"]["rax"]} := by
  apply Fin.ext
  decide

theorem {flags_theorem} :
    bridge{camel}Final.flags.cf = false ∧
    bridge{camel}Final.flags.pf = false ∧
    bridge{camel}Final.flags.af = false ∧
    bridge{camel}Final.flags.zf = false ∧
    bridge{camel}Final.flags.sf = false ∧
    bridge{camel}Final.flags.of = false ∧
    bridge{camel}Final.flags.defined cfBitN = true ∧
    bridge{camel}Final.flags.defined pfBitN = true ∧
    bridge{camel}Final.flags.defined afBitN = true ∧
    bridge{camel}Final.flags.defined zfBitN = true ∧
    bridge{camel}Final.flags.defined sfBitN = true ∧
    bridge{camel}Final.flags.defined ofBitN = true := by
  decide
"""
    )


def _elan_executable() -> str:
    executable = shutil.which("elan")
    if executable is None:
        raise RuntimeError("elan is required for the Python-Lean bridge")
    return executable


def _lean_source() -> str:
    """Return the generated, deterministic Lean witness source."""

    header = """import ActiveAssembly.Semantics
import ActiveAssembly.NormalizedMachine

namespace ActiveAssembly

/- This file is generated by src/lean_bridge.py; do not hand edit. -/
def generatedBridgeInitial : Word64 := word64 {BRIDGE_INITIAL_RAX}

def generatedBridgeProgram : List Op :=
  [.add (word64 2), .inc]

def generatedBridgeExpected : Word64 := word64 {BRIDGE_EXPECTED_RAX}

theorem generated_trace_matches :
    run generatedBridgeProgram generatedBridgeInitial = generatedBridgeExpected := by
  apply Fin.ext
  decide

{registers}

def generatedMachineInitial : MachineStateN :=
  {{ gprs := generatedGpr
    rip := word64n {BRIDGE_INITIAL_RIP}
    flags := FlagsN.mk true false false false false false (fun _ => true)
    memory := fun _ => 0
    stackLower := {BRIDGE_STACK_BOUNDS[0]}
    stackUpper := {BRIDGE_STACK_BOUNDS[1]}
    memoryLower := {BRIDGE_MEMORY_BOUNDS[0]}
    memoryUpper := {BRIDGE_MEMORY_BOUNDS[1]}
    stepIndex := 0
    status := MachineStatusN.running }}

def generatedMachineAfterInc : MachineStateN := incStateN generatedMachineInitial

theorem generated_machine_state_rax_increment :
    generatedMachineAfterInc.gprs 0 = word64n 0 := by
  apply Fin.ext
  decide

theorem generated_machine_state_other_registers_preserved :
    ∀ index : Fin 16, index.val ≠ 0 ->
      generatedMachineAfterInc.gprs index = generatedMachineInitial.gprs index := by
  intro index index_nonzero
  have index_ne_zero : index ≠ 0 := by
    intro equality
    apply index_nonzero
    simpa [equality]
  simp [generatedMachineAfterInc, incStateN, generatedMachineInitial,
    generatedGpr, index_ne_zero]

theorem generated_machine_state_preserves_cf :
    generatedMachineAfterInc.flags.cf = generatedMachineInitial.flags.cf := by
  exact inc_preserves_cf generatedMachineInitial

theorem generated_machine_state_increments_step :
    generatedMachineAfterInc.stepIndex = generatedMachineInitial.stepIndex + 1 := by
  rfl

theorem generated_machine_state_preserves_bounds :
    generatedMachineAfterInc.memoryLower = generatedMachineInitial.memoryLower ∧
    generatedMachineAfterInc.memoryUpper = generatedMachineInitial.memoryUpper ∧
    generatedMachineAfterInc.stackLower = generatedMachineInitial.stackLower ∧
    generatedMachineAfterInc.stackUpper = generatedMachineInitial.stackUpper := by
  simp [generatedMachineAfterInc, incStateN, generatedMachineInitial]
""".format(
        BRIDGE_INITIAL_RAX=BRIDGE_INITIAL_RAX,
        BRIDGE_EXPECTED_RAX=BRIDGE_EXPECTED_RAX,
        registers=_lean_register_definition(),
        BRIDGE_INITIAL_RIP=BRIDGE_INITIAL_RIP,
        BRIDGE_STACK_BOUNDS=BRIDGE_STACK_BOUNDS,
        BRIDGE_MEMORY_BOUNDS=BRIDGE_MEMORY_BOUNDS,
    )
    sections = [header.rstrip()]
    for fixture in BRIDGE_FIXTURES:
        sections.append(_fixture_lean_source(fixture).rstrip())
    return "\n\n".join(sections) + "\n\nend ActiveAssembly\n"


def _fixture_code(fixture: dict[str, Any]) -> bytes:
    return bytes.fromhex(str(fixture["code_hex"]).replace(" ", ""))


def _unicorn_fixture_run(fixture: dict[str, Any]) -> dict[str, Any]:
    """Execute the fixture byte code under real Unicorn and snapshot states."""

    from .emulation import GENERAL_REGISTER_IDS
    from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64, Uc
    import unicorn.x86_const as x86_const

    code = _fixture_code(fixture)
    initial = fixture["initial"]
    stack_lower, stack_upper = (int(bound) for bound in initial["stack_bounds"])
    memory_lower, memory_upper = (int(bound) for bound in initial["memory_bounds"])

    emulator = Uc(UC_ARCH_X86, UC_MODE_64)
    emulator.mem_map(
        min(memory_lower, stack_lower),
        max(memory_upper, stack_upper) - min(memory_lower, stack_lower),
    )
    emulator.mem_write(int(initial["rip"]), code)
    for address, byte in initial["memory"].items():
        emulator.mem_write(int(address), bytes([int(byte)]))
    for name, value in initial["registers"].items():
        emulator.reg_write(GENERAL_REGISTER_IDS[name], int(value))
    emulator.reg_write(x86_const.UC_X86_REG_EFLAGS, int(initial["eflags"]))

    snapshots: list[dict[str, Any]] = []

    def _snapshot(uc: Uc) -> dict[str, Any]:
        snapshot = {
            name: int(uc.reg_read(register_id))
            for name, register_id in GENERAL_REGISTER_IDS.items()
        }
        snapshot["rip"] = int(uc.reg_read(x86_const.UC_X86_REG_RIP))
        snapshot["eflags"] = int(uc.reg_read(x86_const.UC_X86_REG_EFLAGS))
        return snapshot

    def _record_instruction(uc: Uc, address: int, size: int, user_data: object) -> None:
        del address, size, user_data
        snapshots.append(_snapshot(uc))

    emulator.hook_add(UC_HOOK_CODE, _record_instruction)
    emulator.emu_start(int(initial["rip"]), int(initial["rip"]) + len(code))
    final = _snapshot(emulator)
    if len(snapshots) < fixture["instruction_count"]:
        raise RuntimeError(
            f"Unicorn executed fewer instructions than fixture "
            f"{fixture['bridge_id']} declares"
        )
    if fixture["instruction_count"] < len(snapshots):
        boundary = snapshots[fixture["instruction_count"]]
    else:
        # The boundary snapshot must never be the same dict object as the
        # final snapshot: downstream state mutation would silently rewrite
        # both recorded states.
        boundary = dict(final)
    memory = {
        int(address): int.from_bytes(emulator.mem_read(int(address), 1), "little")
        for address in fixture["expected_final"]["memory"]
    }
    missing = sorted(
        set(int(address) for address in fixture["expected_final"]["memory"])
        - set(memory)
    )
    if missing:
        raise RuntimeError(
            f"Unicorn memory readback for fixture {fixture['bridge_id']} is "
            f"missing declared addresses {missing}"
        )
    return {"boundary": boundary, "final": final, "memory": memory}


def _reference_fixture_run(fixture: dict[str, Any]) -> dict[str, Any]:
    """Execute the fixture through the normalized reference interpreter."""

    from .disassembly import disassemble_bytes
    from .formal_correspondence import FormalState, _step

    code = _fixture_code(fixture)
    initial = fixture["initial"]
    events = disassemble_bytes(code, int(initial["rip"]))
    if len(events) < fixture["instruction_count"]:
        raise ValueError(
            f"fixture {fixture['bridge_id']} declares more steps than decodable"
        )
    registers = {
        name: int(initial["registers"].get(name, 0)) for name in BRIDGE_REGISTER_NAMES
    }
    state = FormalState(
        rax=registers["rax"],
        rsp=registers["rsp"],
        rip=int(initial["rip"]),
        eflags=int(initial["eflags"]),
        memory={int(k): int(v) for k, v in initial["memory"].items()},
        registers=registers,
        flags_defined={
            name: bool(initial["flags_defined"])
            for name in ("CF", "PF", "AF", "ZF", "SF", "OF")
        },
        stack_bounds=(int(initial["stack_bounds"][0]), int(initial["stack_bounds"][1])),
        memory_bounds=(
            int(initial["memory_bounds"][0]),
            int(initial["memory_bounds"][1]),
        ),
        step_index=int(initial["step_index"]),
    )
    by_address = {event.address: index for index, event in enumerate(events)}
    boundary = None
    steps = 0
    while True:
        if steps == fixture["instruction_count"]:
            boundary = state.to_dict()
        index = by_address.get(state.rip)
        if index is None:
            break
        event = events[index]
        next_address = events[index + 1].address if index + 1 < len(events) else None
        _step(event, state, next_address)
        steps += 1
    if boundary is None:
        boundary = state.to_dict()
    return {"boundary": boundary, "final": state.to_dict()}


def _normalized_state(observed: dict[str, Any]) -> dict[str, Any]:
    """Flatten Unicorn snapshots (flat registers) and reference states."""

    if "registers" in observed:
        return {
            "registers": observed["registers"],
            "rip": observed["rip"],
            "eflags": observed["eflags"],
            "memory": observed.get("memory", {}),
        }
    return {
        "registers": {
            key: value
            for key, value in observed.items()
            if key not in ("rip", "eflags")
        },
        "rip": observed["rip"],
        "eflags": observed["eflags"],
        "memory": observed.get("memory", {}),
    }


def _components_match(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare registers, rip, masked eflags, and declared memory bytes."""

    from .formal_correspondence import ARCHITECTURAL_FLAGS_MASK, U64_MASK

    observed = _normalized_state(observed)

    for name, value in expected["registers"].items():
        if observed["registers"].get(name) != int(value) & U64_MASK:
            return False
    if observed["rip"] != int(expected["rip"]):
        return False
    if (observed["eflags"] & ARCHITECTURAL_FLAGS_MASK) != int(
        expected["eflags_masked"]
    ):
        return False
    for address, byte in expected.get("memory", {}).items():
        observed_memory = observed.get("memory", {})
        raw = observed_memory.get(address, observed_memory.get(str(address)))
        if raw is None or int(raw) != int(byte):
            return False
    return True


def _verify_bridge_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Verify one fixture against both real Unicorn and the reference run."""

    unicorn = _unicorn_fixture_run(fixture)
    # Independent, JSON-stable copies of the emulator readback: the
    # boundary/final snapshots must never alias one shared dict object (a
    # mutation of one would silently rewrite the others), and memory maps
    # serialize with string addresses so a written payload and its recomputed
    # twin stay identical across a JSON round trip.
    readback = {str(address): byte for address, byte in unicorn["memory"].items()}
    unicorn["boundary"]["memory"] = dict(readback)
    unicorn["final"]["memory"] = dict(readback)
    reference = _reference_fixture_run(fixture)
    expected_boundary = fixture["expected_boundary"]
    expected_final = fixture["expected_final"]
    boundary_ok = _components_match(unicorn["boundary"], expected_boundary)
    final_ok = _components_match(unicorn["final"], expected_final)
    memory_ok = all(
        unicorn["memory"].get(int(address)) == int(byte)
        for address, byte in expected_final.get("memory", {}).items()
    )
    reference_boundary_ok = _components_match(reference["boundary"], expected_boundary)
    reference_final_ok = _components_match(reference["final"], expected_final)
    return {
        "kernel_checked": True,
        "unicorn_boundary": unicorn["boundary"],
        "unicorn_final": unicorn["final"],
        "unicorn_memory_readback": readback,
        "reference_boundary": reference["boundary"],
        "reference_final": reference["final"],
        "boundary_matches_lean_expected": boundary_ok,
        "final_matches_lean_expected": final_ok,
        "memory_matches_lean_expected": memory_ok,
        "reference_boundary_matches": reference_boundary_ok,
        "reference_final_matches": reference_final_ok,
        "trace_matches_lean_expected": (
            boundary_ok
            and final_ok
            and memory_ok
            and reference_boundary_ok
            and reference_final_ok
        ),
    }


def write_generated_lean_fixture(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Serialize the fixed machine fragment used by the bridge receipt."""

    path = project_root / GENERATED_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    source = _lean_source().encode("utf-8")
    path.write_bytes(source)
    state_payload = _python_machine_state_payload()
    state_bytes = json.dumps(
        state_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fixture_rows: list[dict[str, Any]] = []
    for fixture in BRIDGE_FIXTURES:
        fixture_state = _fixture_python_state(fixture)
        fixture_bytes = json.dumps(
            fixture_state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        fixture_rows.append(
            {
                "bridge_id": fixture["bridge_id"],
                "code_hex": fixture["code_hex"],
                "lean_program": list(fixture["lean_program"]),
                "instruction_count": int(fixture["instruction_count"]),
                "theorems": list(fixture["theorems"]),
                "python_initial_machine_state": fixture_state,
                "python_initial_machine_state_sha256": _sha256(fixture_bytes),
                "expected_boundary": {
                    **fixture["expected_boundary"],
                    "memory": {
                        str(address): byte
                        for address, byte in fixture["expected_boundary"][
                            "memory"
                        ].items()
                    },
                },
                "expected_final": {
                    **fixture["expected_final"],
                    "memory": {
                        str(address): byte
                        for address, byte in fixture["expected_final"]["memory"].items()
                    },
                },
            }
        )
    return {
        "path": str(GENERATED_RELATIVE_PATH),
        "source_sha256": _sha256(source),
        "fixture_code_sha256": _sha256(BRIDGE_CODE),
        "initial_rax": BRIDGE_INITIAL_RAX,
        "python_reference_final_rax": BRIDGE_EXPECTED_RAX,
        "lean_expected_final_rax": BRIDGE_EXPECTED_RAX,
        "machine_state_schema": "explicit_16_gpr_rip_flags_memory_stack_step_status",
        "machine_state_register_names": list(BRIDGE_REGISTER_NAMES),
        "python_initial_machine_state_sha256": _sha256(state_bytes),
        "python_initial_machine_state": state_payload,
        "lean_machine_state_theorems": list(BRIDGE_MACHINE_STATE_THEOREMS),
        "fixtures": fixture_rows,
    }


def run_generated_lean_fixture(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Generate and kernel-check the shared Lean fixture, then verify traces."""

    metadata = write_generated_lean_fixture(project_root)
    formal_dir = project_root / "formal"
    generated_path = GENERATED_RELATIVE_PATH.relative_to("formal")
    command = [
        _elan_executable(),
        "run",
        FORMAL_TOOLCHAIN,
        "lake",
        "env",
        "lean",
        str(generated_path),
    ]
    completed = subprocess.run(
        command,
        cwd=formal_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    metadata.update(
        {
            "command": command,
            "returncode": int(completed.returncode),
            "kernel_checked": completed.returncode == 0,
            "compiler_output_sha256": _sha256(output.encode("utf-8")),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "generated Lean bridge failed with exit code "
            f"{completed.returncode}: {output[-4000:]}"
        )
    if metadata["python_reference_final_rax"] != metadata["lean_expected_final_rax"]:
        raise RuntimeError("Python and Lean bridge fixture values disagree")
    for fixture_row in metadata["fixtures"]:
        verification = _verify_bridge_fixture(
            {
                "bridge_id": fixture_row["bridge_id"],
                "code_hex": fixture_row["code_hex"],
                "lean_program": tuple(fixture_row["lean_program"]),
                "instruction_count": fixture_row["instruction_count"],
                "initial": _fixture_initial_from_row(fixture_row),
                "expected_boundary": _fixture_expected_from_row(
                    fixture_row["expected_boundary"]
                ),
                "expected_final": _fixture_expected_from_row(
                    fixture_row["expected_final"]
                ),
                "theorems": tuple(fixture_row["theorems"]),
            }
        )
        fixture_row.update(verification)
        if not verification["trace_matches_lean_expected"]:
            raise RuntimeError(
                "bridge fixture "
                f"{fixture_row['bridge_id']} does not match its Lean expected state"
            )
    return metadata


def _fixture_initial_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "registers": row["python_initial_machine_state"]["registers"],
        "rip": row["python_initial_machine_state"]["rip"],
        "eflags": row["python_initial_machine_state"]["eflags"],
        "memory": row["python_initial_machine_state"]["memory"],
        "stack_bounds": row["python_initial_machine_state"]["stack_bounds"],
        "memory_bounds": row["python_initial_machine_state"]["memory_bounds"],
        "flags_defined": row["python_initial_machine_state"]["flags_defined"],
        "step_index": row["python_initial_machine_state"]["step_index"],
        "status": row["python_initial_machine_state"]["status"],
    }


def _fixture_expected_from_row(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "registers": expected["registers"],
        "rip": expected["rip"],
        "eflags_masked": expected["eflags_masked"],
        "memory": {int(address): byte for address, byte in expected["memory"].items()},
    }


__all__ = [
    "BRIDGE_CODE",
    "BRIDGE_EXPECTED_RAX",
    "BRIDGE_FIXTURES",
    "BRIDGE_FIXTURE_THEOREMS",
    "BRIDGE_INITIAL_RAX",
    "BRIDGE_REGISTER_NAMES",
    "BRIDGE_INITIAL_REGISTER_VALUES",
    "BRIDGE_MACHINE_STATE_THEOREMS",
    "FORMAL_TOOLCHAIN",
    "GENERATED_RELATIVE_PATH",
    "run_generated_lean_fixture",
    "write_generated_lean_fixture",
]
