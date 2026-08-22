"""Real bounded Capstone/Unicorn fixtures for the cross-architecture tranche.

The fixtures encode the same tiny numeric accumulator operation on x86-64,
ARM64, and RISC-V64. Each row is decoded by Capstone and executed for exactly
the two non-returning instructions by Unicorn. This is a runtime/API
invariance check, not a theorem that the instructions have identical complete
architectural effects (flags, exceptions, calling conventions, and memory are
outside this fixture contract).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from capstone import (
    CS_ARCH_ARM64,
    CS_ARCH_RISCV,
    CS_ARCH_X86,
    CS_MODE_64,
    CS_MODE_ARM,
    CS_MODE_RISCV64,
    Cs,
)
from unicorn import (
    UC_ARCH_ARM64,
    UC_ARCH_RISCV,
    UC_ARCH_X86,
    UC_MODE_64,
    UC_MODE_ARM,
    UC_MODE_RISCV64,
    UC_PROT_ALL,
    Uc,
)
from unicorn.arm64_const import UC_ARM64_REG_PC, UC_ARM64_REG_X0
from unicorn.riscv_const import UC_RISCV_REG_A0, UC_RISCV_REG_PC
from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RIP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROSS_ARCHITECTURE_ARTIFACT = "output/data/cross_architecture_fixtures.json"
CROSS_ARCHITECTURE_SCHEMA_VERSION = "1.0.0"
BASE_ADDRESS = 0x1000
EXECUTED_INSTRUCTION_COUNT = 2
EXPECTED_ACCUMULATOR = 8


@dataclass(frozen=True)
class ArchitectureFixture:
    """One transparent byte fixture and its real adapter configuration."""

    architecture: str
    capstone_arch: int
    capstone_mode: int
    unicorn_arch: int
    unicorn_mode: int
    accumulator_register: str
    code: bytes

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["code_hex"] = self.code.hex()
        del data["code"]
        return data


def architecture_fixtures() -> tuple[ArchitectureFixture, ...]:
    """Return frozen same-operation fixtures for three real architectures."""

    return (
        ArchitectureFixture(
            architecture="x86_64",
            capstone_arch=CS_ARCH_X86,
            capstone_mode=CS_MODE_64,
            unicorn_arch=UC_ARCH_X86,
            unicorn_mode=UC_MODE_64,
            accumulator_register="RAX",
            code=bytes.fromhex("b8 05 00 00 00 83 c0 03 c3"),
        ),
        ArchitectureFixture(
            architecture="aarch64",
            capstone_arch=CS_ARCH_ARM64,
            capstone_mode=CS_MODE_ARM,
            unicorn_arch=UC_ARCH_ARM64,
            unicorn_mode=UC_MODE_ARM,
            accumulator_register="X0",
            code=bytes.fromhex("a0 00 80 d2 00 0c 00 91 c0 03 5f d6"),
        ),
        ArchitectureFixture(
            architecture="riscv64",
            capstone_arch=CS_ARCH_RISCV,
            capstone_mode=CS_MODE_RISCV64,
            unicorn_arch=UC_ARCH_RISCV,
            unicorn_mode=UC_MODE_RISCV64,
            accumulator_register="A0",
            code=bytes.fromhex("13 05 50 00 13 05 35 00 67 80 00 00"),
        ),
    )


def _decode(fixture: ArchitectureFixture) -> list[dict[str, object]]:
    decoder = Cs(fixture.capstone_arch, fixture.capstone_mode)
    decoder.detail = True
    instructions = list(decoder.disasm(fixture.code, BASE_ADDRESS))
    if len(instructions) != 3:
        raise ValueError(
            f"{fixture.architecture} decoded {len(instructions)} instructions; expected 3"
        )
    return [
        {
            "address": int(instruction.address),
            "size": int(instruction.size),
            "mnemonic": instruction.mnemonic,
            "operands": instruction.op_str,
            "bytes_hex": bytes(instruction.bytes).hex(),
        }
        for instruction in instructions
    ]


def _execute(fixture: ArchitectureFixture) -> dict[str, object]:
    emulator = Uc(fixture.unicorn_arch, fixture.unicorn_mode)
    emulator.mem_map(BASE_ADDRESS, 0x1000, UC_PROT_ALL)
    emulator.mem_write(BASE_ADDRESS, fixture.code)
    if fixture.architecture == "x86_64":
        emulator.reg_write(UC_X86_REG_RAX, 0)
        emulator.reg_write(UC_X86_REG_RIP, BASE_ADDRESS)
    elif fixture.architecture == "aarch64":
        emulator.reg_write(UC_ARM64_REG_X0, 0)
        emulator.reg_write(UC_ARM64_REG_PC, BASE_ADDRESS)
    elif fixture.architecture == "riscv64":
        emulator.reg_write(UC_RISCV_REG_A0, 0)
        emulator.reg_write(UC_RISCV_REG_PC, BASE_ADDRESS)
    else:
        raise ValueError(f"unsupported fixture architecture: {fixture.architecture}")
    emulator.emu_start(
        BASE_ADDRESS,
        BASE_ADDRESS + len(fixture.code),
        count=EXECUTED_INSTRUCTION_COUNT,
    )
    if fixture.architecture == "x86_64":
        value = emulator.reg_read(UC_X86_REG_RAX)
    elif fixture.architecture == "aarch64":
        value = emulator.reg_read(UC_ARM64_REG_X0)
    else:
        value = emulator.reg_read(UC_RISCV_REG_A0)
    return {
        "initial_accumulator": 0,
        "final_accumulator": int(value),
        "expected_accumulator": EXPECTED_ACCUMULATOR,
        "executed_instruction_count": EXECUTED_INSTRUCTION_COUNT,
        "bounded_execution_passed": int(value) == EXPECTED_ACCUMULATOR,
        "return_instruction_excluded": True,
    }


def cross_architecture_payload() -> dict[str, object]:
    """Decode and execute every frozen architecture fixture."""

    rows: list[dict[str, object]] = []
    for fixture in architecture_fixtures():
        rows.append(
            {
                **fixture.to_dict(),
                "fixture_sha256": hashlib.sha256(fixture.code).hexdigest(),
                "instructions": _decode(fixture),
                "execution": _execute(fixture),
                "api_provenance": {
                    "decoder": "Capstone Cs(...).disasm",
                    "emulator": "Unicorn Uc(...).emu_start",
                },
            }
        )
    return {
        "schema_version": CROSS_ARCHITECTURE_SCHEMA_VERSION,
        "protocol_id": "same-accumulator-operation-v1",
        "source_program": "mov-immediate-then-add-immediate",
        "architectures": [row["architecture"] for row in rows],
        "rows": rows,
        "summary": {
            "architecture_count": len(rows),
            "all_decoded_three_instructions": all(
                len(cast(list[object], row["instructions"])) == 3 for row in rows
            ),
            "all_bounded_executions_passed": all(
                bool(
                    cast(dict[str, object], row["execution"])[
                        "bounded_execution_passed"
                    ]
                )
                for row in rows
            ),
            "same_numeric_result": len(
                {
                    int(
                        cast(
                            int,
                            cast(dict[str, object], row["execution"])[
                                "final_accumulator"
                            ],
                        )
                    )
                    for row in rows
                }
            )
            == 1,
        },
        "scope": (
            "local executable adapter benchmark for one transparent numeric "
            "operation; not cross-ISA semantic equivalence, compiler invariance, "
            "calling-convention analysis, or external validity"
        ),
        "sources": ["capstone", "unicorn"],
    }


def validate_cross_architecture_payload(
    payload: dict[str, Any],
) -> tuple[bool, str]:
    """Validate the cross-architecture protocol without accepting hand-stamps."""

    if payload.get("schema_version") != CROSS_ARCHITECTURE_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(architecture_fixtures()):
        return False, "architecture row count drift"
    if any(not isinstance(row, dict) for row in rows):
        return False, "non-object architecture row"
    expected = {fixture.architecture for fixture in architecture_fixtures()}
    observed = {str(row.get("architecture")) for row in rows}
    if observed != expected:
        return False, f"architectures={sorted(observed)}"
    for row in rows:
        instructions = row.get("instructions")
        execution = row.get("execution")
        if not isinstance(instructions, list) or len(instructions) != 3:
            return False, f"{row.get('architecture')}:instruction contract"
        if not isinstance(execution, dict) or not execution.get(
            "bounded_execution_passed"
        ):
            return False, f"{row.get('architecture')}:execution contract"
        code_hex = row.get("code_hex")
        if not isinstance(code_hex, str) or not code_hex:
            return False, f"{row.get('architecture')}:missing code hash input"
        if row.get("api_provenance") != {
            "decoder": "Capstone Cs(...).disasm",
            "emulator": "Unicorn Uc(...).emu_start",
        }:
            return False, f"{row.get('architecture')}:API provenance drift"
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not all(
        bool(summary.get(key))
        for key in (
            "all_decoded_three_instructions",
            "all_bounded_executions_passed",
            "same_numeric_result",
        )
    ):
        return False, "summary does not reflect executable rows"
    return True, f"architectures={len(rows)} result={EXPECTED_ACCUMULATOR}"


def write_cross_architecture_payload(project_root: Path = PROJECT_ROOT) -> str:
    """Write the source-bound cross-architecture fixture artifact."""

    payload = cross_architecture_payload()
    valid, detail = validate_cross_architecture_payload(payload)
    if not valid:
        raise ValueError(f"cross-architecture payload is invalid: {detail}")
    path = project_root / CROSS_ARCHITECTURE_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CROSS_ARCHITECTURE_ARTIFACT


__all__ = [
    "ArchitectureFixture",
    "CROSS_ARCHITECTURE_ARTIFACT",
    "CROSS_ARCHITECTURE_SCHEMA_VERSION",
    "architecture_fixtures",
    "cross_architecture_payload",
    "validate_cross_architecture_payload",
    "write_cross_architecture_payload",
]
