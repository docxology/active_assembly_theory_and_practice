"""Capstone-backed disassembly for transparent byte fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86_const import X86_OP_IMM, X86_OP_MEM

BASE_ADDRESS = 0x1000

SAMPLE_PROGRAMS: dict[str, bytes] = {
    # mov eax, 1; add eax, 2; shl eax, 1; sub eax, 1
    "x86_accumulator": bytes.fromhex("b8 01 00 00 00 83 c0 02 d1 e0 83 e8 01"),
    # mov dword ptr [rsp + 0x20], 7; mov eax, dword ptr [rsp + 0x20]; add eax, 1
    # This fixture is deliberately transparent and contains no control transfer;
    # it is used to distinguish static memory operands from observed accesses.
    "x86_memory": bytes.fromhex("c7 44 24 20 07 00 00 00 8b 44 24 20 83 c0 01"),
    # mov eax,1; cmp eax,3; jl +3; add eax,1; sub eax,0
    # The signed-less-than branch is taken and skips the add.  The final sub is
    # an in-span exit instruction so the fixture retains a complete label path.
    "x86_guard_taken": bytes.fromhex("b8 01 00 00 00 83 f8 03 7c 03 83 c0 01 83 e8 00"),
}


@dataclass(frozen=True)
class InstructionEvent:
    """One decoded machine instruction."""

    address: int
    size: int
    mnemonic: str
    op_str: str
    bytes_hex: str
    groups: tuple[str, ...]
    registers_read: tuple[str, ...] = ()
    registers_written: tuple[str, ...] = ()
    immediate_operands: tuple[int, ...] = ()
    memory_operands: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["groups"] = list(self.groups)
        data["registers_read"] = list(self.registers_read)
        data["registers_written"] = list(self.registers_written)
        data["immediate_operands"] = list(self.immediate_operands)
        data["memory_operands"] = [dict(operand) for operand in self.memory_operands]
        return data


def sample_bytes(sample_name: str = "x86_accumulator") -> bytes:
    """Return the deterministic byte fixture for a named sample."""

    try:
        return SAMPLE_PROGRAMS[sample_name]
    except KeyError as exc:
        names = ", ".join(sorted(SAMPLE_PROGRAMS))
        raise ValueError(
            f"unknown sample {sample_name!r}; available samples: {names}"
        ) from exc


def disassemble_bytes(
    code: bytes, base_address: int = BASE_ADDRESS
) -> list[InstructionEvent]:
    """Decode x86-64 machine code with Capstone."""

    if not code:
        raise ValueError("code must contain at least one byte")

    decoder = Cs(CS_ARCH_X86, CS_MODE_64)
    decoder.detail = True
    events: list[InstructionEvent] = []
    for instruction in decoder.disasm(code, base_address):
        groups = tuple(
            group_name
            for group in instruction.groups
            if (group_name := decoder.group_name(group))
        )
        registers_read_ids, registers_written_ids = instruction.regs_access()
        registers_read = tuple(
            register_name
            for register_id in registers_read_ids
            if (register_name := decoder.reg_name(register_id))
        )
        registers_written = tuple(
            register_name
            for register_id in registers_written_ids
            if (register_name := decoder.reg_name(register_id))
        )
        memory_operands: list[dict[str, object]] = []
        immediate_operands: list[int] = []
        for operand in instruction.operands:
            if operand.type == X86_OP_IMM:
                immediate_operands.append(int(operand.imm))
                continue
            if operand.type != X86_OP_MEM:
                continue
            memory = operand.mem
            base = decoder.reg_name(memory.base) if memory.base else ""
            index = decoder.reg_name(memory.index) if memory.index else ""
            memory_operands.append(
                {
                    "operand_index": len(memory_operands),
                    "size": int(operand.size),
                    "base": base,
                    "index": index,
                    "scale": int(memory.scale),
                    "displacement": int(memory.disp),
                }
            )
        events.append(
            InstructionEvent(
                address=int(instruction.address),
                size=int(instruction.size),
                mnemonic=str(instruction.mnemonic),
                op_str=str(instruction.op_str),
                bytes_hex=bytes(instruction.bytes).hex(" "),
                groups=groups,
                registers_read=registers_read,
                registers_written=registers_written,
                immediate_operands=tuple(immediate_operands),
                memory_operands=tuple(memory_operands),
            )
        )

    if not events:
        raise ValueError("Capstone decoded zero instructions from non-empty code")
    return events


def disassemble_sample(sample_name: str = "x86_accumulator") -> list[InstructionEvent]:
    """Decode a named sample fixture."""

    return disassemble_bytes(sample_bytes(sample_name))


def instructions_to_json(events: Iterable[InstructionEvent]) -> list[dict[str, object]]:
    """Return JSON-safe instruction rows."""

    return [event.to_dict() for event in events]
