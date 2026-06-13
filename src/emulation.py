"""Unicorn-backed bounded emulation for transparent byte fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from unicorn import Uc, UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RIP, UC_X86_REG_RSP

from .disassembly import BASE_ADDRESS, sample_bytes

MAP_SIZE = 2 * 1024 * 1024
STACK_OFFSET = 0x100000


@dataclass(frozen=True)
class EmulationTrace:
    """Register and instruction trace from bounded emulation."""

    base_address: int
    instruction_count: int
    addresses: tuple[int, ...]
    initial_rax: int
    final_rax: int
    final_rip: int
    registers: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["addresses"] = list(self.addresses)
        return data


def emulate_x86_64(
    code: bytes,
    base_address: int = BASE_ADDRESS,
    *,
    initial_rax: int = 0,
    max_instructions: int = 32,
) -> EmulationTrace:
    """Run a bounded x86-64 emulation and capture instruction addresses."""

    if not code:
        raise ValueError("code must contain at least one byte")
    if max_instructions <= 0:
        raise ValueError("max_instructions must be positive")

    emulator = Uc(UC_ARCH_X86, UC_MODE_64)
    emulator.mem_map(base_address, MAP_SIZE)
    emulator.mem_write(base_address, code)
    emulator.reg_write(UC_X86_REG_RAX, initial_rax)
    emulator.reg_write(UC_X86_REG_RSP, base_address + STACK_OFFSET)

    addresses: list[int] = []

    def _record_instruction(uc: Uc, address: int, size: int, user_data: object) -> None:
        del size, user_data
        addresses.append(int(address))
        if len(addresses) >= max_instructions:
            uc.emu_stop()

    emulator.hook_add(UC_HOOK_CODE, _record_instruction)
    emulator.emu_start(base_address, base_address + len(code), count=max_instructions)

    final_rax = int(emulator.reg_read(UC_X86_REG_RAX))
    final_rip = int(emulator.reg_read(UC_X86_REG_RIP))
    return EmulationTrace(
        base_address=base_address,
        instruction_count=len(addresses),
        addresses=tuple(addresses),
        initial_rax=initial_rax,
        final_rax=final_rax,
        final_rip=final_rip,
        registers={"rax": final_rax, "rip": final_rip},
    )


def emulate_sample(sample_name: str = "x86_accumulator") -> EmulationTrace:
    """Run bounded emulation for a named sample fixture."""

    return emulate_x86_64(sample_bytes(sample_name))

