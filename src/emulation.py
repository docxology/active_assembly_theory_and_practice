"""Unicorn-backed bounded emulation for transparent byte fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from unicorn import (
    Uc,
    UC_ARCH_X86,
    UC_HOOK_CODE,
    UC_HOOK_MEM_READ,
    UC_HOOK_MEM_WRITE,
    UC_MEM_WRITE,
    UC_MODE_64,
)
from unicorn.x86_const import (
    UC_X86_REG_EFLAGS,
    UC_X86_REG_RAX,
    UC_X86_REG_RIP,
    UC_X86_REG_RSP,
)

from .disassembly import BASE_ADDRESS, sample_bytes

MAP_SIZE = 2 * 1024 * 1024
STACK_OFFSET = 0x100000


@dataclass(frozen=True)
class InstructionState:
    """Architectural register state immediately before and after one step.

    The snapshot is intentionally limited to the registers that the current
    bounded formalism can either model directly (``RAX``) or expose as an
    architectural trace boundary (``RIP``, ``RSP``, and ``EFLAGS``).  It is a
    concrete Unicorn observation, not a symbolic state or a complete machine
    state.
    """

    instruction_index: int
    address: int
    pre_rax: int
    post_rax: int
    pre_rip: int
    post_rip: int
    pre_rsp: int
    post_rsp: int
    pre_eflags: int
    post_eflags: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe architectural state row."""

        return asdict(self)


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
    memory_events: tuple[dict[str, int | str], ...] = ()
    instruction_states: tuple[InstructionState, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["addresses"] = list(self.addresses)
        data["memory_events"] = [dict(event) for event in self.memory_events]
        data["instruction_states"] = [
            state.to_dict() for state in self.instruction_states
        ]
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
    pre_states: list[dict[str, int]] = []
    memory_events: list[dict[str, int | str]] = []
    current_instruction: dict[str, int] = {"address": base_address, "index": -1}

    def _record_instruction(uc: Uc, address: int, size: int, user_data: object) -> None:
        del size, user_data
        addresses.append(int(address))
        pre_states.append(
            {
                "rax": int(uc.reg_read(UC_X86_REG_RAX)),
                "rip": int(uc.reg_read(UC_X86_REG_RIP)),
                "rsp": int(uc.reg_read(UC_X86_REG_RSP)),
                "eflags": int(uc.reg_read(UC_X86_REG_EFLAGS)),
            }
        )
        current_instruction["address"] = int(address)
        current_instruction["index"] = len(addresses) - 1
        if len(addresses) >= max_instructions:
            uc.emu_stop()

    def _record_memory(
        uc: Uc,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: object,
    ) -> None:
        del user_data
        observed_value = int(value)
        if access != UC_MEM_WRITE:
            # Unicorn supplies the write value for write callbacks, but the
            # read callback's ``value`` argument is not the loaded operand on
            # all supported versions. Read the mapped bytes before the load so
            # the event records the value the instruction actually observes.
            observed_value = int.from_bytes(
                bytes(uc.mem_read(int(address), int(size))), "little"
            )
        memory_events.append(
            {
                "event_index": len(memory_events),
                "access": "write" if access == UC_MEM_WRITE else "read",
                "address": int(address),
                "size": int(size),
                "value": observed_value,
                "instruction_address": current_instruction["address"],
                "instruction_index": current_instruction["index"],
            }
        )

    emulator.hook_add(UC_HOOK_CODE, _record_instruction)
    emulator.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, _record_memory)
    emulator.emu_start(base_address, base_address + len(code), count=max_instructions)

    final_state = {
        "rax": int(emulator.reg_read(UC_X86_REG_RAX)),
        "rip": int(emulator.reg_read(UC_X86_REG_RIP)),
        "rsp": int(emulator.reg_read(UC_X86_REG_RSP)),
        "eflags": int(emulator.reg_read(UC_X86_REG_EFLAGS)),
    }
    if len(pre_states) != len(addresses):
        raise RuntimeError("Unicorn instruction and state traces are misaligned")
    instruction_states = tuple(
        InstructionState(
            instruction_index=index,
            address=address,
            pre_rax=pre_state["rax"],
            post_rax=(
                pre_states[index + 1]["rax"]
                if index + 1 < len(pre_states)
                else final_state["rax"]
            ),
            pre_rip=pre_state["rip"],
            post_rip=(
                pre_states[index + 1]["rip"]
                if index + 1 < len(pre_states)
                else final_state["rip"]
            ),
            pre_rsp=pre_state["rsp"],
            post_rsp=(
                pre_states[index + 1]["rsp"]
                if index + 1 < len(pre_states)
                else final_state["rsp"]
            ),
            pre_eflags=pre_state["eflags"],
            post_eflags=(
                pre_states[index + 1]["eflags"]
                if index + 1 < len(pre_states)
                else final_state["eflags"]
            ),
        )
        for index, (address, pre_state) in enumerate(zip(addresses, pre_states))
    )
    final_rax = final_state["rax"]
    final_rip = final_state["rip"]
    return EmulationTrace(
        base_address=base_address,
        instruction_count=len(addresses),
        addresses=tuple(addresses),
        initial_rax=initial_rax,
        final_rax=final_rax,
        final_rip=final_rip,
        registers={
            "rax": final_rax,
            "rip": final_rip,
            "rsp": final_state["rsp"],
            "eflags": final_state["eflags"],
        },
        memory_events=tuple(memory_events),
        instruction_states=instruction_states,
    )


def emulate_sample(sample_name: str = "x86_accumulator") -> EmulationTrace:
    """Run bounded emulation for a named sample fixture."""

    return emulate_x86_64(sample_bytes(sample_name))
