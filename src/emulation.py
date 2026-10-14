"""Unicorn-backed bounded emulation for transparent byte fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from unicorn import (
    UC_ARCH_X86,
    UC_HOOK_CODE,
    UC_HOOK_MEM_READ,
    UC_HOOK_MEM_WRITE,
    UC_MEM_WRITE,
    UC_MODE_64,
    Uc,
    x86_const,
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

GENERAL_REGISTER_NAMES = (
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

GENERAL_REGISTER_IDS = {
    name: int(getattr(x86_const, f"UC_X86_REG_{name.upper()}"))
    for name in GENERAL_REGISTER_NAMES
}


@dataclass(frozen=True)
class InstructionState:
    """Architectural register state immediately before and after one step.

    The snapshot includes all sixteen general-purpose registers plus the
    program counter and modeled flags.  It is a concrete Unicorn observation,
    not a symbolic state or a complete machine state.
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
    pre_registers: dict[str, int] | None = None
    post_registers: dict[str, int] | None = None

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
    # The code hook always runs before a memory hook for an instruction.  Keep
    # the pre-hook state explicit instead of using a negative integer sentinel
    # that could be mistaken for a real instruction index.
    current_instruction: dict[str, int | None] = {
        "address": base_address,
        "index": None,
    }

    def _snapshot(uc: Uc) -> dict[str, int]:
        """Capture the declared architectural register and flag state."""

        snapshot = {
            name: int(uc.reg_read(register_id))
            for name, register_id in GENERAL_REGISTER_IDS.items()
        }
        snapshot["rip"] = int(uc.reg_read(UC_X86_REG_RIP))
        snapshot["eflags"] = int(uc.reg_read(UC_X86_REG_EFLAGS))
        return snapshot

    def _record_instruction(uc: Uc, address: int, size: int, user_data: object) -> None:
        del size, user_data
        addresses.append(int(address))
        pre_states.append(_snapshot(uc))
        current_instruction["address"] = int(address)
        current_instruction["index"] = len(addresses) - 1
        # Unicorn's native count limit stops after executing the last allowed
        # instruction. Stopping here would record that instruction's pre-state
        # but suppress its effects, because UC_HOOK_CODE runs before execution.

    def _record_memory(
        uc: Uc,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: object,
    ) -> None:
        del user_data
        instruction_index = current_instruction["index"]
        if instruction_index is None:
            raise RuntimeError("memory hook fired before the code hook")
        instruction_address = current_instruction["address"]
        if instruction_address is None:
            raise RuntimeError("memory hook has no current instruction address")
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
                "instruction_address": instruction_address,
                "instruction_index": instruction_index,
            }
        )

    emulator.hook_add(UC_HOOK_CODE, _record_instruction)
    emulator.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, _record_memory)
    emulator.emu_start(base_address, base_address + len(code), count=max_instructions)

    final_state = _snapshot(emulator)
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
            pre_registers={
                name: value
                for name, value in pre_state.items()
                if name not in {"rip", "eflags"}
            },
            post_registers={
                name: value
                for name, value in (
                    pre_states[index + 1]
                    if index + 1 < len(pre_states)
                    else final_state
                ).items()
                if name not in {"rip", "eflags"}
            },
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
        registers=dict(final_state),
        memory_events=tuple(memory_events),
        instruction_states=instruction_states,
    )


def emulate_sample(sample_name: str = "x86_accumulator") -> EmulationTrace:
    """Run bounded emulation for a named sample fixture."""

    return emulate_x86_64(sample_bytes(sample_name))
