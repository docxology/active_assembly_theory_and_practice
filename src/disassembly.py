"""Capstone-backed disassembly for transparent byte fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from capstone import Cs, CS_ARCH_X86, CS_MODE_64

BASE_ADDRESS = 0x1000

SAMPLE_PROGRAMS: dict[str, bytes] = {
    # mov eax, 1; add eax, 2; shl eax, 1; sub eax, 1
    "x86_accumulator": bytes.fromhex("b8 01 00 00 00 83 c0 02 d1 e0 83 e8 01"),
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["groups"] = list(self.groups)
        return data


def sample_bytes(sample_name: str = "x86_accumulator") -> bytes:
    """Return the deterministic byte fixture for a named sample."""

    try:
        return SAMPLE_PROGRAMS[sample_name]
    except KeyError as exc:
        names = ", ".join(sorted(SAMPLE_PROGRAMS))
        raise ValueError(f"unknown sample {sample_name!r}; available samples: {names}") from exc


def disassemble_bytes(code: bytes, base_address: int = BASE_ADDRESS) -> list[InstructionEvent]:
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
        events.append(
            InstructionEvent(
                address=int(instruction.address),
                size=int(instruction.size),
                mnemonic=str(instruction.mnemonic),
                op_str=str(instruction.op_str),
                bytes_hex=bytes(instruction.bytes).hex(" "),
                groups=groups,
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

