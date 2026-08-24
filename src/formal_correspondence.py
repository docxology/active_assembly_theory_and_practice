"""Bounded Python/Capstone/Unicorn correspondence for a declared x86 subset.

The module contains the executable reference semantics used by the local
correspondence receipt.  It is intentionally explicit about its boundary:
the state is a bounded 64-bit user-mode machine state, not a complete x86-64
semantics, and the final comparison is a computational cross-check against
 bounded Unicorn fixtures.  A generated Lean bridge checks one shared
Word64 boundary fixture; the broader Python/Capstone/Unicorn correspondence
remains computational evidence rather than a kernel theorem.

Two design rules are important here:

* architectural registers are represented by an explicit register file; no
  synthetic memory address can stand for a register;
* operand width is carried through every read and write, so a 32-bit write
  zero-extends while 8- and 16-bit writes preserve the upper register bits.

Unsupported instructions and unsupported flag dependencies fail closed with a
structured reason at the trace boundary.  They are never assigned a guessed
meaning merely to make a receipt green.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, cast

from .disassembly import (
    BASE_ADDRESS,
    SAMPLE_PROGRAMS,
    InstructionEvent,
    disassemble_bytes,
)
from .emulation import STACK_OFFSET, emulate_x86_64
from .formal_verification import formal_verification_payload
from .lean_bridge import BRIDGE_CODE, BRIDGE_EXPECTED_RAX, BRIDGE_INITIAL_RAX
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import SUPPORTED_X86_USER_MODE, scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_CORRESPONDENCE_SCHEMA_VERSION = "2.0.0"
FORMAL_CORRESPONDENCE_ARTIFACT = "output/data/formal_trace_correspondence.json"

U64_MASK = (1 << 64) - 1
U32_MASK = (1 << 32) - 1
ARCHITECTURAL_FLAGS_MASK = (
    (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6) | (1 << 7) | (1 << 11)
)
MODELED_FLAGS = ("CF", "PF", "AF", "ZF", "SF", "OF")
FLAG_BITS = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "OF": 11}
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
MEMORY_LOWER_BOUND = BASE_ADDRESS
MEMORY_UPPER_BOUND = BASE_ADDRESS + (2 * 1024 * 1024)
DEFAULT_MEMORY_BOUNDS = (MEMORY_LOWER_BOUND, MEMORY_UPPER_BOUND)
MAX_REFERENCE_STEPS = 256

_REGISTER_ALIASES: dict[str, tuple[str, int, int]] = {
    "rax": ("rax", 64, 0),
    "eax": ("rax", 32, 0),
    "ax": ("rax", 16, 0),
    "al": ("rax", 8, 0),
    "ah": ("rax", 8, 8),
    "rbx": ("rbx", 64, 0),
    "ebx": ("rbx", 32, 0),
    "bx": ("rbx", 16, 0),
    "bl": ("rbx", 8, 0),
    "bh": ("rbx", 8, 8),
    "rcx": ("rcx", 64, 0),
    "ecx": ("rcx", 32, 0),
    "cx": ("rcx", 16, 0),
    "cl": ("rcx", 8, 0),
    "ch": ("rcx", 8, 8),
    "rdx": ("rdx", 64, 0),
    "edx": ("rdx", 32, 0),
    "dx": ("rdx", 16, 0),
    "dl": ("rdx", 8, 0),
    "dh": ("rdx", 8, 8),
    "rsi": ("rsi", 64, 0),
    "esi": ("rsi", 32, 0),
    "si": ("rsi", 16, 0),
    "sil": ("rsi", 8, 0),
    "rdi": ("rdi", 64, 0),
    "edi": ("rdi", 32, 0),
    "di": ("rdi", 16, 0),
    "dil": ("rdi", 8, 0),
    "rbp": ("rbp", 64, 0),
    "ebp": ("rbp", 32, 0),
    "bp": ("rbp", 16, 0),
    "bpl": ("rbp", 8, 0),
    "rsp": ("rsp", 64, 0),
    "esp": ("rsp", 32, 0),
    "sp": ("rsp", 16, 0),
    "spl": ("rsp", 8, 0),
}
for _name in GENERAL_REGISTER_NAMES:
    if _name.startswith("r") and _name[1:].isdigit():
        _REGISTER_ALIASES.update(
            {
                _name: (_name, 64, 0),
                f"{_name}d": (_name, 32, 0),
                f"{_name}w": (_name, 16, 0),
                f"{_name}b": (_name, 8, 0),
            }
        )

_INTEGER_RE = re.compile(r"^[+-]?(?:0x[0-9a-f]+|[0-9]+)$", re.IGNORECASE)
_SIZE_RE = re.compile(
    r"^(?:byte|word|dword|qword|oword|xmmword)\s+ptr\s+", re.IGNORECASE
)
_SIZE_BYTES = {"byte": 1, "word": 2, "dword": 4, "qword": 8}


@dataclass(frozen=True)
class IRInstruction:
    """Normalized instruction metadata shared by the reference boundary."""

    address: int
    size: int
    mnemonic: str
    operands: tuple[str, ...]
    widths: tuple[int, ...]
    immediate: int | None
    memory_addressing: tuple[dict[str, object], ...]
    flag_reads: tuple[str, ...]
    flag_writes: tuple[str, ...]
    control_flow: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe normalized instruction metadata."""

        data = asdict(self)
        for key in ("operands", "widths", "flag_reads", "flag_writes"):
            data[key] = list(data[key])
        data["memory_addressing"] = [dict(item) for item in self.memory_addressing]
        return data


@dataclass
class FormalState:
    """Explicit bounded machine state used by the Python reference semantics.

    The scalar ``rax``/``rsp``/``rip``/``eflags`` fields are retained as a
    compatibility projection for existing package consumers.  The canonical
    state is ``registers`` plus the byte-addressable ``memory`` map.
    """

    rax: int
    rsp: int
    rip: int
    eflags: int
    memory: dict[int, int]
    registers: dict[str, int] = field(default_factory=dict)
    flags_defined: dict[str, bool] = field(default_factory=dict)
    stack_bounds: tuple[int, int] = (
        BASE_ADDRESS + STACK_OFFSET - 4096,
        BASE_ADDRESS + STACK_OFFSET + 4096,
    )
    memory_bounds: tuple[int, int] = DEFAULT_MEMORY_BOUNDS
    step_index: int = 0
    current_instruction: dict[str, object] | None = None
    status: str = "ok"
    memory_effects: list[dict[str, int | str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize aliases, widths, bounds, and flags at construction."""

        # Scalar RAX/RSP values are the compatibility/default projection.  A
        # partial explicit register file overlays those defaults; it must not
        # silently turn an omitted RSP into zero merely because another GPR
        # was supplied.
        canonical = {name: 0 for name in GENERAL_REGISTER_NAMES}
        canonical["rax"] = int(self.rax)
        canonical["rsp"] = int(self.rsp)
        if self.registers:
            for name, value in self.registers.items():
                base, width, offset = _register_ref(name)
                canonical[base] = _write_register_value(
                    canonical[base], int(value), width, offset
                )
        self.registers = {
            name: int(value) & U64_MASK for name, value in canonical.items()
        }
        self.rax = self.registers["rax"]
        self.rsp = self.registers["rsp"]
        self.rip = int(self.rip) & U64_MASK
        self.eflags = int(self.eflags) & U64_MASK
        if not self.flags_defined:
            self.flags_defined = {name: True for name in MODELED_FLAGS}
        else:
            self.flags_defined = {
                name: bool(self.flags_defined.get(name, False))
                for name in MODELED_FLAGS
            }
        lower, upper = self.memory_bounds
        if lower < 0 or upper <= lower:
            raise ValueError("memory bounds must be a non-empty non-negative interval")
        stack_lower, stack_upper = self.stack_bounds
        if stack_lower < 0 or stack_upper <= stack_lower:
            raise ValueError("stack bounds must be a non-empty non-negative interval")
        if any(
            int(address) < lower or int(address) >= upper or not 0 <= int(value) <= 255
            for address, value in self.memory.items()
        ):
            raise ValueError(
                "memory must contain bounded byte addresses and byte values"
            )
        self.memory = {
            int(address): int(value) for address, value in self.memory.items()
        }

    def clone(self) -> FormalState:
        """Return a deep copy suitable for pre-state capture."""

        return FormalState(
            rax=self.rax,
            rsp=self.rsp,
            rip=self.rip,
            eflags=self.eflags,
            memory=dict(self.memory),
            registers=dict(self.registers),
            flags_defined=dict(self.flags_defined),
            stack_bounds=self.stack_bounds,
            memory_bounds=self.memory_bounds,
            step_index=self.step_index,
            current_instruction=deepcopy(self.current_instruction),
            status=self.status,
            memory_effects=[dict(item) for item in self.memory_effects],
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete normalized state."""

        return {
            "registers": dict(sorted(self.registers.items())),
            "rip": self.rip,
            "eflags": self.eflags,
            "flags_defined": dict(sorted(self.flags_defined.items())),
            "memory": {str(key): value for key, value in sorted(self.memory.items())},
            "stack_bounds": list(self.stack_bounds),
            "memory_bounds": list(self.memory_bounds),
            "step_index": self.step_index,
            "current_instruction": deepcopy(self.current_instruction),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FormalState:
        """Reconstruct a normalized state from its JSON-safe representation."""

        registers = cast(dict[str, int], payload.get("registers", {}))
        memory_payload = cast(dict[str, int], payload.get("memory", {}))
        flags_defined = cast(dict[str, bool], payload.get("flags_defined", {}))
        stack_bounds = tuple(cast(list[int], payload.get("stack_bounds", [])))
        memory_bounds = tuple(cast(list[int], payload.get("memory_bounds", [])))
        if len(stack_bounds) != 2 or len(memory_bounds) != 2:
            raise ValueError(
                "serialized state bounds must contain lower and upper values"
            )
        return cls(
            rax=int(registers.get("rax", 0)),
            rsp=int(registers.get("rsp", 0)),
            rip=int(cast(int, payload.get("rip", 0))),
            eflags=int(cast(int, payload.get("eflags", 0))),
            memory={
                int(address): int(value) for address, value in memory_payload.items()
            },
            registers=registers,
            flags_defined=flags_defined,
            stack_bounds=(int(stack_bounds[0]), int(stack_bounds[1])),
            memory_bounds=(int(memory_bounds[0]), int(memory_bounds[1])),
            step_index=int(cast(int, payload.get("step_index", 0))),
            current_instruction=cast(
                dict[str, object] | None, payload.get("current_instruction")
            ),
            status=str(payload.get("status", "ok")),
        )

    def get_register(self, name: str) -> int:
        """Read an architectural register alias with its declared width."""

        base, width, offset = _register_ref(name)
        return (self.registers[base] >> offset) & ((1 << width) - 1)

    def set_register(self, name: str, value: int) -> None:
        """Write an architectural register alias with x86 width semantics."""

        base, width, offset = _register_ref(name)
        self.registers[base] = (
            _write_register_value(self.registers[base], int(value), width, offset)
            & U64_MASK
        )
        self.rax = self.registers["rax"]
        self.rsp = self.registers["rsp"]


# Public name requested by the protocol. Keeping the alias avoids two subtly
# different machine-state implementations in downstream consumers.
MachineState = FormalState


@dataclass(frozen=True)
class StepResult:
    """Structured result for one reference step."""

    pre_state: dict[str, object]
    instruction: dict[str, object]
    post_state: dict[str, object]
    memory_effects: tuple[dict[str, int | str], ...]
    control_target: int | None
    status: str
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe step result."""

        data = asdict(self)
        data["memory_effects"] = [dict(item) for item in self.memory_effects]
        return data


def _register_ref(name: str) -> tuple[str, int, int]:
    normalized = name.strip().lower().removeprefix("%")
    try:
        return _REGISTER_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported register {name!r}") from exc


def _write_register_value(current: int, value: int, width: int, offset: int = 0) -> int:
    if width not in (8, 16, 32, 64) or offset < 0 or offset + width > 64:
        raise ValueError(
            f"unsupported register width/offset: width={width}, offset={offset}"
        )
    if width == 32 and offset == 0:
        return value & U32_MASK
    mask = ((1 << width) - 1) << offset
    return (current & ~mask) | ((value & ((1 << width) - 1)) << offset)


def _u64(value: int) -> int:
    return int(value) & U64_MASK


def _split_operands(op_str: str) -> tuple[str, ...]:
    """Split Capstone operands without splitting commas in brackets."""

    operands: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(op_str):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            operands.append(op_str[start:index].strip())
            start = index + 1
    final = op_str[start:].strip()
    if final:
        operands.append(final)
    return tuple(operands)


def _clean_operand(token: str) -> str:
    return _SIZE_RE.sub("", token.strip().lower()).strip()


def _token_width(token: str, default: int = 64) -> int:
    raw = token.strip().lower()
    size_match = re.match(r"^(byte|word|dword|qword)\s+ptr", raw)
    if size_match:
        return _SIZE_BYTES[size_match.group(1)] * 8
    clean = _clean_operand(token)
    if "[" in clean:
        return default
    try:
        _, width, _ = _register_ref(clean)
        return width
    except ValueError:
        return default


def _flag_reads_and_writes(mnemonic: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mnemonic = mnemonic.lower()
    reads: tuple[str, ...]
    if mnemonic in {"adc", "sbb"}:
        reads = ("CF",)
    elif mnemonic in {"jl", "jnge", "jge", "jnl"}:
        reads = ("SF", "OF")
    elif mnemonic in {"jle", "jng", "jg", "jnle"}:
        reads = ("SF", "OF", "ZF")
    elif mnemonic in {"jbe", "jna", "ja", "jnbe"}:
        reads = ("CF", "ZF")
    elif mnemonic in {"je", "jz", "jne", "jnz"}:
        reads = ("ZF",)
    elif mnemonic in {"jb", "jc", "jnae", "jae", "jnb", "jnc"}:
        reads = ("CF",)
    elif mnemonic in {"jo", "jno"}:
        reads = ("OF",)
    elif mnemonic in {"js", "jns"}:
        reads = ("SF",)
    elif mnemonic in {"jp", "jpe", "jnp", "jpo"}:
        reads = ("PF",)
    else:
        reads = ()
    writes: tuple[str, ...]
    if mnemonic in {
        "cmp",
        "test",
        "add",
        "adc",
        "sub",
        "sbb",
        "neg",
        "inc",
        "dec",
        "and",
        "or",
        "xor",
        "shl",
        "sal",
        "shr",
        "sar",
    }:
        writes = MODELED_FLAGS
    else:
        writes = ()
    if mnemonic in {"inc", "dec"}:
        writes = tuple(flag for flag in writes if flag != "CF")
    return tuple(reads), tuple(writes)


def normalize_instruction(event: InstructionEvent) -> IRInstruction:
    """Convert a Capstone event into the shared normalized IR."""

    mnemonic = event.mnemonic.lower()
    if mnemonic not in SUPPORTED_X86_USER_MODE and not mnemonic.startswith("j"):
        raise ValueError(f"unsupported mnemonic {mnemonic!r}")
    operands = _split_operands(event.op_str)
    widths = tuple(_token_width(token) for token in operands)
    reads, writes = _flag_reads_and_writes(mnemonic)
    if mnemonic in {"jmp", "call", "ret"} or mnemonic.startswith("j"):
        control_flow = (
            "return"
            if mnemonic == "ret"
            else "call"
            if mnemonic == "call"
            else "branch"
        )
    else:
        control_flow = "fallthrough"
    return IRInstruction(
        address=event.address,
        size=event.size,
        mnemonic=mnemonic,
        operands=operands,
        widths=widths,
        immediate=int(event.immediate_operands[0])
        if event.immediate_operands
        else None,
        memory_addressing=tuple(dict(item) for item in event.memory_operands),
        flag_reads=reads,
        flag_writes=writes,
        control_flow=control_flow,
    )


def _set_flag(
    state: FormalState, flag: str, value: bool, *, defined: bool = True
) -> None:
    bit = FLAG_BITS[flag]
    if value:
        state.eflags |= 1 << bit
    else:
        state.eflags &= ~(1 << bit)
    state.flags_defined[flag] = defined


def _get_flag(state: FormalState, flag: str) -> bool:
    if not state.flags_defined.get(flag, False):
        raise ValueError(f"unsupported read of undefined flag {flag}")
    return bool(state.eflags & (1 << FLAG_BITS[flag]))


def _parity_even(value: int) -> bool:
    return (value & 0xFF).bit_count() % 2 == 0


def _set_logic_flags(state: FormalState, result: int, width: int) -> None:
    mask = (1 << width) - 1
    value = result & mask
    _set_flag(state, "CF", False)
    _set_flag(state, "OF", False)
    _set_flag(state, "ZF", value == 0)
    _set_flag(state, "SF", bool(value & (1 << (width - 1))))
    _set_flag(state, "PF", _parity_even(value))
    _set_flag(state, "AF", False, defined=False)


def _set_arithmetic_flags(
    state: FormalState,
    left: int,
    right: int,
    result: int,
    *,
    width: int,
    subtract: bool,
    carry_in: int = 0,
) -> None:
    """Set x86 arithmetic flags for add/sub with an explicit operand width."""

    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    left &= mask
    right &= mask
    result &= mask
    signed_min = -(1 << (width - 1))
    signed_max = (1 << (width - 1)) - 1
    signed_left = left if left < sign else left - (1 << width)
    signed_right = right if right < sign else right - (1 << width)
    if subtract:
        full_right = right + carry_in
        carry = left < full_right
        auxiliary = (left & 0xF) < ((right & 0xF) + carry_in)
        signed_result = signed_left - signed_right - carry_in
        overflow = not signed_min <= signed_result <= signed_max
    else:
        full_right = right + carry_in
        carry = left + full_right > mask
        auxiliary = (left & 0xF) + (right & 0xF) + carry_in > 0xF
        signed_result = signed_left + signed_right + carry_in
        overflow = not signed_min <= signed_result <= signed_max
    _set_flag(state, "CF", carry)
    _set_flag(state, "AF", auxiliary)
    _set_flag(state, "ZF", result == 0)
    _set_flag(state, "SF", bool(result & sign))
    _set_flag(state, "PF", _parity_even(result))
    _set_flag(state, "OF", overflow)


def _memory_address(
    event: InstructionEvent, state: FormalState, operand_index: int = 0
) -> int:
    if operand_index >= len(event.memory_operands):
        raise ValueError("memory address requested for a non-memory operand")
    operand = cast(dict[str, Any], event.memory_operands[operand_index])
    base = str(operand.get("base", ""))
    index = str(operand.get("index", ""))
    scale = int(operand.get("scale", 1))
    displacement = int(operand.get("displacement", 0))
    address = displacement
    if base:
        address += state.rip if base == "rip" else state.get_register(base)
    if index:
        address += (
            state.rip * scale if index == "rip" else state.get_register(index) * scale
        )
    return _u64(address)


def _memory_width(
    event: InstructionEvent, operand_index: int = 0, default: int = 8
) -> int:
    if operand_index >= len(event.memory_operands):
        return default
    operand = cast(dict[str, Any], event.memory_operands[operand_index])
    width = int(operand.get("size", 0))
    return width if width in (1, 2, 4, 8) else default


def _check_memory_range(state: FormalState, address: int, width: int) -> None:
    lower, upper = state.memory_bounds
    if address < lower or address + width > upper:
        raise ValueError(
            f"memory access [{address:#x}, {address + width:#x}) exceeds bounded memory"
        )


def _check_stack_range(state: FormalState, address: int, width: int) -> None:
    """Reject stack effects that leave the declared bounded stack window."""

    lower, upper = state.stack_bounds
    if address < lower or address + width > upper:
        raise ValueError(
            f"stack access [{address:#x}, {address + width:#x}) exceeds bounded stack"
        )


def _read_memory(state: FormalState, address: int, width: int) -> int:
    if width not in (1, 2, 4, 8):
        raise ValueError(f"unsupported memory width {width}")
    _check_memory_range(state, address, width)
    value = int.from_bytes(
        bytes(state.memory.get(address + offset, 0) for offset in range(width)),
        "little",
    )
    state.memory_effects.append(
        {"access": "read", "address": address, "size": width, "value": value}
    )
    return value


def _write_memory(state: FormalState, address: int, width: int, value: int) -> None:
    if width not in (1, 2, 4, 8):
        raise ValueError(f"unsupported memory width {width}")
    _check_memory_range(state, address, width)
    encoded = int(value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
    state.memory.update({address + offset: byte for offset, byte in enumerate(encoded)})
    state.memory_effects.append(
        {
            "access": "write",
            "address": address,
            "size": width,
            "value": int.from_bytes(encoded, "little"),
        }
    )


def _parse_immediate(token: str, event: InstructionEvent, index: int = 0) -> int:
    clean = _clean_operand(token)
    if _INTEGER_RE.match(clean):
        return int(clean, 0)
    if index < len(event.immediate_operands):
        return int(event.immediate_operands[index])
    raise ValueError(f"unsupported immediate operand {token!r}")


def _is_memory(token: str) -> bool:
    return "[" in token and "]" in token


def _read_operand(
    event: InstructionEvent,
    state: FormalState,
    token: str,
    *,
    memory_index: int,
    immediate_index: int = 0,
    default_width: int = 8,
) -> tuple[int, int]:
    clean = _clean_operand(token)
    if _is_memory(clean):
        width = _memory_width(event, memory_index, default_width)
        return (
            _read_memory(state, _memory_address(event, state, memory_index), width),
            width,
        )
    try:
        _, width, _ = _register_ref(clean)
    except ValueError:
        if _INTEGER_RE.match(clean):
            return int(clean, 0), default_width
        if immediate_index < len(event.immediate_operands):
            return int(event.immediate_operands[immediate_index]), default_width
        raise
    return state.get_register(clean), width // 8


def _write_operand(
    event: InstructionEvent,
    state: FormalState,
    token: str,
    value: int,
    *,
    memory_index: int,
    width: int,
) -> None:
    clean = _clean_operand(token)
    if _is_memory(clean):
        _write_memory(state, _memory_address(event, state, memory_index), width, value)
        return
    state.set_register(clean, value)


def _sign_extend(value: int, source_width: int, destination_width: int) -> int:
    source_mask = (1 << source_width) - 1
    value &= source_mask
    if value & (1 << (source_width - 1)):
        value -= 1 << source_width
    return value & ((1 << destination_width) - 1)


def _condition_holds(state: FormalState, mnemonic: str) -> bool:
    mnemonic = mnemonic.lower()
    if mnemonic in {"jl", "jnge"}:
        return _get_flag(state, "SF") != _get_flag(state, "OF")
    if mnemonic in {"jge", "jnl"}:
        return _get_flag(state, "SF") == _get_flag(state, "OF")
    if mnemonic in {"jle", "jng"}:
        return _get_flag(state, "ZF") or (
            _get_flag(state, "SF") != _get_flag(state, "OF")
        )
    if mnemonic in {"jg", "jnle"}:
        return not _get_flag(state, "ZF") and (
            _get_flag(state, "SF") == _get_flag(state, "OF")
        )
    if mnemonic in {"je", "jz"}:
        return _get_flag(state, "ZF")
    if mnemonic in {"jne", "jnz"}:
        return not _get_flag(state, "ZF")
    if mnemonic in {"jb", "jc", "jnae"}:
        return _get_flag(state, "CF")
    if mnemonic in {"jae", "jnb", "jnc"}:
        return not _get_flag(state, "CF")
    if mnemonic in {"jbe", "jna"}:
        return _get_flag(state, "CF") or _get_flag(state, "ZF")
    if mnemonic in {"ja", "jnbe"}:
        return not _get_flag(state, "CF") and not _get_flag(state, "ZF")
    if mnemonic == "jo":
        return _get_flag(state, "OF")
    if mnemonic == "jno":
        return not _get_flag(state, "OF")
    if mnemonic == "js":
        return _get_flag(state, "SF")
    if mnemonic == "jns":
        return not _get_flag(state, "SF")
    if mnemonic in {"jp", "jpe"}:
        return _get_flag(state, "PF")
    if mnemonic in {"jnp", "jpo"}:
        return not _get_flag(state, "PF")
    if mnemonic == "jmp":
        return True
    if mnemonic.startswith("j"):
        raise ValueError(f"unsupported branch condition {mnemonic!r}")
    return False


def _next_rip(
    state: FormalState, event: InstructionEvent, next_address: int | None
) -> int:
    return next_address if next_address is not None else _u64(state.rip + event.size)


def _step(
    event: InstructionEvent, state: FormalState, next_address: int | None
) -> StepResult:
    """Apply one supported instruction and return a structured step result."""

    pre_state = state.clone()
    state.memory_effects = []
    ir = normalize_instruction(event)
    state.current_instruction = ir.to_dict()
    mnemonic = ir.mnemonic
    operands = ir.operands
    memory_index = 0
    control_target: int | None = None

    if mnemonic == "nop":
        state.rip = _next_rip(state, event, next_address)
    elif mnemonic in {"mov", "movabs"}:
        if len(operands) != 2:
            raise ValueError(f"unsupported move form: {event.op_str}")
        destination, source = operands
        destination_width = _token_width(destination)
        source_value, _ = _read_operand(
            event,
            state,
            source,
            memory_index=memory_index,
            default_width=max(1, destination_width // 8),
        )
        if _is_memory(source):
            memory_index += 1
        _write_operand(
            event,
            state,
            destination,
            source_value,
            memory_index=memory_index,
            width=_memory_width(event, memory_index, max(1, destination_width // 8))
            if _is_memory(destination)
            else max(1, destination_width // 8),
        )
    elif mnemonic in {"movzx", "movsx"}:
        if len(operands) != 2:
            raise ValueError(f"unsupported extension form: {event.op_str}")
        destination, source = operands
        destination_width = _token_width(destination)
        source_value, source_width = _read_operand(
            event,
            state,
            source,
            memory_index=memory_index,
            default_width=max(1, destination_width // 8),
        )
        if _is_memory(source):
            memory_index += 1
        source_bits = source_width * 8
        value = (
            source_value & ((1 << source_bits) - 1)
            if mnemonic == "movzx"
            else _sign_extend(source_value, source_bits, destination_width)
        )
        _write_operand(
            event,
            state,
            destination,
            value,
            memory_index=memory_index,
            width=max(1, destination_width // 8),
        )
    elif mnemonic == "lea":
        if len(operands) != 2 or not _is_memory(operands[1]):
            raise ValueError(f"unsupported lea form: {event.op_str}")
        address = _memory_address(event, state, 0)
        _write_operand(
            event,
            state,
            operands[0],
            address,
            memory_index=0,
            width=max(1, _token_width(operands[0]) // 8),
        )
    elif mnemonic in {"add", "adc", "sub", "sbb", "cmp"}:
        if len(operands) != 2:
            raise ValueError(f"unsupported arithmetic form: {event.op_str}")
        destination, source = operands
        destination_value, width = _read_operand(
            event,
            state,
            destination,
            memory_index=memory_index,
            default_width=max(1, _token_width(destination) // 8),
        )
        if _is_memory(destination):
            memory_index += 1
        source_value, _ = _read_operand(
            event,
            state,
            source,
            memory_index=memory_index,
            default_width=width,
        )
        width_bits = width * 8
        carry_in = int(_get_flag(state, "CF")) if mnemonic in {"adc", "sbb"} else 0
        mask = (1 << width_bits) - 1
        if mnemonic in {"add", "adc"}:
            result = (destination_value + source_value + carry_in) & mask
            _set_arithmetic_flags(
                state,
                destination_value,
                source_value,
                result,
                width=width_bits,
                subtract=False,
                carry_in=carry_in,
            )
        else:
            result = (destination_value - source_value - carry_in) & mask
            _set_arithmetic_flags(
                state,
                destination_value,
                source_value,
                result,
                width=width_bits,
                subtract=True,
                carry_in=carry_in,
            )
        if mnemonic != "cmp":
            _write_operand(
                event, state, destination, result, memory_index=0, width=width
            )
    elif mnemonic in {"inc", "dec", "neg"}:
        if len(operands) != 1:
            raise ValueError(f"unsupported unary arithmetic form: {event.op_str}")
        destination = operands[0]
        value, width = _read_operand(
            event,
            state,
            destination,
            memory_index=0,
            default_width=max(1, _token_width(destination) // 8),
        )
        width_bits = width * 8
        mask = (1 << width_bits) - 1
        if mnemonic == "inc":
            result = (value + 1) & mask
            carry_before = bool(state.eflags & (1 << FLAG_BITS["CF"]))
            carry_was_defined = state.flags_defined.get("CF", False)
            _set_arithmetic_flags(
                state, value, 1, result, width=width_bits, subtract=False
            )
            if carry_was_defined:
                _set_flag(state, "CF", carry_before)
            else:
                state.flags_defined["CF"] = False
        elif mnemonic == "dec":
            result = (value - 1) & mask
            carry_before = bool(state.eflags & (1 << FLAG_BITS["CF"]))
            carry_was_defined = state.flags_defined.get("CF", False)
            _set_arithmetic_flags(
                state, value, 1, result, width=width_bits, subtract=True
            )
            if carry_was_defined:
                _set_flag(state, "CF", carry_before)
            else:
                state.flags_defined["CF"] = False
        else:
            result = (-value) & mask
            _set_arithmetic_flags(
                state, 0, value, result, width=width_bits, subtract=True
            )
        _write_operand(event, state, destination, result, memory_index=0, width=width)
    elif mnemonic in {"and", "or", "xor", "test", "not"}:
        if mnemonic == "not":
            if len(operands) != 1:
                raise ValueError(f"unsupported not form: {event.op_str}")
            destination = operands[0]
            value, width = _read_operand(
                event,
                state,
                destination,
                memory_index=0,
                default_width=max(1, _token_width(destination) // 8),
            )
            _write_operand(
                event,
                state,
                destination,
                (~value) & ((1 << (width * 8)) - 1),
                memory_index=0,
                width=width,
            )
        else:
            if len(operands) != 2:
                raise ValueError(f"unsupported logical form: {event.op_str}")
            destination, source = operands
            left, width = _read_operand(
                event,
                state,
                destination,
                memory_index=0,
                default_width=max(1, _token_width(destination) // 8),
            )
            right, _ = _read_operand(
                event,
                state,
                source,
                memory_index=1 if _is_memory(destination) else 0,
                default_width=width,
            )
            if mnemonic in {"xor", "test"}:
                result = left ^ right
            elif mnemonic == "and":
                result = left & right
            else:
                result = left | right
            _set_logic_flags(state, result, width * 8)
            if mnemonic != "test":
                _write_operand(
                    event, state, destination, result, memory_index=0, width=width
                )
    elif mnemonic in {"shl", "sal", "shr", "sar"}:
        if len(operands) != 2:
            raise ValueError(f"unsupported shift form: {event.op_str}")
        destination, source = operands
        value, width = _read_operand(
            event,
            state,
            destination,
            memory_index=0,
            default_width=max(1, _token_width(destination) // 8),
        )
        shift, _ = _read_operand(
            event,
            state,
            source,
            memory_index=1 if _is_memory(destination) else 0,
            default_width=1,
        )
        width_bits = width * 8
        count = shift & (0x3F if width_bits == 64 else 0x1F)
        mask = (1 << width_bits) - 1
        result = value & mask
        if count:
            if mnemonic in {"shl", "sal"}:
                _set_flag(state, "CF", bool(value & (1 << (width_bits - count))))
                result = (value << count) & mask
                _set_flag(
                    state,
                    "OF",
                    bool((result ^ value) & (1 << (width_bits - 1))),
                    defined=count == 1,
                )
            elif mnemonic == "shr":
                _set_flag(state, "CF", bool(value & (1 << (count - 1))))
                result = value >> count
                _set_flag(
                    state,
                    "OF",
                    bool(value & (1 << (width_bits - 1))),
                    defined=count == 1,
                )
            else:
                _set_flag(state, "CF", bool(value & (1 << (count - 1))))
                signed = (
                    value - (1 << width_bits)
                    if value & (1 << (width_bits - 1))
                    else value
                )
                result = (signed >> count) & mask
                _set_flag(state, "OF", False, defined=count == 1)
            _set_flag(state, "ZF", result == 0)
            _set_flag(state, "SF", bool(result & (1 << (width_bits - 1))))
            _set_flag(state, "PF", _parity_even(result))
            _set_flag(state, "AF", False, defined=False)
        _write_operand(event, state, destination, result, memory_index=0, width=width)
    elif mnemonic == "jmp" or mnemonic.startswith("j"):
        if not operands:
            raise ValueError(f"unsupported branch form: {event.op_str}")
        taken = _condition_holds(state, mnemonic)
        target = _parse_immediate(operands[0], event)
        control_target = target if taken else _next_rip(state, event, next_address)
        state.rip = control_target
        state.step_index += 1
        return StepResult(
            pre_state=pre_state.to_dict(),
            instruction=ir.to_dict(),
            post_state=state.clone().to_dict(),
            memory_effects=tuple(dict(item) for item in state.memory_effects),
            control_target=control_target,
            status="ok",
        )
    elif mnemonic == "push":
        if len(operands) != 1:
            raise ValueError(f"unsupported push form: {event.op_str}")
        value, _ = _read_operand(
            event, state, operands[0], memory_index=0, default_width=8
        )
        new_rsp = _u64(state.rsp - 8)
        _check_stack_range(state, new_rsp, 8)
        state.set_register("rsp", new_rsp)
        _write_memory(state, new_rsp, 8, value)
    elif mnemonic == "pop":
        if len(operands) != 1:
            raise ValueError(f"unsupported pop form: {event.op_str}")
        _check_stack_range(state, state.rsp, 8)
        value = _read_memory(state, state.rsp, 8)
        state.set_register(operands[0], value)
        state.set_register("rsp", _u64(state.rsp + 8))
    elif mnemonic == "call":
        if len(operands) != 1:
            raise ValueError(f"unsupported call form: {event.op_str}")
        target = _parse_immediate(operands[0], event)
        return_address = _next_rip(state, event, next_address)
        new_rsp = _u64(state.rsp - 8)
        _check_stack_range(state, new_rsp, 8)
        state.set_register("rsp", new_rsp)
        _write_memory(state, new_rsp, 8, return_address)
        state.rip = target
        control_target = target
        state.step_index += 1
        return StepResult(
            pre_state=pre_state.to_dict(),
            instruction=ir.to_dict(),
            post_state=state.clone().to_dict(),
            memory_effects=tuple(dict(item) for item in state.memory_effects),
            control_target=control_target,
            status="ok",
        )
    elif mnemonic == "ret":
        _check_stack_range(state, state.rsp, 8)
        target = _read_memory(state, state.rsp, 8)
        state.set_register("rsp", _u64(state.rsp + 8))
        state.rip = target
        control_target = target
        state.step_index += 1
        return StepResult(
            pre_state=pre_state.to_dict(),
            instruction=ir.to_dict(),
            post_state=state.clone().to_dict(),
            memory_effects=tuple(dict(item) for item in state.memory_effects),
            control_target=control_target,
            status="ok",
        )
    else:
        raise ValueError(f"unsupported mnemonic {mnemonic!r}")

    state.rip = _next_rip(state, event, next_address)
    state.step_index += 1
    return StepResult(
        pre_state=pre_state.to_dict(),
        instruction=ir.to_dict(),
        post_state=state.clone().to_dict(),
        memory_effects=tuple(dict(item) for item in state.memory_effects),
        control_target=control_target,
        status="ok",
    )


def _row_from_step(event: InstructionEvent, result: StepResult) -> dict[str, object]:
    pre = cast(dict[str, object], result.pre_state)
    post = cast(dict[str, object], result.post_state)
    pre_registers = cast(dict[str, int], pre["registers"])
    post_registers = cast(dict[str, int], post["registers"])
    return {
        "address": event.address,
        "pre_rax": int(pre_registers["rax"]),
        "post_rax": int(post_registers["rax"]),
        "pre_rip": int(cast(int, pre["rip"])),
        "post_rip": int(cast(int, post["rip"])),
        "pre_rsp": int(pre_registers["rsp"]),
        "post_rsp": int(post_registers["rsp"]),
        "pre_eflags": int(cast(int, pre["eflags"])),
        "post_eflags": int(cast(int, post["eflags"])),
        "pre_registers": dict(pre_registers),
        "post_registers": dict(post_registers),
        "pre_state": pre,
        "post_state": post,
        "instruction": result.instruction,
        "memory_effects": [dict(item) for item in result.memory_effects],
        "control_target": result.control_target,
        "status": result.status,
    }


def _reference_trace(
    code: bytes, initial_rax: int
) -> tuple[list[dict[str, object]], str | None]:
    """Execute the normalized reference interpreter over a bounded trace."""

    if not 0 <= int(initial_rax) <= U64_MASK:
        raise ValueError("initial_rax must be an unsigned 64-bit integer")
    instructions = disassemble_bytes(code)
    by_address = {event.address: index for index, event in enumerate(instructions)}
    state = FormalState(
        rax=initial_rax,
        rsp=BASE_ADDRESS + STACK_OFFSET,
        rip=instructions[0].address,
        eflags=2,
        memory={},
    )
    rows: list[dict[str, object]] = []
    for _ in range(MAX_REFERENCE_STEPS):
        index = by_address.get(state.rip)
        if index is None:
            break
        event = instructions[index]
        next_address = (
            instructions[index + 1].address if index + 1 < len(instructions) else None
        )
        try:
            result = _step(event, state, next_address)
        except ValueError as exc:
            state.status = "unsupported"
            return rows, str(exc)
        rows.append(_row_from_step(event, result))
    else:
        return rows, "reference step bound exceeded"
    return rows, None


def _memory_events_by_instruction(
    events: tuple[dict[str, int | str], ...],
) -> dict[int, list[dict[str, int | str]]]:
    grouped: dict[int, list[dict[str, int | str]]] = {}
    for event in events:
        index = int(event["instruction_index"])
        grouped.setdefault(index, []).append(
            {
                key: value
                for key, value in event.items()
                if key in {"access", "address", "size", "value"}
            }
        )
    return grouped


def _compare(code: bytes, initial_rax: int) -> dict[str, Any]:
    """Compare every declared state component against bounded Unicorn."""

    unicorn = emulate_x86_64(
        code, initial_rax=initial_rax, max_instructions=MAX_REFERENCE_STEPS
    )
    reference_rows, unsupported = _reference_trace(code, initial_rax)
    unicorn_rows = [state.to_dict() for state in unicorn.instruction_states]
    unicorn_memory = _memory_events_by_instruction(unicorn.memory_events)
    unicorn_final_registers = {
        key: value
        for key, value in unicorn.registers.items()
        if key in GENERAL_REGISTER_NAMES
    }
    register_equal = all(
        reference.get("post_registers") == unicorn_row.get("post_registers")
        for reference, unicorn_row in zip(reference_rows, unicorn_rows)
    ) and (
        not reference_rows
        or reference_rows[-1]["post_registers"] == unicorn_final_registers
    )
    flags_equal = all(
        (int(cast(int, reference["post_eflags"])) & ARCHITECTURAL_FLAGS_MASK)
        == (int(cast(int, unicorn_row["post_eflags"])) & ARCHITECTURAL_FLAGS_MASK)
        for reference, unicorn_row in zip(reference_rows, unicorn_rows)
    )
    memory_equal = all(
        reference.get("memory_effects", []) == unicorn_memory.get(index, [])
        for index, reference in enumerate(reference_rows)
    ) and len(unicorn_memory) == sum(
        bool(row.get("memory_effects")) for row in reference_rows
    )
    checks: dict[str, object] = {
        "unsupported": unsupported,
        "step_count_equal": len(reference_rows) == len(unicorn_rows),
        "addresses_equal": [row["address"] for row in reference_rows]
        == [row["address"] for row in unicorn_rows],
        "register_file_equal": register_equal,
        "rip_and_control_targets_equal": all(
            int(cast(int, reference["post_rip"]))
            == int(cast(int, unicorn_row["post_rip"]))
            for reference, unicorn_row in zip(reference_rows, unicorn_rows)
        ),
        "modeled_flags_equal": flags_equal,
        "memory_effects_equal": memory_equal,
        "state_serialization_present": all(
            isinstance(row.get("pre_state"), dict)
            and isinstance(row.get("post_state"), dict)
            for row in reference_rows
        ),
    }
    checks["matched_at_every_step"] = bool(
        unsupported is None
        and all(bool(value) for key, value in checks.items() if key != "unsupported")
    )
    return {
        "initial_rax": int(initial_rax),
        "code_sha256": hashlib.sha256(code).hexdigest(),
        "reference_steps": reference_rows,
        "unicorn_steps": unicorn_rows,
        "unicorn_final": unicorn.to_dict(),
        "checks": checks,
    }


def formal_correspondence_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the source-bound bounded correspondence receipt."""

    rows: list[dict[str, Any]] = []
    interventions = (
        0,
        1,
        2,
        (1 << 31) - 1,
        1 << 31,
        (1 << 32) - 1,
        1 << 63,
        U64_MASK,
    )
    for sample_id, code in sorted(SAMPLE_PROGRAMS.items()):
        for initial_rax in interventions:
            row = _compare(code, initial_rax)
            row["sample_id"] = sample_id
            rows.append(row)
    formal_root = project_root if (project_root / "formal").is_dir() else PROJECT_ROOT
    formal = formal_verification_payload(formal_root)
    bridge_rows, bridge_error = _reference_trace(BRIDGE_CODE, BRIDGE_INITIAL_RAX)
    bridge_reference_value = (
        int(cast(dict[str, Any], bridge_rows[-1]["post_registers"])["rax"])
        if bridge_rows and bridge_error is None
        else None
    )
    bridge = dict(cast(dict[str, Any], formal["cross_language_bridge"]))
    bridge.update(
        {
            "python_reference_trace_final_rax": bridge_reference_value,
            "python_reference_trace_error": bridge_error,
            "trace_matches_lean_expected": bridge_reference_value
            == BRIDGE_EXPECTED_RAX,
        }
    )
    protocol = scientific_protocol_payload()
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("emulation.py"),
        Path(__file__).with_name("disassembly.py"),
    )
    input_hashes = dict(formal.get("source_hashes", {}))
    input_hashes.update(
        {
            str(path.relative_to(PROJECT_ROOT)): sha256_path(path)
            for path in source_paths
        }
    )
    versions = {
        **toolchain_versions(("capstone", "numpy", "unicorn")),
        "lean_toolchain": str(formal.get("toolchain", "unavailable")),
    }
    all_matched = all(row["checks"]["matched_at_every_step"] for row in rows)
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=input_hashes,
        toolchain_versions=versions,
        status="validated_bounded" if all_matched else "failed",
        limitations=(
            "The correspondence is bounded to transparent repository-owned fixtures and the declared user-mode integer subset.",
            "Unicorn comparison is computational evidence; it is not a kernel proof of complete x86-64 semantics.",
        ),
        unsupported_cases=(
            "SIMD, floating point, privileged/system instructions, unmodeled exceptions, and unbounded aliasing",
            "Instructions whose flag dependencies are outside the modeled definedness contract",
        ),
        underpowered_cases=(
            "No external compiler or architecture claim is promoted by this fixture receipt.",
        ),
    )
    return {
        **envelope,
        "artifact": FORMAL_CORRESPONDENCE_ARTIFACT,
        "scope": "declared transparent x86-64 fixture subset; not complete x86-64 semantics",
        "architecture": "x86_64",
        "supported_instruction_subset": list(SUPPORTED_X86_USER_MODE),
        "state": [
            *GENERAL_REGISTER_NAMES,
            "RIP",
            *MODELED_FLAGS,
            "flag_definedness",
            "bounded_byte_memory",
            "step_index",
            "status",
        ],
        "memory_bounds": {
            "lower": MEMORY_LOWER_BOUND,
            "upper_exclusive": MEMORY_UPPER_BOUND,
        },
        "stack_bounds": {
            "lower": BASE_ADDRESS + STACK_OFFSET - 4096,
            "upper_exclusive": BASE_ADDRESS + STACK_OFFSET + 4096,
        },
        "lean_prerequisite": {
            "status": formal.get("endpoint_status", formal.get("status")),
            "toolchain": formal.get("toolchain"),
            "theorems": formal.get("theorems"),
            "proof_boundary": "Lean prerequisite; Python/Unicorn rows remain computational cross-checks",
        },
        "cross_language_bridge": bridge,
        "fixture_count": len(SAMPLE_PROGRAMS),
        "intervention_count": len(interventions),
        "interventions": list(interventions),
        "step_rows": rows,
        "all_steps_matched": all_matched,
    }


def validate_formal_correspondence_payload(payload: object) -> tuple[bool, str]:
    """Validate the full correspondence receipt and every matched row."""

    if not isinstance(payload, dict):
        return False, "formal correspondence payload must be an object"
    if payload.get("schema_version") != FORMAL_CORRESPONDENCE_SCHEMA_VERSION:
        return False, "formal correspondence schema mismatch"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"formal correspondence receipt: {receipt_detail}"
    if payload.get("status") != "validated_bounded":
        return False, "formal correspondence is not validated_bounded"
    if payload.get("all_steps_matched") is not True:
        return False, "formal correspondence contains an unmatched step"
    bridge = payload.get("cross_language_bridge")
    if (
        not isinstance(bridge, dict)
        or bridge.get("trace_matches_lean_expected") is not True
    ):
        return False, "Python reference trace does not match the generated Lean bridge"
    if (
        not isinstance(payload.get("supported_instruction_subset"), list)
        or not payload["supported_instruction_subset"]
    ):
        return False, "formal correspondence supported subset is missing"
    if not set(GENERAL_REGISTER_NAMES).issubset(set(payload.get("state", []))):
        return False, "formal correspondence state omits an architectural register"
    rows = payload.get("step_rows")
    if not isinstance(rows, list) or not rows:
        return False, "formal correspondence step rows are missing"
    for row in rows:
        if not isinstance(row, dict):
            return False, "formal correspondence row is not an object"
        checks = row.get("checks")
        if (
            not isinstance(checks, dict)
            or checks.get("matched_at_every_step") is not True
        ):
            return False, "formal correspondence row is not matched"
        if checks.get("unsupported") is not None:
            return False, "formal correspondence row is unsupported"
        for key in ("reference_steps", "unicorn_steps", "unicorn_final"):
            if key not in row:
                return False, f"formal correspondence row missing {key}"
    if payload.get("fixture_count") != len(SAMPLE_PROGRAMS):
        return False, "formal correspondence fixture count drift"
    if payload.get("intervention_count") != len(payload.get("interventions", [])):
        return False, "formal correspondence intervention count drift"
    return True, "ok"


def write_formal_correspondence(project_root: Path = PROJECT_ROOT) -> Path:
    """Write and validate the deterministic correspondence receipt."""

    payload = formal_correspondence_payload(project_root)
    valid, detail = validate_formal_correspondence_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / FORMAL_CORRESPONDENCE_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "ARCHITECTURAL_FLAGS_MASK",
    "FormalState",
    "GENERAL_REGISTER_NAMES",
    "IRInstruction",
    "MachineState",
    "MODELED_FLAGS",
    "StepResult",
    "U32_MASK",
    "U64_MASK",
    "_memory_address",
    "_memory_width",
    "_read_memory",
    "_reference_trace",
    "_set_arithmetic_flags",
    "_step",
    "_write_memory",
    "formal_correspondence_payload",
    "normalize_instruction",
    "validate_formal_correspondence_payload",
    "write_formal_correspondence",
]
