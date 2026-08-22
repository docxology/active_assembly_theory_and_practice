"""Bounded, auditable transfer functions for the project's x86-64 fixtures.

The project deliberately does not pretend to be a complete x86 semantics
engine.  This module implements a small closed-world subset that is sufficient
for the transparent fixtures in :mod:`src.corpus`: immediate moves and
arithmetic/bitwise updates on ``EAX``/``RAX``, bounded stack-relative memory
stores and loads, shifts, comparisons, and the conditional-branch decision
used by the guard fixtures.  Every unsupported instruction is reported
explicitly rather than silently receiving a guessed meaning.

The symbolic expression carried alongside the concrete value is intentionally
simple.  It is a readable transfer-function trace, not SMT solving.  The final
concrete value is checked against the independent Unicorn trace, which makes
the semantic layer useful as an auditable explanation without confusing it
with a complete reverse-engineering or binary-lifting backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .disassembly import InstructionEvent
from .emulation import EmulationTrace, InstructionState

SEMANTICS_SCHEMA_VERSION = "1.1"
_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1
_EFLAGS_MASKS = {
    "carry": 1 << 0,
    "zero": 1 << 6,
    "sign": 1 << 7,
    "overflow": 1 << 11,
}


@dataclass(frozen=True)
class SemanticEvent:
    """One bounded transfer-function application for a decoded instruction."""

    index: int
    address: int
    mnemonic: str
    operands: str
    executed: bool
    supported: bool
    operation: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    pre_rax: int
    post_rax: int
    symbolic_expression: str
    flags: dict[str, bool | None]
    unicorn_pre_rax: int | None
    unicorn_post_rax: int | None
    unicorn_post_eflags: int | None
    rax_pre_matches: bool | None
    rax_post_matches: bool | None
    modeled_flag_comparisons: int
    modeled_flag_agreements: int
    modeled_flag_mismatches: int
    modeled_flags_match: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe semantic event."""

        data = asdict(self)
        data["reads"] = list(self.reads)
        data["writes"] = list(self.writes)
        data["flags"] = dict(self.flags)
        return data


@dataclass(frozen=True)
class SymbolicExecutionReport:
    """Result of bounded transfer-function execution and Unicorn comparison."""

    schema_version: str
    initial_rax: int
    final_rax: int
    unicorn_final_rax: int
    final_rax_matches: bool
    executed_addresses: tuple[int, ...]
    events: tuple[SemanticEvent, ...]
    supported_instruction_count: int
    unsupported_instruction_count: int
    stepwise_rax_agreement_count: int
    stepwise_rax_comparable_count: int
    stepwise_rax_mismatch_count: int
    stepwise_rax_matches: bool
    modeled_flag_comparisons: int
    modeled_flag_agreements: int
    modeled_flag_mismatches: int
    modeled_flags_match: bool
    scope: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe semantic report."""

        data = asdict(self)
        data["executed_addresses"] = [int(value) for value in self.executed_addresses]
        data["events"] = [event.to_dict() for event in self.events]
        data["scope"] = str(self.scope)
        return data


def _operands(operand_text: str) -> list[str]:
    """Split Capstone's simple comma-delimited operand spelling."""

    return [part.strip().lower() for part in operand_text.split(",") if part.strip()]


def _parse_immediate(operand: str) -> int | None:
    """Parse a decimal/hex immediate while rejecting registers and memory."""

    candidate = operand.strip().lower()
    if not candidate or any(token in candidate for token in ("[", "]", "ptr")):
        return None
    try:
        return int(candidate, 0)
    except ValueError:
        return None


def _format_immediate(value: int) -> str:
    """Format an immediate deterministically for symbolic expressions."""

    return str(value) if value < 10 else hex(value)


def _width_bits(register: str) -> int:
    """Return the architectural width for a supported accumulator register."""

    if register == "eax":
        return 32
    if register == "rax":
        return 64
    raise ValueError(f"unsupported accumulator register: {register}")


def _width_mask(width_bits: int) -> int:
    """Return the unsigned mask for a 32- or 64-bit operation."""

    if width_bits not in {32, 64}:
        raise ValueError(f"unsupported operation width: {width_bits}")
    return (1 << width_bits) - 1


def _sign_bit(width_bits: int) -> int:
    """Return the sign-bit mask for a supported operation width."""

    return 1 << (width_bits - 1)


def _signed(value: int, width_bits: int) -> int:
    """Interpret a masked integer as a signed value at the requested width."""

    mask = _width_mask(width_bits)
    masked = value & mask
    sign = _sign_bit(width_bits)
    return masked - (1 << width_bits) if masked & sign else masked


def _base_flags(
    result: int,
    width_bits: int,
    *,
    overflow: bool | None,
    carry: bool | None,
    less_signed: bool | None,
) -> dict[str, bool | None]:
    """Build the branch-relevant flags for a masked arithmetic result."""

    mask = _width_mask(width_bits)
    return {
        "zero": (result & mask) == 0,
        "sign": bool(result & _sign_bit(width_bits)),
        "overflow": overflow,
        "carry": carry,
        "less_signed": less_signed,
    }


def _arithmetic_flags(
    left: int,
    right: int,
    result: int,
    width_bits: int,
    mnemonic: str,
) -> dict[str, bool | None]:
    """Return x86-style flags for supported add/sub/bitwise updates."""

    mask = _width_mask(width_bits)
    sign = _sign_bit(width_bits)
    left_masked = left & mask
    right_masked = right & mask
    result_masked = result & mask
    if mnemonic == "add":
        overflow = bool(
            (~(left_masked ^ right_masked) & (left_masked ^ result_masked) & sign)
        )
        carry = left_masked + right_masked > mask
        less_signed = None
    elif mnemonic == "sub":
        overflow = bool(
            (left_masked ^ right_masked) & (left_masked ^ result_masked) & sign
        )
        carry = left_masked < right_masked
        less_signed = bool(bool(result_masked & sign) != overflow)
    else:
        # AND/OR/XOR clear CF and OF on x86; signed ordering is not defined by
        # these operations, so a later signed branch must not reuse it.
        overflow = False
        carry = False
        less_signed = None
    return _base_flags(
        result_masked,
        width_bits,
        overflow=overflow,
        carry=carry,
        less_signed=less_signed,
    )


def _subtraction_flags(
    left: int, right: int, width_bits: int
) -> dict[str, bool | None]:
    """Return flags for a width-correct ``cmp left, right`` operation."""

    result = (left - right) & _width_mask(width_bits)
    return _arithmetic_flags(left, right, result, width_bits, "sub")


def _test_flags(value: int, mask: int, width_bits: int) -> dict[str, bool | None]:
    """Return the flags defined by a width-correct ``test`` operation."""

    result = (value & mask) & _width_mask(width_bits)
    return _base_flags(
        result,
        width_bits,
        overflow=False,
        carry=False,
        less_signed=None,
    )


def _shift_flags(
    before: int,
    result: int,
    shift: int,
    width_bits: int,
    mnemonic: str,
) -> dict[str, bool | None]:
    """Return the defined subset of flags after a logical/arithmetic shift."""

    mask = _width_mask(width_bits)
    if shift == 0:
        # x86 leaves all flags unchanged for a zero effective shift count. The
        # caller preserves the previous flag state in this case.
        raise ValueError("zero-count shifts preserve the previous flags")
    if mnemonic in {"shl", "sal"}:
        carry = (
            bool((before & mask) & (1 << (width_bits - shift)))
            if shift <= width_bits
            else False
        )
        overflow = (
            bool(bool(result & _sign_bit(width_bits)) != carry) if shift == 1 else None
        )
    else:
        carry = bool(before & (1 << (shift - 1))) if shift <= width_bits else False
        if mnemonic == "shr" and shift == 1:
            overflow = bool(before & _sign_bit(width_bits))
        elif mnemonic == "sar" and shift == 1:
            # Intel defines OF=0 for a one-bit arithmetic right shift.
            overflow = False
        else:
            overflow = None
    return _base_flags(
        result,
        width_bits,
        overflow=overflow,
        carry=carry,
        less_signed=None,
    )


def _branch_taken(mnemonic: str, flags: dict[str, bool | None]) -> bool | None:
    """Evaluate the supported conditional branches from the latest flags."""

    if mnemonic in {"jl", "jnge"}:
        value = flags.get("less_signed")
        return None if value is None else bool(value)
    if mnemonic in {"jge", "jnl"}:
        value = flags.get("less_signed")
        return None if value is None else not bool(value)
    if mnemonic in {"je", "jz"}:
        value = flags.get("zero")
        return None if value is None else bool(value)
    if mnemonic in {"jne", "jnz"}:
        value = flags.get("zero")
        return None if value is None else not bool(value)
    if mnemonic in {"jg", "jnle"}:
        zero = flags.get("zero")
        less_signed = flags.get("less_signed")
        return (
            None
            if zero is None or less_signed is None
            else not bool(zero) and not bool(less_signed)
        )
    if mnemonic in {"jle", "jng"}:
        zero = flags.get("zero")
        less_signed = flags.get("less_signed")
        return (
            None
            if zero is None or less_signed is None
            else bool(zero) or bool(less_signed)
        )
    if mnemonic in {"js"}:
        value = flags.get("sign")
        return None if value is None else bool(value)
    if mnemonic in {"jns"}:
        value = flags.get("sign")
        return None if value is None else not bool(value)
    if mnemonic in {"jb", "jnae", "jc"}:
        value = flags.get("carry")
        return None if value is None else bool(value)
    if mnemonic in {"jae", "jnb", "jnc"}:
        value = flags.get("carry")
        return None if value is None else not bool(value)
    if mnemonic in {"ja", "jnbe"}:
        zero = flags.get("zero")
        carry = flags.get("carry")
        return (
            None
            if zero is None or carry is None
            else not bool(zero) and not bool(carry)
        )
    if mnemonic in {"jbe", "jna"}:
        zero = flags.get("zero")
        carry = flags.get("carry")
        return None if zero is None or carry is None else bool(zero) or bool(carry)
    if mnemonic in {"jo"}:
        value = flags.get("overflow")
        return None if value is None else bool(value)
    if mnemonic in {"jno"}:
        value = flags.get("overflow")
        return None if value is None else not bool(value)
    return None


def _write_register(value: int, register: str) -> int:
    """Apply x86-64 zero-extension for EAX and masking for RAX."""

    return (value & _U32_MASK) if register == "eax" else (value & _U64_MASK)


def _compare_modeled_flags(
    flags: dict[str, bool | None], eflags: int | None
) -> tuple[int, int, int, bool | None]:
    """Compare defined transfer-function flags with post-step EFLAGS.

    Only flags that the bounded interpreter explicitly defines are compared.
    Undefined or preserved flags remain outside the numerator and denominator;
    this avoids turning a deliberately partial semantics implementation into a
    false complete-ISA claim.
    """

    if eflags is None:
        return 0, 0, 0, None
    comparisons = agreements = mismatches = 0
    for name, mask in _EFLAGS_MASKS.items():
        expected = flags.get(name)
        if expected is None:
            continue
        comparisons += 1
        if bool(eflags & mask) == bool(expected):
            agreements += 1
        else:
            mismatches += 1
    return (
        comparisons,
        agreements,
        mismatches,
        None if comparisons == 0 else mismatches == 0,
    )


def symbolic_execute(
    instructions: Iterable[InstructionEvent],
    trace: EmulationTrace,
    *,
    initial_rax: int | None = None,
) -> SymbolicExecutionReport:
    """Execute the bounded transfer-function subset over a Unicorn path.

    Static instructions whose address is absent from ``trace.addresses`` are
    retained in the report but do not mutate state.  This preserves the
    distinction between decoded code and the dynamically observed path.  The
    function is intentionally fail-closed for the semantics claim: unsupported
    mnemonics are counted and described instead of being assigned an invented
    transfer function.
    """

    events = tuple(instructions)
    if not events:
        raise ValueError("instructions must not be empty")
    start_rax = trace.initial_rax if initial_rax is None else int(initial_rax)
    if start_rax < 0 or start_rax > _U64_MASK:
        raise ValueError("initial_rax must be an unsigned 64-bit integer")

    executed_addresses = tuple(int(address) for address in trace.addresses)
    executed_set = set(executed_addresses)
    trace_states_by_address: dict[int, list[InstructionState]] = {}
    for state in trace.instruction_states:
        trace_states_by_address.setdefault(int(state.address), []).append(state)
    rax = start_rax
    expression = hex(start_rax)
    flags: dict[str, bool | None] = {
        "zero": None,
        "sign": None,
        "overflow": None,
        "carry": None,
        "less_signed": None,
        "branch_taken": None,
    }
    semantic_events: list[SemanticEvent] = []
    # Memory is deliberately bounded to addresses observed by Unicorn hooks.
    # The interpreter does not invent an initial heap or perform alias
    # analysis; a load is supported only after a matching bounded store has
    # populated the same observed address.
    memory_values: dict[int, int] = {}
    memory_events_by_instruction: dict[int, list[dict[str, int | str]]] = {}
    for memory_event in trace.memory_events:
        instruction_address = int(memory_event["instruction_address"])
        memory_events_by_instruction.setdefault(instruction_address, []).append(
            dict(memory_event)
        )

    for index, event in enumerate(events):
        mnemonic = event.mnemonic.lower()
        operands = _operands(event.op_str)
        executed = event.address in executed_set
        pre_rax = rax
        operation = "unexecuted" if not executed else "unsupported"
        supported = not executed
        post_rax = rax
        event_flags = dict(flags)
        unicorn_pre_rax: int | None = None
        unicorn_post_rax: int | None = None
        unicorn_post_eflags: int | None = None
        rax_pre_matches: bool | None = None
        rax_post_matches: bool | None = None
        modeled_flag_comparisons = 0
        modeled_flag_agreements = 0
        modeled_flag_mismatches = 0
        modeled_flags_match: bool | None = None

        matching_states = trace_states_by_address.get(int(event.address), [])
        trace_state = matching_states[0] if len(matching_states) == 1 else None
        if executed and trace_state is not None:
            unicorn_pre_rax = int(trace_state.pre_rax)

        if executed:
            supported = False
            if mnemonic in {"mov", "movabs"} and len(operands) == 2:
                destination, source = operands
                immediate = _parse_immediate(source)
                bounded_memory_events = memory_events_by_instruction.get(
                    int(event.address), []
                )
                memory_writes = [
                    memory_event
                    for memory_event in bounded_memory_events
                    if memory_event.get("access") == "write"
                ]
                memory_reads = [
                    memory_event
                    for memory_event in bounded_memory_events
                    if memory_event.get("access") == "read"
                ]
                if "[" in destination and immediate is not None and memory_writes:
                    memory_event = memory_writes[0]
                    address = int(memory_event["address"])
                    observed_value = int(memory_event["value"])
                    width_bits = int(memory_event["size"]) * 8
                    expected_value = (
                        immediate & ((1 << width_bits) - 1)
                        if 0 < width_bits <= 64
                        else None
                    )
                    if expected_value is not None and observed_value == expected_value:
                        memory_values[address] = observed_value
                        operation = f"M[0x{address:x}] := immediate"
                        supported = True
                elif destination in {"eax", "rax"} and "[" in source and memory_reads:
                    memory_event = memory_reads[0]
                    address = int(memory_event["address"])
                    observed_value = int(memory_event["value"])
                    stored_value = memory_values.get(address)
                    if stored_value is not None and stored_value == observed_value:
                        post_rax = _write_register(observed_value, destination)
                        expression = f"M[0x{address:x}]"
                        operation = f"{destination} := M[0x{address:x}]"
                        supported = True
                elif destination in {"eax", "rax"} and immediate is not None:
                    post_rax = _write_register(immediate, operands[0])
                    expression = _format_immediate(immediate)
                    operation = f"{operands[0]} := immediate"
                    supported = True
            elif mnemonic in {"add", "sub", "and", "or", "xor"} and len(operands) == 2:
                immediate = _parse_immediate(operands[1])
                if operands[0] in {"eax", "rax"} and immediate is not None:
                    width_bits = _width_bits(operands[0])
                    width_mask = _width_mask(width_bits)
                    left = pre_rax & width_mask
                    if mnemonic == "add":
                        result = left + immediate
                    elif mnemonic == "sub":
                        result = left - immediate
                    elif mnemonic == "and":
                        result = left & immediate
                    elif mnemonic == "or":
                        result = left | immediate
                    else:
                        result = left ^ immediate
                    post_rax = _write_register(result, operands[0])
                    expression = (
                        f"({expression} {mnemonic} {_format_immediate(immediate)})"
                    )
                    operation = f"{operands[0]} := {mnemonic} immediate"
                    flags = _arithmetic_flags(
                        left, immediate, result, width_bits, mnemonic
                    )
                    flags["branch_taken"] = None
                    event_flags = dict(flags)
                    supported = True
            elif mnemonic in {"shl", "sal", "shr", "sar"} and len(operands) == 2:
                immediate = _parse_immediate(operands[1])
                if operands[0] in {"eax", "rax"} and immediate is not None:
                    width_bits = _width_bits(operands[0])
                    width_mask = _width_mask(width_bits)
                    left = pre_rax & width_mask
                    count_mask = 0x1F if width_bits == 32 else 0x3F
                    shift = max(0, immediate) & count_mask
                    if shift == 0:
                        result = left
                    elif mnemonic in {"shl", "sal"}:
                        result = left << shift
                    elif mnemonic == "shr":
                        result = left >> shift
                    else:
                        result = _signed(left, width_bits) >> shift
                    post_rax = _write_register(result, operands[0])
                    expression = (
                        f"({expression} {mnemonic} {_format_immediate(immediate)})"
                    )
                    operation = f"{operands[0]} := {mnemonic} immediate"
                    if shift:
                        flags = _shift_flags(left, result, shift, width_bits, mnemonic)
                        flags["branch_taken"] = None
                        event_flags = dict(flags)
                    supported = True
            elif mnemonic == "cmp" and len(operands) == 2:
                immediate = _parse_immediate(operands[1])
                if operands[0] in {"eax", "rax"} and immediate is not None:
                    width_bits = _width_bits(operands[0])
                    flags = _subtraction_flags(pre_rax, immediate, width_bits)
                    flags["branch_taken"] = None
                    event_flags = dict(flags)
                    operation = "compare; update flags"
                    supported = True
            elif mnemonic == "test" and len(operands) == 2:
                immediate = _parse_immediate(operands[1])
                register_mask = (
                    pre_rax
                    if operands[1] == operands[0] and operands[0] in {"eax", "rax"}
                    else None
                )
                if operands[0] in {"eax", "rax"} and (
                    immediate is not None or register_mask is not None
                ):
                    width_bits = _width_bits(operands[0])
                    mask_value = immediate if immediate is not None else register_mask
                    assert mask_value is not None
                    flags = _test_flags(pre_rax, mask_value, width_bits)
                    flags["branch_taken"] = None
                    event_flags = dict(flags)
                    operation = "test; update flags"
                    supported = True
            elif mnemonic.startswith("j"):
                taken = _branch_taken(mnemonic, flags)
                event_flags = dict(flags)
                event_flags["branch_taken"] = taken
                flags = event_flags
                operation = "conditional branch"
                supported = taken is not None or mnemonic in {"jmp", "ret"}
            elif mnemonic in {"nop", "ret"}:
                operation = "control transfer" if mnemonic == "ret" else "no-op"
                supported = True

        if not supported and executed:
            operation = f"unsupported: {mnemonic}"

        if executed and trace_state is not None:
            unicorn_post_rax = int(trace_state.post_rax)
            unicorn_post_eflags = int(trace_state.post_eflags)
            rax_pre_matches = pre_rax == unicorn_pre_rax
            rax_post_matches = post_rax == unicorn_post_rax
            (
                modeled_flag_comparisons,
                modeled_flag_agreements,
                modeled_flag_mismatches,
                modeled_flags_match,
            ) = _compare_modeled_flags(event_flags, unicorn_post_eflags)

        reads = tuple(event.registers_read)
        writes = tuple(event.registers_written)
        semantic_events.append(
            SemanticEvent(
                index=index,
                address=int(event.address),
                mnemonic=event.mnemonic,
                operands=event.op_str,
                executed=executed,
                supported=supported,
                operation=operation,
                reads=reads,
                writes=writes,
                pre_rax=int(pre_rax),
                post_rax=int(post_rax),
                symbolic_expression=expression,
                flags=event_flags,
                unicorn_pre_rax=unicorn_pre_rax,
                unicorn_post_rax=unicorn_post_rax,
                unicorn_post_eflags=unicorn_post_eflags,
                rax_pre_matches=rax_pre_matches,
                rax_post_matches=rax_post_matches,
                modeled_flag_comparisons=modeled_flag_comparisons,
                modeled_flag_agreements=modeled_flag_agreements,
                modeled_flag_mismatches=modeled_flag_mismatches,
                modeled_flags_match=modeled_flags_match,
            )
        )
        rax = int(post_rax)

    supported_count = sum(
        event.executed and event.supported for event in semantic_events
    )
    unsupported_count = sum(
        event.executed and not event.supported for event in semantic_events
    )
    rax_comparable_count = sum(
        event.rax_post_matches is not None for event in semantic_events
    )
    rax_agreement_count = sum(
        event.rax_post_matches is True for event in semantic_events
    )
    rax_mismatch_count = sum(
        event.rax_post_matches is False for event in semantic_events
    )
    flag_comparisons = sum(event.modeled_flag_comparisons for event in semantic_events)
    flag_agreements = sum(event.modeled_flag_agreements for event in semantic_events)
    flag_mismatches = sum(event.modeled_flag_mismatches for event in semantic_events)
    final_rax = int(rax)
    return SymbolicExecutionReport(
        schema_version=SEMANTICS_SCHEMA_VERSION,
        initial_rax=int(start_rax),
        final_rax=final_rax,
        unicorn_final_rax=int(trace.final_rax),
        final_rax_matches=final_rax == int(trace.final_rax),
        executed_addresses=executed_addresses,
        events=tuple(semantic_events),
        supported_instruction_count=int(supported_count),
        unsupported_instruction_count=int(unsupported_count),
        stepwise_rax_agreement_count=int(rax_agreement_count),
        stepwise_rax_comparable_count=int(rax_comparable_count),
        stepwise_rax_mismatch_count=int(rax_mismatch_count),
        stepwise_rax_matches=(rax_comparable_count > 0 and rax_mismatch_count == 0),
        modeled_flag_comparisons=int(flag_comparisons),
        modeled_flag_agreements=int(flag_agreements),
        modeled_flag_mismatches=int(flag_mismatches),
        modeled_flags_match=(flag_comparisons > 0 and flag_mismatches == 0),
        scope=(
            "bounded x86-64 transfer functions for immediate EAX/RAX updates, "
            "bounded stack-relative memory, shifts, comparisons, and supported "
            "conditional branches with stepwise RAX and modeled-EFLAGS checks"
        ),
    )


def validate_symbolic_report(report: SymbolicExecutionReport) -> tuple[bool, str]:
    """Return an auditable contract result for a semantic execution report."""

    if not report.events:
        return False, "no semantic events"
    if report.unsupported_instruction_count:
        return False, f"unsupported_executed={report.unsupported_instruction_count}"
    if not report.final_rax_matches:
        return (
            False,
            f"symbolic_final_rax={report.final_rax} unicorn_final_rax={report.unicorn_final_rax}",
        )
    executed_count = sum(event.executed for event in report.events)
    if report.stepwise_rax_comparable_count != executed_count:
        return (
            False,
            "stepwise RAX comparison is incomplete: "
            f"comparable={report.stepwise_rax_comparable_count} executed={executed_count}",
        )
    if not report.stepwise_rax_matches:
        return (
            False,
            f"stepwise RAX mismatch: mismatches={report.stepwise_rax_mismatch_count}",
        )
    if not report.modeled_flags_match:
        return (
            False,
            f"modeled EFLAGS mismatch: mismatches={report.modeled_flag_mismatches}",
        )
    return True, (
        f"events={len(report.events)} executed={len(report.executed_addresses)} "
        f"stepwise_rax={report.stepwise_rax_agreement_count}/"
        f"{report.stepwise_rax_comparable_count} "
        f"modeled_flags={report.modeled_flag_agreements}/"
        f"{report.modeled_flag_comparisons} final_rax={report.final_rax}"
    )
