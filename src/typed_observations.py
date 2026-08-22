"""Typed, provenance-preserving observations for the Phase A roadmap.

The original observation channel intentionally collapses an instruction into one
of six categories.  That channel remains useful as a compact reference model,
but it discards information that is already available from the real Capstone
decode and bounded Unicorn trace.  This module exposes those fields as a typed
record without introducing a second decoder or making a semantic claim about
the instruction.

The encoder is deliberately label-blind: it consumes only
``InstructionEvent`` records and, optionally, an ``EmulationTrace``.  Hand-
authored phase labels are not accepted as an argument and therefore cannot
enter the serialized observation record by accident.  Missing dynamic data is
represented by ``None`` and is handled explicitly by the factorized model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence, cast

from .disassembly import InstructionEvent
from .emulation import EmulationTrace

TYPED_OBSERVATION_SCHEMA_VERSION = "1.0.0"
MISSING_VALUE = "__MISSING__"
UNKNOWN_VALUE = "__UNKNOWN__"

# The field order is part of the machine-readable contract and is used by the
# factorized model, figures, validation, and manuscript variable generator.
TYPED_OBSERVATION_FIELDS = (
    "instruction_family",
    "operand_class",
    "register_read_set",
    "register_write_set",
    "flag_effect",
    "memory_addressing_mode",
    "branch_target_class",
    "dynamic_status",
)

STATIC_TYPED_OBSERVATION_FIELDS = tuple(
    field for field in TYPED_OBSERVATION_FIELDS if field != "dynamic_status"
)

TYPED_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "instruction_semantics": (
        "instruction_family",
        "operand_class",
        "flag_effect",
        "memory_addressing_mode",
    ),
    "register_context": ("register_read_set", "register_write_set"),
    "control_flow": ("branch_target_class",),
    "dynamic_context": ("dynamic_status",),
}

_FLAG_REGISTERS = {"eflags", "rflags"}
_STACK_REGISTERS = {"esp", "ebp", "rsp", "rbp"}


@dataclass(frozen=True)
class TypedObservation:
    """One decoded instruction represented as typed static/dynamic evidence."""

    index: int
    address: int
    mnemonic: str
    instruction_family: str
    operand_class: str
    register_read_set: tuple[str, ...]
    register_write_set: tuple[str, ...]
    flag_effect: str
    memory_addressing_mode: str
    branch_target_class: str
    dynamic_status: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe row without any phase-label field."""

        data = asdict(self)
        data["register_read_set"] = list(self.register_read_set)
        data["register_write_set"] = list(self.register_write_set)
        return data


def _instruction_family(event: InstructionEvent) -> str:
    """Assign a broad ISA-facing family from the decoded mnemonic."""

    mnemonic = event.mnemonic.lower()
    if mnemonic in {"mov", "movabs", "lea", "push", "pop"}:
        return "data_transfer"
    if mnemonic in {"add", "sub", "inc", "dec", "adc", "sbb"}:
        return "arithmetic"
    if mnemonic in {
        "shl",
        "shr",
        "sal",
        "sar",
        "rol",
        "ror",
        "and",
        "or",
        "xor",
    }:
        return "bitwise"
    if mnemonic in {"cmp", "test"}:
        return "comparison"
    if mnemonic.startswith("j") or mnemonic in {"call", "ret", "sysret"}:
        return "control_transfer"
    return "other"


def _operand_class(event: InstructionEvent) -> str:
    """Classify operand evidence without parsing arbitrary assembly text."""

    kinds: set[str] = set()
    if event.immediate_operands:
        kinds.add("immediate")
    if event.memory_operands:
        kinds.add("memory")
    if event.registers_read or event.registers_written:
        kinds.add("register")
    if not kinds:
        return "none"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _flag_effect(event: InstructionEvent) -> str:
    """Summarize Capstone's implicit flag register access."""

    reads_flags = bool(_FLAG_REGISTERS & set(event.registers_read))
    writes_flags = bool(_FLAG_REGISTERS & set(event.registers_written))
    if reads_flags and writes_flags:
        return "read_write"
    if reads_flags:
        return "read"
    if writes_flags:
        return "write"
    return "none"


def _memory_addressing_mode(event: InstructionEvent) -> str:
    """Classify decoded memory operands by addressing form.

    This is a bounded syntactic classification.  It does not infer aliases,
    stack frames, heap objects, or effective addresses beyond the explicit
    operand metadata emitted by Capstone.
    """

    if not event.memory_operands:
        return "none"
    modes: set[str] = set()
    for operand in event.memory_operands:
        base = str(operand.get("base", ""))
        index = str(operand.get("index", ""))
        displacement = int(cast(int, operand.get("displacement", 0)))
        if index:
            modes.add("indexed")
        elif base in _STACK_REGISTERS:
            modes.add("stack_relative")
        elif not base:
            modes.add("absolute")
        elif displacement:
            modes.add("base_plus_displacement")
        else:
            modes.add("register_indirect")
    return next(iter(modes)) if len(modes) == 1 else "mixed_memory"


def _direct_target(event: InstructionEvent) -> int | None:
    """Parse a direct target when Capstone exposes one in ``op_str``."""

    if not (
        event.mnemonic.lower().startswith("j") or event.mnemonic.lower() in {"call"}
    ):
        return None
    token = event.op_str.split(",", 1)[0].strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def _branch_target_class(
    event: InstructionEvent, instruction_addresses: set[int]
) -> str:
    """Classify direct branch targets relative to the decoded span."""

    mnemonic = event.mnemonic.lower()
    if not (mnemonic.startswith("j") or mnemonic in {"call", "ret", "sysret"}):
        return "none"
    if mnemonic in {"ret", "sysret"}:
        return "return"
    target = _direct_target(event)
    if target is None:
        return "indirect"
    return "internal" if target in instruction_addresses else "external"


def encode_typed_observations(
    instructions: Sequence[InstructionEvent] | Iterable[InstructionEvent],
    trace: EmulationTrace | None = None,
) -> list[TypedObservation]:
    """Encode decoded instructions into label-blind typed observations.

    Args:
        instructions: Real Capstone instruction events.
        trace: Optional bounded Unicorn trace.  When absent, ``dynamic_status``
            is ``None`` rather than being guessed from the static record.

    Raises:
        ValueError: If the instruction sequence is empty or has duplicate
            addresses.
    """

    events = tuple(instructions)
    if not events:
        raise ValueError("instructions must not be empty")
    addresses = [int(event.address) for event in events]
    if len(set(addresses)) != len(addresses):
        raise ValueError("instruction addresses must be unique")
    address_set = set(addresses)
    executed_addresses = set(trace.addresses) if trace is not None else set()
    observations: list[TypedObservation] = []
    for index, event in enumerate(events):
        observations.append(
            TypedObservation(
                index=index,
                address=int(event.address),
                mnemonic=event.mnemonic,
                instruction_family=_instruction_family(event),
                operand_class=_operand_class(event),
                register_read_set=tuple(sorted(set(event.registers_read))),
                register_write_set=tuple(sorted(set(event.registers_written))),
                flag_effect=_flag_effect(event),
                memory_addressing_mode=_memory_addressing_mode(event),
                branch_target_class=_branch_target_class(event, address_set),
                dynamic_status=(
                    None
                    if trace is None
                    else (
                        "executed"
                        if event.address in executed_addresses
                        else "not_executed"
                    )
                ),
            )
        )
    return observations


def typed_field_value(observation: TypedObservation, field: str) -> str:
    """Return a stable categorical token for a typed observation field."""

    if field not in TYPED_OBSERVATION_FIELDS:
        raise ValueError(f"unknown typed observation field: {field}")
    value = getattr(observation, field)
    if value is None:
        return MISSING_VALUE
    if isinstance(value, tuple):
        return "|".join(value) if value else "<none>"
    return str(value)


def validate_typed_observations(
    observations: Sequence[TypedObservation],
) -> tuple[bool, str]:
    """Validate structure and explicitly reject phase-label leakage."""

    if not observations:
        return False, "typed observation sequence is empty"
    for expected_index, observation in enumerate(observations):
        if observation.index != expected_index:
            return False, "typed observation indexes are not contiguous"
        if not observation.mnemonic:
            return False, "typed observation mnemonic is empty"
        if observation.dynamic_status not in {None, "executed", "not_executed"}:
            return False, "invalid dynamic status"
        row = observation.to_dict()
        if any(key in row for key in ("phase_label", "phase_labels", "truth")):
            return False, "phase-label field leaked into typed observation"
    return True, "ok"


def mask_typed_field(
    observations: Sequence[TypedObservation], field: str
) -> list[TypedObservation]:
    """Return observations with one field removed under the missing-value policy."""

    if field not in TYPED_OBSERVATION_FIELDS:
        raise ValueError(f"unknown typed observation field: {field}")
    return [
        replace(observation, **cast(Any, {field: None})) for observation in observations
    ]
