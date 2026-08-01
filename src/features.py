"""Categorical observation features derived from instruction and trace events."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Iterable

from .disassembly import InstructionEvent
from .emulation import EmulationTrace

OBSERVATION_LABELS = (
    "load_immediate",
    "arithmetic_update",
    "bitwise_transform",
    "comparison",
    "control_transfer",
    "other",
)

HIDDEN_STATE_LABELS = (
    "setup",
    "accumulator_update",
    "representation_transform",
    "guard_check",
    "exit",
)

IMMEDIATE_RE = re.compile(r"(^|[\s,])(?:0x[0-9a-fA-F]+|\d+)(?:$|[\s,])")


@dataclass(frozen=True)
class AssemblyObservation:
    """Categorical observation used by the Active Inference bridge."""

    index: int
    address: int
    mnemonic: str
    operand_text: str
    category: str
    observation_id: int
    executed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return asdict(self)


def categorize_instruction(event: InstructionEvent) -> str:
    """Map a decoded instruction to a small observation vocabulary."""

    mnemonic = event.mnemonic.lower()
    if mnemonic in {"mov", "movabs", "lea", "push", "pop"}:
        if IMMEDIATE_RE.search(event.op_str):
            return "load_immediate"
        return "other"
    if mnemonic in {"add", "sub", "inc", "dec", "adc", "sbb"}:
        return "arithmetic_update"
    if mnemonic in {"shl", "shr", "sal", "sar", "rol", "ror", "and", "or", "xor"}:
        return "bitwise_transform"
    if mnemonic in {"cmp", "test"}:
        return "comparison"
    if mnemonic.startswith("j") or mnemonic in {"call", "ret"}:
        return "control_transfer"
    return "other"


def encode_observations(
    instructions: Iterable[InstructionEvent],
    trace: EmulationTrace,
) -> list[AssemblyObservation]:
    """Encode decoded instructions as categorical observations."""

    executed_addresses = set(trace.addresses)
    observations: list[AssemblyObservation] = []
    for index, event in enumerate(instructions):
        category = categorize_instruction(event)
        observations.append(
            AssemblyObservation(
                index=index,
                address=event.address,
                mnemonic=event.mnemonic,
                operand_text=event.op_str,
                category=category,
                observation_id=OBSERVATION_LABELS.index(category),
                executed=event.address in executed_addresses,
            )
        )
    if not observations:
        raise ValueError(
            "at least one instruction is required for observation encoding"
        )
    return observations


def observation_counts(observations: Iterable[AssemblyObservation]) -> dict[str, int]:
    """Count observations by category."""

    counts = Counter(observation.category for observation in observations)
    return {label: counts.get(label, 0) for label in OBSERVATION_LABELS}
