"""Stable JSON trace schema shared by assembly adapters and inference backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

from ..disassembly import BASE_ADDRESS, InstructionEvent, sample_bytes
from ..emulation import EmulationTrace
from ..features import AssemblyObservation, HIDDEN_STATE_LABELS, OBSERVATION_LABELS

TRACE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class PackageProvenance:
    """Package or command provenance for one evidence-producing adapter."""

    name: str
    adapter: str
    version: str
    optional: bool
    available: bool
    command: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe package provenance row."""

        return asdict(self)


@dataclass(frozen=True)
class BasicBlock:
    """Static basic-block placeholder for a decoded instruction span."""

    block_id: str
    start_address: int
    end_address: int
    instruction_indexes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe block row."""

        data = asdict(self)
        data["instruction_indexes"] = list(self.instruction_indexes)
        return data


@dataclass(frozen=True)
class CfgEdge:
    """Directed control-flow edge between static basic blocks."""

    source_block_id: str
    target_block_id: str
    edge_type: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe edge row."""

        return asdict(self)


@dataclass(frozen=True)
class TraceBundle:
    """Complete assembly evidence bundle for one sample and inference run."""

    schema_version: str
    sample_id: str
    architecture: str
    base_address: int
    binary_sha256: str
    static_instruction_events: tuple[InstructionEvent, ...]
    basic_blocks: tuple[BasicBlock, ...]
    cfg_edges: tuple[CfgEdge, ...]
    dynamic_register_trace: tuple[dict[str, Any], ...]
    memory_events: tuple[dict[str, Any], ...]
    symbolic_facts: tuple[dict[str, Any], ...]
    package_provenance: tuple[PackageProvenance, ...]
    observations: tuple[AssemblyObservation, ...]
    observation_labels: tuple[str, ...]
    hidden_state_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace bundle."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "architecture": self.architecture,
            "base_address": self.base_address,
            "binary_sha256": self.binary_sha256,
            "static_instruction_events": [
                event.to_dict() for event in self.static_instruction_events
            ],
            "basic_blocks": [block.to_dict() for block in self.basic_blocks],
            "cfg_edges": [edge.to_dict() for edge in self.cfg_edges],
            "dynamic_register_trace": list(self.dynamic_register_trace),
            "memory_events": list(self.memory_events),
            "symbolic_facts": list(self.symbolic_facts),
            "package_provenance": [
                provenance.to_dict() for provenance in self.package_provenance
            ],
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "observation_labels": list(self.observation_labels),
            "hidden_state_labels": list(self.hidden_state_labels),
        }


def binary_hash(sample_name: str) -> str:
    """Return the SHA-256 hash of the sample fixture bytes."""

    return hashlib.sha256(sample_bytes(sample_name)).hexdigest()


def _single_linear_block(
    instructions: tuple[InstructionEvent, ...],
) -> tuple[BasicBlock, ...]:
    if not instructions:
        return ()
    first = instructions[0]
    last = instructions[-1]
    return (
        BasicBlock(
            block_id="block_0000",
            start_address=first.address,
            end_address=last.address + last.size,
            instruction_indexes=tuple(range(len(instructions))),
        ),
    )


def _register_trace(trace: EmulationTrace) -> tuple[dict[str, Any], ...]:
    return (
        {
            "event": "initial",
            "address": trace.base_address,
            "registers": {"rax": trace.initial_rax},
        },
        {
            "event": "final",
            "address": trace.final_rip,
            "instruction_count": trace.instruction_count,
            "registers": dict(trace.registers),
        },
    )


def build_trace_bundle(
    sample_name: str,
    instructions: tuple[InstructionEvent, ...],
    trace: EmulationTrace,
    observations: tuple[AssemblyObservation, ...],
    package_provenance: tuple[PackageProvenance, ...],
) -> TraceBundle:
    """Build the stable trace bundle consumed by Python and Julia backends."""

    if len(instructions) != len(observations):
        raise ValueError(
            "instruction and observation streams must have the same length"
        )
    return TraceBundle(
        schema_version=TRACE_SCHEMA_VERSION,
        sample_id=sample_name,
        architecture="x86_64",
        base_address=BASE_ADDRESS,
        binary_sha256=binary_hash(sample_name),
        static_instruction_events=instructions,
        basic_blocks=_single_linear_block(instructions),
        cfg_edges=(),
        dynamic_register_trace=_register_trace(trace),
        memory_events=(),
        symbolic_facts=(),
        package_provenance=package_provenance,
        observations=observations,
        observation_labels=OBSERVATION_LABELS,
        hidden_state_labels=HIDDEN_STATE_LABELS,
    )


def trace_bundle_to_json(bundle: TraceBundle) -> dict[str, Any]:
    """Return a JSON-safe trace bundle payload."""

    return bundle.to_dict()
