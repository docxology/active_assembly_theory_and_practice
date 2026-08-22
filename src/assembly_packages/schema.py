"""Stable JSON trace schema shared by assembly adapters and inference backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

from ..disassembly import BASE_ADDRESS, InstructionEvent, sample_bytes
from ..emulation import EmulationTrace
from ..features import AssemblyObservation, HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from ..typed_observations import TypedObservation, encode_typed_observations

TRACE_SCHEMA_VERSION = "1.4.0"


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
    """Static basic-block record for a decoded instruction span."""

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
    typed_observations: tuple[TypedObservation, ...]
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
            "typed_observations": [
                observation.to_dict() for observation in self.typed_observations
            ],
            "observation_labels": list(self.observation_labels),
            "hidden_state_labels": list(self.hidden_state_labels),
        }


def binary_hash(sample_name: str) -> str:
    """Return the SHA-256 hash of the sample fixture bytes."""

    return hashlib.sha256(sample_bytes(sample_name)).hexdigest()


def _direct_target(operand_text: str) -> int | None:
    """Parse a direct Capstone branch target when it is statically visible."""

    token = operand_text.split(",", 1)[0].strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def build_static_cfg(
    instructions: tuple[InstructionEvent, ...],
) -> tuple[tuple[BasicBlock, ...], tuple[CfgEdge, ...]]:
    """Partition decoded instructions into bounded basic blocks and edges.

    This is intentionally a small direct-target CFG extractor.  It handles the
    branch forms present in the transparent fixtures and records unresolved or
    out-of-range targets as ``external_<address>`` edges rather than inventing
    a destination block.  It is therefore useful evidence about the fixture,
    not a claim of complete binary CFG recovery.
    """

    if not instructions:
        return (), ()
    address_to_index = {
        event.address: index for index, event in enumerate(instructions)
    }
    leaders = {0}
    for index, event in enumerate(instructions):
        mnemonic = event.mnemonic.lower()
        is_branch = mnemonic.startswith("j")
        is_terminator = is_branch or mnemonic in {"ret", "sysret"}
        if not is_terminator:
            continue
        target = _direct_target(event.op_str) if is_branch else None
        if target is not None and target in address_to_index:
            leaders.add(address_to_index[target])
        if index + 1 < len(instructions):
            leaders.add(index + 1)

    ordered_leaders = sorted(leaders)
    blocks: list[BasicBlock] = []
    block_by_instruction: dict[int, BasicBlock] = {}
    for block_number, start_index in enumerate(ordered_leaders):
        end_index = (
            ordered_leaders[block_number + 1]
            if block_number + 1 < len(ordered_leaders)
            else len(instructions)
        )
        block = BasicBlock(
            block_id=f"block_{block_number:04d}",
            start_address=instructions[start_index].address,
            end_address=instructions[end_index - 1].address
            + instructions[end_index - 1].size,
            instruction_indexes=tuple(range(start_index, end_index)),
        )
        blocks.append(block)
        for instruction_index in block.instruction_indexes:
            block_by_instruction[instruction_index] = block

    edges: list[CfgEdge] = []
    for block_number, block in enumerate(blocks):
        last_index = block.instruction_indexes[-1]
        last_instruction = instructions[last_index]
        mnemonic = last_instruction.mnemonic.lower()
        next_block = (
            blocks[block_number + 1] if block_number + 1 < len(blocks) else None
        )
        if mnemonic.startswith("j"):
            target = _direct_target(last_instruction.op_str)
            if target is not None and target in address_to_index:
                target_block = block_by_instruction[address_to_index[target]]
                target_id = target_block.block_id
            elif target is not None:
                target_id = f"external_0x{target:x}"
            else:
                target_id = "external_indirect"
            edge_type = "jump" if mnemonic == "jmp" else "taken"
            edges.append(CfgEdge(block.block_id, target_id, edge_type))
            if mnemonic != "jmp" and next_block is not None:
                edges.append(
                    CfgEdge(block.block_id, next_block.block_id, "fallthrough")
                )
        elif mnemonic not in {"ret", "sysret"} and next_block is not None:
            edges.append(CfgEdge(block.block_id, next_block.block_id, "fallthrough"))

    return tuple(blocks), tuple(edges)


def _register_trace(trace: EmulationTrace) -> tuple[dict[str, Any], ...]:
    if trace.instruction_states:
        first = trace.instruction_states[0]
        initial_registers = {
            "rax": first.pre_rax,
            "rip": first.pre_rip,
            "rsp": first.pre_rsp,
            "eflags": first.pre_eflags,
        }
    else:
        initial_registers = {"rax": trace.initial_rax, "rip": trace.base_address}
    rows: list[dict[str, Any]] = [
        {
            "event": "initial",
            "address": trace.base_address,
            "registers": initial_registers,
        }
    ]
    rows.extend(
        {
            "event": "instruction",
            "instruction_index": state.instruction_index,
            "address": state.address,
            "registers_before": {
                "rax": state.pre_rax,
                "rip": state.pre_rip,
                "rsp": state.pre_rsp,
                "eflags": state.pre_eflags,
            },
            "registers_after": {
                "rax": state.post_rax,
                "rip": state.post_rip,
                "rsp": state.post_rsp,
                "eflags": state.post_eflags,
            },
        }
        for state in trace.instruction_states
    )
    rows.append(
        {
            "event": "final",
            "address": trace.final_rip,
            "instruction_count": trace.instruction_count,
            "registers": dict(trace.registers),
        }
    )
    return tuple(rows)


def build_trace_bundle(
    sample_name: str,
    instructions: tuple[InstructionEvent, ...],
    trace: EmulationTrace,
    observations: tuple[AssemblyObservation, ...],
    package_provenance: tuple[PackageProvenance, ...],
    symbolic_facts: tuple[dict[str, Any], ...] = (),
    typed_observations: tuple[TypedObservation, ...] = (),
) -> TraceBundle:
    """Build the stable trace bundle consumed by Python and Julia backends."""

    if len(instructions) != len(observations):
        raise ValueError(
            "instruction and observation streams must have the same length"
        )
    if not typed_observations:
        typed_observations = tuple(encode_typed_observations(instructions, trace))
    if len(instructions) != len(typed_observations):
        raise ValueError(
            "instruction and typed-observation streams must have the same length"
        )
    basic_blocks, cfg_edges = build_static_cfg(instructions)
    return TraceBundle(
        schema_version=TRACE_SCHEMA_VERSION,
        sample_id=sample_name,
        architecture="x86_64",
        base_address=BASE_ADDRESS,
        binary_sha256=binary_hash(sample_name),
        static_instruction_events=instructions,
        basic_blocks=basic_blocks,
        cfg_edges=cfg_edges,
        dynamic_register_trace=_register_trace(trace),
        memory_events=tuple(trace.memory_events),
        symbolic_facts=symbolic_facts,
        package_provenance=package_provenance,
        observations=observations,
        typed_observations=typed_observations,
        observation_labels=OBSERVATION_LABELS,
        hidden_state_labels=HIDDEN_STATE_LABELS,
    )


def trace_bundle_to_json(bundle: TraceBundle) -> dict[str, Any]:
    """Return a JSON-safe trace bundle payload."""

    return bundle.to_dict()
