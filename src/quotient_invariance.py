"""Bounded operational-equivalence witnesses for the active P3 program.

This is the first executable prerequisite for quotient-state identifiability.
It defines a deliberately narrow output relation (the final 64-bit ``RAX``
value) and evaluates transparent transformation pairs under a finite family of
initial-state interventions using the real Capstone and Unicorn paths.

The artifact is a *protocol and witness set*, not a semantic-learning result.
It does not infer labels, claim complete x86 equivalence, or measure posterior
invariance.  Equivalent pairs and near-miss pairs are retained together so a
future model cannot earn an invariance claim by becoming blind to every
difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, cast

from .disassembly import disassemble_bytes
from .emulation import emulate_x86_64
from .features import encode_observations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUOTIENT_PROTOCOL_SCHEMA_VERSION = "1.0.0"
QUOTIENT_PROTOCOL_ARTIFACT = "output/data/quotient_invariance_protocol.json"
INTERVENTION_INITIAL_RAX: tuple[int, ...] = (
    0,
    1,
    (1 << 32) - 1,
    (1 << 64) - 1,
)
U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class TransformationPair:
    """One transparent reference/variant pair with a declared output relation."""

    pair_id: str
    motif: str
    expected_equivalent: bool
    reference_code: bytes
    variant_code: bytes
    relation: str

    def to_dict(self) -> dict[str, object]:
        """Return the pair specification without executable byte objects."""

        data = asdict(self)
        data["reference_code_hex"] = self.reference_code.hex()
        data["variant_code_hex"] = self.variant_code.hex()
        del data["reference_code"]
        del data["variant_code"]
        return data


def _pair_specs() -> tuple[TransformationPair, ...]:
    """Return the frozen equivalent and near-miss fixture specifications."""

    return (
        TransformationPair(
            pair_id="add_vs_two_inc",
            motif="constant_accumulator_update",
            expected_equivalent=True,
            reference_code=bytes.fromhex("b8 01 00 00 00 83 c0 02"),
            variant_code=bytes.fromhex("b8 01 00 00 00 ff c0 ff c0"),
            relation="final_RAX_equal_for_each_declared_initial_state",
        ),
        TransformationPair(
            pair_id="neutral_nop_insertion",
            motif="neutral_representation_change",
            expected_equivalent=True,
            reference_code=bytes.fromhex("b8 07 00 00 00 83 c0 05"),
            variant_code=bytes.fromhex("b8 07 00 00 00 90 83 c0 05"),
            relation="final_RAX_equal_for_each_declared_initial_state",
        ),
        TransformationPair(
            pair_id="temporary_register_rewrite",
            motif="temporary_register_introduction",
            expected_equivalent=True,
            reference_code=bytes.fromhex("b8 03 00 00 00 83 c0 04"),
            variant_code=bytes.fromhex("b9 03 00 00 00 83 c1 04 89 c8"),
            relation="final_RAX_equal_for_each_declared_initial_state",
        ),
        TransformationPair(
            pair_id="near_miss_substitution",
            motif="changed_arithmetic_effect",
            expected_equivalent=False,
            reference_code=bytes.fromhex("b8 05 00 00 00 83 c0 03"),
            variant_code=bytes.fromhex("b8 05 00 00 00 83 e8 03"),
            relation="final_RAX_must_differ_for_at_least_one_declared_initial_state",
        ),
    )


def _code_fingerprint(code: bytes) -> str:
    """Return a stable digest for a transparent code witness."""

    return hashlib.sha256(code).hexdigest()


def _program_record(code: bytes, initial_rax: int) -> dict[str, object]:
    """Decode and execute one transparent program under one intervention."""

    instructions = tuple(disassemble_bytes(code))
    trace = emulate_x86_64(code, initial_rax=initial_rax)
    observations = encode_observations(instructions, trace)
    terminal_register_projection = {
        key: int(value) for key, value in sorted(trace.registers.items())
    }
    instruction_state_projection = [
        {
            "address": int(state.address),
            "pre_rax": int(state.pre_rax),
            "post_rax": int(state.post_rax),
            "pre_eflags": int(state.pre_eflags),
            "post_eflags": int(state.post_eflags),
        }
        for state in trace.instruction_states
    ]
    surface_digest = hashlib.sha256(
        json.dumps(
            {
                "terminal_register_projection": terminal_register_projection,
                "instruction_state_projection": instruction_state_projection,
                "memory_event_count": len(trace.memory_events),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "initial_rax": int(initial_rax),
        "final_rax": int(trace.final_rax),
        "terminal_register_projection": terminal_register_projection,
        "instruction_state_projection": instruction_state_projection,
        "memory_event_count": len(trace.memory_events),
        "architectural_surface_sha256": surface_digest,
        "executed_instruction_count": int(trace.instruction_count),
        "static_instruction_count": len(instructions),
        "observation_categories": [
            observation.category for observation in observations
        ],
        "executed_addresses": [int(address) for address in trace.addresses],
    }


def _validate_intervention_catalog(
    values: Iterable[int] | None,
) -> tuple[int, ...]:
    """Return a unique, bounded intervention catalog for one witness run."""

    catalog = tuple(INTERVENTION_INITIAL_RAX if values is None else values)
    if not catalog:
        raise ValueError("intervention catalog must not be empty")
    if any(isinstance(value, bool) for value in catalog):
        raise ValueError("intervention values must be integers, not booleans")
    normalized = tuple(int(value) for value in catalog)
    if any(value < 0 or value > U64_MAX for value in normalized):
        raise ValueError("intervention values must be unsigned 64-bit integers")
    if len(normalized) != len(set(normalized)):
        raise ValueError("intervention catalog values must be unique")
    return normalized


def evaluate_transformation_pair(
    pair: TransformationPair,
    *,
    initial_rax_values: Iterable[int] | None = None,
) -> dict[str, object]:
    """Evaluate one pair and retain both equivalence and surface evidence."""

    intervention_catalog = _validate_intervention_catalog(initial_rax_values)
    interventions: list[dict[str, object]] = []
    for initial_rax in intervention_catalog:
        reference = _program_record(pair.reference_code, initial_rax)
        variant = _program_record(pair.variant_code, initial_rax)
        interventions.append(
            {
                "initial_rax": int(initial_rax),
                "reference": reference,
                "variant": variant,
                "final_rax_equal": reference["final_rax"] == variant["final_rax"],
            }
        )
    equality_rows = [bool(row["final_rax_equal"]) for row in interventions]
    observed_equivalent = all(equality_rows)
    return {
        **pair.to_dict(),
        "reference_sha256": _code_fingerprint(pair.reference_code),
        "variant_sha256": _code_fingerprint(pair.variant_code),
        "initial_state_interventions": list(intervention_catalog),
        "interventions": interventions,
        "observed_equivalent": observed_equivalent,
        "near_miss_separated": not observed_equivalent,
        "observation_surface_changes": any(
            cast(dict[str, object], row["reference"])["observation_categories"]
            != cast(dict[str, object], row["variant"])["observation_categories"]
            for row in interventions
        ),
        "diagnostic_surface_changes": {
            "terminal_register_projection": any(
                cast(dict[str, object], row["reference"])[
                    "terminal_register_projection"
                ]
                != cast(dict[str, object], row["variant"])[
                    "terminal_register_projection"
                ]
                for row in interventions
            ),
            "instruction_state_projection": any(
                cast(dict[str, object], row["reference"])[
                    "instruction_state_projection"
                ]
                != cast(dict[str, object], row["variant"])[
                    "instruction_state_projection"
                ]
                for row in interventions
            ),
            "memory_event_count": any(
                cast(dict[str, object], row["reference"])["memory_event_count"]
                != cast(dict[str, object], row["variant"])["memory_event_count"]
                for row in interventions
            ),
        },
    }


def quotient_invariance_payload(
    pairs: Iterable[TransformationPair] | None = None,
) -> dict[str, object]:
    """Return the bounded quotient protocol and its executable witnesses."""

    pair_specs = tuple(_pair_specs() if pairs is None else pairs)
    if not pair_specs:
        raise ValueError("at least one transformation pair is required")
    pair_rows = [
        evaluate_transformation_pair(
            pair,
            initial_rax_values=INTERVENTION_INITIAL_RAX,
        )
        for pair in pair_specs
    ]
    equivalent_rows = [row for row in pair_rows if row["expected_equivalent"]]
    near_miss_rows = [row for row in pair_rows if not row["expected_equivalent"]]
    return {
        "schema_version": QUOTIENT_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": "bounded-rmax-quotient-v1",
        "architecture": "x86_64",
        "output_relation": "final_RAX",
        "diagnostic_surfaces": [
            "terminal_register_projection",
            "instruction_state_projection",
            "memory_event_count",
            "observation_categories",
        ],
        "initial_state_interventions": [
            int(value) for value in INTERVENTION_INITIAL_RAX
        ],
        "label_source": (
            "declared output relation and transparent transformation witnesses; "
            "no inferred or hand-authored latent phase labels"
        ),
        "pair_count": len(pair_rows),
        "expected_equivalent_count": len(equivalent_rows),
        "observed_equivalent_count": sum(
            bool(row["observed_equivalent"]) for row in equivalent_rows
        ),
        "near_miss_count": len(near_miss_rows),
        "near_miss_separated_count": sum(
            bool(row["near_miss_separated"]) for row in near_miss_rows
        ),
        "scope": (
            "bounded operational-equivalence protocol over transparent x86-64 "
            "register computations; not complete-ISA equivalence, semantic "
            "discovery, cross-architecture transfer, or a posterior-invariance result"
        ),
        "pairs": pair_rows,
    }


def validate_quotient_invariance_payload(
    payload: dict[str, object],
) -> tuple[bool, str]:
    """Validate the protocol contract and its expected equivalent/near-miss split."""

    if payload.get("schema_version") != QUOTIENT_PROTOCOL_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    interventions = payload.get("initial_state_interventions")
    pairs = payload.get("pairs")
    if not isinstance(interventions, list) or not interventions:
        return False, "initial_state_interventions must be non-empty"
    if interventions != list(INTERVENTION_INITIAL_RAX):
        return False, "intervention catalog differs from canonical protocol"
    if payload.get("diagnostic_surfaces") != [
        "terminal_register_projection",
        "instruction_state_projection",
        "memory_event_count",
        "observation_categories",
    ]:
        return False, "diagnostic surface manifest drift"
    if not isinstance(pairs, list) or not pairs:
        return False, "pairs must be a non-empty list"
    required = {
        "pair_id",
        "expected_equivalent",
        "observed_equivalent",
        "near_miss_separated",
        "diagnostic_surface_changes",
        "interventions",
        "reference_sha256",
        "variant_sha256",
        "initial_state_interventions",
    }
    ids: list[str] = []
    errors: list[str] = []
    for row in pairs:
        if not isinstance(row, dict):
            errors.append("non-object pair")
            continue
        pair_id = str(row.get("pair_id", ""))
        ids.append(pair_id)
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{pair_id}:missing={missing}")
            continue
        rows = row.get("interventions")
        row_catalog = row.get("initial_state_interventions")
        if row_catalog != list(INTERVENTION_INITIAL_RAX):
            errors.append(f"{pair_id}:intervention-catalog")
        if not isinstance(rows, list) or len(rows) != len(INTERVENTION_INITIAL_RAX):
            errors.append(f"{pair_id}:intervention-row-count")
            continue
        if bool(row["expected_equivalent"]):
            if not bool(row["observed_equivalent"]):
                errors.append(f"{pair_id}:equivalent-witness-failed")
        elif not bool(row["near_miss_separated"]):
            errors.append(f"{pair_id}:near-miss-not-separated")
        for intervention, expected_initial in zip(
            rows, INTERVENTION_INITIAL_RAX, strict=True
        ):
            if (
                not isinstance(intervention, dict)
                or int(intervention.get("initial_rax", -1)) != expected_initial
            ):
                errors.append(f"{pair_id}:intervention-alignment")
                break
            if isinstance(intervention, dict):
                for role in ("reference", "variant"):
                    record = intervention.get(role)
                    if not isinstance(record, dict) or any(
                        key not in record
                        for key in (
                            "terminal_register_projection",
                            "instruction_state_projection",
                            "memory_event_count",
                            "architectural_surface_sha256",
                        )
                    ):
                        errors.append(f"{pair_id}:surface-projection")
                        break
    if len(ids) != len(set(ids)):
        errors.append("duplicate pair IDs")
    pair_count = payload.get("pair_count", -1)
    expected_equivalent_count = payload.get("expected_equivalent_count", -1)
    if (pair_count if isinstance(pair_count, int) else -1) != len(pairs):
        errors.append("pair_count mismatch")
    expected_count = sum(
        bool(row.get("expected_equivalent")) for row in pairs if isinstance(row, dict)
    )
    if (
        expected_equivalent_count if isinstance(expected_equivalent_count, int) else -1
    ) != expected_count:
        errors.append("expected_equivalent_count mismatch")
    if errors:
        return False, "; ".join(errors[:8])
    return True, f"pairs={len(pairs)} interventions={len(INTERVENTION_INITIAL_RAX)}"


def write_quotient_invariance_protocol(project_root: Path = PROJECT_ROOT) -> str:
    """Write the executable quotient protocol artifact."""

    path = project_root / QUOTIENT_PROTOCOL_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = quotient_invariance_payload()
    valid, detail = validate_quotient_invariance_payload(payload)
    if not valid:
        raise ValueError(f"canonical quotient protocol is invalid: {detail}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return QUOTIENT_PROTOCOL_ARTIFACT


__all__ = [
    "INTERVENTION_INITIAL_RAX",
    "QUOTIENT_PROTOCOL_ARTIFACT",
    "QUOTIENT_PROTOCOL_SCHEMA_VERSION",
    "TransformationPair",
    "evaluate_transformation_pair",
    "quotient_invariance_payload",
    "validate_quotient_invariance_payload",
    "write_quotient_invariance_protocol",
]
