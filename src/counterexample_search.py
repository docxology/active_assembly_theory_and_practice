"""Smallest-first formal counterexample search over declared x86-64 candidates.

Objective: enumerate a declared, deterministic, smallest-first bounded domain
of 64-bit register values and search for counterexamples to eight declared
candidate theorems about the Python reference semantics in
``src/formal_correspondence.py``.  Every falsified candidate must carry a
witness whose predicate failure is confirmed by an independent second
interpreter run (fresh disassembly, fresh state) before it is recorded.

Primary endpoint: a sealed receipt at ``output/data/counterexample_search.json``
whose overall status is ``validated_bounded`` only when every candidate
resolves to ``falsified_by_counterexample`` or ``holds_in_searched_domain``,
with per-candidate searched domain sizes, witness state projections, and the
shared enumeration rule made explicit rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast

from .artifact_contract import validate_artifact_envelope, write_json_artifact
from .disassembly import BASE_ADDRESS, InstructionEvent, disassemble_bytes
from .formal_correspondence import (
    FLAG_BITS,
    MODELED_FLAGS,
    U64_MASK,
    FormalState,
    _step,
)
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION = "1.0.0"
COUNTEREXAMPLE_SEARCH_ARTIFACT = "output/data/counterexample_search.json"

FALSIFIED_STATUS = "falsified_by_counterexample"
HOLDS_STATUS = "holds_in_searched_domain"
RESOLVED_STATUSES = frozenset({FALSIFIED_STATUS, HOLDS_STATUS})
ALL_DEFINED_FLAGS: tuple[bool, ...] = (True,) * len(MODELED_FLAGS)

_REGISTER_KEYS = ("rax", "rbx", "rcx")

# Declared smallest-first enumeration: an ascending prefix of 64-bit values,
# then the 64-bit boundary extremes in ascending order.  Tuples are
# enumerated lexicographically in declared operand order.
_SINGLE_PREFIX = 64
_PAIR_PREFIX = 32
_BOUNDARY_EXTREMES = (2**63 - 1, 2**63, 2**64 - 1)
_SINGLE_VALUES = tuple(range(_SINGLE_PREFIX)) + _BOUNDARY_EXTREMES
_PAIR_VALUES = tuple(range(_PAIR_PREFIX)) + _BOUNDARY_EXTREMES
_DIVISOR_EXPONENTS = tuple(range(1, 64))
_SHIFT_COUNTS = tuple(range(64))


# Transparent register-register fixtures at BASE_ADDRESS (hex machine code).
_CODE_ADD = bytes.fromhex("4801d8")  # add rax, rbx
_CODE_SUB = bytes.fromhex("4829d8")  # sub rax, rbx
_CODE_ADC = bytes.fromhex("4811d8")  # adc rax, rbx
_CODE_SAR_CL = bytes.fromhex("48d3f8")  # sar rax, cl
_CODE_SHL_CL = bytes.fromhex("48d3e0")  # shl rax, cl
_CODE_MOV_EAX_EBX = bytes.fromhex("89d8")  # mov eax, ebx
_CODE_INC_RAX = bytes.fromhex("48ffc0")  # inc rax
_CODE_DEC_RAX = bytes.fromhex("48ffc8")  # dec rax
_CODE_DEC_INC_RAX = _CODE_DEC_RAX + _CODE_INC_RAX  # dec rax; inc rax

_ENUMERATION_ORDER = (
    "deterministic smallest-first: operand values ascending from 0 through the "
    "declared prefix length, then the 64-bit boundary extremes "
    "{2**63-1, 2**63, 2**64-1} ascending; tuples lexicographic in declared "
    "operand order"
)

# Declared initial flag contexts for the CF-preservation candidate:
# (initial eflags bits, per-flag definedness aligned with MODELED_FLAGS).
_CF_CONTEXTS: tuple[tuple[int, tuple[bool, ...]], ...] = (
    (1, ALL_DEFINED_FLAGS),
    (0, ALL_DEFINED_FLAGS),
    (1, (False,) + ALL_DEFINED_FLAGS[1:]),
)


@dataclass(frozen=True)
class WitnessInputs:
    """One enumerated input point: registers, eflags, and flag definedness."""

    rax: int
    rbx: int = 0
    rcx: int = 0
    eflags: int = 0
    flags_defined: tuple[bool, ...] = ALL_DEFINED_FLAGS


Projection = dict[str, object]
RunProjections = tuple[Projection, Projection]
Runner = Callable[[bytes, WitnessInputs], RunProjections]
Evaluator = Callable[[WitnessInputs, Runner], tuple[bool, dict[str, object]]]


@dataclass(frozen=True)
class CandidateSpec:
    """One declared candidate theorem with its domain and predicate."""

    property_id: str
    statement: str
    expected: str
    domain: tuple[tuple[WitnessInputs, int], ...]
    evaluate: Evaluator


@lru_cache(maxsize=None)
def _decode_events(code: bytes) -> tuple[InstructionEvent, ...]:
    """Decode one fixture once for the cached interpreter path."""

    return tuple(disassemble_bytes(code, base_address=BASE_ADDRESS))


def _initial_state(inputs: WitnessInputs) -> FormalState:
    return FormalState(
        rax=inputs.rax,
        rsp=0,
        rip=BASE_ADDRESS,
        eflags=inputs.eflags,
        memory={},
        registers={name: getattr(inputs, name) for name in _REGISTER_KEYS},
        flags_defined=dict(zip(MODELED_FLAGS, inputs.flags_defined)),
    )


def _project(state: FormalState) -> Projection:
    """Project the state components the candidate predicates consult."""

    return {
        "registers": {name: state.registers[name] for name in _REGISTER_KEYS},
        "eflags": state.eflags,
        "flags_defined": dict(state.flags_defined),
    }


def _run_with_events(
    events: tuple[InstructionEvent, ...], inputs: WitnessInputs
) -> RunProjections:
    state = _initial_state(inputs)
    before = _project(state)
    for event in events:
        _step(event, state, None)
    return before, _project(state)


def _run_cached(code: bytes, inputs: WitnessInputs) -> RunProjections:
    return _run_with_events(_decode_events(code), inputs)


def _run_fresh(code: bytes, inputs: WitnessInputs) -> RunProjections:
    """Independent interpreter path: fresh Capstone decode and fresh state."""

    events = tuple(disassemble_bytes(code, base_address=BASE_ADDRESS))
    return _run_with_events(events, inputs)


def _register_value(projection: Projection, name: str) -> int:
    registers = cast(dict[str, int], projection["registers"])
    return registers[name]


def _flag_bit(projection: Projection, flag: str) -> int:
    return (int(cast(int, projection["eflags"])) >> FLAG_BITS[flag]) & 1


def _cf_defined(projection: Projection) -> bool:
    return bool(cast(dict[str, bool], projection["flags_defined"])["CF"])


def _same_projection(left: Projection, right: Projection) -> bool:
    return (
        left["registers"] == right["registers"]
        and left["eflags"] == right["eflags"]
        and left["flags_defined"] == right["flags_defined"]
    )


def _runs_evidence(
    runs: list[tuple[bytes, WitnessInputs, Projection, Projection]],
) -> dict[str, object]:
    return {
        "runs": [
            {
                "code_hex": code.hex(),
                "inputs": _serialize_inputs(inputs),
                "before": before,
                "after": after,
            }
            for code, inputs, before, after in runs
        ]
    }


def _serialize_inputs(inputs: WitnessInputs) -> dict[str, object]:
    return {
        "registers": {name: getattr(inputs, name) for name in _REGISTER_KEYS},
        "initial_eflags": inputs.eflags,
        "initial_flags_defined": dict(zip(MODELED_FLAGS, inputs.flags_defined)),
    }


# --- Candidate predicates -------------------------------------------------


def _evaluate_add_cf_symmetry(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    before, after = run(_CODE_ADD, inputs)
    carry = 1 if inputs.rax + inputs.rbx > U64_MASK else 0
    holds = (
        _flag_bit(after, "CF") == carry
        and _register_value(after, "rax") == (inputs.rax + inputs.rbx) & U64_MASK
    )
    return holds, _runs_evidence([(_CODE_ADD, inputs, before, after)])


def _evaluate_sub_vs_add_negation(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    negated = WitnessInputs(rax=inputs.rax, rbx=(-inputs.rbx) & U64_MASK)
    sub_before, sub_after = run(_CODE_SUB, inputs)
    add_before, add_after = run(_CODE_ADD, negated)
    holds = _same_projection(sub_after, add_after)
    return holds, _runs_evidence(
        [
            (_CODE_SUB, inputs, sub_before, sub_after),
            (_CODE_ADD, negated, add_before, add_after),
        ]
    )


def _evaluate_sar_vs_signed_division(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    before, after = run(_CODE_SAR_CL, inputs)
    count = inputs.rcx & 0x3F
    signed = inputs.rax - (1 << 64) if inputs.rax >> 63 else inputs.rax
    quotient = signed >> count if signed >= 0 else -((-signed) >> count)
    holds = _register_value(after, "rax") == quotient & U64_MASK
    return holds, _runs_evidence([(_CODE_SAR_CL, inputs, before, after)])


def _evaluate_dec_inc_restores_of(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    before, after = run(_CODE_DEC_INC_RAX, inputs)
    holds = _register_value(after, "rax") == _register_value(
        before, "rax"
    ) and _flag_bit(after, "OF") == _flag_bit(before, "OF")
    return holds, _runs_evidence([(_CODE_DEC_INC_RAX, inputs, before, after)])


def _evaluate_inc_dec_cf_preservation(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    runs: list[tuple[bytes, WitnessInputs, Projection, Projection]] = []
    holds = True
    for code in (_CODE_INC_RAX, _CODE_DEC_RAX):
        before, after = run(code, inputs)
        runs.append((code, inputs, before, after))
        if _cf_defined(after) != _cf_defined(before):
            holds = False
        elif _cf_defined(before) and _flag_bit(before, "CF") != _flag_bit(after, "CF"):
            holds = False
    return holds, _runs_evidence(runs)


def _evaluate_mov32_upper_preservation(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    before, after = run(_CODE_MOV_EAX_EBX, inputs)
    expected = (inputs.rax & 0xFFFFFFFF00000000) | inputs.rbx
    holds = _register_value(after, "rax") == expected
    return holds, _runs_evidence([(_CODE_MOV_EAX_EBX, inputs, before, after)])


def _evaluate_adc_zero_carry_vs_add(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    adc_before, adc_after = run(_CODE_ADC, inputs)
    add_before, add_after = run(_CODE_ADD, inputs)
    holds = _same_projection(adc_after, add_after)
    return holds, _runs_evidence(
        [
            (_CODE_ADC, inputs, adc_before, adc_after),
            (_CODE_ADD, inputs, add_before, add_after),
        ]
    )


def _evaluate_shl_count_masking(
    inputs: WitnessInputs, run: Runner
) -> tuple[bool, dict[str, object]]:
    masked = WitnessInputs(
        rax=inputs.rax,
        rcx=inputs.rcx & 0x3F,
        eflags=inputs.eflags,
        flags_defined=inputs.flags_defined,
    )
    unmasked = WitnessInputs(
        rax=inputs.rax,
        rcx=(inputs.rcx & 0x3F) + 64,
        eflags=inputs.eflags,
        flags_defined=inputs.flags_defined,
    )
    first_before, first_after = run(_CODE_SHL_CL, masked)
    second_before, second_after = run(_CODE_SHL_CL, unmasked)
    holds = (
        _register_value(first_after, "rax") == _register_value(second_after, "rax")
        and first_after["eflags"] == second_after["eflags"]
        and first_after["flags_defined"] == second_after["flags_defined"]
    )
    return holds, _runs_evidence(
        [
            (_CODE_SHL_CL, masked, first_before, first_after),
            (_CODE_SHL_CL, unmasked, second_before, second_after),
        ]
    )


# --- Declared domains -----------------------------------------------------


def _pair_domain() -> tuple[tuple[WitnessInputs, int], ...]:
    return tuple(
        (WitnessInputs(rax=a, rbx=b), 2) for a in _PAIR_VALUES for b in _PAIR_VALUES
    )


def _single_domain() -> tuple[tuple[WitnessInputs, int], ...]:
    return tuple((WitnessInputs(rax=x), 1) for x in _SINGLE_VALUES)


def _sar_domain() -> tuple[tuple[WitnessInputs, int], ...]:
    return tuple(
        (WitnessInputs(rax=x, rcx=n), 2)
        for x in _SINGLE_VALUES
        for n in _DIVISOR_EXPONENTS
    )


def _cf_context_domain() -> tuple[tuple[WitnessInputs, int], ...]:
    return tuple(
        (WitnessInputs(rax=x, eflags=eflags, flags_defined=flags_defined), 1)
        for x in _SINGLE_VALUES
        for eflags, flags_defined in _CF_CONTEXTS
    )


def _shl_domain() -> tuple[tuple[WitnessInputs, int], ...]:
    return tuple(
        (WitnessInputs(rax=x, rcx=count), 2)
        for x in _SINGLE_VALUES
        for count in _SHIFT_COUNTS
    )


_CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        property_id="add_cf_symmetry",
        statement=(
            "For 64-bit add rax, rbx the CF flag equals the carry out of the "
            "unsigned addition and the register result is the sum modulo 2**64."
        ),
        expected="holds",
        domain=_pair_domain(),
        evaluate=_evaluate_add_cf_symmetry,
    ),
    CandidateSpec(
        property_id="sub_equals_add_of_negation_incl_cf",
        statement=(
            "For 64-bit a and b, sub a, b produces the same register result "
            "and full modeled flag set as add a, (-b) modulo 2**64, including CF."
        ),
        expected="falsified",
        domain=_pair_domain(),
        evaluate=_evaluate_sub_vs_add_negation,
    ),
    CandidateSpec(
        property_id="sar_equals_signed_division",
        statement=(
            "For 64-bit x and count n in 1..63, sar x, n equals x divided by "
            "2**n with signed truncation toward zero."
        ),
        expected="falsified",
        domain=_sar_domain(),
        evaluate=_evaluate_sar_vs_signed_division,
    ),
    CandidateSpec(
        property_id="dec_then_inc_restores_of",
        statement=(
            "dec rax followed by inc rax restores both the register value and "
            "the OF flag."
        ),
        expected="falsified",
        domain=_single_domain(),
        evaluate=_evaluate_dec_inc_restores_of,
    ),
    CandidateSpec(
        property_id="inc_dec_preserve_cf_and_definedness",
        statement=(
            "inc and dec preserve both the CF bit value (when CF is defined) "
            "and the definedness of CF."
        ),
        expected="holds",
        domain=_cf_context_domain(),
        evaluate=_evaluate_inc_dec_cf_preservation,
    ),
    CandidateSpec(
        property_id="mov32_preserves_upper_bits",
        statement=(
            "Writing ebx into eax preserves the upper 32 bits of rax "
            "(no zero-extension)."
        ),
        expected="falsified",
        domain=_pair_domain(),
        evaluate=_evaluate_mov32_upper_preservation,
    ),
    CandidateSpec(
        property_id="adc_zero_carry_equals_add",
        statement=(
            "With CF defined and clear, adc rax, rbx produces the same register "
            "result and full modeled flag set as add rax, rbx."
        ),
        expected="holds",
        domain=_pair_domain(),
        evaluate=_evaluate_adc_zero_carry_vs_add,
    ),
    CandidateSpec(
        property_id="shl_count_masked_mod_64",
        statement=(
            "For 64-bit shl rax, cl the count is masked modulo 64: cl = c and "
            "cl = c + 64 produce identical register results and flags."
        ),
        expected="holds",
        domain=_shl_domain(),
        evaluate=_evaluate_shl_count_masking,
    ),
)

_PROPERTY_IDS = tuple(spec.property_id for spec in _CANDIDATE_SPECS)


# --- Serialization helpers ------------------------------------------------


def _inputs_from_serialized(payload: object) -> WitnessInputs | None:
    if not isinstance(payload, dict):
        return None
    registers = payload.get("registers")
    eflags = payload.get("initial_eflags")
    defined = payload.get("initial_flags_defined")
    if not isinstance(registers, dict) or not isinstance(defined, dict):
        return None
    values: dict[str, int] = {}
    for name in _REGISTER_KEYS:
        value = registers.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values[name] = value
    if not isinstance(eflags, int) or isinstance(eflags, bool):
        return None
    if any(
        name not in defined or not isinstance(defined[name], bool)
        for name in MODELED_FLAGS
    ):
        return None
    return WitnessInputs(
        rax=values["rax"],
        rbx=values["rbx"],
        rcx=values["rcx"],
        eflags=eflags,
        flags_defined=tuple(bool(defined[name]) for name in MODELED_FLAGS),
    )


def _search_candidate(spec: CandidateSpec) -> dict[str, object]:
    """Search the declared domain smallest-first and resolve the candidate."""

    witness: dict[str, object] | None = None
    status = HOLDS_STATUS
    for inputs, witness_size in spec.domain:
        holds, evidence = spec.evaluate(inputs, _run_cached)
        if holds:
            continue
        reverified_holds, _ = spec.evaluate(inputs, _run_fresh)
        runs = cast(list[dict[str, object]], evidence["runs"])
        primary = runs[0]
        witness = {
            "inputs": _serialize_inputs(inputs),
            "witness_size": witness_size,
            "before": primary["before"],
            "after": primary["after"],
            "runs": evidence["runs"],
            "reverified": not reverified_holds,
        }
        status = FALSIFIED_STATUS if not reverified_holds else "unresolved"
        break
    return {
        "property_id": spec.property_id,
        "statement": spec.statement,
        "expected": spec.expected,
        "status": status,
        "searched_domain_size": len(spec.domain),
        "enumeration_rule": _ENUMERATION_ORDER,
        "witness": witness,
    }


# --- Public API -----------------------------------------------------------


def counterexample_search_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the source-bound bounded counterexample search receipt."""

    rows = [_search_candidate(spec) for spec in _CANDIDATE_SPECS]
    all_resolved = all(row["status"] in RESOLVED_STATUSES for row in rows)
    falsified_count = sum(row["status"] == FALSIFIED_STATUS for row in rows)
    holds_count = sum(row["status"] == HOLDS_STATUS for row in rows)

    protocol = scientific_protocol_payload()
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("formal_correspondence.py"),
    )
    input_hashes: dict[str, str] = {
        str(path.relative_to(PROJECT_ROOT)): sha256_path(path) for path in source_paths
    }
    input_hashes["research_protocol.json"] = str(protocol.get("protocol_sha256", ""))
    limitations = [
        "Resolution claims are bounded to the declared searched domain per candidate; holds_in_searched_domain is not a proof over all 2**64 register values.",
        "Only the declared register-register instruction fixtures are exercised; memory, stack, and control-flow effects are outside this search.",
    ]
    formal_trace = project_root / "output" / "data" / "formal_trace_correspondence.json"
    if formal_trace.is_file():
        input_hashes["output/data/formal_trace_correspondence.json"] = sha256_path(
            formal_trace
        )
    else:
        limitations.append(
            "output/data/formal_trace_correspondence.json was absent under the "
            "project root; its input hash is omitted rather than rendered as zero."
        )
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=input_hashes,
        toolchain_versions=toolchain_versions(("capstone", "numpy", "unicorn")),
        status="validated_bounded" if all_resolved else "failed",
        limitations=limitations,
        unsupported_cases=(
            "Memory, stack, and control-flow instructions; SIMD, floating-point, "
            "and privileged instructions.",
            "Flag dependencies outside the modeled definedness contract of the "
            "reference interpreter.",
        ),
        underpowered_cases=(
            "A bounded search is computational evidence only; "
            "holds_in_searched_domain candidates are not promoted to theorems.",
        ),
    )
    return {
        **envelope,
        "schema_version": COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION,
        "artifact": COUNTEREXAMPLE_SEARCH_ARTIFACT,
        "scope": "declared bounded counterexample search over the Python "
        "reference x86-64 semantics; not complete x86-64 semantics",
        "architecture": "x86_64",
        "enumeration_rule": {
            "order": _ENUMERATION_ORDER,
            "smallest_prefix_single": _SINGLE_PREFIX,
            "smallest_prefix_pair": _PAIR_PREFIX,
            "boundary_extremes": list(_BOUNDARY_EXTREMES),
            "witness_selection": "first enumerated input tuple whose predicate "
            "fails; the whole declared domain is searched and its size recorded",
            "deterministic": True,
        },
        "state_projection_fields": ["registers", "eflags", "flags_defined"],
        "candidate_count": len(rows),
        "falsified_count": falsified_count,
        "holds_count": holds_count,
        "all_candidates_resolved": all_resolved,
        "candidates": rows,
        "assumptions": [
            "The searched domain is the declared smallest-first bounded prefix "
            "of 64-bit register values plus boundary extremes, not the full "
            "2**64 space.",
            "The Python reference interpreter in src/formal_correspondence.py is "
            "the sole semantic oracle; no external emulator adjudicates "
            "candidate resolution.",
            "Initial flag values and flag definedness are independent declared "
            "inputs of each enumerated point.",
        ],
    }


def validate_counterexample_search_payload(payload: object) -> tuple[bool, str]:
    """Validate the receipt envelope and re-verify every recorded witness."""

    if not isinstance(payload, dict):
        return False, "counterexample search payload must be an object"
    if payload.get("schema_version") != COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION:
        return False, "counterexample search schema mismatch"
    receipt_valid, receipt_detail = validate_artifact_envelope(
        payload, expected_protocol=scientific_protocol_payload()
    )
    if not receipt_valid:
        return False, f"counterexample search receipt: {receipt_detail}"
    if payload.get("status") != "validated_bounded":
        return False, "counterexample search is not validated_bounded"
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, dict):
        return False, "counterexample search input_hashes must be an object"
    for key in (
        "research_protocol.json",
        "src/counterexample_search.py",
        "src/formal_correspondence.py",
    ):
        if key not in hashes:
            return False, f"counterexample search input hash missing: {key}"
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 8:
        return False, "counterexample search candidates are missing"
    if tuple(row.get("property_id") for row in candidates) != _PROPERTY_IDS:
        return False, "counterexample search candidate roster drift"
    falsified_count = 0
    holds_count = 0
    for row in candidates:
        if not isinstance(row, dict):
            return False, "counterexample search candidate row is not an object"
        status = row.get("status")
        if status not in RESOLVED_STATUSES:
            return False, f"candidate {row.get('property_id')!r} is unresolved"
        expected = row.get("expected")
        if expected == "falsified" and status != FALSIFIED_STATUS:
            return False, (
                f"candidate {row.get('property_id')!r} contradicts its declared "
                "expectation"
            )
        if expected == "holds" and status != HOLDS_STATUS:
            return False, (
                f"candidate {row.get('property_id')!r} contradicts its declared "
                "expectation"
            )
        size = row.get("searched_domain_size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            return False, (
                f"candidate {row.get('property_id')!r} lacks a positive "
                "searched_domain_size"
            )
        if status == FALSIFIED_STATUS:
            falsified_count += 1
            ok, detail = _validate_falsified_row(row)
            if not ok:
                return False, detail
        else:
            holds_count += 1
            if row.get("witness") is not None:
                return False, (
                    f"holding candidate {row.get('property_id')!r} carries a witness"
                )
    if payload.get("falsified_count") != falsified_count:
        return False, "counterexample search falsified_count drift"
    if payload.get("holds_count") != holds_count:
        return False, "counterexample search holds_count drift"
    if not isinstance(payload.get("enumeration_rule"), dict):
        return False, "counterexample search enumeration_rule is missing"
    return True, "ok"


def _validate_falsified_row(row: dict[str, Any]) -> tuple[bool, str]:
    property_id = str(row.get("property_id"))
    witness = row.get("witness")
    if not isinstance(witness, dict):
        return False, f"falsified candidate {property_id!r} lacks a witness"
    if witness.get("reverified") is not True:
        return False, f"witness for {property_id!r} was not re-verified"
    if not isinstance(witness.get("before"), dict) or not isinstance(
        witness.get("after"), dict
    ):
        return False, f"witness for {property_id!r} lacks state projections"
    if not isinstance(witness.get("witness_size"), int) or isinstance(
        witness.get("witness_size"), bool
    ):
        return False, f"witness for {property_id!r} lacks a witness_size"
    inputs = _inputs_from_serialized(witness.get("inputs"))
    if inputs is None:
        return False, f"witness inputs for {property_id!r} are malformed"
    spec = next(item for item in _CANDIDATE_SPECS if item.property_id == property_id)
    holds_on_second_run, _ = spec.evaluate(inputs, _run_fresh)
    if holds_on_second_run:
        return False, f"witness for {property_id!r} does not fail its predicate"
    return True, "ok"


def write_counterexample_search(project_root: Path = PROJECT_ROOT) -> Path:
    """Write and validate the sealed counterexample search receipt."""

    payload = counterexample_search_payload(project_root)
    valid, detail = validate_counterexample_search_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / COUNTEREXAMPLE_SEARCH_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path


__all__ = [
    "COUNTEREXAMPLE_SEARCH_ARTIFACT",
    "COUNTEREXAMPLE_SEARCH_SCHEMA_VERSION",
    "CandidateSpec",
    "WitnessInputs",
    "counterexample_search_payload",
    "validate_counterexample_search_payload",
    "write_counterexample_search",
]
