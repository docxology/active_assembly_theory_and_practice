"""Fail-closed theorem-or-countertheorem evaluation for the open roadmap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .formal_verification import formal_verification_payload
from .provenance import generator_revision, toolchain_versions
from .representation_invariance import (
    _family_specs,
    _pair_for_role,
)
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THEOREM_PHASE_SCHEMA_VERSION = "1.0.0"
THEOREM_PHASE_ARTIFACT = "output/data/theorem_or_countertheorem.json"


def theorem_or_countertheorem_payload(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Attempt a deliberately strong finite claim and emit its counterexample."""

    protocol = scientific_protocol_payload()
    formal = formal_verification_payload(
        project_root if (project_root / "formal").is_dir() else PROJECT_ROOT
    )
    candidate = (
        "The single intervention initial_rax=0 separates every declared near-miss "
        "family in the source-bound quotient benchmark."
    )
    counterexample: dict[str, Any] | None = None
    for family in _family_specs():
        row = _pair_for_role(
            family,
            role="near_miss",
            variant_code=family.near_miss_code,
            expected_equivalent=False,
            initial_rax_values=(0, 1),
        )
        interventions = cast(list[dict[str, Any]], row["interventions"])
        zero_state = next(item for item in interventions if item["initial_rax"] == 0)
        if zero_state["final_rax_equal"] and not all(
            item["final_rax_equal"] for item in interventions
        ):
            counterexample = {
                "family_id": family.family_id,
                "reference_code_hex": family.reference_code.hex(),
                "near_miss_code_hex": family.near_miss_code.hex(),
                "interventions": interventions,
                "witness": (
                    "initial_rax=0 is observationally non-separating, while "
                    "initial_rax=1 separates the pair"
                ),
            }
            break
    if counterexample is None:
        raise RuntimeError("the declared bounded counterexample was not found")
    return {
        "schema_version": THEOREM_PHASE_SCHEMA_VERSION,
        "artifact": THEOREM_PHASE_ARTIFACT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "generator_revision": generator_revision(Path(__file__)),
        "input_hashes": {
            "formal_sources": formal["source_hashes"],
            "protocol": protocol["protocol_sha256"],
        },
        "toolchain_versions": {
            **toolchain_versions(("capstone", "numpy", "unicorn")),
            "lean_toolchain": str(formal["toolchain"]),
        },
        "assumptions": [
            "bounded transparent fixture bytes",
            "deterministic Unicorn execution",
            "declared initial-state intervention catalog",
            "terminal RAX relation as the quotient observation",
            "no undefined, exceptional, privileged, SIMD, or floating-point effects",
        ],
        "candidate_theorems": [
            {
                "statement": "Word64 modular accumulator identities in the declared Lean fragment",
                "status": "proved",
                "proof_receipt": "output/data/formal_verification.json",
                "theorems": list(cast(list[Any], formal["theorems"])),
            },
            {
                "statement": "Exact optimality of the finite sensing policy",
                "status": "computationally_validated_only",
                "proof_receipt": "output/data/sensor_cost_receipt.json",
                "scope": "finite declared sensor/action space",
            },
            {
                "statement": candidate,
                "status": "falsified_by_counterexample",
                "counterexample_id": counterexample["family_id"],
            },
        ],
        "counterexample": counterexample,
        "promotion_status": "fail_closed_no_broad_theorem",
        "unresolved": [
            "general trace-refinement theorem for the full declared user-mode subset",
            "compiler/source/architecture transfer guarantee",
            "independent replay of the local harness",
        ],
    }


def validate_theorem_or_countertheorem_payload(
    payload: object,
) -> tuple[bool, str]:
    """Require a checked proof, a concrete counterexample, or explicit unresolved status."""

    if not isinstance(payload, dict):
        return False, "theorem phase payload must be an object"
    if payload.get("schema_version") != THEOREM_PHASE_SCHEMA_VERSION:
        return False, "theorem phase schema mismatch"
    protocol = scientific_protocol_payload()
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        return False, "theorem phase protocol hash mismatch"
    if payload.get("promotion_status") != "fail_closed_no_broad_theorem":
        return False, "theorem phase promotion is not fail closed"
    counterexample = payload.get("counterexample")
    if not isinstance(counterexample, dict):
        return False, "bounded counterexample is missing"
    interventions = counterexample.get("interventions")
    if not isinstance(interventions, list) or len(interventions) < 2:
        return False, "counterexample interventions are incomplete"
    by_initial = {
        item.get("initial_rax"): item
        for item in interventions
        if isinstance(item, dict)
    }
    if by_initial.get(0, {}).get("final_rax_equal") is not True:
        return False, "counterexample does not defeat the zero-state claim"
    if all(
        item.get("final_rax_equal") for item in interventions if isinstance(item, dict)
    ):
        return False, "counterexample does not contain a separating intervention"
    return True, "ok"


def write_theorem_or_countertheorem(
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Write the theorem/countertheorem receipt."""

    payload = theorem_or_countertheorem_payload(project_root)
    valid, detail = validate_theorem_or_countertheorem_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / THEOREM_PHASE_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "THEOREM_PHASE_ARTIFACT",
    "THEOREM_PHASE_SCHEMA_VERSION",
    "theorem_or_countertheorem_payload",
    "validate_theorem_or_countertheorem_payload",
    "write_theorem_or_countertheorem",
]
