"""Exact finite intervention-power analysis for the quotient benchmark.

The representation-invariance benchmark already records the outcome of every
declared initial-state intervention.  This module turns that outcome matrix
into a falsifiable identifiability diagnostic: it enumerates every non-empty
intervention subset, measures which declared near-miss pairs it separates, and
reports the smallest separating subsets when they exist.

This is deliberately a finite combinatorial result.  It does not infer a
universal intervention law, and it does not treat a separating subset as proof
that the underlying semantic quotient is identifiable outside the frozen
benchmark.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, cast

from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .representation_invariance import representation_invariance_payload
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVENTION_ANALYSIS_SCHEMA_VERSION = "2.0.0"
INTERVENTION_ANALYSIS_ARTIFACT = "output/data/intervention_power.json"


def _canonical_digest(value: object) -> str:
    """Return a stable digest for a JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _as_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _outcome_matrix(
    benchmark: dict[str, object],
) -> tuple[list[int], list[dict[str, object]]]:
    """Extract the declared terminal-output equality matrix."""

    raw_interventions = benchmark.get("initial_state_interventions")
    raw_pairs = benchmark.get("pairs")
    if not isinstance(raw_interventions, list) or not raw_interventions:
        raise ValueError("representation benchmark has no intervention catalog")
    catalog = [int(value) for value in raw_interventions]
    if len(set(catalog)) != len(catalog):
        raise ValueError("intervention catalog contains duplicate values")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("representation benchmark has no pair rows")

    rows: list[dict[str, object]] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"pair {pair_index} is not an object")
        family_id = raw_pair.get("family_id")
        expected_equivalent = raw_pair.get("expected_equivalent")
        raw_rows = raw_pair.get("interventions")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"pair {pair_index} has no family identifier")
        expected = _as_bool(
            expected_equivalent, field=f"pair {pair_index}.expected_equivalent"
        )
        if not isinstance(raw_rows, list) or len(raw_rows) != len(catalog):
            raise ValueError(f"pair {family_id} has an incomplete intervention row")
        outcomes: dict[str, bool] = {}
        observed_values: list[int] = []
        for row in raw_rows:
            if not isinstance(row, dict):
                raise ValueError(f"pair {family_id} has a malformed intervention")
            initial = row.get("initial_rax")
            equality = row.get("final_rax_equal")
            if not isinstance(initial, int) or isinstance(initial, bool):
                raise ValueError(f"pair {family_id} has a non-integer intervention")
            if initial not in catalog:
                raise ValueError(
                    f"pair {family_id} contains an undeclared intervention"
                )
            if initial in observed_values:
                raise ValueError(f"pair {family_id} repeats intervention {initial}")
            observed_values.append(initial)
            outcomes[str(initial)] = _as_bool(
                equality, field=f"pair {family_id}.final_rax_equal"
            )
        if set(observed_values) != set(catalog):
            raise ValueError(f"pair {family_id} does not cover the full catalog")
        rows.append(
            {
                "family_id": family_id,
                "expected_equivalent": expected,
                "outcomes": outcomes,
            }
        )
    return catalog, rows


def _subset_rows(
    catalog: list[int], pair_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Enumerate every non-empty subset and evaluate both endpoints."""

    evaluated: list[dict[str, object]] = []
    for size in range(1, len(catalog) + 1):
        for subset in combinations(catalog, size):
            equivalent_rows = [
                row for row in pair_rows if bool(row["expected_equivalent"])
            ]
            near_miss_rows = [
                row for row in pair_rows if not bool(row["expected_equivalent"])
            ]

            def outcomes(row: dict[str, object]) -> dict[str, bool]:
                return {
                    str(value): bool(
                        cast(dict[str, object], row["outcomes"])[str(value)]
                    )
                    for value in subset
                }

            equivalent_consistency = sum(
                all(outcomes(row)[str(value)] for value in subset)
                for row in equivalent_rows
            )
            separated = [
                row
                for row in near_miss_rows
                if any(not outcomes(row)[str(value)] for value in subset)
            ]
            unresolved = [
                str(row["family_id"])
                for row in near_miss_rows
                if not any(not outcomes(row)[str(value)] for value in subset)
            ]
            equivalent_count = len(equivalent_rows)
            near_miss_count = len(near_miss_rows)
            evaluated.append(
                {
                    "subset_id": "i[" + ",".join(str(value) for value in subset) + "]",
                    "values": list(subset),
                    "size": size,
                    "equivalent_consistency": (
                        equivalent_consistency / equivalent_count
                        if equivalent_count
                        else None
                    ),
                    "near_miss_separation": (
                        len(separated) / near_miss_count if near_miss_count else None
                    ),
                    "near_miss_unresolved_families": unresolved,
                    "fully_separating": (
                        equivalent_consistency == equivalent_count
                        and len(separated) == near_miss_count
                    ),
                }
            )
    return evaluated


def _subset_curve(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize best/worst separation power at each subset cardinality."""

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(cast(int, row["size"])), []).append(row)
    curve: list[dict[str, object]] = []
    for size, size_rows in sorted(grouped.items()):
        separation = [
            float(cast(float, row["near_miss_separation"])) for row in size_rows
        ]
        consistency = [
            float(cast(float, row["equivalent_consistency"])) for row in size_rows
        ]
        curve.append(
            {
                "size": size,
                "subset_count": len(size_rows),
                "min_near_miss_separation": min(separation),
                "mean_near_miss_separation": sum(separation) / len(separation),
                "max_near_miss_separation": max(separation),
                "min_equivalent_consistency": min(consistency),
                "max_equivalent_consistency": max(consistency),
                "fully_separating_subset_count": sum(
                    bool(row["fully_separating"]) for row in size_rows
                ),
            }
        )
    return curve


def intervention_power_payload(
    benchmark: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the exact finite intervention-power receipt."""

    benchmark_payload = benchmark or representation_invariance_payload()
    catalog, pair_rows = _outcome_matrix(benchmark_payload)
    subsets = _subset_rows(catalog, pair_rows)
    separating = [row for row in subsets if bool(row["fully_separating"])]
    minimum_size = min(
        (int(cast(int, row["size"])) for row in separating), default=None
    )
    minimal_sets = [
        list(cast(list[int], row["values"]))
        for row in separating
        if int(cast(int, row["size"])) == minimum_size
    ]
    near_miss_rows = [row for row in pair_rows if not bool(row["expected_equivalent"])]
    per_pair_minimum: list[dict[str, object]] = []
    for pair in near_miss_rows:
        outcomes = cast(dict[str, bool], pair["outcomes"])
        separators = [value for value in catalog if not outcomes[str(value)]]
        per_pair_minimum.append(
            {
                "family_id": pair["family_id"],
                "separating_interventions": separators,
                "minimum_cardinality": 1 if separators else None,
            }
        )
    one_state = next((row for row in subsets if row["values"] == [catalog[0]]), None)
    if one_state is None:
        raise RuntimeError("single-intervention negative control was not enumerated")
    status = "validated_bounded" if separating else "underpowered"
    protocol = scientific_protocol_payload()
    input_hash = _canonical_digest(
        {
            "initial_state_interventions": benchmark_payload.get(
                "initial_state_interventions"
            ),
            "families": benchmark_payload.get("families"),
            "pairs": benchmark_payload.get("pairs"),
        }
    )
    limitations = [
        "The relation is the declared bounded terminal-output relation.",
        "The intervention family is finite and repository-owned.",
        "Separation does not imply semantic identifiability outside the benchmark.",
        "No independent reimplementation or external source-program sample is included.",
    ]
    underpowered_cases = (
        [
            "The finite matrix is not evidence of universal identifiability or a complete semantic quotient."
        ]
        if status == "validated_bounded"
        else [
            "The declared finite matrix contains no fully separating intervention subset.",
        ]
    )
    input_hashes = {
        "representation_invariance_benchmark": input_hash,
        "src/intervention_analysis.py": sha256_path(Path(__file__)),
    }
    toolchains = toolchain_versions(("capstone", "numpy", "unicorn"))
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes=input_hashes,
        toolchain_versions=toolchains,
        status=status,
        limitations=limitations,
        unsupported_cases=(
            "Malformed or unsupported outcome rows fail closed before subset enumeration.",
        ),
        underpowered_cases=underpowered_cases,
    )
    return {
        **envelope,
        "schema_version": INTERVENTION_ANALYSIS_SCHEMA_VERSION,
        "artifact": INTERVENTION_ANALYSIS_ARTIFACT,
        "endpoint_status": "validated_finite_intervention_analysis"
        if separating
        else "underpowered_or_falsified_full_catalog",
        "scope": (
            "exact enumeration of non-empty subsets of the frozen initial-state "
            "intervention catalog; finite protocol diagnostic only"
        ),
        "intervention_catalog": catalog,
        "pair_count": len(pair_rows),
        "equivalent_pair_count": sum(
            bool(row["expected_equivalent"]) for row in pair_rows
        ),
        "near_miss_pair_count": len(near_miss_rows),
        "outcome_matrix": pair_rows,
        "subset_count": len(subsets),
        "subset_evaluations": subsets,
        "subset_curve": _subset_curve(subsets),
        "negative_control": {
            "values": [catalog[0]],
            "evaluation": one_state,
            "near_miss_separation": one_state["near_miss_separation"],
            "unresolved_near_miss_families": one_state["near_miss_unresolved_families"],
            "interpretation": (
                "A one-state catalog is retained as a deliberately insufficient "
                "control; unresolved near misses are evidence of intervention "
                "sensitivity, not a failed software run."
            ),
        },
        "minimal_separating_cardinality": minimum_size,
        "minimal_separating_sets": minimal_sets,
        "per_near_miss_minimal_interventions": per_pair_minimum,
        "promotion_status": "fail_closed_no_identifiability_promotion",
        "limitations": limitations,
        "unsupported_cases": list(envelope["unsupported_cases"]),
        "underpowered_cases": underpowered_cases,
    }


def validate_intervention_power_payload(payload: object) -> tuple[bool, str]:
    """Validate the exact-enumeration and fail-closed contracts."""

    if not isinstance(payload, dict):
        return False, "intervention-power payload must be an object"
    if payload.get("schema_version") != INTERVENTION_ANALYSIS_SCHEMA_VERSION:
        return False, "intervention-power schema mismatch"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"intervention-power receipt: {receipt_detail}"
    if payload.get("status") not in {"validated_bounded", "underpowered"}:
        return False, "intervention-power evidence status is invalid"
    catalog = payload.get("intervention_catalog")
    if (
        not isinstance(catalog, list)
        or not catalog
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in catalog
        )
        or len(set(catalog)) != len(catalog)
    ):
        return False, "intervention catalog is malformed"
    subset_rows = payload.get("subset_evaluations")
    if not isinstance(subset_rows, list) or len(subset_rows) != 2 ** len(catalog) - 1:
        return False, "subset enumeration is incomplete"
    seen: set[tuple[int, ...]] = set()
    for row in subset_rows:
        if not isinstance(row, dict):
            return False, "subset row is not an object"
        values = row.get("values")
        if (
            not isinstance(values, list)
            or not values
            or any(value not in catalog for value in values)
            or len(set(values)) != len(values)
            or int(row.get("size", -1)) != len(values)
        ):
            return False, "subset values are malformed"
        key = tuple(values)
        if key in seen:
            return False, "subset enumeration contains duplicates"
        seen.add(key)
        for metric in ("equivalent_consistency", "near_miss_separation"):
            value = row.get(metric)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
            ):
                return False, f"subset metric {metric} is invalid"
        unresolved = row.get("near_miss_unresolved_families")
        if not isinstance(unresolved, list) or any(
            not isinstance(value, str) for value in unresolved
        ):
            return False, "subset unresolved-family list is malformed"
    pair_rows = payload.get("outcome_matrix")
    if not isinstance(pair_rows, list) or not pair_rows:
        return False, "outcome matrix is missing"
    expected_count = payload.get("pair_count")
    if expected_count != len(pair_rows):
        return False, "pair count does not match outcome matrix"
    for row in pair_rows:
        if not isinstance(row, dict):
            return False, "outcome row is not an object"
        outcomes = row.get("outcomes")
        if (
            not isinstance(outcomes, dict)
            or set(outcomes) != {str(value) for value in catalog}
            or any(not isinstance(value, bool) for value in outcomes.values())
        ):
            return False, "outcome row does not cover the intervention catalog"
    negative_control = payload.get("negative_control")
    if not isinstance(negative_control, dict):
        return False, "negative control is missing"
    if negative_control.get("values") != [catalog[0]]:
        return False, "negative control does not use the first declared intervention"
    negative_separation = negative_control.get("near_miss_separation")
    if (
        not isinstance(negative_separation, (int, float))
        or isinstance(negative_separation, bool)
        or not 0.0 <= float(negative_separation) <= 1.0
    ):
        return False, "negative-control separation is malformed"
    unresolved = negative_control.get("unresolved_near_miss_families")
    if not isinstance(unresolved, list) or any(
        not isinstance(value, str) for value in unresolved
    ):
        return False, "negative-control unresolved families are malformed"
    if payload.get("promotion_status") != "fail_closed_no_identifiability_promotion":
        return False, "intervention analysis cannot promote identifiability"
    return True, f"subsets={len(subset_rows)} pairs={len(pair_rows)}"


def write_intervention_power_payload(project_root: Path = PROJECT_ROOT) -> Path:
    """Write and validate the finite intervention-power receipt."""

    payload = intervention_power_payload()
    valid, detail = validate_intervention_power_payload(payload)
    if not valid:
        raise ValueError(detail)
    path = project_root / INTERVENTION_ANALYSIS_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "INTERVENTION_ANALYSIS_ARTIFACT",
    "INTERVENTION_ANALYSIS_SCHEMA_VERSION",
    "intervention_power_payload",
    "validate_intervention_power_payload",
    "write_intervention_power_payload",
]
