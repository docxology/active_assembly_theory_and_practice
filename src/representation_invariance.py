"""Family-level representation-invariance protocol for the active P3 tranche.

The first quotient protocol evaluates a small witness set.  This module adds
the next scientific layer: transformation families, source-family-disjoint
development/held-out splits, predeclared endpoints, and descriptive separation
statistics.  It still uses only transparent byte fixtures and the real
Capstone/Unicorn adapters.  No model is fitted and no family label is learned;
the artifact is a preregistered benchmark contract and protocol diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Iterable, cast

from .quotient_invariance import (
    INTERVENTION_INITIAL_RAX,
    TransformationPair,
    U64_MAX,
    evaluate_transformation_pair,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPRESENTATION_INVARIANCE_SCHEMA_VERSION = "1.1.0"
REPRESENTATION_INVARIANCE_ARTIFACT = (
    "output/data/representation_invariance_benchmark.json"
)
DEVELOPMENT_SPLIT = "development"
HELD_OUT_SPLIT = "held_out"
BENCHMARK_SPLITS = (DEVELOPMENT_SPLIT, HELD_OUT_SPLIT)
NON_SEPARATING_INTERVENTION_INITIAL_RAX: tuple[int, ...] = (0,)
BENCHMARK_ID = "source-family-quotient-v1"
WILSON_CONFIDENCE_LEVEL = 0.95
EQUIVALENT_RELATION = "final_RAX_equal_for_each_declared_initial_state"
NEAR_MISS_RELATION = "final_RAX_must_differ_for_at_least_one_declared_initial_state"


@dataclass(frozen=True)
class TransformationFamily:
    """One source-family fixture with an equivalent and a near-miss variant."""

    family_id: str
    motif: str
    split: str
    reference_code: bytes
    equivalent_code: bytes
    near_miss_code: bytes

    def to_dict(self) -> dict[str, object]:
        """Return the family manifest without executable byte objects."""

        data = asdict(self)
        data["reference_code_hex"] = self.reference_code.hex()
        data["equivalent_code_hex"] = self.equivalent_code.hex()
        data["near_miss_code_hex"] = self.near_miss_code.hex()
        del data["reference_code"]
        del data["equivalent_code"]
        del data["near_miss_code"]
        return data


def _family_specs() -> tuple[TransformationFamily, ...]:
    """Return the frozen source-family and split manifest."""

    return (
        TransformationFamily(
            family_id="constant_addition",
            motif="constant_accumulator_update",
            split=DEVELOPMENT_SPLIT,
            reference_code=bytes.fromhex("b8 01 00 00 00 83 c0 02"),
            equivalent_code=bytes.fromhex("b8 01 00 00 00 ff c0 ff c0"),
            near_miss_code=bytes.fromhex("b8 01 00 00 00 83 e8 02"),
        ),
        TransformationFamily(
            family_id="neutral_layout",
            motif="neutral_representation_change",
            split=DEVELOPMENT_SPLIT,
            reference_code=bytes.fromhex("b8 07 00 00 00 83 c0 05"),
            equivalent_code=bytes.fromhex("b8 07 00 00 00 90 83 c0 05"),
            near_miss_code=bytes.fromhex("b8 07 00 00 00 83 c0 06"),
        ),
        TransformationFamily(
            family_id="constant_fold",
            motif="constant_expression_reduction",
            split=DEVELOPMENT_SPLIT,
            reference_code=bytes.fromhex("b8 02 00 00 00 83 c0 03"),
            equivalent_code=bytes.fromhex("b8 05 00 00 00"),
            near_miss_code=bytes.fromhex("b8 06 00 00 00"),
        ),
        TransformationFamily(
            family_id="constant_subtraction",
            motif="constant_subtractive_update",
            split=DEVELOPMENT_SPLIT,
            reference_code=bytes.fromhex("b8 0a 00 00 00 83 e8 03"),
            equivalent_code=bytes.fromhex("b8 07 00 00 00"),
            near_miss_code=bytes.fromhex("b8 06 00 00 00"),
        ),
        TransformationFamily(
            family_id="temporary_register",
            motif="temporary_register_introduction",
            split=HELD_OUT_SPLIT,
            reference_code=bytes.fromhex("b8 03 00 00 00 83 c0 04"),
            equivalent_code=bytes.fromhex("b9 03 00 00 00 83 c1 04 89 c8"),
            near_miss_code=bytes.fromhex("b9 03 00 00 00 83 c1 05 89 c8"),
        ),
        TransformationFamily(
            family_id="zeroing_idiom",
            motif="register_zeroing",
            split=HELD_OUT_SPLIT,
            reference_code=bytes.fromhex("31 c0"),
            equivalent_code=bytes.fromhex("29 c0"),
            near_miss_code=bytes.fromhex("09 c0"),
        ),
        TransformationFamily(
            family_id="doubling_idiom",
            motif="arithmetic_doubling",
            split=HELD_OUT_SPLIT,
            reference_code=bytes.fromhex("d1 e0"),
            equivalent_code=bytes.fromhex("01 c0"),
            near_miss_code=bytes.fromhex("d1 c8"),
        ),
        TransformationFamily(
            family_id="low_byte_mask",
            motif="low_byte_projection",
            split=HELD_OUT_SPLIT,
            reference_code=bytes.fromhex("25 ff 00 00 00"),
            equivalent_code=bytes.fromhex("0f b6 c0"),
            near_miss_code=bytes.fromhex("25 fe 00 00 00"),
        ),
    )


def _sha256_bytes(value: bytes) -> str:
    """Return a stable digest for a transparent byte fixture."""

    return hashlib.sha256(value).hexdigest()


def _manifest_sha256(families: Iterable[TransformationFamily]) -> str:
    """Hash the canonical family manifest before executable evaluation."""

    manifest = [family.to_dict() for family in families]
    return _manifest_sha256_rows(manifest)


def _manifest_sha256_rows(families: Iterable[dict[str, object]]) -> str:
    """Hash serialized family rows exactly as they appear in the receipt."""

    manifest = list(families)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float = WILSON_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return a Wilson score interval for one family-level binary endpoint.

    The interval is intentionally descriptive.  The benchmark has only a small
    number of independent source families, so this calculation is a transparent
    uncertainty display rather than a population-level inferential claim.
    """

    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("Wilson successes must lie between zero and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "Wilson confidence level must lie strictly between zero and one"
        )
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return (center - half_width, center + half_width)


def _relation_for_role(role: str) -> str:
    """Return the only relation permitted for one declared pair role."""

    if role == "equivalent":
        return EQUIVALENT_RELATION
    if role == "near_miss":
        return NEAR_MISS_RELATION
    raise ValueError(f"unknown transformation role: {role}")


def _pair_for_role(
    family: TransformationFamily,
    *,
    role: str,
    variant_code: bytes,
    expected_equivalent: bool,
    initial_rax_values: Iterable[int] = INTERVENTION_INITIAL_RAX,
) -> dict[str, object]:
    """Evaluate one family variant and attach split/provenance metadata."""

    pair = TransformationPair(
        pair_id=f"{family.family_id}__{role}",
        motif=family.motif,
        expected_equivalent=expected_equivalent,
        reference_code=family.reference_code,
        variant_code=variant_code,
        relation=_relation_for_role(role),
    )
    row = evaluate_transformation_pair(
        pair,
        initial_rax_values=initial_rax_values,
    )
    row.update(
        {
            "family_id": family.family_id,
            "split": family.split,
            "variant_role": role,
            "reference_program_sha256": _sha256_bytes(family.reference_code),
        }
    )
    return row


def _pair_rows(
    families: Iterable[TransformationFamily],
    *,
    initial_rax_values: Iterable[int] = INTERVENTION_INITIAL_RAX,
) -> list[dict[str, object]]:
    """Evaluate every equivalent and near-miss member of every family."""

    catalog = tuple(initial_rax_values)
    rows: list[dict[str, object]] = []
    for family in families:
        rows.append(
            _pair_for_role(
                family,
                role="equivalent",
                variant_code=family.equivalent_code,
                expected_equivalent=True,
                initial_rax_values=catalog,
            )
        )
        rows.append(
            _pair_for_role(
                family,
                role="near_miss",
                variant_code=family.near_miss_code,
                expected_equivalent=False,
                initial_rax_values=catalog,
            )
        )
    return rows


def _minimum_output_difference(row: dict[str, object]) -> int:
    """Return the smallest absolute final-RAX difference for one pair."""

    interventions = cast(list[object], row["interventions"])
    differences: list[int] = []
    for intervention_value in interventions:
        intervention = cast(dict[str, object], intervention_value)
        reference = cast(dict[str, object], intervention["reference"])
        variant = cast(dict[str, object], intervention["variant"])
        differences.append(
            abs(
                int(cast(int, reference["final_rax"]))
                - int(cast(int, variant["final_rax"]))
            )
        )
    if not differences:
        raise ValueError("a pair must contain at least one intervention")
    return min(differences)


def _summary(pair_rows: list[dict[str, object]]) -> dict[str, object]:
    """Compute descriptive endpoint summaries at the source-family level."""

    equivalent = [row for row in pair_rows if row["variant_role"] == "equivalent"]
    near_miss = [row for row in pair_rows if row["variant_role"] == "near_miss"]
    if not equivalent or not near_miss:
        raise ValueError("benchmark requires equivalent and near-miss rows")
    family_ids = {
        str(row.get("family_id", "")) for row in pair_rows if row.get("family_id")
    }
    if not family_ids or len(family_ids) != len(pair_rows) / 2:
        raise ValueError(
            "benchmark summary requires exactly one equivalent and one near-miss per family"
        )
    blind_near_miss = [
        row for row in near_miss if not bool(row["observation_surface_changes"])
    ]
    equivalent_successes = sum(bool(row["observed_equivalent"]) for row in equivalent)
    near_miss_successes = sum(bool(row["near_miss_separated"]) for row in near_miss)
    uncertainty = {
        "confidence_level": WILSON_CONFIDENCE_LEVEL,
        "unit": "source_program_family",
        "method": (
            "Wilson score interval over one binary endpoint per source-program family"
        ),
        "descriptive_only": True,
        "source_family_count": len(family_ids),
        "equivalence_output_consistency": {
            "successes": equivalent_successes,
            "trials": len(equivalent),
            "interval": list(
                _wilson_interval(
                    equivalent_successes,
                    len(equivalent),
                )
            ),
        },
        "near_miss_separation_rate": {
            "successes": near_miss_successes,
            "trials": len(near_miss),
            "interval": list(_wilson_interval(near_miss_successes, len(near_miss))),
        },
    }
    return {
        "equivalence_output_consistency": fmean(
            bool(row["observed_equivalent"]) for row in equivalent
        ),
        "near_miss_separation_rate": fmean(
            bool(row["near_miss_separated"]) for row in near_miss
        ),
        "equivalent_pair_surface_change_rate": fmean(
            bool(row["observation_surface_changes"]) for row in equivalent
        ),
        "near_miss_pair_surface_change_rate": fmean(
            bool(row["observation_surface_changes"]) for row in near_miss
        ),
        "surface_blind_near_miss_count": len(blind_near_miss),
        "surface_blind_near_miss_rate": len(blind_near_miss) / len(near_miss),
        "near_miss_min_output_difference_mean": fmean(
            _minimum_output_difference(row) for row in near_miss
        ),
        "uncertainty": uncertainty,
    }


def representation_invariance_payload(
    families: Iterable[TransformationFamily] | None = None,
) -> dict[str, object]:
    """Return the source-family-disjoint P3 benchmark and protocol diagnostics."""

    family_specs = tuple(_family_specs() if families is None else families)
    if not family_specs:
        raise ValueError("at least one transformation family is required")
    family_ids = [family.family_id for family in family_specs]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("transformation family IDs must be unique")
    unknown_splits = sorted(
        {family.split for family in family_specs} - set(BENCHMARK_SPLITS)
    )
    if unknown_splits:
        raise ValueError(f"unsupported benchmark splits: {unknown_splits}")
    pair_rows = _pair_rows(family_specs)
    negative_control_pair_rows = _pair_rows(
        family_specs,
        initial_rax_values=NON_SEPARATING_INTERVENTION_INITIAL_RAX,
    )
    families_by_split = {
        split: sorted(
            family.family_id for family in family_specs if family.split == split
        )
        for split in BENCHMARK_SPLITS
    }
    summary = _summary(pair_rows)
    return {
        "schema_version": REPRESENTATION_INVARIANCE_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "architecture": "x86_64",
        "output_relation": "final_RAX",
        "initial_state_interventions": [
            int(value) for value in INTERVENTION_INITIAL_RAX
        ],
        "intervention_catalogs": {
            "primary": {
                "catalog_id": "boundary_four_state_v1",
                "values": [int(value) for value in INTERVENTION_INITIAL_RAX],
                "purpose": "declared primary witness catalog",
            },
            "non_separating_negative_control": {
                "catalog_id": "single_zero_state_v1",
                "values": [
                    int(value) for value in NON_SEPARATING_INTERVENTION_INITIAL_RAX
                ],
                "purpose": (
                    "deliberately insufficient catalog; at least one declared "
                    "near miss should remain unresolved"
                ),
            },
        },
        "sampling_unit": "source_program_family",
        "family_split_policy": (
            "all variants of a transformation family remain in one split; "
            "development and held-out family IDs are disjoint"
        ),
        "families_by_split": families_by_split,
        "source_family_disjoint": not (
            set(families_by_split[DEVELOPMENT_SPLIT])
            & set(families_by_split[HELD_OUT_SPLIT])
        ),
        "family_manifest_sha256": _manifest_sha256(family_specs),
        "family_count": len(family_specs),
        "pair_count": len(pair_rows),
        "equivalent_pair_count": sum(
            row["variant_role"] == "equivalent" for row in pair_rows
        ),
        "near_miss_pair_count": sum(
            row["variant_role"] == "near_miss" for row in pair_rows
        ),
        "primary_endpoints": [
            {
                "name": "equivalence_output_consistency",
                "estimand": "mean observed-equivalent indicator over source families",
                "unit": "source_program_family",
                "interpretation": "protocol witness consistency only",
            },
            {
                "name": "near_miss_separation_rate",
                "estimand": "mean separated indicator over source families",
                "unit": "source_program_family",
                "interpretation": "declared-output separation only",
            },
            {
                "name": "invariance_gap",
                "estimand": "future model endpoint: transformed-vs-reference posterior divergence",
                "unit": "source family",
                "interpretation": "not computed by this protocol receipt",
            },
            {
                "name": "separation_gap",
                "estimand": "future model endpoint: near-miss-vs-equivalent posterior separation",
                "unit": "source family",
                "interpretation": "not computed by this protocol receipt",
            },
        ],
        "summary": summary,
        "negative_control": {
            "catalog_id": "single_zero_state_v1",
            "initial_state_interventions": [
                int(value) for value in NON_SEPARATING_INTERVENTION_INITIAL_RAX
            ],
            "summary": _summary(negative_control_pair_rows),
            "pairs": negative_control_pair_rows,
            "interpretation": (
                "The one-state catalog is intentionally not separating: unresolved "
                "near misses are an expected control result, not a model failure."
            ),
        },
        "families": [family.to_dict() for family in family_specs],
        "pairs": pair_rows,
        "scope": (
            "source-family-disjoint transparent x86-64 protocol diagnostics; "
            "no fitted model, latent semantic discovery, complete-ISA equivalence, "
            "cross-architecture transfer, or posterior invariance claim"
        ),
    }


def _expected_summary(pair_rows: list[dict[str, object]]) -> dict[str, object]:
    """Recompute summary endpoints during validation."""

    return _summary(pair_rows)


def _decode_manifest_hex(value: object) -> bytes | None:
    """Decode one canonical lowercase hexadecimal manifest field."""

    if not isinstance(value, str) or not value or len(value) % 2:
        return None
    if any(character not in "0123456789abcdef" for character in value):
        return None
    try:
        return bytes.fromhex(value)
    except ValueError:
        return None


def _is_uint64(value: object) -> bool:
    """Return whether a receipt value is an unsigned 64-bit integer."""

    return (
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= U64_MAX
    )


def validate_representation_invariance_payload(
    payload: dict[str, object],
) -> tuple[bool, str]:
    """Validate provenance, witnesses, split membership, and endpoint receipts.

    Validation is intentionally structural rather than an executor for arbitrary
    payload bytes.  Canonical receipts are produced by
    :func:`representation_invariance_payload`, which uses the real Capstone and
    Unicorn adapters; this validator then proves that the serialized receipt has
    not drifted away from its declared family manifest or witness rows.
    """

    if payload.get("schema_version") != REPRESENTATION_INVARIANCE_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    if payload.get("initial_state_interventions") != list(INTERVENTION_INITIAL_RAX):
        return False, "primary intervention catalog differs from canonical protocol"
    intervention_catalogs = payload.get("intervention_catalogs")
    if not isinstance(intervention_catalogs, dict):
        return False, "intervention_catalogs must be an object"
    primary_catalog = intervention_catalogs.get("primary")
    negative_catalog = intervention_catalogs.get("non_separating_negative_control")
    if (
        not isinstance(primary_catalog, dict)
        or primary_catalog.get("catalog_id") != "boundary_four_state_v1"
    ):
        return False, "primary intervention catalog metadata is invalid"
    if primary_catalog.get("values") != list(INTERVENTION_INITIAL_RAX):
        return False, "primary intervention catalog metadata differs from canonical"
    if (
        not isinstance(negative_catalog, dict)
        or negative_catalog.get("catalog_id") != "single_zero_state_v1"
    ):
        return False, "negative-control intervention catalog metadata is invalid"
    if negative_catalog.get("values") != list(NON_SEPARATING_INTERVENTION_INITIAL_RAX):
        return False, "negative-control intervention catalog metadata differs"
    metadata_errors: list[str] = []
    for field, expected in (
        ("benchmark_id", BENCHMARK_ID),
        ("architecture", "x86_64"),
        ("output_relation", "final_RAX"),
        ("sampling_unit", "source_program_family"),
    ):
        if payload.get(field) != expected:
            metadata_errors.append(f"{field} differs from canonical protocol")
    families = payload.get("families")
    pairs = payload.get("pairs")
    if not isinstance(families, list) or not families:
        return False, "families must be a non-empty list"
    if not isinstance(pairs, list) or not pairs:
        return False, "pairs must be a non-empty list"
    errors: list[str] = list(metadata_errors)
    family_rows: list[dict[str, object]] = []
    family_ids = [
        str(cast(dict[str, object], family).get("family_id", ""))
        for family in families
        if isinstance(family, dict)
    ]
    if len(family_ids) != len(families) or len(family_ids) != len(set(family_ids)):
        errors.append("family IDs must be unique and present")
    family_by_id: dict[str, dict[str, object]] = {}
    family_code_bytes: dict[str, dict[str, bytes]] = {}
    for family_value in families:
        if not isinstance(family_value, dict):
            errors.append("non-object family")
            continue
        family = cast(dict[str, object], family_value)
        family_rows.append(family)
        family_id = str(family.get("family_id", ""))
        if family_id in family_by_id:
            continue
        family_by_id[family_id] = family
        if not family_id or not isinstance(family.get("motif"), str):
            errors.append(f"{family_id}:family metadata missing")
        if family.get("split") not in BENCHMARK_SPLITS:
            errors.append(f"{family_id}:unknown-family-split")
        decoded_codes: dict[str, bytes] = {}
        for field in (
            "reference_code_hex",
            "equivalent_code_hex",
            "near_miss_code_hex",
        ):
            decoded = _decode_manifest_hex(family.get(field))
            if decoded is None:
                errors.append(f"{family_id}:{field}-invalid-hex")
            else:
                decoded_codes[field] = decoded
        if len(decoded_codes) == 3:
            family_code_bytes[family_id] = decoded_codes
    families_by_split = payload.get("families_by_split")
    if not isinstance(families_by_split, dict):
        errors.append("families_by_split must be an object")
        observed_split_ids: dict[str, set[object]] = {
            split: set() for split in BENCHMARK_SPLITS
        }
    else:
        if set(families_by_split) != set(BENCHMARK_SPLITS):
            errors.append("families_by_split keys differ from canonical splits")
        observed_split_ids = {}
        for split in BENCHMARK_SPLITS:
            values = families_by_split.get(split)
            if not isinstance(values, list):
                errors.append(f"families_by_split[{split}] must be a list")
                observed_split_ids[split] = set()
                continue
            values_set = set(values)
            if len(values_set) != len(values):
                errors.append(f"families_by_split[{split}] contains duplicate IDs")
            observed_split_ids[split] = values_set
        if observed_split_ids[DEVELOPMENT_SPLIT] & observed_split_ids[HELD_OUT_SPLIT]:
            errors.append("source-family split overlap")
        if set().union(*observed_split_ids.values()) != set(family_ids):
            errors.append("families_by_split does not cover family IDs")
        for family in family_rows:
            family_id = str(family.get("family_id", ""))
            family_split = family.get("split")
            if (
                isinstance(family_split, str)
                and family_split in BENCHMARK_SPLITS
                and family_id not in observed_split_ids[family_split]
            ):
                errors.append(f"{family_id}:family split does not match split manifest")
            if (
                isinstance(family_split, str)
                and family_split in BENCHMARK_SPLITS
                and any(
                    family_id in observed_split_ids[other]
                    for other in BENCHMARK_SPLITS
                    if other != family_split
                )
            ):
                errors.append(f"{family_id}:family appears in the wrong split manifest")
    if family_rows and payload.get("family_manifest_sha256") != _manifest_sha256_rows(
        family_rows
    ):
        errors.append("family_manifest_sha256 mismatch")
    if payload.get("source_family_disjoint") is not True:
        errors.append("source_family_disjoint must be true")
    endpoint_names = {
        str(cast(dict[str, object], endpoint).get("name", ""))
        for endpoint in cast(list[object], payload.get("primary_endpoints", []))
        if isinstance(endpoint, dict)
    }
    if (
        not {
            "equivalence_output_consistency",
            "near_miss_separation_rate",
            "invariance_gap",
            "separation_gap",
        }
        <= endpoint_names
    ):
        errors.append("primary endpoint registry is incomplete")
    pair_ids: list[str] = []
    pair_family_roles: dict[str, set[str]] = {}

    def validate_pair_rows(
        rows: list[object],
        *,
        expected_catalog: list[int],
        require_near_miss_separation: bool,
        label: str,
    ) -> dict[str, set[str]]:
        """Validate one primary or negative-control witness collection."""

        roles_by_family: dict[str, set[str]] = {}
        local_ids: list[str] = []
        for pair_value in rows:
            if not isinstance(pair_value, dict):
                errors.append(f"{label} contains a non-object pair")
                continue
            pair = cast(dict[str, object], pair_value)
            pair_id = str(pair.get("pair_id", ""))
            family_id = str(pair.get("family_id", ""))
            role = str(pair.get("variant_role", ""))
            local_ids.append(pair_id)
            roles_by_family.setdefault(family_id, set()).add(role)
            if family_id not in family_by_id:
                errors.append(f"{pair_id}:unknown-family")
            if role not in {"equivalent", "near_miss"}:
                errors.append(f"{pair_id}:unknown-role")
            if pair.get("split") not in BENCHMARK_SPLITS:
                errors.append(f"{pair_id}:unknown-split")
            family = family_by_id.get(family_id)
            if family is not None:
                if pair.get("split") != family.get("split"):
                    errors.append(f"{pair_id}:pair split does not match family split")
                if pair.get("motif") != family.get("motif"):
                    errors.append(f"{pair_id}:motif does not match family manifest")
            if role in {"equivalent", "near_miss"}:
                expected_equivalent = role == "equivalent"
                if not isinstance(pair.get("expected_equivalent"), bool):
                    errors.append(f"{pair_id}:expected_equivalent is not boolean")
                elif pair["expected_equivalent"] != expected_equivalent:
                    errors.append(f"{pair_id}:expected_equivalent-role mismatch")
                if pair.get("relation") != _relation_for_role(role):
                    errors.append(f"{pair_id}:relation-role mismatch")
                if family is not None and pair_id != f"{family_id}__{role}":
                    errors.append(f"{pair_id}:pair ID-role mismatch")
            if pair.get("initial_state_interventions") != expected_catalog:
                errors.append(f"{pair_id}:intervention-catalog")
                if label == "negative_control":
                    errors.append("negative_control pair catalog mismatch")
            if not isinstance(pair.get("observed_equivalent"), bool):
                errors.append(f"{pair_id}:observed-equivalent is not boolean")
            if not isinstance(pair.get("near_miss_separated"), bool):
                errors.append(f"{pair_id}:near-miss-separated is not boolean")
            if not isinstance(pair.get("observation_surface_changes"), bool):
                errors.append(f"{pair_id}:surface-change flag is not boolean")
            if role == "equivalent" and pair.get("observed_equivalent") is not True:
                errors.append(f"{pair_id}:equivalent-witness-failed")
            if (
                role == "near_miss"
                and require_near_miss_separation
                and pair.get("near_miss_separated") is not True
            ):
                errors.append(f"{pair_id}:near-miss-not-separated")

            if family is not None and role in {"equivalent", "near_miss"}:
                family_codes = family_code_bytes.get(family_id, {})
                expected_reference_hex = family.get("reference_code_hex")
                expected_variant_hex = family.get(
                    "equivalent_code_hex"
                    if role == "equivalent"
                    else "near_miss_code_hex"
                )
                if pair.get("reference_code_hex") != expected_reference_hex:
                    errors.append(f"{pair_id}:reference code drift")
                if pair.get("variant_code_hex") != expected_variant_hex:
                    errors.append(f"{pair_id}:variant code drift")
                expected_reference = family_codes.get("reference_code_hex")
                expected_variant = family_codes.get(
                    "equivalent_code_hex"
                    if role == "equivalent"
                    else "near_miss_code_hex"
                )
                if expected_reference is not None and pair.get(
                    "reference_sha256"
                ) != _sha256_bytes(expected_reference):
                    errors.append(f"{pair_id}:reference SHA-256 drift")
                if expected_reference is not None and pair.get(
                    "reference_program_sha256"
                ) != _sha256_bytes(expected_reference):
                    errors.append(f"{pair_id}:reference program SHA-256 drift")
                if expected_variant is not None and pair.get(
                    "variant_sha256"
                ) != _sha256_bytes(expected_variant):
                    errors.append(f"{pair_id}:variant SHA-256 drift")

            intervention_rows = pair.get("interventions")
            if not isinstance(intervention_rows, list) or len(intervention_rows) != len(
                expected_catalog
            ):
                errors.append(f"{pair_id}:intervention-row-count")
                continue
            equality_rows: list[bool] = []
            surface_changes: list[bool] = []
            rows_well_formed = True
            for intervention, expected_initial in zip(
                intervention_rows, expected_catalog, strict=True
            ):
                if not isinstance(intervention, dict):
                    errors.append(f"{pair_id}:intervention-row-object")
                    rows_well_formed = False
                    continue
                intervention_row = cast(dict[str, object], intervention)
                if intervention_row.get("initial_rax") != expected_initial:
                    errors.append(f"{pair_id}:intervention-alignment")
                    rows_well_formed = False
                reference = intervention_row.get("reference")
                variant = intervention_row.get("variant")
                if not isinstance(reference, dict) or not isinstance(variant, dict):
                    errors.append(f"{pair_id}:intervention-program-record")
                    rows_well_formed = False
                    continue
                reference_record = cast(dict[str, object], reference)
                variant_record = cast(dict[str, object], variant)
                if (
                    reference_record.get("initial_rax") != expected_initial
                    or variant_record.get("initial_rax") != expected_initial
                ):
                    errors.append(f"{pair_id}:program-record-alignment")
                    rows_well_formed = False
                reference_final = reference_record.get("final_rax")
                variant_final = variant_record.get("final_rax")
                if not _is_uint64(reference_final) or not _is_uint64(variant_final):
                    errors.append(f"{pair_id}:final-RAX-out-of-range")
                    rows_well_formed = False
                    continue
                final_equal = reference_final == variant_final
                if intervention_row.get("final_rax_equal") != final_equal:
                    errors.append(f"{pair_id}:final-RAX equality drift")
                    rows_well_formed = False
                equality_rows.append(final_equal)
                reference_categories = reference_record.get("observation_categories")
                variant_categories = variant_record.get("observation_categories")
                if not isinstance(reference_categories, list) or not isinstance(
                    variant_categories, list
                ):
                    errors.append(f"{pair_id}:observation-category record invalid")
                    rows_well_formed = False
                    continue
                surface_changes.append(reference_categories != variant_categories)
            if rows_well_formed and len(equality_rows) == len(expected_catalog):
                observed_equivalent = all(equality_rows)
                if pair.get("observed_equivalent") != observed_equivalent:
                    errors.append(f"{pair_id}:observed-equivalent drift")
                if pair.get("near_miss_separated") != (not observed_equivalent):
                    errors.append(f"{pair_id}:near-miss-separated drift")
                if pair.get("observation_surface_changes") != any(surface_changes):
                    errors.append(f"{pair_id}:surface-change drift")
                if role == "equivalent" and not observed_equivalent:
                    errors.append(f"{pair_id}:equivalent-witness-failed")
                if (
                    role == "near_miss"
                    and require_near_miss_separation
                    and not observed_equivalent
                ):
                    pass
                elif role == "near_miss" and require_near_miss_separation:
                    errors.append(f"{pair_id}:near-miss-not-separated")
        if len(local_ids) != len(set(local_ids)):
            errors.append(f"{label} duplicate pair IDs")
        return roles_by_family

    pair_family_roles = validate_pair_rows(
        pairs,
        expected_catalog=list(INTERVENTION_INITIAL_RAX),
        require_near_miss_separation=True,
        label="primary pairs",
    )
    pair_ids = [
        str(cast(dict[str, object], pair).get("pair_id", ""))
        for pair in pairs
        if isinstance(pair, dict)
    ]
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("duplicate pair IDs")
    if any(
        roles != {"equivalent", "near_miss"} for roles in pair_family_roles.values()
    ):
        errors.append("each family must contain one equivalent and one near-miss role")
    if set(pair_family_roles) != set(family_ids):
        errors.append("pair family coverage mismatch")
    expected_pair_count = len(pairs)
    if payload.get("pair_count") != expected_pair_count:
        errors.append("pair_count mismatch")
    if payload.get("family_count") != len(families):
        errors.append("family_count mismatch")
    try:
        expected_summary = _expected_summary(
            [cast(dict[str, object], pair) for pair in pairs]
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary recomputation failed: {exc}")
        expected_summary = {}
    if payload.get("summary") != expected_summary:
        errors.append("summary mismatch")
    negative_control = payload.get("negative_control")
    if not isinstance(negative_control, dict):
        errors.append("negative_control must be an object")
    else:
        negative_values = negative_control.get("initial_state_interventions")
        negative_pairs = negative_control.get("pairs")
        negative_summary = negative_control.get("summary")
        expected_negative_values = list(NON_SEPARATING_INTERVENTION_INITIAL_RAX)
        if negative_control.get("catalog_id") != "single_zero_state_v1":
            errors.append("negative_control catalog ID mismatch")
        if negative_values != expected_negative_values:
            errors.append("negative_control intervention catalog mismatch")
        if not isinstance(negative_pairs, list) or not negative_pairs:
            errors.append("negative_control pairs must be a non-empty list")
        else:
            negative_pair_dicts = [
                cast(dict[str, object], pair)
                for pair in negative_pairs
                if isinstance(pair, dict)
            ]
            negative_family_roles = validate_pair_rows(
                list(negative_pairs),
                expected_catalog=expected_negative_values,
                require_near_miss_separation=False,
                label="negative_control",
            )
            if set(negative_family_roles) != set(family_ids):
                errors.append("negative_control family coverage mismatch")
            if any(
                roles != {"equivalent", "near_miss"}
                for roles in negative_family_roles.values()
            ):
                errors.append("negative_control role coverage mismatch")
            try:
                expected_negative_summary = _expected_summary(negative_pair_dicts)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"negative_control summary recomputation failed: {exc}")
            else:
                if negative_summary != expected_negative_summary:
                    errors.append("negative_control summary mismatch")
            if not any(
                not bool(pair.get("near_miss_separated"))
                for pair in negative_pair_dicts
                if pair.get("variant_role") == "near_miss"
            ):
                errors.append("negative_control did not retain an unresolved near miss")
    if errors:
        return False, "; ".join(errors[:8])
    return (
        True,
        f"families={len(families)} pairs={len(pairs)} source_family_disjoint=true",
    )


def write_representation_invariance_benchmark(
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Write the deterministic source-family benchmark receipt."""

    path = project_root / REPRESENTATION_INVARIANCE_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = representation_invariance_payload()
    valid, detail = validate_representation_invariance_payload(payload)
    if not valid:
        raise ValueError(f"canonical representation benchmark is invalid: {detail}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return REPRESENTATION_INVARIANCE_ARTIFACT


__all__ = [
    "BENCHMARK_SPLITS",
    "DEVELOPMENT_SPLIT",
    "HELD_OUT_SPLIT",
    "NON_SEPARATING_INTERVENTION_INITIAL_RAX",
    "REPRESENTATION_INVARIANCE_ARTIFACT",
    "REPRESENTATION_INVARIANCE_SCHEMA_VERSION",
    "TransformationFamily",
    "representation_invariance_payload",
    "validate_representation_invariance_payload",
    "write_representation_invariance_benchmark",
]
