"""Versioned scientific protocol shared by the open-roadmap producers.

The existing package has several independently validated research tranches.
This module supplies the missing common contract for the next tranche.  It is
deliberately data-only: it freezes assumptions and provenance without turning
any of the labels into learned semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import EVIDENCE_STATUSES, RECEIPT_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_SCHEMA_VERSION = "2.0.0"
PROTOCOL_ID = "active-assembly-open-roadmap-v2"
PROTOCOL_ARTIFACT = "output/data/research_protocol.json"

MODEL_FAMILIES = (
    "calibrated_interpretable",
    "temporal_probabilistic",
    "neural_sequence",
)
PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")
SEEDS = (11, 23, 47, 89, 131)
BOOTSTRAP_REPLICATES = 1000
PRACTICAL_MARGIN = 0.05
CONFIDENCE_LEVEL = 0.95
SENSOR_CERTIFICATE_TOLERANCE = 0.05

SUPPORTED_X86_USER_MODE = (
    "mov",
    "movabs",
    "movzx",
    "movsx",
    "lea",
    "add",
    "sub",
    "adc",
    "sbb",
    "inc",
    "dec",
    "neg",
    "not",
    "and",
    "or",
    "xor",
    "cmp",
    "test",
    "shl",
    "shr",
    "sal",
    "sar",
    "jmp",
    "jcc",
    "call",
    "ret",
    "push",
    "pop",
    "nop",
)

UNSUPPORTED_X86_CLASSES = (
    "simd",
    "floating_point",
    "privileged_or_system",
    "self_modifying_code",
    "unbounded_aliasing",
    "unmodeled_exception_or_undefined_behavior",
)


@dataclass(frozen=True)
class ScientificProtocol:
    """Immutable protocol values serialized into every new receipt."""

    protocol_id: str
    schema_version: str
    observational_contract: dict[str, Any]
    quotient_contract: dict[str, Any]
    split_contract: dict[str, Any]
    model_contract: dict[str, Any]
    statistics_contract: dict[str, Any]
    sensing_contract: dict[str, Any]
    formal_contract: dict[str, Any]
    evidence_contract: dict[str, Any]
    receipt_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe protocol object."""

        return asdict(self)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_scientific_protocol() -> ScientificProtocol:
    """Build the frozen protocol used by all v2 research producers."""

    return ScientificProtocol(
        protocol_id=PROTOCOL_ID,
        schema_version=PROTOCOL_SCHEMA_VERSION,
        observational_contract={
            "label_blind": True,
            "missing_token": "__MISSING__",
            "unseen_token": "__UNKNOWN__",
            "memory_model": "bounded_byte_addressable_fixture_memory",
            "initial_register_interventions": [
                0,
                1,
                2,
                2**31 - 1,
                2**31,
                2**32 - 1,
                2**63,
                2**64 - 1,
            ],
            "sensor_fields": [
                "instruction_family",
                "operand_class",
                "register_read_set",
                "register_write_set",
                "flag_effect",
                "memory_addressing_mode",
                "branch_target_class",
                "dynamic_status",
            ],
        },
        quotient_contract={
            "label_source": "formal_output_relation",
            "endpoint": "same_formal_quotient",
            "relation": "equal_terminal_observation_under_all_declared_interventions",
            "positive_transformations": "source-bound_or_declared_transparent_fixture",
            "negative_controls": [
                "one_state_non_separating_catalog",
                "one_observationally_material_near_miss",
                "family_label_permutation",
            ],
            "labels_are_not": [
                "mnemonic_names",
                "hand_authored_phase_labels",
                "model_predictions",
            ],
        },
        split_contract={
            "unit": "source_program_family",
            "training_fraction": 0.5,
            "development_fraction": 0.25,
            "test_fraction": 0.25,
            "held_out_family_required": True,
            "three_way_partition": "training/development/held_out; no family overlap",
            "minimum_independent_families": 64,
            "minimum_examples_per_family": 64,
            "minimum_held_out_families": 16,
            "minimum_equivalent_rows_per_family": 32,
            "minimum_near_miss_rows_per_family": 32,
            "minimum_transformations_per_family": 4,
            "minimum_near_miss_controls_per_family": 4,
            "real_program_minimum_projects": 5,
            "real_program_minimum_labelable_examples": 100,
        },
        model_contract={
            "families": list(MODEL_FAMILIES),
            "co_primary": True,
            "seeds": list(SEEDS),
            "hyperparameters_use_development_only": True,
            "calibrated_interpretable": {
                "algorithm": "regularized_binary_logistic_gradient_descent",
                "feature_contract": "typed_trace_summary_without_quotient_label",
                "stopping_rule": "fixed_80_epoch_budget_no_test_adaptation",
            },
            "temporal_probabilistic": {
                "algorithm": "class_conditional_first_order_markov_likelihood",
                "feature_contract": "typed_trace_token_sequence",
                "stopping_rule": "closed_form_smoothed_counts_no_test_adaptation",
            },
            "neural_sequence": {
                "algorithm": "one_layer_tanh_rnn_cpu_numpy",
                "hidden_size": 16,
                "optimizer": "fixed_step_gradient_descent",
                "epochs": 8,
                "stopping_rule": "fixed_8_epoch_budget_no_test_adaptation",
            },
            "surface_baselines": [
                "opcode_histogram",
                "byte_ngrams_hashed_128_bins",
                "mnemonic_counts",
                "instruction_length",
                "control_flow_summary",
                "memory_event_counts",
            ],
        },
        statistics_contract={
            "primary_metrics": list(PRIMARY_METRICS),
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap": "source_family_clustered_percentile",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "calibration_bins": 10,
            "practical_comparison_margin": PRACTICAL_MARGIN,
            "multiplicity": "Holm across model-by-primary-metric comparisons",
            "paired_comparison_sampling_unit": "source_family",
            "no_post_hoc_winner_promotion": True,
        },
        sensing_contract={
            "primary_cost": "deterministic_vector",
            "cost_components": [
                "sensor_invocations",
                "decoded_instructions",
                "bounded_execution_steps",
                "register_flag_observations",
                "branch_observations",
                "memory_hook_events",
                "serialized_bytes",
                "unsupported_attempts",
            ],
            "calibration": "optional_fixed-hardware_wall_clock_receipt",
            "certificate_tolerance": SENSOR_CERTIFICATE_TOLERANCE,
            "certificate_scope": "frozen_finite_state_action_space_only",
        },
        formal_contract={
            "architecture": "x86_64",
            "scope": "broad_declared_user_mode_integer_subset",
            "supported_mnemonics": list(SUPPORTED_X86_USER_MODE),
            "unsupported_classes": list(UNSUPPORTED_X86_CLASSES),
            "state": [
                "program_counter",
                "all_16_64_bit_general_purpose_registers",
                "modeled_flags_with_definedness",
                "bounded_byte_addressable_memory",
                "bounded_stack",
                "decoded_instruction_and_step_index",
                "explicit_exceptional_or_unsupported_status",
            ],
            "memory_window_bytes": 65536,
            "stack_window_bytes": 4096,
            "maximum_steps": 256,
            "operand_widths": [8, 16, 32, 64],
            "modeled_flags": ["CF", "PF", "AF", "ZF", "SF", "OF"],
            "correspondence": "Lean_reference_python_trace_bounded_Unicorn_fixture",
        },
        evidence_contract={
            "canonical_fixture_policy": "transparent_repository_owned_bytes_only",
            "external_track": "license_cleared_source_local_compile_static_analysis",
            "arbitrary_external_execution": False,
            "network_required_for_replay": False,
            "application_claim_gate": "external_validity_receipt_and_independent_replay",
        },
        receipt_contract={
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "required_fields": [
                "schema_version",
                "protocol_id",
                "protocol_sha256",
                "generator_revision",
                "input_hashes",
                "toolchain_versions",
                "status",
                "limitations",
                "unsupported_cases",
                "underpowered_cases",
            ],
            "status_vocabulary": sorted(EVIDENCE_STATUSES),
            "mixed_protocol_outputs_allowed": False,
            "v1_migration": "read_only_diagnostic",
        },
    )


def scientific_protocol_payload() -> dict[str, Any]:
    """Return the protocol with a machine-independent content hash.

    The protocol hash covers the frozen scientific contract only.  Runtime
    provenance such as Python/platform and package versions remains in the
    payload but is deliberately not hashed into the protocol identity; those
    values are receipts of the executing environment, not changes to the
    experiment definition.  This lets a Linux replay validate the same
    protocol as the macOS producer while still recording both toolchains.
    """

    protocol_body = build_scientific_protocol().to_dict()
    protocol_hash = _canonical_hash(protocol_body)
    body = dict(protocol_body)
    # ``schema_version`` belongs to the protocol itself.  The explicit receipt
    # key keeps the same provenance contract available to domain artifacts that
    # legitimately use a different local schema version.
    body["receipt_schema_version"] = RECEIPT_SCHEMA_VERSION
    body["generator_revision"] = generator_revision(Path(__file__))
    body["input_hashes"] = {"src/scientific_protocol.py": sha256_path(Path(__file__))}
    body["toolchain_versions"] = toolchain_versions(("capstone", "numpy", "unicorn"))
    body["status"] = "validated"
    body["artifact_id"] = PROTOCOL_ARTIFACT
    body["producer"] = "src/scientific_protocol.py"
    body["limitations"] = [
        "The protocol freezes a bounded research contract; it is not evidence of any scientific endpoint by itself.",
        "Unavailable compiler targets and unsupported instruction effects remain explicit rather than imputed.",
    ]
    body["unsupported_cases"] = list(UNSUPPORTED_X86_CLASSES)
    body["underpowered_cases"] = [
        "Independent replication, external-validity, and broad-theorem gates remain unearned until their receipts pass.",
    ]
    body["protocol_sha256"] = protocol_hash
    return body


def validate_scientific_protocol_payload(payload: object) -> tuple[bool, str]:
    """Validate structure and the self-contained protocol digest."""

    if not isinstance(payload, dict):
        return False, "scientific protocol must be an object"
    expected = scientific_protocol_payload()
    canonical_protocol = build_scientific_protocol().to_dict()
    for key in canonical_protocol:
        if key not in payload:
            return False, f"scientific protocol missing {key}"
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        return False, "scientific protocol schema mismatch"
    recorded_hash = payload.get("protocol_sha256")
    if recorded_hash != _canonical_hash(canonical_protocol):
        return False, "scientific protocol hash mismatch"
    for key, expected_value in canonical_protocol.items():
        if payload.get(key) != expected_value:
            return False, f"scientific protocol {key} drift"
    if payload.get("protocol_id") != PROTOCOL_ID:
        return False, "scientific protocol id drift"
    if payload.get("model_contract", {}).get("families") != list(MODEL_FAMILIES):
        return False, "scientific protocol model family drift"
    if payload.get("statistics_contract", {}).get("primary_metrics") != list(
        PRIMARY_METRICS
    ):
        return False, "scientific protocol primary metric drift"
    if not payload.get("observational_contract", {}).get("label_blind"):
        return False, "scientific protocol must remain label-blind"
    if payload.get("evidence_contract", {}).get("arbitrary_external_execution"):
        return False, "arbitrary external execution must remain disabled"
    if payload.get("status") not in EVIDENCE_STATUSES:
        return False, "scientific protocol evidence status is invalid"
    if (
        payload.get("receipt_contract", {}).get("schema_version")
        != RECEIPT_SCHEMA_VERSION
    ):
        return False, "scientific protocol receipt contract drift"
    if (
        payload.get("receipt_contract", {}).get("mixed_protocol_outputs_allowed")
        is not False
    ):
        return False, "mixed protocol outputs must remain disabled"
    if payload.get("underpowered_cases") == []:
        return False, "protocol must retain current unearned gates"
    # Source identity and fixed metadata must remain reproducible.  Toolchain
    # metadata is intentionally allowed to differ across a host and a replay
    # guest; it is a capability receipt, not part of the protocol identity.
    for key in (
        "generator_revision",
        "input_hashes",
        "receipt_schema_version",
        "artifact_id",
        "producer",
        "limitations",
        "unsupported_cases",
        "underpowered_cases",
    ):
        if payload.get(key) != expected.get(key):
            return (
                False,
                f"scientific protocol {key} is stale relative to the canonical producer",
            )
    if not isinstance(payload.get("toolchain_versions"), dict):
        return False, "scientific protocol toolchain receipt is invalid"
    return True, "ok"


def write_scientific_protocol(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the source-bound research protocol receipt."""

    path = project_root / PROTOCOL_ARTIFACT
    # Import lazily to avoid a module cycle: artifact_contract consumes the
    # canonical protocol when sealing the broader project data surface.
    from .artifact_contract import write_json_atomic

    write_json_atomic(path, scientific_protocol_payload())
    return path
