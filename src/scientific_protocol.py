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

from .provenance import generator_revision, toolchain_versions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "active-assembly-open-roadmap-v1"
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe protocol object."""

        return asdict(self)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_scientific_protocol() -> ScientificProtocol:
    """Build the frozen protocol used by P3--P7 producers."""

    return ScientificProtocol(
        protocol_id=PROTOCOL_ID,
        schema_version=PROTOCOL_SCHEMA_VERSION,
        observational_contract={
            "label_blind": True,
            "missing_token": "__MISSING__",
            "unseen_token": "__UNKNOWN__",
            "memory_model": "bounded_byte_addressable_fixture_memory",
            "initial_register_interventions": [0, 1, 2**32 - 1, 2**64 - 1],
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
            "minimum_independent_families": 8,
            "minimum_examples": 64,
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
        },
        statistics_contract={
            "primary_metrics": list(PRIMARY_METRICS),
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap": "source_family_clustered_percentile",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "calibration_bins": 10,
            "practical_comparison_margin": PRACTICAL_MARGIN,
            "multiplicity": "Holm across model-by-primary-metric comparisons",
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
                "general_purpose_registers",
                "CF_ZF_SF_OF_flags_when_declared",
                "bounded_memory",
                "bounded_stack",
            ],
            "correspondence": "Lean_reference_python_trace_bounded_Unicorn_fixture",
        },
        evidence_contract={
            "canonical_fixture_policy": "transparent_repository_owned_bytes_only",
            "external_track": "license_cleared_source_local_compile_static_analysis",
            "arbitrary_external_execution": False,
            "network_required_for_replay": False,
            "application_claim_gate": "external_validity_receipt_and_independent_replay",
        },
    )


def scientific_protocol_payload() -> dict[str, Any]:
    """Return the protocol with a stable content hash."""

    body = build_scientific_protocol().to_dict()
    body["generator_revision"] = generator_revision(Path(__file__))
    body["input_hashes"] = {}
    body["toolchain_versions"] = toolchain_versions(("capstone", "numpy", "unicorn"))
    body["protocol_sha256"] = _canonical_hash(body)
    return body


def validate_scientific_protocol_payload(payload: object) -> tuple[bool, str]:
    """Validate structure and the self-contained protocol digest."""

    if not isinstance(payload, dict):
        return False, "scientific protocol must be an object"
    expected = scientific_protocol_payload()
    for key in expected:
        if key not in payload:
            return False, f"scientific protocol missing {key}"
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        return False, "scientific protocol schema mismatch"
    recorded_hash = payload.get("protocol_sha256")
    body = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    if recorded_hash != _canonical_hash(body):
        return False, "scientific protocol hash mismatch"
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
    return True, "ok"


def write_scientific_protocol(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the source-bound research protocol receipt."""

    path = project_root / PROTOCOL_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scientific_protocol_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
