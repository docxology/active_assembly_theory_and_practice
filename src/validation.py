"""Analysis validation report for scholarship, traces, inference, and figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import matplotlib.image as mpimg
import numpy as np

from .artifact_contract import (
    artifact_domain_view,
    validate_data_artifact_surface,
    write_json_artifact,
)
from .assembly_packages.schema import TRACE_SCHEMA_VERSION, binary_hash
from .action_sensing import (
    ACTION_SENSING_SCHEMA_VERSION,
    action_conditioned_sensing_payload,
    validate_action_conditioned_sensing_payload,
)
from .assembly_formalism import validate_assembly_listing_payload
from .build_provenance import verify_build_provenance
from .claim_register import CLAIM_RECORDS
from .corpus_coverage import (
    CORPUS_COVERAGE_SCHEMA_VERSION,
    corpus_coverage_payload,
    validate_corpus_coverage_payload,
)
from .corpus import build_corpus
from .docx_validation import (
    docx_render_payload,
    validate_docx_render_payload,
    validate_docx_render_receipt,
)
from .external_sources import (
    EXTERNAL_CAPABILITY_ARTIFACT,
    EXTERNAL_ACQUISITION_ARTIFACT,
    EXTERNAL_MANIFEST_SCHEMA_VERSION,
    EXTERNAL_STATIC_RECEIPT_ARTIFACT,
    compiler_capabilities_payload,
    validate_external_acquisition_freshness,
    external_source_validation_payload,
    validate_compiler_capabilities_payload,
    validate_external_static_sources,
    validate_external_static_validation_payload,
    validate_external_source_validation_payload,
)
from .external_labeling import (
    LABEL_CONTRACT_ARTIFACT,
    external_label_contract_freshness,
    validate_external_label_contract_payload,
)
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .factorized_model import FACTOR_MODEL_SCHEMA_VERSION
from .information_policy import (
    INFORMATION_POLICY_SCHEMA_VERSION,
    active_information_payload,
    validate_active_information_payload,
)
from .intervention_analysis import (
    INTERVENTION_ANALYSIS_SCHEMA_VERSION,
    intervention_power_payload,
    validate_intervention_power_payload,
)
from .formal_verification import (
    FORMAL_SCHEMA_VERSION,
    formal_verification_payload,
    validate_formal_verification_payload,
)
from .formal_correspondence import (
    FORMAL_CORRESPONDENCE_SCHEMA_VERSION,
    formal_correspondence_payload,
    validate_formal_correspondence_payload,
)
from .compiler_transfer import (
    COMPILER_TRANSFER_SCHEMA_VERSION,
    compiler_transfer_payload,
    validate_compiler_transfer_payload,
)
from .figures import FIGURE_CAPTIONS, plot_validation_summary
from .figure_contract import FIGURE_REGISTRY_SCHEMA_VERSION
from .manuscript_registry import (
    COMPATIBILITY_ONLY_TOKENS,
    MANUSCRIPT_VARIABLE_REGISTRY,
)
from .parameter_learning import (
    PARAMETER_LEARNING_SCHEMA_VERSION,
    parameter_learning_payload,
    validate_parameter_learning_payload,
)
from .multi_arch import (
    CROSS_ARCHITECTURE_SCHEMA_VERSION,
    cross_architecture_payload,
    validate_cross_architecture_payload,
)
from .quotient_invariance import (
    QUOTIENT_PROTOCOL_SCHEMA_VERSION,
    quotient_invariance_payload,
    validate_quotient_invariance_payload,
)
from .representation_invariance import (
    REPRESENTATION_INVARIANCE_SCHEMA_VERSION,
    representation_invariance_payload,
    validate_representation_invariance_payload,
)
from .quotient_classifier import (
    QUOTIENT_CLASSIFIER_SCHEMA_VERSION,
    quotient_classifier_payload,
    validate_quotient_classifier_payload,
)
from .replication import (
    INDEPENDENT_REPLAY_ARTIFACT,
    REPLICATION_SCHEMA_VERSION,
    replication_bundle_payload,
    validate_independent_replay_artifact,
    validate_replication_bundle,
)
from .reproducibility import MANIFEST_TARGETS, verify_reproducibility_manifest
from .rxinfer_bridge import (
    file_sha256,
    load_free_energy_records,
    validate_categorical_rxinfer_outputs,
    validate_free_energy_records,
)
from .scholarship import scholarship_keys, scholarship_payload
from .scientific_program import (
    SCIENTIFIC_PROGRAM_SCHEMA_VERSION,
    scientific_program_payload,
    validate_scientific_program_payload,
)
from .scientific_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    scientific_protocol_payload,
    validate_scientific_protocol_payload,
)
from .sensor_costs import (
    SENSOR_COST_SCHEMA_VERSION,
    sensor_cost_payload,
    validate_sensor_cost_payload,
)
from .theorem_phase import (
    THEOREM_PHASE_SCHEMA_VERSION,
    theorem_or_countertheorem_payload,
    validate_theorem_or_countertheorem_payload,
)
from .symbolic_semantics import SEMANTICS_SCHEMA_VERSION
from .typed_observations import (
    TYPED_OBSERVATION_FIELDS,
    TYPED_OBSERVATION_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MANUSCRIPT_SOURCE_EXCLUDES = {"AGENTS.md", "README.md", "SYNTAX.md"}
MANUSCRIPT_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


@dataclass(frozen=True)
class ValidationCheck:
    """One validation assertion with a compact diagnostic string."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe check row."""

        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for generated analysis outputs."""

    generated_at: str
    checks: tuple[ValidationCheck, ...]
    summary: dict[str, Any]
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "generated_at": self.generated_at,
            "checks": [check.to_dict() for check in self.checks],
            "summary": dict(self.summary),
            "artifact_paths": dict(self.artifact_paths),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_payload_equal(left: object, right: object) -> bool:
    """Compare regenerated domain data while ignoring receipt-only fields."""

    return artifact_domain_view(left) == artifact_domain_view(right)


def _check(name: str, condition: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(condition), detail=detail)


def _trace_checks(trace_path: Path) -> list[ValidationCheck]:
    payload = _load_json(trace_path)
    required = {
        "schema_version",
        "sample_id",
        "architecture",
        "binary_sha256",
        "static_instruction_events",
        "observations",
        "typed_observations",
        "package_provenance",
        "basic_blocks",
    }
    missing = sorted(required - set(payload))
    observations = payload.get("observations", [])
    typed_observations = payload.get("typed_observations", [])
    instructions = payload.get("static_instruction_events", [])
    provenance = {row["name"]: row for row in payload.get("package_provenance", [])}
    required_packages = {"capstone", "unicorn"}
    optional_roadmap_packages = {
        "angr",
        "rizin",
        "radare2",
        "llvm-objdump",
        "ghidra",
        "bap",
        "gtirb",
        "qiling",
        "miasm",
        "dyninst",
        "retdec",
        "mcsema",
        "keystone",
        "nasm",
    }
    try:
        expected_binary_hash = binary_hash(str(payload.get("sample_id")))
    except ValueError:
        expected_binary_hash = None
    observation_semantics_ok = bool(observations) and bool(instructions)
    if observation_semantics_ok:
        for observation in observations:
            index = int(observation.get("index", -1))
            category = str(observation.get("category", ""))
            observation_id = int(observation.get("observation_id", -1))
            if (
                index < 0
                or index >= len(instructions)
                or category not in OBSERVATION_LABELS
                or observation_id != OBSERVATION_LABELS.index(category)
                or observation.get("address") != instructions[index].get("address")
            ):
                observation_semantics_ok = False
                break
    typed_semantics_ok = (
        len(typed_observations) == len(instructions)
        and bool(typed_observations)
        and all(
            isinstance(observation, dict)
            and set(TYPED_OBSERVATION_FIELDS).issubset(observation)
            and not any(
                key in observation for key in ("phase_label", "phase_labels", "truth")
            )
            for observation in typed_observations
        )
    )
    label_vocab_ok = payload.get("observation_labels") == list(
        OBSERVATION_LABELS
    ) and payload.get("hidden_state_labels") == list(HIDDEN_STATE_LABELS)
    dynamic_trace = payload.get("dynamic_register_trace", [])
    instruction_trace_rows = [
        row
        for row in dynamic_trace[1:-1]
        if isinstance(row, dict) and row.get("event") == "instruction"
    ]
    dynamic_trace_ok = (
        len(dynamic_trace) >= 2
        and dynamic_trace[0].get("event") == "initial"
        and dynamic_trace[-1].get("event") == "final"
        and int(dynamic_trace[-1].get("instruction_count", -1))
        == len(instruction_trace_rows)
        and [row.get("instruction_index") for row in instruction_trace_rows]
        == list(range(len(instruction_trace_rows)))
        and all(
            isinstance(row.get("registers_before"), dict)
            and isinstance(row.get("registers_after"), dict)
            and {"rax", "rip", "rsp", "eflags"}.issubset(row["registers_before"])
            and {"rax", "rip", "rsp", "eflags"}.issubset(row["registers_after"])
            for row in instruction_trace_rows
        )
    )
    block_rows = payload.get("basic_blocks", [])
    covered_indexes = [
        int(index)
        for block in block_rows
        if isinstance(block, dict)
        for index in block.get("instruction_indexes", [])
    ]
    block_ids = {
        str(block.get("block_id")) for block in block_rows if isinstance(block, dict)
    }
    edge_rows = payload.get("cfg_edges", [])
    cfg_ok = (
        bool(block_rows)
        and sorted(covered_indexes) == list(range(len(instructions)))
        and len(covered_indexes) == len(set(covered_indexes))
        and all(
            isinstance(edge, dict)
            and str(edge.get("source_block_id")) in block_ids
            and (
                str(edge.get("target_block_id")) in block_ids
                or str(edge.get("target_block_id")).startswith("external_")
            )
            for edge in edge_rows
        )
    )
    symbolic_facts = payload.get("symbolic_facts", [])
    symbolic_facts_ok = (
        len(symbolic_facts) == len(instructions)
        and all(
            isinstance(fact, dict)
            and "index" in fact
            and "address" in fact
            and "supported" in fact
            and "executed" in fact
            for fact in symbolic_facts
        )
        and not any(
            bool(fact.get("executed")) and not bool(fact.get("supported"))
            for fact in symbolic_facts
            if isinstance(fact, dict)
        )
    )
    return [
        _check("trace_required_keys", len(missing) == 0, f"missing={missing}"),
        _check(
            "trace_schema_version",
            payload.get("schema_version") == TRACE_SCHEMA_VERSION,
            f"schema={payload.get('schema_version')}",
        ),
        _check(
            "trace_binary_hash",
            payload.get("binary_sha256") == expected_binary_hash,
            f"sha256={payload.get('binary_sha256')}",
        ),
        _check(
            "trace_instruction_observation_alignment",
            len(instructions) == len(observations) and len(instructions) > 0,
            f"instructions={len(instructions)} observations={len(observations)}",
        ),
        _check(
            "trace_required_package_provenance",
            required_packages.issubset(provenance)
            and all(provenance[name]["available"] for name in required_packages),
            f"packages={sorted(provenance)}",
        ),
        _check(
            "trace_optional_roadmap_provenance",
            optional_roadmap_packages.issubset(provenance)
            and all(provenance[name]["optional"] for name in optional_roadmap_packages),
            f"optional_packages={sorted(optional_roadmap_packages & set(provenance))}",
        ),
        _check(
            "trace_basic_block_coverage",
            cfg_ok,
            f"blocks={len(block_rows)} edges={len(edge_rows)} covered={len(covered_indexes)}",
        ),
        _check(
            "trace_observation_semantics",
            observation_semantics_ok and label_vocab_ok,
            f"observations={len(observations)} labels_ok={label_vocab_ok}",
        ),
        _check(
            "trace_typed_observation_alignment",
            typed_semantics_ok,
            f"typed_observations={len(typed_observations)}",
        ),
        _check(
            "trace_dynamic_register_contract",
            dynamic_trace_ok,
            f"dynamic_register_events={len(dynamic_trace)}",
        ),
        _check(
            "trace_symbolic_facts_contract",
            symbolic_facts_ok,
            f"symbolic_facts={len(symbolic_facts)}",
        ),
    ]


def validate_trace_file(trace_path: Path) -> list[ValidationCheck]:
    """Validate one trace bundle file and return check rows."""

    return _trace_checks(trace_path)


def _assembly_listing_checks(output_dir: Path) -> list[ValidationCheck]:
    """Verify the manuscript-facing assembly listing is trace-derived."""

    listing_path = output_dir / "data" / "assembly_listing.json"
    trace_path = output_dir / "data" / "trace_bundle.json"
    if not listing_path.exists() or not trace_path.exists():
        return [
            _check(
                "assembly_listing_contract",
                False,
                f"listing={listing_path.exists()} trace={trace_path.exists()}",
            )
        ]
    try:
        listing = _load_json(listing_path)
        trace = _load_json(trace_path)
        passed, detail = validate_assembly_listing_payload(listing, trace)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return [_check("assembly_listing_contract", False, str(exc))]
    return [_check("assembly_listing_contract", passed, detail)]


def _symbolic_execution_checks(output_dir: Path) -> list[ValidationCheck]:
    """Validate the bounded semantics artifact against its trace projection."""

    symbolic_path = output_dir / "data" / "symbolic_execution.json"
    trace_path = output_dir / "data" / "trace_bundle.json"
    if not symbolic_path.exists() or not trace_path.exists():
        return [
            _check(
                "symbolic_semantics_contract",
                False,
                f"symbolic={symbolic_path.exists()} trace={trace_path.exists()}",
            )
        ]
    try:
        symbolic = _load_json(symbolic_path)
        trace = _load_json(trace_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [_check("symbolic_semantics_contract", False, str(exc))]
    events = symbolic.get("events", [])
    facts = trace.get("symbolic_facts", [])
    dynamic_trace = trace.get("dynamic_register_trace", [])
    dynamic_instruction_rows = [
        row
        for row in dynamic_trace[1:-1]
        if isinstance(row, dict) and row.get("event") == "instruction"
    ]
    contract_ok = (
        symbolic.get("schema_version") == SEMANTICS_SCHEMA_VERSION
        and bool(events)
        and events == facts
        and bool(symbolic.get("final_rax_matches"))
        and int(symbolic.get("unsupported_instruction_count", 1)) == 0
        and len(events) == len(trace.get("static_instruction_events", []))
    )
    executed_events = [
        event for event in events if isinstance(event, dict) and event.get("executed")
    ]
    stepwise_ok = (
        bool(symbolic.get("stepwise_rax_matches"))
        and int(symbolic.get("stepwise_rax_comparable_count", -1))
        == len(executed_events)
        and int(symbolic.get("stepwise_rax_mismatch_count", 1)) == 0
        and int(symbolic.get("modeled_flag_mismatches", 1)) == 0
        and len(dynamic_instruction_rows) == len(executed_events)
        and all(
            isinstance(event, dict)
            and isinstance(row, dict)
            and event.get("address") == row.get("address")
            and event.get("unicorn_pre_rax")
            == row.get("registers_before", {}).get("rax")
            and event.get("unicorn_post_rax")
            == row.get("registers_after", {}).get("rax")
            and event.get("rax_pre_matches") is True
            and event.get("rax_post_matches") is True
            for event, row in zip(executed_events, dynamic_instruction_rows)
        )
    )
    return [
        _check(
            "symbolic_semantics_contract",
            contract_ok,
            f"events={len(events)} unsupported={symbolic.get('unsupported_instruction_count')} "
            f"final_match={symbolic.get('final_rax_matches')}",
        ),
        _check(
            "symbolic_stepwise_trace_refinement",
            stepwise_ok,
            f"executed={len(executed_events)} comparable={symbolic.get('stepwise_rax_comparable_count')} "
            f"rax_mismatches={symbolic.get('stepwise_rax_mismatch_count')} "
            f"flag_mismatches={symbolic.get('modeled_flag_mismatches')}",
        ),
    ]


def _held_out_model_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate held-out fitting metadata and byte-identical regeneration."""

    path = project_root / "output" / "data" / "held_out_model.json"
    if not path.exists():
        return [
            _check("held_out_model_contract", False, "missing held_out_model.json"),
            _check("held_out_model_freshness", False, "missing held_out_model.json"),
        ]
    try:
        payload = _load_json(path)
        rows = payload.get("per_fixture", [])
        leakage_ok = bool(rows) and all(
            row.get("held_out_fixture") not in row.get("training_fixtures", [])
            and len(row.get("training_fixtures", []))
            == payload.get("fixture_count", 0) - 1
            for row in rows
            if isinstance(row, dict)
        )
        from .advanced_inference import held_out_model_payload

        fresh = held_out_model_payload()
        fresh_ok = _domain_payload_equal(payload, fresh)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return [
            _check("held_out_model_contract", False, str(exc)),
            _check("held_out_model_freshness", False, str(exc)),
        ]
    return [
        _check(
            "held_out_model_contract",
            leakage_ok and len(rows) == int(payload.get("fixture_count", 0)),
            f"rows={len(rows)} fixture_count={payload.get('fixture_count')} leakage_free={leakage_ok}",
        ),
        _check(
            "held_out_model_freshness",
            fresh_ok,
            f"recompute_matches_disk={fresh_ok}",
        ),
    ]


def _typed_observation_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the typed schema, its label-blind payload, and its held-out fit."""

    data_dir = project_root / "output" / "data"
    typed_path = data_dir / "typed_observations.json"
    factorized_path = data_dir / "factorized_observation_model.json"
    if not typed_path.exists() or not factorized_path.exists():
        missing = [
            path.name for path in (typed_path, factorized_path) if not path.exists()
        ]
        return [
            _check("typed_observation_schema_contract", False, f"missing={missing}"),
            _check("typed_observation_schema_freshness", False, f"missing={missing}"),
            _check("factorized_observation_contract", False, f"missing={missing}"),
            _check("factorized_observation_freshness", False, f"missing={missing}"),
            _check("factorized_observation_leakage_guard", False, f"missing={missing}"),
        ]
    try:
        typed_payload = _load_json(typed_path)
        factorized_payload = _load_json(factorized_path)
        typed_rows = typed_payload.get("per_fixture", {})
        typed_contract = (
            typed_payload.get("schema_version") == TYPED_OBSERVATION_SCHEMA_VERSION
            and typed_payload.get("fields") == list(TYPED_OBSERVATION_FIELDS)
            and typed_payload.get("phase_label_keys") == []
            and bool(typed_rows)
            and all(
                isinstance(record, dict)
                and int(record.get("instruction_count", 0))
                == len(record.get("observations", []))
                and all(
                    isinstance(observation, dict)
                    and set(TYPED_OBSERVATION_FIELDS).issubset(observation)
                    and not any(
                        key in observation
                        for key in ("phase_label", "phase_labels", "truth")
                    )
                    for observation in record.get("observations", [])
                )
                for record in typed_rows.values()
            )
        )
        factorized_rows = factorized_payload.get("per_fixture", [])
        factorized_summary = factorized_payload.get("summary", {})
        factorized_contract = (
            factorized_payload.get("schema_version") == FACTOR_MODEL_SCHEMA_VERSION
            and factorized_payload.get("fields") == list(TYPED_OBSERVATION_FIELDS)
            and factorized_payload.get("static_fields")
            and len(factorized_rows) == int(factorized_payload.get("fixture_count", 0))
            and set(factorized_payload.get("field_group_ablation", {}))
            == {
                "instruction_semantics",
                "register_context",
                "control_flow",
                "dynamic_context",
            }
            and all(
                np.isfinite(float(factorized_summary[key]))
                for key in (
                    "mean_collapsed_accuracy",
                    "mean_static_factorized_accuracy",
                    "mean_factorized_accuracy",
                    "mean_hierarchical_accuracy",
                    "mean_missing_memory_field_accuracy",
                )
            )
        )
        leakage_guard = (
            factorized_payload.get("leakage_checks", {}).get(
                "held_out_excluded_from_training"
            )
            is True
            and factorized_payload.get("leakage_checks", {}).get(
                "phase_label_keys_absent_from_typed_records"
            )
            is True
            and all(
                isinstance(row, dict)
                and row.get("held_out_fixture") not in row.get("training_fixtures", [])
                for row in factorized_rows
            )
        )
        from .advanced_inference import (
            factorized_observation_payload,
            typed_observation_payload,
        )

        fresh_typed = typed_observation_payload()
        fresh_factorized = factorized_observation_payload()
        typed_fresh = _domain_payload_equal(typed_payload, fresh_typed)
        factorized_fresh = _domain_payload_equal(factorized_payload, fresh_factorized)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        detail = str(exc)
        return [
            _check("typed_observation_schema_contract", False, detail),
            _check("typed_observation_schema_freshness", False, detail),
            _check("factorized_observation_contract", False, detail),
            _check("factorized_observation_freshness", False, detail),
            _check("factorized_observation_leakage_guard", False, detail),
        ]
    return [
        _check(
            "typed_observation_schema_contract",
            typed_contract,
            f"fixtures={len(typed_rows)} fields={len(typed_payload.get('fields', []))}",
        ),
        _check(
            "typed_observation_schema_freshness",
            typed_fresh,
            f"recompute_matches_disk={typed_fresh}",
        ),
        _check(
            "factorized_observation_contract",
            bool(factorized_contract),
            f"rows={len(factorized_rows)} groups={len(factorized_payload.get('field_group_ablation', {}))}",
        ),
        _check(
            "factorized_observation_freshness",
            factorized_fresh,
            f"recompute_matches_disk={factorized_fresh}",
        ),
        _check(
            "factorized_observation_leakage_guard",
            leakage_guard,
            f"rows={len(factorized_rows)} leakage_free={leakage_guard}",
        ),
    ]


def _corpus_coverage_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate and recompute the fixture-design coverage inventory."""

    path = project_root / "output" / "data" / "corpus_coverage.json"
    if not path.exists():
        return [
            _check("corpus_coverage_contract", False, "missing corpus_coverage.json"),
            _check("corpus_coverage_freshness", False, "missing corpus_coverage.json"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_corpus_coverage_payload(payload)
        fresh = corpus_coverage_payload()
        fresh_ok = _domain_payload_equal(payload, fresh)
        contract = (
            valid and payload.get("schema_version") == CORPUS_COVERAGE_SCHEMA_VERSION
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        detail = str(exc)
        return [
            _check("corpus_coverage_contract", False, detail),
            _check("corpus_coverage_freshness", False, detail),
        ]
    return [
        _check("corpus_coverage_contract", contract, detail),
        _check(
            "corpus_coverage_freshness",
            fresh_ok,
            f"recompute_matches_disk={fresh_ok}",
        ),
    ]


def _action_sensing_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the held-out action-conditioned sensing artifact."""

    path = project_root / "output" / "data" / "action_sensing.json"
    if not path.exists():
        return [
            _check("action_sensing_contract", False, "missing action_sensing.json"),
            _check("action_sensing_freshness", False, "missing action_sensing.json"),
            _check(
                "action_sensing_leakage_guard",
                False,
                "missing action_sensing.json",
            ),
        ]
    try:
        payload = _load_json(path)
        contract, detail = validate_action_conditioned_sensing_payload(payload)
        fresh_payload = action_conditioned_sensing_payload(tuple(build_corpus()))
        fresh = _domain_payload_equal(payload, fresh_payload)
        summary = payload.get("summary", {})
        rows = payload.get("per_fixture", [])
        leakage_free = bool(summary.get("leakage_free")) and all(
            bool(row.get("leakage_free")) for row in rows if isinstance(row, dict)
        )
        schema_ok = payload.get("schema_version") == ACTION_SENSING_SCHEMA_VERSION
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        detail = str(exc)
        return [
            _check("action_sensing_contract", False, detail),
            _check("action_sensing_freshness", False, detail),
            _check("action_sensing_leakage_guard", False, detail),
        ]
    return [
        _check(
            "action_sensing_contract",
            contract and schema_ok,
            f"schema={payload.get('schema_version')} {detail}",
        ),
        _check(
            "action_sensing_freshness",
            fresh,
            f"recompute_matches_disk={fresh}",
        ),
        _check(
            "action_sensing_leakage_guard",
            leakage_free,
            f"leakage_free={leakage_free}",
        ),
    ]


def _parameter_learning_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate and recompute the observation-only EM artifact."""

    path = project_root / "output" / "data" / "parameter_learning.json"
    if not path.exists():
        return [
            _check(
                "parameter_learning_contract",
                False,
                "missing parameter_learning.json",
            ),
            _check(
                "parameter_learning_freshness",
                False,
                "missing parameter_learning.json",
            ),
            _check(
                "parameter_learning_leakage_guard",
                False,
                "missing parameter_learning.json",
            ),
        ]
    try:
        payload = _load_json(path)
        contract, detail = validate_parameter_learning_payload(payload)
        fresh_payload = parameter_learning_payload(tuple(build_corpus()))
        fresh = _domain_payload_equal(payload, fresh_payload)
        rows = payload.get("per_fixture", [])
        leakage_free = payload.get("phase_labels_used_for_fit") is False and all(
            isinstance(row, dict)
            and isinstance(row.get("fit_input_contract"), dict)
            and row["fit_input_contract"].get("phase_labels_used_for_fit") is False
            and row["fit_input_contract"].get("labels_present_in_fit_input") is False
            and row["fit_input_contract"].get("observation_only") is True
            and row.get("held_out_fixture") not in row.get("training_fixtures", [])
            for row in rows
        )
        schema_ok = payload.get("schema_version") == PARAMETER_LEARNING_SCHEMA_VERSION
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        detail = str(exc)
        return [
            _check("parameter_learning_contract", False, detail),
            _check("parameter_learning_freshness", False, detail),
            _check("parameter_learning_leakage_guard", False, detail),
        ]
    return [
        _check(
            "parameter_learning_contract",
            contract and schema_ok,
            f"schema={payload.get('schema_version')} {detail}",
        ),
        _check(
            "parameter_learning_freshness",
            fresh,
            f"recompute_matches_disk={fresh}",
        ),
        _check(
            "parameter_learning_leakage_guard",
            leakage_free,
            f"observation_only={leakage_free}",
        ),
    ]


def _cfg_evidence_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the branch-bearing CFG artifact and its deterministic source."""

    path = project_root / "output" / "data" / "cfg_evidence.json"
    if not path.exists():
        return [
            _check("cfg_evidence_contract", False, "missing cfg_evidence.json"),
            _check("cfg_evidence_freshness", False, "missing cfg_evidence.json"),
        ]
    try:
        payload = _load_json(path)
        instructions = payload.get("static_instruction_events", [])
        blocks = payload.get("basic_blocks", [])
        edges = payload.get("cfg_edges", [])
        covered = [
            int(index)
            for block in blocks
            if isinstance(block, dict)
            for index in block.get("instruction_indexes", [])
        ]
        external_edge = any(
            str(edge.get("target_block_id", "")).startswith("external_")
            for edge in edges
            if isinstance(edge, dict)
        )
        contract_ok = (
            payload.get("schema_version") == "1.0"
            and payload.get("sample_id") == "x86_guard"
            and len(instructions) == 4
            and len(blocks) >= 2
            and sorted(covered) == list(range(len(instructions)))
            and len(covered) == len(set(covered))
            and external_edge
            and payload.get("dynamic_addresses")
        )
        from .advanced_inference import cfg_evidence_payload

        fresh = cfg_evidence_payload()
        fresh_ok = _domain_payload_equal(payload, fresh)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return [
            _check("cfg_evidence_contract", False, str(exc)),
            _check("cfg_evidence_freshness", False, str(exc)),
        ]
    return [
        _check(
            "cfg_evidence_contract",
            bool(contract_ok),
            f"blocks={len(blocks)} edges={len(edges)} external_edge={external_edge}",
        ),
        _check(
            "cfg_evidence_freshness", fresh_ok, f"recompute_matches_disk={fresh_ok}"
        ),
    ]


def _memory_evidence_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate static memory operands against real dynamic access hooks."""

    path = project_root / "output" / "data" / "memory_evidence.json"
    if not path.exists():
        return [
            _check("memory_evidence_contract", False, "missing memory_evidence.json"),
            _check("memory_evidence_freshness", False, "missing memory_evidence.json"),
        ]
    try:
        payload = _load_json(path)
        instructions = payload.get("static_instruction_events", [])
        indexes = {
            int(index) for index in payload.get("static_memory_instruction_indexes", [])
        }
        events = payload.get("dynamic_memory_events", [])
        summary = payload.get("summary", {})
        expected_hash = hashlib.sha256(
            bytes.fromhex(str(payload.get("code_hex", "")))
        ).hexdigest()
        join_ok = bool(events) and all(
            isinstance(event, dict)
            and int(event.get("instruction_index", -1)) in indexes
            and int(event.get("address", 0))
            == int(payload.get("stack_pointer", 0))
            + int(event.get("stack_relative_address", 0))
            and any(
                int(operand.get("displacement", 0))
                == int(event.get("stack_relative_address", 0))
                and int(operand.get("size", -1)) == int(event.get("size", -2))
                for operand in instructions[int(event["instruction_index"])].get(
                    "memory_operands", []
                )
            )
            for event in events
        )
        contract_ok = (
            payload.get("schema_version") == "1.0"
            and payload.get("sample_id") == "x86_memory"
            and payload.get("architecture") == "x86_64"
            and payload.get("binary_sha256") == expected_hash
            and len(instructions) == 3
            and indexes == {0, 1}
            and summary.get("dynamic_event_count") == len(events) == 2
            and summary.get("read_count") == 1
            and summary.get("write_count") == 1
            and payload.get("final_rax") == 8
            and join_ok
        )
        from .advanced_inference import memory_evidence_payload

        fresh = memory_evidence_payload()
        fresh_ok = _domain_payload_equal(payload, fresh)
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        return [
            _check("memory_evidence_contract", False, str(exc)),
            _check("memory_evidence_freshness", False, str(exc)),
        ]
    return [
        _check(
            "memory_evidence_contract",
            bool(contract_ok),
            f"instructions={len(instructions)} events={len(events)} join_ok={join_ok}",
        ),
        _check(
            "memory_evidence_freshness",
            fresh_ok,
            f"recompute_matches_disk={fresh_ok}",
        ),
    ]


def _cover_image_checks(project_root: Path) -> list[ValidationCheck]:
    """Verify that the configured manuscript cover is a present, nonblank PNG."""

    config_path = project_root / "manuscript" / "config.yaml"
    image_path = project_root / "manuscript" / "assets" / "active_assembly_cover.png"
    try:
        config_text = config_path.read_text(encoding="utf-8")
        image = np.asarray(mpimg.imread(image_path), dtype=float)
        configured = 'image: "assets/active_assembly_cover.png"' in config_text
        alt_configured = 'alt: "Abstract cover illustration' in config_text
        present_and_nonblank = (
            image_path.is_file() and image.size > 0 and float(np.var(image)) > 0.00001
        )
    except (OSError, ValueError) as exc:
        return [_check("cover_image_contract", False, str(exc))]
    return [
        _check(
            "cover_image_contract",
            configured and alt_configured and present_and_nonblank,
            f"configured={configured} bytes={image_path.stat().st_size if image_path.is_file() else 0} "
            f"variance={float(np.var(image)) if image.size else 0.0:.6f} alt={alt_configured}",
        )
    ]


def _rxinfer_checks(trace_path: Path, rxinfer_dir: Path) -> list[ValidationCheck]:
    records = load_free_energy_records(rxinfer_dir / "free_energy_trace.jsonl")
    diagnostics = _load_json(rxinfer_dir / "diagnostics.json")
    posteriors = _load_json(rxinfer_dir / "posteriors.json")
    model_config = _load_json(rxinfer_dir / "model_config.json")
    expected_data_hash = file_sha256(trace_path)
    checks: list[ValidationCheck] = []
    try:
        validate_free_energy_records(records, expected_data_hash=expected_data_hash)
    except Exception as exc:  # noqa: BLE001 - validation report must serialize failure text
        checks.append(_check("rxinfer_free_energy_records", False, str(exc)))
    else:
        checks.append(
            _check(
                "rxinfer_free_energy_records",
                True,
                f"records={len(records)} final={records[-1]['value']}",
            )
        )
    checks.append(
        _check(
            "rxinfer_diagnostics_consistent",
            diagnostics.get("iteration_count") == len(records)
            and diagnostics.get("all_free_energy_finite") is True
            and diagnostics.get("data_hash") == expected_data_hash,
            f"iterations={diagnostics.get('iteration_count')} all_finite={diagnostics.get('all_free_energy_finite')}",
        )
    )
    companion_files = (
        "posteriors.json",
        "model_config.json",
        "stdout.log",
        "stderr.log",
    )
    companion_files_ok = all(
        (rxinfer_dir / filename).exists() for filename in companion_files
    )
    posterior_rows = posteriors.get("posteriors", {}).get("μ", [])
    config_values = model_config.get("observation_values", [])
    checks.append(
        _check(
            "rxinfer_companion_artifacts",
            companion_files_ok
            and posteriors.get("run_id") == diagnostics.get("run_id")
            and len(posterior_rows) == diagnostics.get("iteration_count")
            and len(config_values) > 0,
            f"companions={companion_files_ok} posterior_rows={len(posterior_rows)}",
        )
    )
    return checks


def _categorical_rxinfer_checks(
    trace_path: Path, rxinfer_dir: Path
) -> list[ValidationCheck]:
    """Validate the real, namespaced categorical RxInfer companion run."""

    required_paths = (
        rxinfer_dir / "categorical_free_energy_trace.jsonl",
        rxinfer_dir / "categorical_posteriors.json",
        rxinfer_dir / "categorical_diagnostics.json",
        rxinfer_dir / "categorical_model_config.json",
        rxinfer_dir / "stdout.log",
        rxinfer_dir / "stderr.log",
    )
    if not all(path.exists() for path in required_paths):
        missing = [path.name for path in required_paths if not path.exists()]
        return [
            _check(
                "rxinfer_categorical_free_energy_records",
                False,
                f"missing={missing}",
            ),
            _check(
                "rxinfer_categorical_companion_artifacts",
                False,
                f"missing={missing}",
            ),
            _check(
                "rxinfer_categorical_diagnostics_consistent",
                False,
                f"missing={missing}",
            ),
        ]
    try:
        expected_data_hash = file_sha256(trace_path)
        records = validate_categorical_rxinfer_outputs(
            rxinfer_dir, expected_data_hash=expected_data_hash
        )
        diagnostics = _load_json(rxinfer_dir / "categorical_diagnostics.json")
        posteriors = _load_json(rxinfer_dir / "categorical_posteriors.json")
        config = _load_json(rxinfer_dir / "categorical_model_config.json")
        records_ok = bool(records) and all(
            record.get("model_name") == "categorical_phase_hmm"
            and record.get("mode") == "categorical"
            and record.get("is_finite") is True
            for record in records
        )
        companion_ok = (
            posteriors.get("run_id") == diagnostics.get("run_id")
            and config.get("model_name") == "categorical_phase_hmm"
            and config.get("mode") == "categorical"
            and len(config.get("observation_values", [])) > 0
        )
        diagnostics_ok = (
            diagnostics.get("model_name") == "categorical_phase_hmm"
            and diagnostics.get("mode") == "categorical"
            and diagnostics.get("iteration_count") == len(records)
            and diagnostics.get("all_free_energy_finite") is True
            and diagnostics.get("data_hash") == expected_data_hash
            and bool(diagnostics.get("config_hash"))
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        detail = str(exc)
        return [
            _check("rxinfer_categorical_free_energy_records", False, detail),
            _check("rxinfer_categorical_companion_artifacts", False, detail),
            _check("rxinfer_categorical_diagnostics_consistent", False, detail),
        ]
    return [
        _check(
            "rxinfer_categorical_free_energy_records",
            records_ok,
            f"records={len(records)} model=categorical_phase_hmm",
        ),
        _check(
            "rxinfer_categorical_companion_artifacts",
            companion_ok,
            f"run_id={posteriors.get('run_id')}",
        ),
        _check(
            "rxinfer_categorical_diagnostics_consistent",
            diagnostics_ok,
            f"iterations={diagnostics.get('iteration_count')} all_finite={diagnostics.get('all_free_energy_finite')}",
        ),
    ]


def _figure_checks(figure_dir: Path) -> list[ValidationCheck]:
    expected = tuple(f"{stem}.png" for stem in FIGURE_CAPTIONS)
    checks: list[ValidationCheck] = []
    for filename in expected:
        path = figure_dir / filename
        if not path.exists():
            checks.append(_check(f"figure_{path.stem}", False, "missing"))
            continue
        image = np.asarray(mpimg.imread(path), dtype=float)
        checks.append(
            _check(
                f"figure_{path.stem}",
                path.stat().st_size > 0
                and image.size > 0
                and float(np.var(image)) > 0.00001,
                f"bytes={path.stat().st_size} variance={float(np.var(image)):.6f}",
            )
        )
    return checks


def _figure_caption_checks(figure_dir: Path) -> list[ValidationCheck]:
    """Require a one-to-one, present caption row for every generated PNG."""

    caption_path = figure_dir / "README.md"
    if not caption_path.exists():
        return [_check("figure_caption_ledger", False, "missing README.md")]
    caption_text = caption_path.read_text(encoding="utf-8")
    rows = re.findall(r"\| `([^`]+\.png)` \((present|missing)\) \|", caption_text)
    listed = {filename for filename, _status in rows}
    statuses = {filename: status for filename, status in rows}
    expected = {f"{stem}.png" for stem in FIGURE_CAPTIONS}
    actual = {path.name for path in figure_dir.glob("*.png")}
    status_ok = all(statuses.get(filename) == "present" for filename in expected)
    return [
        _check(
            "figure_caption_ledger",
            listed == expected == actual and status_ok,
            f"expected={len(expected)} listed={len(listed)} actual={len(actual)} "
            f"missing={sorted(expected - listed)} stale={sorted(actual - expected)}",
        )
    ]


def _figure_registry_checks(output_dir: Path) -> list[ValidationCheck]:
    """Require complete source-bound metadata for every generated figure."""

    registry_path = output_dir / "data" / "figure_registry.json"
    if not registry_path.exists():
        return [
            _check("figure_registry_contract", False, "missing figure_registry.json")
        ]
    try:
        payload = _load_json(registry_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [_check("figure_registry_contract", False, str(exc))]
    expected = {f"{stem}.png" for stem in FIGURE_CAPTIONS}
    rows = payload.get("figures", [])
    filenames = [
        Path(str(row.get("png_path", ""))).name
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("png_path"), str)
    ]
    duplicate_filenames = sorted(
        filename for filename in set(filenames) if filenames.count(filename) > 1
    )
    # Validate the canonical v2 names directly.  The older caption ledger
    # remains a compatibility view, but it must not be allowed to make a v1
    # row appear source-bound in the canonical registry.
    required_fields = {
        "figure_id",
        "label",
        "generator",
        "input_artifacts",
        "input_hashes",
        "population",
        "sample_hierarchy",
        "metrics",
        "uncertainty",
        "missing_semantics",
        "evidence_status",
        "receipt_status",
        "scholarship_keys",
        "caption",
        "alt_text",
        "long_description",
        "claim_ids",
        "validation_checks",
        "visual_qa",
        "png_path",
        "caption_word_count",
        "alt_text_word_count",
        "long_description_word_count",
        "palette_contract",
        "table_companion_path",
        "source_note",
        "visual_qa_status",
    }
    rows_ok = True
    metadata_errors: list[str] = []
    known_citation_keys = scholarship_keys()
    for row in rows:
        missing = sorted(required_fields - set(row))
        visual_qa = row.get("visual_qa", {})
        citation_keys = row.get("scholarship_keys")
        if missing:
            metadata_errors.append(f"{row.get('figure_id')}:missing={missing}")
        if not isinstance(citation_keys, list) or any(
            key not in known_citation_keys for key in citation_keys
        ):
            metadata_errors.append(
                f"{row.get('figure_id')}:unknown-citation-keys={citation_keys}"
            )
        if (
            not row.get("input_artifacts")
            or not row.get("claim_ids")
            or not row.get("validation_checks")
        ):
            metadata_errors.append(f"{row.get('figure_id')}:missing-lineage")
        if (
            not str(row.get("alt_text", "")).strip()
            or not str(row.get("long_description", "")).strip()
        ):
            metadata_errors.append(f"{row.get('figure_id')}:missing-accessibility")
        path_value = row.get("png_path")
        path = (
            output_dir.parent / path_value
            if isinstance(path_value, str)
            else output_dir / "figures" / ""
        )
        expected_hash = visual_qa.get("sha256")
        actual_hash = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if (
            not path.is_file()
            or visual_qa.get("status") != "programmatic_nonblank"
            or not isinstance(expected_hash, str)
            or expected_hash != actual_hash
        ):
            metadata_errors.append(f"{row.get('figure_id')}:visual-qa-or-hash")
    rows_ok = (
        payload.get("schema_version") == FIGURE_REGISTRY_SCHEMA_VERSION
        and payload.get("figure_count") == len(expected)
        and set(filenames) == expected
        and len(filenames) == len(set(filenames))
        and not duplicate_filenames
        and not metadata_errors
    )
    template_registry_path = output_dir / "figures" / "figure_registry.json"
    template_registry_errors: list[str] = []
    try:
        template_payload = _load_json(template_registry_path)
        template_rows = template_payload.get("figures", [])
        canonical_by_filename = {
            Path(str(row.get("png_path"))).name: row
            for row in rows
            if isinstance(row, dict) and row.get("png_path")
        }
        for template_row in template_rows:
            filename = str(template_row.get("filename", ""))
            canonical = canonical_by_filename.get(filename)
            if canonical is None:
                template_registry_errors.append(f"{filename}:not-in-canonical-registry")
                continue
            if (
                str(template_row.get("alt", "")).strip()
                != str(canonical.get("alt_text", "")).strip()
            ):
                template_registry_errors.append(f"{filename}:alt-mismatch")
        if len(template_rows) != len(rows):
            template_registry_errors.append(
                f"row-count={len(template_rows)} expected={len(rows)}"
            )
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        template_registry_errors.append(str(exc))
    rows_ok = rows_ok and not template_registry_errors
    return [
        _check(
            "figure_registry_contract",
            rows_ok,
            f"expected={len(expected)} rows={len(rows)} errors={(metadata_errors + template_registry_errors)[:4]}",
        )
    ]


def _scholarship_checks(project_root: Path) -> list[ValidationCheck]:
    references = (project_root / "manuscript" / "references.bib").read_text(
        encoding="utf-8"
    )
    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((project_root / "manuscript").glob("*.md"))
    )
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", references))
    cited_keys = {
        key
        for match in re.findall(r"@([A-Za-z0-9_:-]+)", manuscript_text)
        for key in [match.rstrip(".,;]")]
        if not key.startswith(("fig:", "tab:", "sec:", "eq:", "lst:", "def:", "prop:"))
    }
    ledger_keys = scholarship_keys()
    missing_bib = sorted(ledger_keys - bib_keys)
    missing_usage = sorted((ledger_keys & bib_keys) - cited_keys)
    unresolved_citations = sorted(cited_keys - bib_keys)
    payload = scholarship_payload()
    sources = payload["sources"]
    allowed_source_types = {
        "peer_reviewed_theory",
        "peer_reviewed_statistics",
        "software_paper",
        "software_documentation",
        "peer_reviewed_binary_analysis",
        "peer_reviewed_formal_methods",
        "peer_reviewed_compiler_correctness",
        "peer_reviewed_reproducibility",
        "software_repository",
        "technical_report",
        "methodology_textbook",
        "formal_methods_documentation",
    }
    allowed_evidence_tiers = {
        "theory",
        "implemented_backend",
        "implemented_dependency",
        "integration_guidance",
        "comparison_baseline",
        "binary_analysis_context",
        "assembly_formalism",
        "statistical_method",
        "roadmap_adapter",
        "formal_verification",
        "compiler_formalism",
        "provenance_method",
    }
    allowed_project_stages = {
        "implemented",
        "comparison",
        "roadmap",
        "context",
    }
    doi_pattern = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
    source_quality_ok = all(
        source["source_type"] in allowed_source_types
        and source["evidence_tier"] in allowed_evidence_tiers
        and source["project_stage"] in allowed_project_stages
        and str(source["url"]).startswith("https://")
        and (source["doi"] is None or bool(doi_pattern.match(str(source["doi"]))))
        and len(str(source["project_role"])) >= 20
        and len(str(source["claim_boundary"])) >= 20
        for source in sources
    )
    project_stage_counts = payload["project_stage_counts"]
    evidence_tier_counts = payload["evidence_tier_counts"]
    evidence_partition_ok = (
        set(project_stage_counts).issubset(allowed_project_stages)
        and set(evidence_tier_counts).issubset(allowed_evidence_tiers)
        and int(project_stage_counts.get("implemented", 0)) >= 4
        and int(project_stage_counts.get("roadmap", 0)) >= 10
        and int(project_stage_counts.get("context", 0)) >= 3
        and int(project_stage_counts.get("comparison", 0)) >= 1
    )
    # Cautious-boundary language is protected conceptually (not by exact prose),
    # so a legitimate rephrase does not silently trip the gate.
    reverse_engineering_boundary = bool(
        re.search(
            r"not\s+(?:a\s+)?general\s+reverse[-\s]?engineering",
            manuscript_text,
            flags=re.IGNORECASE,
        )
    )
    categorical_hmm_boundary = bool(
        re.search(
            r"(?:not\s+(?:yet\s+)?(?:the\s+)?planned\s+categorical(?:[-\s]+)hidden\s+markov|"
            r"fixed[-\s]+parameter\s+categorical\s+(?:program[-\s]+)?phase[-\s]+HMM|"
            r"categorical.*does\s+not\s+establish\s+learned\s+semantic)",
            manuscript_text,
            flags=re.IGNORECASE,
        )
    )
    return [
        _check(
            "scholarship_source_quality",
            source_quality_ok and int(payload["source_count"]) >= 20,
            f"sources={payload['source_count']} types={sorted(payload['source_type_counts'])}",
        ),
        _check(
            "scholarship_evidence_partition",
            evidence_partition_ok,
            f"stages={project_stage_counts} tiers={evidence_tier_counts}",
        ),
        _check(
            "scholarship_bibliography_coverage",
            len(missing_bib) == 0,
            f"missing_bib={missing_bib}",
        ),
        _check(
            "scholarship_citation_closure",
            len(unresolved_citations) == 0,
            f"unresolved_citations={unresolved_citations}",
        ),
        _check(
            "scholarship_manuscript_usage",
            len(missing_usage) == 0,
            f"uncited_ledger_keys={missing_usage}",
        ),
        _check(
            "scholarship_claim_boundaries",
            reverse_engineering_boundary and categorical_hmm_boundary,
            "cautious claim boundaries present",
        ),
    ]


def _manuscript_source_checks(project_root: Path) -> list[ValidationCheck]:
    manuscript_dir = project_root / "manuscript"
    expected_sections = {
        "00_abstract.md",
        "01_introduction.md",
        "02_methodology.md",
        "03_results.md",
        "04_discussion.md",
        "05_validation.md",
        "06_limits_and_roadmap.md",
        "99_references.md",
    }
    source_paths = tuple(
        path
        for path in manuscript_dir.glob("*.md")
        if path.name not in MANUSCRIPT_SOURCE_EXCLUDES
    )
    files = {path.name for path in source_paths}
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    # The canonical figure registry is the source of truth for the manuscript
    # surface.  Requiring the caption and path tokens as a pair prevents a
    # figure from silently retaining a legacy literal caption or a legacy path
    # while its registry metadata has moved on.  Atlas-only rows are generated
    # and validated but need not be embedded in the narrative source.
    try:
        from .figure_contract import build_figure_specs

        figure_specs = build_figure_specs(project_root / "output")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        figure_specs = []
    expected_figures = {
        (
            f"FIGURE_{spec.figure_id.upper()}_CAPTION",
            f"FIGURE_{spec.figure_id.upper()}_PATH",
        )
        for spec in figure_specs
        if spec.placement != "atlas"
    }
    missing_sections = sorted(expected_sections - files)
    missing_figures = sorted(
        f"{caption_token}|{path_token}"
        for caption_token, path_token in expected_figures
        if f"{{{{{caption_token}}}}}" not in text or f"{{{{{path_token}}}}}" not in text
    )
    raw_figure_refs = sorted(set(re.findall(r"`(fig:[A-Za-z0-9_-]+)`", text)))
    return [
        _check(
            "manuscript_source_surface",
            len(missing_sections) == 0
            and len(missing_figures) == 0
            and not raw_figure_refs,
            f"missing_sections={missing_sections} missing_figures={missing_figures} "
            f"raw_figure_refs={raw_figure_refs}",
        )
    ]


def _manuscript_variable_coverage_checks(project_root: Path) -> list[ValidationCheck]:
    """Ensure source tokens and generated values close through the registry."""

    source_paths = tuple(
        path
        for path in (project_root / "manuscript").glob("*.md")
        if path.name not in MANUSCRIPT_SOURCE_EXCLUDES
    )
    required = {
        token
        for path in source_paths
        for token in MANUSCRIPT_TOKEN_RE.findall(path.read_text(encoding="utf-8"))
    }
    variables_path = project_root / "output" / "data" / "manuscript_variables.json"
    if not variables_path.exists():
        return [
            _check(
                "manuscript_variable_coverage",
                False,
                f"missing manuscript_variables.json required={len(required)}",
            )
        ]
    try:
        raw_variables = _load_json(variables_path)
        variables = artifact_domain_view(raw_variables)
        if not isinstance(variables, dict):
            raise TypeError("manuscript variables receipt payload must be an object")
    except (OSError, json.JSONDecodeError) as exc:
        return [_check("manuscript_variable_coverage", False, str(exc))]
    missing = sorted(required - set(variables))
    atlas_only = {token for token in variables if token.startswith("FIGURE_")}
    compatibility_only = {
        token for token in variables if token in COMPATIBILITY_ONLY_TOKENS
    }
    unused = sorted((set(variables) - required) - atlas_only - compatibility_only)
    pending = sorted(
        token for token, value in variables.items() if str(value).lower() == "pending"
    )
    if pending:
        return [
            _check(
                "manuscript_variable_coverage",
                not missing and not unused,
                f"pre-validation values pending: missing={missing} unused={unused}",
            ),
            _check(
                "manuscript_variable_registry",
                True,
                "deferred until validation values are produced",
            ),
        ]
    registry_path = project_root / MANUSCRIPT_VARIABLE_REGISTRY
    registry_errors: list[str] = []
    registry_payload: dict[str, Any] = {}
    if not registry_path.exists():
        registry_errors.append("missing registry")
    else:
        try:
            registry_payload = _load_json(registry_path)
        except (OSError, json.JSONDecodeError) as exc:
            registry_errors.append(str(exc))
    records = registry_payload.get("records", [])
    record_tokens = [record.get("token") for record in records]
    if len(record_tokens) != len(set(record_tokens)):
        registry_errors.append("duplicate record tokens")
    if set(record_tokens) != set(variables):
        registry_errors.append("registry record/generated token mismatch")
    if registry_payload.get("missing_tokens") != missing:
        registry_errors.append("registry missing-token summary is stale")
    if registry_payload.get("unused_generated_tokens") != unused:
        registry_errors.append("registry unused-token summary is stale")
    recorded_atlas_only = set(registry_payload.get("atlas_only_tokens", []))
    if recorded_atlas_only != atlas_only - set(required):
        registry_errors.append("registry atlas-only token summary is stale")
    recorded_compatibility_only = set(
        registry_payload.get("compatibility_only_tokens", [])
    )
    if recorded_compatibility_only != compatibility_only - set(required):
        registry_errors.append("registry compatibility-only token summary is stale")
    if not registry_payload.get("summary", {}).get("no_pending_values", False):
        registry_errors.append("registry contains pending values")
    record_consumer_errors = [
        str(record.get("token"))
        for record in records
        if record.get("token") in required and not record.get("consumers")
    ]
    if record_consumer_errors:
        registry_errors.append(f"missing consumers={record_consumer_errors}")
    return [
        _check(
            "manuscript_variable_coverage",
            not missing and not unused and not pending,
            f"required={len(required)} generated={len(variables)} missing={missing} unused={unused} pending={pending}",
        ),
        _check(
            "manuscript_variable_registry",
            not registry_errors,
            f"path={registry_path} errors={registry_errors}",
        ),
    ]


def _manuscript_token_checks(output_dir: Path) -> list[ValidationCheck]:
    manuscript_dir = output_dir / "manuscript"
    if not manuscript_dir.exists():
        return [_check("rendered_manuscript_tokens", True, "not hydrated yet")]
    unresolved: list[str] = []
    for path in manuscript_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\{\{[A-Z0-9_]+\}\}", text):
            unresolved.append(path.name)
    return [
        _check(
            "rendered_manuscript_tokens",
            not unresolved,
            f"unresolved_files={unresolved}",
        )
    ]


def _claim_register_checks(
    project_root: Path,
    observed_check_names: set[str],
) -> list[ValidationCheck]:
    report_path = project_root / "output" / "data" / "claim_register.json"
    if not report_path.exists():
        return [_check("claim_register_contract", False, "missing claim_register.json")]
    payload = _load_json(report_path)
    claims = payload.get("claims", [])
    expected_ids = {claim.claim_id for claim in CLAIM_RECORDS}
    observed_ids = {claim.get("claim_id") for claim in claims}
    ledger_keys = scholarship_keys()
    all_sources_known = all(
        set(claim.get("support_keys", [])).issubset(ledger_keys) for claim in claims
    )
    all_artifacts_present = all(
        all(
            (project_root / str(path)).exists()
            for path in claim.get("artifact_paths", [])
        )
        for claim in claims
    )
    all_checks_known = all(
        set(claim.get("validation_checks", [])).issubset(observed_check_names)
        for claim in claims
    )
    all_validated = all(claim.get("status") == "validated" for claim in claims)
    return [
        _check(
            "claim_register_contract",
            payload.get("claim_count") == len(CLAIM_RECORDS)
            and observed_ids == expected_ids
            and all_sources_known
            and all_artifacts_present
            and all_checks_known
            and all_validated,
            (
                f"claims={payload.get('claim_count')} sources={all_sources_known} "
                f"artifacts={all_artifacts_present} checks={all_checks_known}"
            ),
        )
    ]


def _reproducibility_checks(project_root: Path) -> list[ValidationCheck]:
    manifest_path = project_root / "output" / "data" / "reproducibility_manifest.json"
    if not manifest_path.exists():
        return [_check("reproducibility_artifact_manifest", False, "missing manifest")]
    payload = _load_json(manifest_path)
    errors = verify_reproducibility_manifest(project_root)
    rows = payload.get("artifacts", [])
    hashes_present = all(
        (not bool(row.get("required", True)) and row.get("status") == "missing")
        or (
            isinstance(row.get("sha256"), str)
            and len(str(row.get("sha256"))) == 64
            and int(row.get("bytes", 0)) > 0
        )
        for row in rows
    )
    return [
        _check(
            "reproducibility_artifact_manifest",
            not errors
            and payload.get("artifact_count") == len(MANIFEST_TARGETS)
            and payload.get("all_required_present") is True
            and hashes_present,
            f"artifacts={payload.get('artifact_count')} errors={errors[:3]}",
        )
    ]


def _docx_render_checks(project_root: Path) -> list[ValidationCheck]:
    """Check the optional post-render DOCX lane without blocking pre-render work.

    The project-side generator necessarily runs before the sibling template
    renderer.  A clean checkout therefore has no DOCX yet, and a source edit
    can temporarily leave an older DOCX receipt behind while the hydrated tree
    is being regenerated.  Those two states are explicitly marked deferred;
    the dedicated post-render script remains fail-closed and must be run after
    the template has emitted the document.
    """

    payload = docx_render_payload(project_root)
    if not payload.get("artifact_exists"):
        detail = "deferred until sibling template renders the DOCX"
        return [
            _check("docx_render_contract", True, detail),
            _check("docx_render_receipt_freshness", True, detail),
        ]
    contract_errors = validate_docx_render_payload(payload)
    receipt_errors = validate_docx_render_receipt(project_root, payload)
    pre_render = os.environ.get("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER") == "1"
    if pre_render and receipt_errors:
        receipt_detail = (
            f"deferred during pre-render hydration: errors={receipt_errors[:3]}"
        )
        receipt_passed = True
    else:
        receipt_detail = f"errors={receipt_errors[:3]}"
        receipt_passed = not receipt_errors
    if pre_render and contract_errors:
        contract_detail = (
            f"deferred during pre-render hydration: errors={contract_errors[:3]}"
        )
        contract_passed = True
    else:
        contract_detail = (
            f"artifact={payload.get('artifact_path')} errors={contract_errors[:3]}"
        )
        contract_passed = not contract_errors
    return [
        _check(
            "docx_render_contract",
            contract_passed,
            contract_detail,
        ),
        _check(
            "docx_render_receipt_freshness",
            receipt_passed,
            receipt_detail,
        ),
    ]


def _documentation_checks(project_root: Path) -> list[ValidationCheck]:
    docs_dir = project_root / "docs"
    expected_docs = {
        "README.md",
        "analysis_validation.md",
        "architecture.md",
        "assembly_packages.md",
        "claim_map.md",
        "claim_register.md",
        "reproducibility.md",
        "rxinfer_roadmap.md",
        "scholarship.md",
        "roadmap.md",
        "scientific_program.md",
        "verification.md",
    }
    observed_docs = {path.name for path in docs_dir.glob("*.md")}
    missing_docs = sorted(expected_docs - observed_docs)
    claim_map_path = docs_dir / "claim_map.md"
    claim_map_text = (
        claim_map_path.read_text(encoding="utf-8") if claim_map_path.exists() else ""
    )
    return [
        _check(
            "documentation_contract_surface",
            not missing_docs
            and "output/rxinfer/free_energy_trace.jsonl" in claim_map_text
            and "output/data/reproducibility_manifest.json" in claim_map_text
            and "Passing validation is a project contract" in claim_map_text,
            f"missing_docs={missing_docs}",
        )
    ]


def _scientific_program_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the machine-readable long-horizon research contract."""

    path = project_root / "output" / "data" / "scientific_program.json"
    if not path.exists():
        return [
            _check(
                "scientific_program_contract", False, "missing scientific_program.json"
            ),
            _check("scientific_program_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            _check("scientific_program_contract", False, str(exc)),
            _check("scientific_program_freshness", False, "artifact unreadable"),
        ]
    valid, detail = validate_scientific_program_payload(payload)
    fresh = scientific_program_payload()
    fresh_match = _domain_payload_equal(payload, fresh)
    return [
        _check(
            "scientific_program_contract",
            valid
            and payload.get("schema_version") == SCIENTIFIC_PROGRAM_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "scientific_program_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _quotient_invariance_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the first bounded P3 equivalence/near-miss protocol."""

    path = project_root / "output" / "data" / "quotient_invariance_protocol.json"
    if not path.exists():
        return [
            _check(
                "quotient_invariance_contract",
                False,
                "missing quotient_invariance_protocol.json",
            ),
            _check("quotient_invariance_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            _check("quotient_invariance_contract", False, str(exc)),
            _check("quotient_invariance_freshness", False, "artifact unreadable"),
        ]
    valid, detail = validate_quotient_invariance_payload(payload)
    fresh = quotient_invariance_payload()
    fresh_match = _domain_payload_equal(payload, fresh)
    return [
        _check(
            "quotient_invariance_contract",
            valid and payload.get("schema_version") == QUOTIENT_PROTOCOL_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "quotient_invariance_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _representation_invariance_checks(
    project_root: Path,
) -> list[ValidationCheck]:
    """Validate the source-family-disjoint P3 benchmark receipt."""

    path = project_root / "output" / "data" / "representation_invariance_benchmark.json"
    if not path.exists():
        return [
            _check(
                "representation_invariance_contract",
                False,
                "missing representation_invariance_benchmark.json",
            ),
            _check(
                "representation_invariance_freshness",
                False,
                "artifact missing",
            ),
        ]
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            _check("representation_invariance_contract", False, str(exc)),
            _check("representation_invariance_freshness", False, "artifact unreadable"),
        ]
    valid, detail = validate_representation_invariance_payload(payload)
    fresh = representation_invariance_payload()
    fresh_match = _domain_payload_equal(payload, fresh)
    return [
        _check(
            "representation_invariance_contract",
            valid
            and payload.get("schema_version")
            == REPRESENTATION_INVARIANCE_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "representation_invariance_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _intervention_power_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate exact finite intervention-subset enumeration and freshness."""

    path = project_root / "output" / "data" / "intervention_power.json"
    if not path.exists():
        return [
            _check("intervention_power_contract", False, "artifact missing"),
            _check("intervention_power_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_intervention_power_payload(payload)
        fresh = intervention_power_payload()
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("intervention_power_contract", False, str(exc)),
            _check("intervention_power_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "intervention_power_contract",
            valid
            and payload.get("schema_version") == INTERVENTION_ANALYSIS_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "intervention_power_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _cross_architecture_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the real multi-architecture Capstone/Unicorn fixture receipt."""

    path = project_root / "output" / "data" / "cross_architecture_fixtures.json"
    if not path.exists():
        return [
            _check("cross_architecture_contract", False, "artifact missing"),
            _check("cross_architecture_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_cross_architecture_payload(payload)
        fresh = cross_architecture_payload()
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("cross_architecture_contract", False, str(exc)),
            _check("cross_architecture_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "cross_architecture_contract",
            valid
            and payload.get("schema_version") == CROSS_ARCHITECTURE_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "cross_architecture_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _formal_verification_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the compiled Lean receipt and its theorem/source manifest."""

    path = project_root / "output" / "data" / "formal_verification.json"
    if not path.exists():
        return [
            _check("formal_verification_contract", False, "artifact missing"),
            _check("formal_verification_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_formal_verification_payload(payload)
        fresh = formal_verification_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("formal_verification_contract", False, str(exc)),
            _check(
                "formal_verification_freshness", False, "formal build or receipt failed"
            ),
        ]
    return [
        _check(
            "formal_verification_contract",
            valid and payload.get("schema_version") == FORMAL_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "formal_verification_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _active_information_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the exact finite-horizon information-policy artifact."""

    path = project_root / "output" / "data" / "active_information_frontier.json"
    if not path.exists():
        return [
            _check("active_information_contract", False, "artifact missing"),
            _check("active_information_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_active_information_payload(payload)
        fresh = active_information_payload()
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return [
            _check("active_information_contract", False, str(exc)),
            _check("active_information_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "active_information_contract",
            valid
            and payload.get("schema_version") == INFORMATION_POLICY_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "active_information_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _stable_sensor_cost_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove intentionally variable wall-clock samples for freshness checks."""

    stable = json.loads(json.dumps(payload))
    for row in stable.get("sensors", []):
        if isinstance(row, dict):
            row.pop("wall_clock_nanoseconds_per_query", None)
    stable.get("calibration", {}).pop("python", None)
    stable.get("calibration", {}).pop("platform", None)
    return stable


def _scientific_protocol_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "research_protocol.json"
    if not path.exists():
        return [
            _check("scientific_protocol_contract", False, "artifact missing"),
            _check("scientific_protocol_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_scientific_protocol_payload(payload)
        fresh = scientific_protocol_payload()
        fresh_match = json.dumps(payload, sort_keys=True) == json.dumps(
            fresh, sort_keys=True
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("scientific_protocol_contract", False, str(exc)),
            _check("scientific_protocol_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "scientific_protocol_contract",
            valid and payload.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "scientific_protocol_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _quotient_classifier_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "quotient_classifier.json"
    if not path.exists():
        return [
            _check("quotient_classifier_contract", False, "artifact missing"),
            _check("quotient_classifier_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_quotient_classifier_payload(payload)
        fresh = quotient_classifier_payload()
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("quotient_classifier_contract", False, str(exc)),
            _check("quotient_classifier_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "quotient_classifier_contract",
            valid
            and payload.get("schema_version") == QUOTIENT_CLASSIFIER_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "quotient_classifier_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _formal_correspondence_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "formal_trace_correspondence.json"
    if not path.exists():
        return [
            _check("formal_correspondence_contract", False, "artifact missing"),
            _check("formal_correspondence_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_formal_correspondence_payload(payload)
        fresh = formal_correspondence_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("formal_correspondence_contract", False, str(exc)),
            _check("formal_correspondence_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "formal_correspondence_contract",
            valid
            and payload.get("schema_version") == FORMAL_CORRESPONDENCE_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "formal_correspondence_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _compiler_transfer_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "compiler_transfer.json"
    if not path.exists():
        return [
            _check("compiler_transfer_contract", False, "artifact missing"),
            _check("compiler_transfer_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_compiler_transfer_payload(payload)
        fresh = compiler_transfer_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("compiler_transfer_contract", False, str(exc)),
            _check("compiler_transfer_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "compiler_transfer_contract",
            valid and payload.get("schema_version") == COMPILER_TRANSFER_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "compiler_transfer_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _sensor_cost_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "sensor_cost_receipt.json"
    if not path.exists():
        return [
            _check("sensor_cost_contract", False, "artifact missing"),
            _check("sensor_cost_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_sensor_cost_payload(payload)
        # Reuse the matching local calibration receipt. Timing samples are
        # measured observations, not deterministic source values; regenerating
        # them during a freshness check would manufacture a false mismatch.
        fresh = sensor_cost_payload(project_root, reuse_existing=True)
        fresh_match = _stable_sensor_cost_payload(
            payload
        ) == _stable_sensor_cost_payload(fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("sensor_cost_contract", False, str(exc)),
            _check("sensor_cost_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "sensor_cost_contract",
            valid and payload.get("schema_version") == SENSOR_COST_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "sensor_cost_freshness",
            fresh_match,
            f"stable_recompute_matches_disk={fresh_match}",
        ),
    ]


def _external_source_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the offline, static-only external-source promotion boundary."""

    path = project_root / "output" / "data" / "external_source_validation.json"
    if not path.exists():
        return [
            _check("external_source_contract", False, "artifact missing"),
            _check("external_source_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_external_source_validation_payload(payload)
        fresh = external_source_validation_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("external_source_contract", False, str(exc)),
            _check("external_source_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "external_source_contract",
            valid and payload.get("schema_version") == EXTERNAL_MANIFEST_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "external_source_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _external_acquisition_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the offline freshness of the explicit acquisition receipt."""

    path = project_root / EXTERNAL_ACQUISITION_ARTIFACT
    if not path.exists():
        return [
            _check("external_acquisition_contract", False, "artifact missing"),
            _check("external_acquisition_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_external_acquisition_freshness(payload, project_root)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return [
            _check("external_acquisition_contract", False, str(exc)),
            _check("external_acquisition_freshness", False, "artifact unreadable"),
        ]
    return [
        _check("external_acquisition_contract", valid, detail),
        _check(
            "external_acquisition_freshness",
            valid,
            "offline acquisition cache and manifest inputs match"
            if valid
            else "offline acquisition cache freshness failed",
        ),
    ]


def _external_label_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate source-bound labels and every referenced static artifact."""

    path = project_root / LABEL_CONTRACT_ARTIFACT
    if not path.exists():
        return [
            _check("external_label_contract", False, "artifact missing"),
            _check("external_label_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_external_label_contract_payload(payload)
        fresh, fresh_detail = external_label_contract_freshness(payload, project_root)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return [
            _check("external_label_contract", False, str(exc)),
            _check("external_label_freshness", False, "artifact unreadable"),
        ]
    return [
        _check("external_label_contract", valid, detail),
        _check("external_label_freshness", fresh, fresh_detail),
    ]


def _external_static_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate static-only compilation rows and their reproducible receipt."""

    path = project_root / EXTERNAL_STATIC_RECEIPT_ARTIFACT
    if not path.exists():
        return [
            _check("external_static_contract", False, "artifact missing"),
            _check("external_static_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_external_static_validation_payload(payload)
        fresh = validate_external_static_sources(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return [
            _check("external_static_contract", False, str(exc)),
            _check("external_static_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "external_static_contract",
            valid and payload.get("schema_version") == EXTERNAL_MANIFEST_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "external_static_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _external_capability_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate capability receipts without treating advertised targets as built."""

    path = project_root / EXTERNAL_CAPABILITY_ARTIFACT
    if not path.exists():
        return [
            _check("external_capability_contract", False, "artifact missing"),
            _check("external_capability_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_compiler_capabilities_payload(payload)
        fresh = compiler_capabilities_payload()
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return [
            _check("external_capability_contract", False, str(exc)),
            _check("external_capability_freshness", False, "artifact unreadable"),
        ]
    return [
        _check("external_capability_contract", valid, detail),
        _check(
            "external_capability_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _independent_replay_checks(project_root: Path) -> list[ValidationCheck]:
    """Keep the separate-replay gate visible without fabricating evidence."""

    path = project_root / INDEPENDENT_REPLAY_ARTIFACT
    if not path.exists():
        return [
            _check(
                "independent_replay_contract",
                True,
                "not_yet_observed: no separately operated receipt has been ingested",
            ),
            _check(
                "independent_replay_freshness",
                True,
                "not_yet_observed: optional promotion artifact is absent",
            ),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_independent_replay_artifact(payload, project_root)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return [
            _check("independent_replay_contract", False, str(exc)),
            _check("independent_replay_freshness", False, "artifact unreadable"),
        ]
    return [
        _check("independent_replay_contract", valid, detail),
        _check(
            "independent_replay_freshness",
            valid,
            "protected scoring receipt is internally consistent",
        ),
    ]


def _replication_checks(project_root: Path) -> list[ValidationCheck]:
    path = project_root / "output" / "data" / "replication_manifest.json"
    if not path.exists():
        return [
            _check("replication_contract", False, "artifact missing"),
            _check("replication_freshness", False, "artifact missing"),
        ]
    try:
        valid, detail = validate_replication_bundle(project_root)
        payload = _load_json(path)
        fresh = replication_bundle_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("replication_contract", False, str(exc)),
            _check("replication_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "replication_contract",
            valid and payload.get("schema_version") == REPLICATION_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "replication_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _theorem_countertheorem_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the checked-proof/counterexample promotion boundary."""

    path = project_root / "output" / "data" / "theorem_or_countertheorem.json"
    if not path.exists():
        return [
            _check("theorem_countertheorem_contract", False, "artifact missing"),
            _check("theorem_countertheorem_freshness", False, "artifact missing"),
        ]
    try:
        payload = _load_json(path)
        valid, detail = validate_theorem_or_countertheorem_payload(payload)
        fresh = theorem_or_countertheorem_payload(project_root)
        fresh_match = _domain_payload_equal(payload, fresh)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return [
            _check("theorem_countertheorem_contract", False, str(exc)),
            _check("theorem_countertheorem_freshness", False, "artifact unreadable"),
        ]
    return [
        _check(
            "theorem_countertheorem_contract",
            valid and payload.get("schema_version") == THEOREM_PHASE_SCHEMA_VERSION,
            detail,
        ),
        _check(
            "theorem_countertheorem_freshness",
            fresh_match,
            f"recompute_matches_disk={fresh_match}",
        ),
    ]


def _build_provenance_checks(project_root: Path) -> list[ValidationCheck]:
    """Ensure the current candidate has a source/template lineage receipt."""

    errors = verify_build_provenance(project_root, strict=False)
    return [
        _check(
            "build_provenance_current",
            not errors,
            f"errors={errors[:4]}",
        )
    ]


def _advanced_inference_checks(project_root: Path) -> list[ValidationCheck]:
    """Validate the temporal, active, evaluation, and adequacy artifacts.

    Every check RECOMPUTES the relevant property from the artifact on disk (and,
    for the headline, re-derives it from the live generator) so the gate fails if
    a number was hand-edited, went stale, or was laundered green-by-construction.
    """

    data_dir = project_root / "output" / "data"
    temporal_path = data_dir / "temporal_inference.json"
    segments_path = data_dir / "phase_segments.json"
    active_path = data_dir / "active_selection.json"
    evaluation_path = data_dir / "evaluation_report.json"
    adequacy_path = data_dir / "generative_adequacy.json"

    required = (
        temporal_path,
        segments_path,
        active_path,
        evaluation_path,
        adequacy_path,
    )
    present = all(path.exists() for path in required)
    if not present:
        missing = [p.name for p in required if not p.exists()]
        return [
            _check("advanced_temporal_artifacts_present", False, f"missing={missing}"),
            _check(
                "advanced_temporal_beats_identity_transition",
                False,
                "artifacts missing",
            ),
            _check(
                "advanced_negative_controls_discriminate", False, "artifacts missing"
            ),
            _check("advanced_phase_segmentation_partition", False, "artifacts missing"),
            _check("advanced_active_never_worse", False, "artifacts missing"),
            _check("advanced_evaluation_freshness", False, "artifacts missing"),
            _check("generative_adequacy_contract", False, "artifacts missing"),
            _check("generative_adequacy_freshness", False, "artifacts missing"),
        ]

    evaluation = _load_json(evaluation_path)
    temporal = _load_json(temporal_path)
    segments_payload = _load_json(segments_path)
    active = _load_json(active_path)
    adequacy = _load_json(adequacy_path)

    summary = evaluation.get("summary", {})
    mean_temporal = float(summary.get("mean_temporal_accuracy", 0.0))
    mean_identity_transition = float(
        summary.get("mean_identity_transition_accuracy", 1.0)
    )
    mean_memoryless = float(summary.get("mean_memoryless_accuracy", 0.0))
    temporal_advantage = float(
        summary.get("temporal_advantage_over_best_non_temporal", 1.0)
    )
    shuffle = float(
        evaluation.get("shuffled_label_control", {}).get("mean_accuracy", 1.0)
    )
    # Compare like-to-like (accuracy vs accuracy), not accuracy vs balanced accuracy.
    uniform_accuracy = float(
        evaluation.get("uniform_belief_control", {}).get("accuracy", 1.0)
    )

    temporal_fixtures = temporal.get("per_fixture", {})
    artifacts_ok = (
        bool(temporal.get("model"))
        and bool(temporal_fixtures)
        and all(
            {"filter", "smoothing", "viterbi_path", "truth_labels"}.issubset(record)
            for record in temporal_fixtures.values()
        )
        and bool(evaluation.get("per_fixture"))
    )

    # Phase-segment partition: each fixture's segments must exactly tile its range.
    partition_ok = bool(segments_payload.get("per_fixture"))
    for record in segments_payload.get("per_fixture", {}).values():
        covered: list[int] = []
        for segment in record.get("segments", []):
            covered.extend(
                range(int(segment["start_index"]), int(segment["end_index"]) + 1)
            )
        if covered != list(range(len(covered))) or not covered:
            partition_ok = False
            break

    active_summary = active.get("summary", {})
    active_never_worse = bool(active_summary.get("active_never_worse", False))

    # Freshness: re-derive the FULL evaluation payload from the live generator and
    # compare it byte-for-byte to the on-disk artifact, so a hand edit anywhere in
    # the report (not just the two summary means) fails the gate.
    from .advanced_inference import evaluation_report_payload

    fresh_payload = evaluation_report_payload()
    freshness_ok = _domain_payload_equal(fresh_payload, evaluation)
    fresh_adequacy = fresh_payload["generative_adequacy"]
    adequacy_contract_ok = (
        adequacy.get("method") == "fixed_parameter_generative_adequacy"
        and adequacy.get("interval") == "central_95_percent_simulation_interval"
        and int(adequacy.get("replicates", 0)) > 0
        and int(adequacy.get("fixture_count", 0))
        == len(adequacy.get("per_fixture", []))
        and int(adequacy.get("tested_metric_count", 0)) > 0
        and 0.0 <= float(adequacy.get("inside_interval_fraction", -1.0)) <= 1.0
        and all(
            {"fixture", "length", "observed", "predictive", "inside_interval"}.issubset(
                row
            )
            for row in adequacy.get("per_fixture", [])
            if isinstance(row, dict)
        )
    )
    adequacy_freshness_ok = _domain_payload_equal(fresh_adequacy, adequacy)

    # Honest-comparison invariant: the artifact records the strongest non-temporal
    # baseline (the memoryless per-step decoder) and the temporal advantage over it,
    # so the headline cannot hide that a memoryless decoder may beat the temporal
    # model on this corpus. This binds the corrected, non-overclaiming narrative.
    honest_comparison_ok = (
        "mean_memoryless_accuracy" in summary
        and mean_memoryless > 0.0
        and abs(
            temporal_advantage
            - (mean_temporal - max(mean_identity_transition, mean_memoryless))
        )
        < 1e-9
    )

    # Clean-ablation invariant: the static and temporal decoders must share the
    # same likelihood A and prior D, so the accuracy gap isolates the transition
    # matrix B (temporal structure) and is not confounded by a different
    # observation model.
    from .active_selection import expected_free_energy
    from .generative_model import build_generative_model
    from .temporal_model import build_temporal_model

    static_model = build_generative_model()
    temporal_model = build_temporal_model()
    isolation_ok = bool(
        np.allclose(static_model.A, temporal_model.A)
        and np.allclose(static_model.D, temporal_model.D)
    )

    # Functional-policy invariant: the expected free energy must genuinely vary
    # across candidate observations, so the active null at corpus scale is a real
    # null and not a collapsed (constant-EFE) policy reported as one.
    flat_preference = np.ones(len(static_model.observation_labels), dtype=float)
    belief = np.asarray(static_model.D, dtype=float)
    belief = belief / belief.sum()
    efe_values = [
        expected_free_energy(
            belief, static_model.A, flat_preference, i, i
        ).expected_free_energy
        for i in range(len(static_model.observation_labels))
    ]
    policy_functional = (max(efe_values) - min(efe_values)) > 1e-6

    # Segmentation evaluation: the smoothed beliefs must genuinely recover the
    # ground-truth phase boundaries (recomputed from the live generator, not a
    # hardcoded literal).
    seg_pooled = fresh_payload["segmentation"]["pooled"]
    seg_f1 = float(seg_pooled["boundary_f1"])
    seg_step = float(seg_pooled["step_accuracy"])
    segmentation_measured = seg_f1 > 0 and seg_step > 0

    # Ambiguity ablation: the temporal advantage over the memoryless decoder
    # must EMERGE as observations become ambiguous (positive gap at high blend)
    # and be ABSENT at the authored likelihood (gap <= 0 at blend 0) — binding
    # both directions of the mechanism honestly.
    ablation_payload = fresh_payload["ambiguity_ablation"]
    ablation_rows = list(ablation_payload["rows"])
    blend0_gap = float(ablation_rows[0]["temporal_minus_memoryless"])
    max_blend_row = max(ablation_rows, key=lambda row: float(row["blend"]))
    high_blend_gap = float(max_blend_row["temporal_minus_memoryless"])
    crossover_blend = ablation_payload["crossover_blend"]
    seg_blend0_temporal = float(ablation_rows[0]["temporal_segmentation_f1"])
    seg_blend0_memoryless = float(ablation_rows[0]["memoryless_segmentation_f1"])
    seg_high_temporal = float(max_blend_row["temporal_segmentation_f1"])
    seg_high_memoryless = float(max_blend_row["memoryless_segmentation_f1"])
    ablation_shows_advantage = (
        high_blend_gap > 0
        and blend0_gap <= 1e-6
        and crossover_blend is not None
        # Segmentation follows the same mechanism: at the informative likelihood
        # the memoryless per-step segmentation is at least as good, but under
        # ambiguity memoryless boundary recovery collapses while the temporal
        # smoothed segmentation retains/improves.
        and seg_high_temporal > seg_high_memoryless
        and seg_blend0_temporal <= seg_blend0_memoryless + 1e-6
    )

    # Corpus stability: the headline "temporal beats the identity-transition filter, and is
    # well above chance" must survive every single-fixture holdout, confirmed by
    # recomputation (not a hardcoded literal).
    stability = fresh_payload["corpus_stability"]
    stability_holdouts = list(stability["per_holdout"])
    stability_span = float(stability["temporal_pooled_range"]["span"])
    stability_ok = (
        len(stability_holdouts) > 0
        and bool(stability["all_holdouts_temporal_above_identity_transition"])
        and bool(stability["all_holdouts_temporal_above_chance"])
    )

    # Active layer under ambiguity: the EFE policy must be never worse at every
    # blend, must show a real advantage (fewer steps to confidence) somewhere in
    # the sweep, and must have NO advantage at the authored likelihood — binding
    # the "activity matters only under differential informativeness" mechanism.
    active_never_worse = bool(ablation_payload["active_never_worse"])
    active_max_fraction = float(ablation_payload["active_max_advantage_fraction"])
    active_authored = int(ablation_payload["authored_blend_active_advantage"])
    active_advantage_emerges = (
        active_never_worse and active_max_fraction > 0 and active_authored == 0
    )

    # Per-phase recovery: every phase must be reported, the decently-recovered
    # phases must be mastered (best-recall phase fully recovered), and the hardest
    # phase must be measured as underdetermined (recall below half) — turning the
    # qualitative \"exit / guard underdetermined\" limit into a recomputed number.
    phase_recovery = fresh_payload["phase_recovery"]
    phase_per = phase_recovery["per_phase"]
    phase_macro = float(phase_recovery["macro_f1"])
    hardest_phase = str(phase_recovery["hardest_phase"])
    hardest_recall = float(phase_per[hardest_phase]["recall"])
    best_recall_phase = str(phase_recovery["best_recall_phase"])
    best_recall = float(phase_per[best_recall_phase]["recall"])
    phase_recovery_ok = (
        set(phase_per.keys()) == set(HIDDEN_STATE_LABELS)
        and phase_macro > 0.4
        and hardest_recall < 0.5
        and best_recall >= 1.0
    )

    # Decoder calibration: on this corpus calibration tracks accuracy — the
    # memoryless decoder is best calibrated and the identity-transition comparator worst.
    # Binding the honest ordering (recomputed, not hardcoded) prevents overstating
    # the temporal model's calibration.
    cal_decoders = fresh_payload["decoder_calibration"]["decoders"]
    cal_temporal = float(cal_decoders["temporal"]["ece"])
    cal_identity_transition = float(cal_decoders["identity_transition"]["ece"])
    cal_memoryless = float(cal_decoders["memoryless"]["ece"])
    calibration_ordering_ok = cal_memoryless < cal_temporal < cal_identity_transition

    # Observation informativeness: every category must be reported with a
    # positive-gain spread, and the ranking must be recomputed (not hardcoded).
    obs_info = fresh_payload["observation_informativeness"]
    obs_rows = list(obs_info["observations"])
    obs_info_ok = (
        len(obs_rows) >= 5
        and float(obs_info["info_gain_range"]) > 0
        and all(
            "observation" in row and "info_gain" in row and "best_phase" in row
            for row in obs_rows
        )
    )

    # Bayesian model evidence: the temporal HMM must be the best generative model
    # (highest log marginal likelihood) by recomputation, with a positive margin.
    evidence = fresh_payload["model_evidence"]
    ev = evidence["log_evidence"]
    evidence_ok = (
        float(ev["temporal"]) > float(ev["identity_transition"])
        and float(ev["temporal"]) > float(ev["memoryless"])
        and float(evidence["temporal_margin_over_best_alternative"]) > 0
    )

    # Transition ablation: as B -> uniform, temporal must converge exactly to
    # memoryless (B @ posterior = uniform), and temporal must beat static at the
    # authored B (the mechanism the ablation isolates).
    trans = fresh_payload["transition_ablation"]
    trans_rows = list(trans["rows"])
    trans_ok = (
        len(trans_rows) >= 5
        and trans["convergence_blend"] is not None
        and abs(float(trans_rows[-1]["temporal_minus_memoryless"])) < 1e-6
        and float(trans_rows[0]["temporal_minus_identity_transition"]) > 0
    )

    return [
        _check(
            "advanced_temporal_isolates_transition",
            isolation_ok,
            f"static_and_temporal_share_A_and_D={isolation_ok}",
        ),
        _check(
            "advanced_active_policy_functional",
            policy_functional,
            f"efe_spread={max(efe_values) - min(efe_values):.4f}",
        ),
        _check(
            "advanced_temporal_artifacts_present",
            artifacts_ok,
            f"temporal_fixtures={len(temporal_fixtures)}",
        ),
        _check(
            "advanced_temporal_beats_identity_transition",
            mean_temporal > mean_identity_transition,
            f"temporal={mean_temporal:.4f} identity_transition={mean_identity_transition:.4f} "
            "(narrow claim: temporal beats the identity-transition comparator only)",
        ),
        _check(
            "advanced_memoryless_baseline_honest",
            honest_comparison_ok,
            f"memoryless={mean_memoryless:.4f} temporal={mean_temporal:.4f} "
            f"temporal_advantage_over_best_non_temporal={temporal_advantage:.4f}",
        ),
        _check(
            "advanced_negative_controls_discriminate",
            (mean_temporal - shuffle) >= 0.15 and mean_temporal > uniform_accuracy,
            f"temporal={mean_temporal:.4f} shuffle={shuffle:.4f} uniform_acc={uniform_accuracy:.4f}",
        ),
        _check(
            "advanced_phase_segmentation_partition",
            partition_ok,
            f"fixtures={len(segments_payload.get('per_fixture', {}))}",
        ),
        _check(
            "advanced_active_never_worse",
            active_never_worse,
            f"never_worse={active_never_worse} ties={active_summary.get('ties')}",
        ),
        _check(
            "advanced_segmentation_measured",
            segmentation_measured,
            f"boundary_f1={seg_f1:.3f} step_acc={seg_step:.3f}",
        ),
        _check(
            "advanced_ambiguity_ablation_shows_advantage",
            ablation_shows_advantage,
            f"crossover={crossover_blend} blend0_gap={blend0_gap:+.3f} "
            f"high_blend_gap={high_blend_gap:+.3f} "
            f"seg_blend0(t/m)={seg_blend0_temporal:.2f}/{seg_blend0_memoryless:.2f} "
            f"seg_high(t/m)={seg_high_temporal:.2f}/{seg_high_memoryless:.2f}",
        ),
        _check(
            "advanced_transition_ablation_converges",
            trans_ok,
            f"convergence_blend={trans['convergence_blend']} "
            f"peak={float(trans['peak_temporal_accuracy']):.3f}",
        ),
        _check(
            "advanced_model_evidence_temporal_best",
            evidence_ok,
            f"temporal={float(ev['temporal']):.3f} "
            f"identity_transition={float(ev['identity_transition']):.3f} "
            f"memoryless={float(ev['memoryless']):.3f}",
        ),
        _check(
            "advanced_observation_informativeness_measured",
            obs_info_ok,
            f"range={float(obs_info['info_gain_range']):.3f} "
            f"most={obs_info['most_informative']} least={obs_info['least_informative']}",
        ),
        _check(
            "advanced_decoder_calibration_ordering",
            calibration_ordering_ok,
            "ece(memoryless/temporal/identity_transition)="
            f"{cal_memoryless:.3f}/{cal_temporal:.3f}/{cal_identity_transition:.3f}",
        ),
        _check(
            "advanced_phase_recovery_measured",
            phase_recovery_ok,
            f"macro_f1={phase_macro:.2f} hardest={hardest_phase} "
            f"hardest_recall={hardest_recall:.2f} best_recall={best_recall:.2f}",
        ),
        _check(
            "advanced_active_advantage_emerges_under_ambiguity",
            active_advantage_emerges,
            f"never_worse={active_never_worse} max_frac={active_max_fraction:.2f} "
            f"authored_adv={active_authored}",
        ),
        _check(
            "advanced_corpus_stability_measured",
            stability_ok,
            f"temporal_pooled_span={stability_span:.3f} "
            f"holdouts={len(stability_holdouts)}",
        ),
        _check(
            "advanced_evaluation_freshness",
            freshness_ok,
            f"recompute_matches_disk={freshness_ok}",
        ),
        _check(
            "generative_adequacy_contract",
            adequacy_contract_ok,
            f"fixtures={len(adequacy.get('per_fixture', []))} replicates={adequacy.get('replicates')}",
        ),
        _check(
            "generative_adequacy_freshness",
            adequacy_freshness_ok,
            f"recompute_matches_disk={adequacy_freshness_ok}",
        ),
    ]


def build_validation_report(project_root: Path = PROJECT_ROOT) -> ValidationReport:
    """Validate generated analysis artifacts and return a structured report."""

    output_dir = project_root / "output"
    trace_path = output_dir / "data" / "trace_bundle.json"
    rxinfer_dir = output_dir / "rxinfer"
    checks: list[ValidationCheck] = []
    receipt_errors = validate_data_artifact_surface(project_root)
    checks.append(
        _check(
            "artifact_receipt_surface",
            not receipt_errors,
            "all project data JSON artifacts carry the protocol-v2 receipt contract"
            if not receipt_errors
            else "; ".join(
                f"{row['path']}: {row['detail']}" for row in receipt_errors[:8]
            ),
        )
    )
    checks.extend(_trace_checks(trace_path))
    checks.extend(_assembly_listing_checks(output_dir))
    checks.extend(_symbolic_execution_checks(output_dir))
    checks.extend(_rxinfer_checks(trace_path, rxinfer_dir))
    checks.extend(
        _categorical_rxinfer_checks(trace_path, output_dir / "rxinfer_categorical")
    )
    checks.extend(_figure_checks(output_dir / "figures"))
    checks.extend(_figure_caption_checks(output_dir / "figures"))
    checks.extend(_figure_registry_checks(output_dir))
    checks.extend(_scholarship_checks(project_root))
    checks.extend(_manuscript_source_checks(project_root))
    checks.extend(_manuscript_variable_coverage_checks(project_root))
    checks.extend(_manuscript_token_checks(output_dir))
    checks.extend(_reproducibility_checks(project_root))
    checks.extend(_docx_render_checks(project_root))
    checks.extend(_advanced_inference_checks(project_root))
    checks.extend(_held_out_model_checks(project_root))
    checks.extend(_typed_observation_checks(project_root))
    checks.extend(_corpus_coverage_checks(project_root))
    checks.extend(_action_sensing_checks(project_root))
    checks.extend(_parameter_learning_checks(project_root))
    checks.extend(_cfg_evidence_checks(project_root))
    checks.extend(_memory_evidence_checks(project_root))
    checks.extend(_cover_image_checks(project_root))
    checks.extend(_scientific_program_checks(project_root))
    checks.extend(_quotient_invariance_checks(project_root))
    checks.extend(_representation_invariance_checks(project_root))
    checks.extend(_intervention_power_checks(project_root))
    checks.extend(_cross_architecture_checks(project_root))
    checks.extend(_formal_verification_checks(project_root))
    checks.extend(_active_information_checks(project_root))
    checks.extend(_scientific_protocol_checks(project_root))
    checks.extend(_quotient_classifier_checks(project_root))
    checks.extend(_formal_correspondence_checks(project_root))
    checks.extend(_compiler_transfer_checks(project_root))
    checks.extend(_sensor_cost_checks(project_root))
    checks.extend(_external_acquisition_checks(project_root))
    checks.extend(_external_source_checks(project_root))
    checks.extend(_external_label_checks(project_root))
    checks.extend(_external_static_checks(project_root))
    checks.extend(_external_capability_checks(project_root))
    checks.extend(_replication_checks(project_root))
    checks.extend(_independent_replay_checks(project_root))
    checks.extend(_theorem_countertheorem_checks(project_root))
    checks.extend(
        _claim_register_checks(project_root, {check.name for check in checks})
    )
    checks.extend(_documentation_checks(project_root))
    checks.extend(_build_provenance_checks(project_root))
    passed = sum(check.passed for check in checks)
    failed = len(checks) - passed
    report_path = output_dir / "data" / "validation_report.json"
    figure_path = output_dir / "figures" / "validation_summary.png"
    return ValidationReport(
        generated_at=datetime.now(UTC).isoformat(),
        checks=tuple(checks),
        summary={
            "passed": passed,
            "failed": failed,
            "total": len(checks),
            "all_passed": failed == 0,
        },
        artifact_paths={
            "validation_report": str(report_path.relative_to(project_root)),
            "validation_summary": str(figure_path.relative_to(project_root)),
        },
    )


def write_validation_report(project_root: Path = PROJECT_ROOT) -> ValidationReport:
    """Write validation JSON and summary figure."""

    output_dir = project_root / "output"
    report_path = output_dir / "data" / "validation_report.json"
    figure_path = output_dir / "figures" / "validation_summary.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # The summary figure is itself one of the validated outputs. Build once to
    # obtain the check rows, materialize that figure and its caption row, then
    # rebuild so a clean checkout is validated against the complete surface.
    first_pass = build_validation_report(project_root)
    plot_validation_summary(
        [check.to_dict() for check in first_pass.checks], figure_path
    )
    from .figure_contract import write_figure_bundle

    write_figure_bundle(output_dir)
    report = build_validation_report(project_root)
    write_json_artifact(
        report_path,
        report.to_dict(),
        project_root=project_root,
        producer_path=Path(__file__),
    )
    plot_validation_summary([check.to_dict() for check in report.checks], figure_path)
    # The validation figure itself is part of the caption ledger. Refresh after
    # plotting so a clean checkout cannot retain a stale "missing" status.
    write_figure_bundle(output_dir)
    from .build_provenance import write_build_provenance

    write_build_provenance(project_root)
    return report
