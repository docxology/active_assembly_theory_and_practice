"""Compose the temporal, active, evaluation, and semantics pipeline.

This thin orchestrator wires the temporal, active-selection, evaluation,
semantics, typed-observation, and corpus-coverage modules together and writes
their deterministic artifacts under ``output/data``:

* ``temporal_inference.json`` — per-fixture HMM forward filter, smoothing, and
  Viterbi decoding from :mod:`temporal_model` (the explicit ``B``-transition model
  compared with an identity-transition comparator),
* ``phase_segments.json`` — per-fixture phase segmentation of the smoothed
  beliefs from :mod:`phase_segmentation`,
* ``active_selection.json`` — per-fixture active-disassembly ordering and the
  honest active-vs-fixed step comparison from :mod:`active_selection`,
* ``evaluation_report.json`` — corpus decoding accuracy, calibration,
  free-energy convergence, model comparison, and two negative controls from
  :mod:`evaluation`.
* ``cfg_evidence.json`` — direct-target CFG evidence for the branch-bearing
  fixture, including static blocks, edge types, and the bounded dynamic path.
* ``held_out_model.json`` — smoothed training-only HMM fits and
  leave-one-fixture-out predictions from :mod:`model_fitting`.
* ``quotient_invariance_protocol.json`` — bounded equivalent-pair and near-miss
  witnesses for the next quotient-state research tranche.
* ``representation_invariance_benchmark.json`` — source-family-disjoint
  transformation families, preregistered P3 endpoints, and protocol diagnostics.
* ``intervention_power.json`` — exact enumeration of every non-empty intervention
  subset, its near-miss separation power, and finite minimal separators.
* ``cross_architecture_fixtures.json`` — real Capstone/Unicorn execution of one
  transparent numeric operation on x86-64, AArch64, and RISC-V64.
* ``formal_verification.json`` — a source-bound receipt for the compiled Lean
  operational subset when the canonical project output is generated.
* ``scientific_program.json`` — machine-readable conjectures, falsifiers,
  dependencies, and promotion gates for the long-horizon program.

Every value written here is computed from the live modules; nothing is
hand-stamped. The orchestrator owns no domain logic.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from .artifact_contract import write_json_artifact
from .active_selection import (
    CONFIDENCE_THRESHOLD,
    active_observation_order,
    fixed_order_steps_to_confidence,
)
from .action_sensing import action_conditioned_sensing_payload
from .assembly_packages.schema import build_static_cfg
from .corpus import build_corpus, fixture_observation_ids
from .corpus_coverage import corpus_coverage_payload
from .disassembly import disassemble_bytes, sample_bytes
from .emulation import emulate_x86_64
from .evaluation import evaluation_payload, generative_adequacy
from .factorized_model import (
    build_typed_observation_corpus,
    leave_one_fixture_out_factorized_evaluation,
)
from .formal_verification import write_formal_verification
from .formal_correspondence import write_formal_correspondence
from .generative_model import build_generative_model
from .model_fitting import leave_one_fixture_out_evaluation
from .multi_arch import cross_architecture_payload, validate_cross_architecture_payload
from .compiler_transfer import write_compiler_transfer
from .external_sources import (
    write_external_corpus_preparation_audit,
    write_external_acquisition_payload,
    write_compiler_capabilities,
    write_external_source_validation_payload,
    write_external_static_validation,
)
from .external_labeling import write_external_label_contract
from .information_policy import (
    active_information_payload,
    validate_active_information_payload,
)
from .intervention_analysis import write_intervention_power_payload
from .parameter_learning import parameter_learning_payload
from .phase_segmentation import segment_phases
from .quotient_invariance import write_quotient_invariance_protocol
from .quotient_classifier import write_quotient_classifier_report
from .replication import write_replication_bundle
from .representation_invariance import write_representation_invariance_benchmark
from .scientific_protocol import write_scientific_protocol
from .scientific_program import write_scientific_program
from .sensor_costs import write_sensor_cost_payload
from .sensing_stress import write_sensing_stress
from .counterexample_search import write_counterexample_search
from .theorem_phase import write_theorem_or_countertheorem
from .temporal_model import (
    build_temporal_model,
    forward_backward,
    forward_filter,
    viterbi_decode,
)
from .typed_observations import (
    TYPED_FIELD_GROUPS,
    TYPED_OBSERVATION_FIELDS,
    TYPED_OBSERVATION_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def _write_json(path: Path, payload: object) -> str:
    write_json_artifact(
        path,
        payload,
        project_root=PROJECT_ROOT,
        producer_path=Path(__file__),
    )
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def temporal_inference_payload() -> dict[str, object]:
    """Per-fixture forward filter, smoothing, and Viterbi decoding."""

    model = build_temporal_model()
    corpus = build_corpus()
    per_fixture: dict[str, object] = {}
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        filter_result = forward_filter(observation_ids, model)
        smoothing = forward_backward(observation_ids, model)
        viterbi_path, viterbi_log_prob = viterbi_decode(observation_ids, model)
        per_fixture[fixture.name] = {
            "observation_ids": observation_ids,
            "truth_labels": list(fixture.phase_labels),
            "filter": filter_result.to_dict(),
            "smoothing": smoothing.to_dict(),
            "viterbi_path": list(viterbi_path),
            "viterbi_log_prob": float(viterbi_log_prob),
        }
    return {
        "model": model.to_dict(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "per_fixture": per_fixture,
    }


def phase_segments_payload() -> dict[str, object]:
    """Per-fixture phase segmentation of the smoothed beliefs."""

    model = build_temporal_model()
    corpus = build_corpus()
    per_fixture: dict[str, object] = {}
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        smoothing = forward_backward(observation_ids, model)
        segments = segment_phases(
            smoothing.smoothed_posteriors, model.hidden_state_labels
        )
        per_fixture[fixture.name] = {
            "segment_count": len(segments),
            "segments": [segment.to_dict() for segment in segments],
        }
    return {"per_fixture": per_fixture}


def active_selection_payload() -> dict[str, object]:
    """Per-fixture active-disassembly ordering and the active-vs-fixed comparison.

    Honest reporting: with a flat preference and the peaked likelihood ``A`` of
    short fixtures, the first observation already saturates confidence, so the
    active ordering and the fixed left-to-right scan reach the threshold in the
    same number of steps. The active *mechanism* is unit-verified in
    ``tests/test_active_selection.py``; this artifact records the corpus outcome
    as measured, including ties.
    """

    model = build_generative_model()
    preference = np.ones(len(model.observation_labels), dtype=float)
    corpus = build_corpus()
    per_fixture: dict[str, dict[str, object]] = {}
    active_wins = 0
    ties = 0
    for fixture in corpus:
        observation_ids = [int(o) for o in fixture_observation_ids(fixture)]
        selection = active_observation_order(
            observation_ids,
            model.A,
            model.D,
            preference,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        predictive_selection = active_observation_order(
            observation_ids,
            model.A,
            model.D,
            preference,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            policy="predictive",
        )
        fixed_steps = fixed_order_steps_to_confidence(
            observation_ids, model.A, model.D, CONFIDENCE_THRESHOLD
        )
        active_steps = selection.steps_to_confidence
        if (
            active_steps is not None
            and fixed_steps is not None
            and active_steps < fixed_steps
        ):
            active_wins += 1
        elif active_steps == fixed_steps:
            ties += 1
        per_fixture[fixture.name] = {
            "selection": selection.to_dict(),
            "predictive_selection": predictive_selection.to_dict(),
            "active_steps_to_confidence": active_steps,
            "predictive_steps_to_confidence": predictive_selection.steps_to_confidence,
            "fixed_steps_to_confidence": fixed_steps,
            "active_no_worse_than_fixed": (
                active_steps is not None
                and fixed_steps is not None
                and active_steps <= fixed_steps
            ),
        }
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "preference": [float(value) for value in preference],
        "summary": {
            "fixture_count": len(corpus),
            "active_strictly_fewer_steps": int(active_wins),
            "ties": int(ties),
            "active_never_worse": all(
                row["active_no_worse_than_fixed"] for row in per_fixture.values()
            ),
        },
        "per_fixture": per_fixture,
    }


def held_out_model_payload() -> dict[str, object]:
    """Fit and evaluate a smoothed HMM with strict fixture-level holdouts."""

    return leave_one_fixture_out_evaluation(build_corpus())


def cfg_evidence_payload() -> dict[str, object]:
    """Return source-bound CFG evidence for the branch-bearing fixture."""

    fixture = next(fixture for fixture in build_corpus() if fixture.name == "x86_guard")
    instructions = tuple(disassemble_bytes(fixture.code))
    trace = emulate_x86_64(fixture.code)
    blocks, edges = build_static_cfg(instructions)
    return {
        "schema_version": "1.0",
        "sample_id": fixture.name,
        "architecture": "x86_64",
        "base_address": int(trace.base_address),
        "code_hex": fixture.code.hex(),
        "binary_sha256": sha256(fixture.code).hexdigest(),
        "static_instruction_events": [
            instruction.to_dict() for instruction in instructions
        ],
        "basic_blocks": [block.to_dict() for block in blocks],
        "cfg_edges": [edge.to_dict() for edge in edges],
        "dynamic_addresses": [int(address) for address in trace.addresses],
        "dynamic_final_rax": int(trace.final_rax),
        "scope": "bounded direct-target CFG over the decoded x86_guard fixture",
    }


def memory_evidence_payload() -> dict[str, object]:
    """Join static memory operands to dynamic Unicorn access events.

    The memory projection is intentionally separate from the categorical
    inference artifact: it tests the evidence contract for effective-address
    calculation and observed load/store values without implying that the phase
    model has learned a general memory semantics. Every dynamic row is linked to the Capstone
    instruction index that triggered it and to the bounded stack address used
    by Unicorn.
    """

    sample_name = "x86_memory"
    code = sample_bytes(sample_name)
    instructions = tuple(disassemble_bytes(code))
    trace = emulate_x86_64(code)
    rsp = int(trace.registers["rsp"])
    static_memory_indexes = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.memory_operands
    ]
    dynamic_events: list[dict[str, object]] = []
    for event in trace.memory_events:
        row = cast(dict[str, object], dict(event))
        address = int(cast(int, row["address"]))
        row.update(
            {
                "address_hex": f"0x{address:x}",
                "stack_relative_address": address - rsp,
                "instruction_address_hex": (
                    f"0x{int(cast(int, row['instruction_address'])):x}"
                ),
            }
        )
        dynamic_events.append(row)
    reads = sum(event["access"] == "read" for event in dynamic_events)
    writes = sum(event["access"] == "write" for event in dynamic_events)
    return {
        "schema_version": "1.0",
        "sample_id": sample_name,
        "architecture": "x86_64",
        "base_address": int(trace.base_address),
        "binary_sha256": sha256(code).hexdigest(),
        "code_hex": code.hex(),
        "stack_pointer": rsp,
        "final_rax": int(trace.final_rax),
        "static_instruction_events": [
            instruction.to_dict() for instruction in instructions
        ],
        "static_memory_instruction_indexes": static_memory_indexes,
        "dynamic_memory_events": dynamic_events,
        "summary": {
            "static_memory_instruction_count": len(static_memory_indexes),
            "dynamic_event_count": len(dynamic_events),
            "read_count": reads,
            "write_count": writes,
            "distinct_addresses": len(
                {int(cast(int, event["address"])) for event in dynamic_events}
            ),
        },
        "scope": (
            "bounded Capstone effective-address metadata joined to Unicorn memory "
            "hooks for the transparent x86_memory fixture"
        ),
    }


def evaluation_report_payload() -> dict[str, Any]:
    """Corpus evaluation metrics with negative controls."""

    model = build_temporal_model()
    corpus = build_corpus()
    return evaluation_payload(corpus, model)


def generative_adequacy_payload() -> dict[str, object]:
    """Return the fixed-parameter simulation-based adequacy check."""

    return generative_adequacy(build_corpus(), build_temporal_model())


def typed_observation_payload() -> dict[str, object]:
    """Return the label-blind typed observation record for every fixture."""

    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    return {
        "schema_version": TYPED_OBSERVATION_SCHEMA_VERSION,
        "fields": list(TYPED_OBSERVATION_FIELDS),
        "field_groups": {
            group: list(fields) for group, fields in TYPED_FIELD_GROUPS.items()
        },
        "dynamic_field": "dynamic_status",
        "phase_label_keys": [],
        "scope": (
            "label-blind typed records derived from Capstone detail and bounded "
            "Unicorn execution over transparent fixtures"
        ),
        "per_fixture": {
            fixture.name: {
                "instruction_count": len(typed[fixture.name]),
                "observations": [
                    observation.to_dict() for observation in typed[fixture.name]
                ],
            }
            for fixture in corpus
        },
    }


def factorized_observation_payload() -> dict[str, object]:
    """Return held-out collapsed/factorized/hierarchical comparisons."""

    return leave_one_fixture_out_factorized_evaluation(build_corpus())


def corpus_coverage_payload_current() -> dict[str, object]:
    """Return the source-bound fixture-design coverage inventory."""

    return corpus_coverage_payload(build_corpus())


def action_sensing_payload_current() -> dict[str, object]:
    """Return the leakage-safe action-conditioned sensing experiment."""

    return action_conditioned_sensing_payload(build_corpus())


def parameter_learning_payload_current() -> dict[str, object]:
    """Return the observation-only ordered-HMM parameter-learning experiment."""

    return parameter_learning_payload(build_corpus())


def cross_architecture_payload_current() -> dict[str, object]:
    """Return and validate the real multi-architecture adapter benchmark."""

    payload = cross_architecture_payload()
    valid, detail = validate_cross_architecture_payload(payload)
    if not valid:
        raise ValueError(f"cross-architecture payload is invalid: {detail}")
    return payload


def active_information_payload_current() -> dict[str, object]:
    """Return and validate the exact finite-horizon information frontier."""

    payload = active_information_payload()
    valid, detail = validate_active_information_payload(payload)
    if not valid:
        raise ValueError(f"active-information payload is invalid: {detail}")
    return payload


def run_advanced_inference(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Write the current inference artifacts and return their relative paths."""

    data_dir = output_dir / "data"
    # Seal the protocol before any dependent artifact is written.  This avoids
    # a subtle stale-hash window when a producer is called after a protocol
    # source edit but before the full manuscript pipeline is regenerated.
    write_scientific_protocol(output_dir.parent)

    def local_path(value: Path) -> str:
        """Return a project-relative path, including temporary test roots."""

        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value.relative_to(output_dir.parent))

    paths = {
        "temporal_inference": _write_json(
            data_dir / "temporal_inference.json", temporal_inference_payload()
        ),
        "phase_segments": _write_json(
            data_dir / "phase_segments.json", phase_segments_payload()
        ),
        "active_selection": _write_json(
            data_dir / "active_selection.json", active_selection_payload()
        ),
        "evaluation_report": _write_json(
            data_dir / "evaluation_report.json", evaluation_report_payload()
        ),
        "generative_adequacy": _write_json(
            data_dir / "generative_adequacy.json", generative_adequacy_payload()
        ),
        "typed_observations": _write_json(
            data_dir / "typed_observations.json", typed_observation_payload()
        ),
        "factorized_observation_model": _write_json(
            data_dir / "factorized_observation_model.json",
            factorized_observation_payload(),
        ),
        "corpus_coverage": _write_json(
            data_dir / "corpus_coverage.json", corpus_coverage_payload_current()
        ),
        "action_sensing": _write_json(
            data_dir / "action_sensing.json", action_sensing_payload_current()
        ),
        "parameter_learning": _write_json(
            data_dir / "parameter_learning.json", parameter_learning_payload_current()
        ),
        "cross_architecture_fixtures": _write_json(
            data_dir / "cross_architecture_fixtures.json",
            cross_architecture_payload_current(),
        ),
        "active_information_frontier": _write_json(
            data_dir / "active_information_frontier.json",
            active_information_payload_current(),
        ),
        "quotient_invariance_protocol": write_quotient_invariance_protocol(
            output_dir.parent
        ),
        "representation_invariance_benchmark": write_representation_invariance_benchmark(
            output_dir.parent
        ),
        "intervention_power": str(
            local_path(write_intervention_power_payload(output_dir.parent))
        ),
        "scientific_program": write_scientific_program(output_dir.parent),
        "research_protocol": local_path(write_scientific_protocol(output_dir.parent)),
        "quotient_classifier": str(
            local_path(write_quotient_classifier_report(output_dir.parent))
        ),
        "formal_trace_correspondence": str(
            local_path(write_formal_correspondence(output_dir.parent))
        ),
        "compiler_transfer": str(
            local_path(write_compiler_transfer(output_dir.parent))
        ),
        "sensor_cost_receipt": str(
            local_path(write_sensor_cost_payload(output_dir.parent))
        ),
        "sensing_stress": str(local_path(write_sensing_stress(output_dir.parent))),
        "counterexample_search": str(
            local_path(write_counterexample_search(output_dir.parent))
        ),
        "replication_manifest": str(
            local_path(write_replication_bundle(output_dir.parent))
        ),
        "theorem_or_countertheorem": str(
            local_path(write_theorem_or_countertheorem(output_dir.parent))
        ),
        "cfg_evidence": _write_json(
            data_dir / "cfg_evidence.json", cfg_evidence_payload()
        ),
        "memory_evidence": _write_json(
            data_dir / "memory_evidence.json", memory_evidence_payload()
        ),
        "held_out_model": _write_json(
            data_dir / "held_out_model.json", held_out_model_payload()
        ),
    }
    # The canonical checkout has the pinned external-source manifest.  Small
    # temporary roots used by unit tests intentionally do not; absence there
    # must not turn a local fixture test into an implicit network operation.
    external_manifest = output_dir.parent / "configs" / "external_sources.yaml"
    if external_manifest.is_file():
        if not (data_dir / "external_acquisition.json").is_file():
            # A fresh offline checkout needs an explicit blocked/cache receipt.
            # Network acquisition remains exclusive to the opt-in CLI command.
            paths["external_acquisition"] = str(
                local_path(write_external_acquisition_payload(output_dir.parent))
            )
        external_design = output_dir.parent / "configs" / "external_label_design.yaml"
        if external_design.is_file():
            paths["external_label_contract"] = str(
                local_path(write_external_label_contract(output_dir.parent))
            )
        paths["external_capabilities"] = str(
            local_path(write_compiler_capabilities(output_dir.parent))
        )
        paths["external_static_validation"] = str(
            local_path(write_external_static_validation(output_dir.parent))
        )
        paths["external_source_validation"] = str(
            local_path(write_external_source_validation_payload(output_dir.parent))
        )
        paths["external_corpus_preparation_audit"] = str(
            local_path(write_external_corpus_preparation_audit(output_dir.parent))
        )
    if output_dir.resolve() == OUTPUT_DIR.resolve():
        paths["formal_verification"] = write_formal_verification(PROJECT_ROOT)
    return paths
