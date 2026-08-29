"""End-to-end tests for real assembly analysis dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from src.active_inference_bridge import infer_hidden_states
from src.assembly_formalism import (
    assembly_listing_markdown,
    assembly_listing_payload,
    build_assembly_listing,
    validate_assembly_listing_payload,
)
from src.assembly_packages.schema import TRACE_SCHEMA_VERSION, binary_hash
from src.claim_register import (
    CLAIM_RECORDS,
    claim_register_payload,
    write_claim_register,
)
from src.corpus import build_corpus
from src.disassembly import disassemble_sample, sample_bytes
from src.emulation import emulate_sample, emulate_x86_64
from src.experiments import PROJECT_ROOT, run_experiment
from src.features import (
    HIDDEN_STATE_LABELS,
    AssemblyObservation,
    encode_observations,
    observation_counts,
)
from src.figures import FIGURE_CAPTIONS, plot_observation_pipeline
from src.generative_model import build_generative_model
from src.manuscript_variables import _mapping_or_empty, generate_variables
from src.reproducibility import (
    MANIFEST_TARGETS,
    reproducibility_payload,
    verify_reproducibility_manifest,
    write_reproducibility_manifest,
)
from src.rxinfer_bridge import load_free_energy_records, validate_free_energy_records
from src.scholarship import (
    scholarship_keys,
    scholarship_payload,
    write_scholarship_sources,
)
from src.validation import validate_trace_file, write_validation_report


def test_capstone_decodes_expected_mnemonic_sequence() -> None:
    instructions = disassemble_sample()

    assert [event.mnemonic for event in instructions] == ["mov", "add", "shl", "sub"]
    assert [event.address for event in instructions] == [0x1000, 0x1005, 0x1008, 0x100A]
    assert instructions[0].op_str == "eax, 1"
    assert instructions[0].registers_written == ("eax",)
    assert instructions[0].immediate_operands == (1,)
    assert instructions[1].registers_read == ("eax",)
    assert "eax" in instructions[1].registers_written
    assert "rflags" in instructions[1].registers_written


def test_unicorn_bounded_emulation_reaches_expected_register_state() -> None:
    trace = emulate_sample()

    assert trace.instruction_count == 4
    assert trace.addresses == (0x1000, 0x1005, 0x1008, 0x100A)
    assert trace.initial_rax == 0
    assert trace.final_rax == 5
    assert trace.registers["rax"] == 5
    assert len(trace.instruction_states) == trace.instruction_count
    assert trace.instruction_states[0].pre_rax == trace.initial_rax
    assert trace.instruction_states[-1].post_rax == trace.final_rax
    assert all(
        left.post_rax == right.pre_rax
        for left, right in zip(trace.instruction_states, trace.instruction_states[1:])
    )


def test_unicorn_memory_hooks_capture_real_store_load_values() -> None:
    """Static memory operands join to the actual bounded Unicorn events."""

    instructions = disassemble_sample("x86_memory")
    trace = emulate_x86_64(sample_bytes("x86_memory"))

    assert [event.mnemonic for event in instructions] == ["mov", "mov", "add"]
    assert all(event.memory_operands for event in instructions[:2])
    assert instructions[0].immediate_operands == (7,)
    assert instructions[1].immediate_operands == ()
    assert instructions[2].memory_operands == ()
    assert [(event["access"], event["value"]) for event in trace.memory_events] == [
        ("write", 7),
        ("read", 7),
    ]
    assert all(
        event["address"] == trace.registers["rsp"] + 0x20
        for event in trace.memory_events
    )
    assert trace.final_rax == 8


def test_feature_encoding_covers_instruction_stream() -> None:
    instructions = disassemble_sample()
    trace = emulate_sample()
    observations = encode_observations(instructions, trace)

    assert [observation.category for observation in observations] == [
        "load_immediate",
        "arithmetic_update",
        "bitwise_transform",
        "arithmetic_update",
    ]
    assert all(observation.executed for observation in observations)
    assert all(observation.has_immediate_operand for observation in observations)
    assert all(observation.uses_accumulator for observation in observations)
    assert observation_counts(observations)["arithmetic_update"] == 2


def test_assembly_listing_is_a_source_bound_projection_of_real_trace() -> None:
    instructions = disassemble_sample()
    trace = emulate_sample()
    observations = encode_observations(instructions, trace)
    payload = assembly_listing_payload(
        "x86_accumulator", instructions, observations, trace
    )
    listing = assembly_listing_markdown(payload)

    assert payload["instruction_count"] == 4
    assert "0x1000" in listing
    assert "mov eax, 1" in listing
    assert "category=load_immediate" in listing
    trace_bundle = run_experiment().trace_bundle
    trace_payload = trace_bundle.to_dict()
    valid, detail = validate_assembly_listing_payload(payload, trace_payload)
    assert valid, detail


def test_assembly_listing_rejects_static_observation_address_drift() -> None:
    instructions = disassemble_sample()
    trace = emulate_sample()
    observations = encode_observations(instructions, trace)
    observations[0] = AssemblyObservation(
        index=0,
        address=0x2000,
        mnemonic=observations[0].mnemonic,
        operand_text=observations[0].operand_text,
        category=observations[0].category,
        observation_id=observations[0].observation_id,
        executed=observations[0].executed,
    )
    with pytest.raises(ValueError, match="addresses differ"):
        build_assembly_listing(instructions, observations, trace)


def test_assembly_listing_rejects_malformed_payloads_and_empty_inputs() -> None:
    """The source-bound listing validator fails closed on malformed evidence."""

    instructions = disassemble_sample()
    trace = emulate_sample()
    observations = encode_observations(instructions, trace)
    payload = assembly_listing_payload(
        "x86_accumulator", instructions, observations, trace
    )
    trace_payload = {
        "static_instruction_events": [event.to_dict() for event in instructions],
        "observations": [observation.to_dict() for observation in observations],
    }

    assert validate_assembly_listing_payload({}, {})[0] is False
    assert (
        validate_assembly_listing_payload(
            {**payload, "schema_version": "0.0"}, trace_payload
        )[0]
        is False
    )
    assert (
        validate_assembly_listing_payload(
            {**payload, "instructions": []}, trace_payload
        )[0]
        is False
    )
    drifted = [dict(row) for row in payload["instructions"]]
    drifted[0]["mnemonic"] = "not_mov"
    assert (
        validate_assembly_listing_payload(
            {**payload, "instructions": drifted}, trace_payload
        )[0]
        is False
    )

    with pytest.raises(ValueError, match="must not be empty"):
        build_assembly_listing([], [], trace)
    with pytest.raises(ValueError, match="equal length"):
        build_assembly_listing(instructions, observations[:-1], trace)
    with pytest.raises(ValueError, match="instruction rows"):
        assembly_listing_markdown({"sample_id": "bad", "instructions": []})
    with pytest.raises(ValueError, match="instruction rows must be objects"):
        assembly_listing_markdown({"sample_id": "bad", "instructions": ["bad"]})


def test_observation_pipeline_rejects_conflicting_category_for_id(
    tmp_path: Path,
) -> None:
    observations = [
        AssemblyObservation(0, 0x1000, "mov", "eax, 1", "load_immediate", 0, True),
        AssemblyObservation(1, 0x1001, "add", "eax, 1", "arithmetic_update", 0, True),
    ]
    with pytest.raises(ValueError, match="must map to one category"):
        plot_observation_pipeline(observations, tmp_path / "conflict.png")


def test_generative_model_probability_tensors_normalize() -> None:
    model = build_generative_model()

    assert model.A.shape == (6, 5)
    assert model.D.shape == (5,)
    assert np.allclose(model.A.sum(axis=0), 1.0)
    assert np.isclose(model.D.sum(), 1.0)


def test_pymdp_variational_updates_are_finite() -> None:
    observations = encode_observations(disassemble_sample(), emulate_sample())
    model = build_generative_model()
    inference = infer_hidden_states(observations, model)

    assert len(inference.posterior_by_step) == len(observations)
    assert all(np.isclose(sum(row), 1.0) for row in inference.posterior_by_step)
    assert all(math.isfinite(value) for value in inference.free_energy_by_step)
    assert inference.final_state in model.hidden_state_labels


def test_run_experiment_writes_required_artifacts() -> None:
    result = run_experiment()

    expected_data = {
        "disassembly_trace.json",
        "emulation_trace.json",
        "inference_result.json",
        "trace_bundle.json",
    }
    written_data = {Path(path).name for path in result.data_paths.values()}
    assert expected_data.issubset(written_data)
    for path in [*result.data_paths.values(), *result.figure_paths.values()]:
        artifact = PROJECT_ROOT / path
        assert artifact.exists(), artifact
        assert artifact.stat().st_size > 0

    inference_payload = json.loads(
        (PROJECT_ROOT / result.data_paths["inference_result"]).read_text()
    )
    assert inference_payload["inference"]["final_state"] == result.inference.final_state


def test_trace_bundle_schema_records_package_provenance_and_hashes() -> None:
    result = run_experiment()
    bundle_path = PROJECT_ROOT / result.data_paths["trace_bundle"]
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == TRACE_SCHEMA_VERSION
    assert payload["sample_id"] == "x86_accumulator"
    assert payload["architecture"] == "x86_64"
    assert payload["binary_sha256"] == binary_hash("x86_accumulator")
    assert payload["basic_blocks"][0]["instruction_indexes"] == [0, 1, 2, 3]
    assert len(payload["static_instruction_events"]) == 4
    assert len(payload["observations"]) == 4
    assert len(payload["typed_observations"]) == 4
    assert payload["typed_observations"][1]["flag_effect"] == "write"
    assert len(payload["symbolic_facts"]) == 4
    assert len(payload["dynamic_register_trace"]) == 6
    assert all(
        row["event"] == "instruction" for row in payload["dynamic_register_trace"][1:-1]
    )
    assert payload["dynamic_register_trace"][1]["registers_after"]["rax"] == 1
    assert payload["cfg_edges"] == []
    provenance = {row["name"]: row for row in payload["package_provenance"]}
    assert provenance["capstone"]["available"] is True
    assert provenance["capstone"]["optional"] is False
    assert provenance["unicorn"]["available"] is True
    assert provenance["angr"]["optional"] is True
    for optional_name in (
        "ghidra",
        "bap",
        "gtirb",
        "qiling",
        "miasm",
        "dyninst",
        "retdec",
        "mcsema",
    ):
        assert provenance[optional_name]["optional"] is True
    assert (PROJECT_ROOT / result.figure_paths["package_provenance"]).exists()


def test_optional_replay_sections_fail_closed_to_empty_mappings() -> None:
    section = {"control_scope": "same_host_separately_controlled_guest"}

    assert _mapping_or_empty(section) == section
    assert _mapping_or_empty(["malformed"]) == {}


def test_manuscript_variables_are_generated_from_live_result() -> None:
    variables = generate_variables()

    assert variables["PROJECT_TITLE"] == "Active Assembly Theory and Practice"
    assert variables["INSTRUCTION_COUNT"] == "4"
    assert variables["FINAL_RAX"] == "5"
    assert variables["TRACE_SCHEMA_VERSION"] == TRACE_SCHEMA_VERSION
    assert variables["RXINFER_FREE_ENERGY_COUNT"] == "8"
    assert variables["RXINFER_VERSION"]
    assert variables["SCHOLARSHIP_SOURCE_COUNT"] == str(len(scholarship_keys()))
    assert variables["CLAIM_COUNT"] == str(len(CLAIM_RECORDS))
    assert variables["VALIDATED_CLAIM_COUNT"] == str(len(CLAIM_RECORDS))
    assert variables["GENERATIVE_ADEQUACY_REPLICATES"] == "256"
    assert variables["GENERATIVE_ADEQUACY_METRIC_COUNT"] == str(
        len(build_corpus()) * (3 + len(HIDDEN_STATE_LABELS))
    )
    assert variables["GENERATIVE_ADEQUACY_FIGURE"].endswith("generative_adequacy.png")
    assert variables["REPRODUCIBILITY_ARTIFACT_COUNT"] == str(len(MANIFEST_TARGETS))
    assert variables["VALIDATION_ALL_PASSED"] == "true"
    assert variables["SYMBOLIC_FINAL_RAX_MATCH"] == "yes"
    assert variables["SYMBOLIC_UNSUPPORTED_COUNT"] == "0"
    assert variables["MEMORY_DYNAMIC_EVENT_COUNT"] == "2"
    assert variables["MEMORY_READ_COUNT"] == "1"
    assert variables["MEMORY_WRITE_COUNT"] == "1"
    assert variables["RXINFER_CATEGORICAL_MODEL_NAME"] == "categorical phase hmm"
    assert variables["RXINFER_CATEGORICAL_FREE_ENERGY_COUNT"] == "8"
    assert variables["HELD_OUT_MEAN_ACCURACY"]
    assert variables["ACTION_SENSING_LEAKAGE_FREE"] == "yes"
    assert variables["ACTION_SENSING_ACTIVE_MEAN_ACTIONS"]
    assert variables["ACTION_SENSING_FIXED_MEAN_ACTIONS"]
    assert variables["ACTION_SENSING_RANDOM_CONFIDENCE_RATE"]
    assert variables["PARAMETER_LEARNING_LEAKAGE_FREE"] == "yes"
    assert variables["PARAMETER_LEARNING_HOLDOUT_COUNT"] == "6"
    assert variables["PARAMETER_LEARNING_EM_MEAN_ACCURACY"] == "0.428"
    assert variables["PARAMETER_LEARNING_SUPERVISED_MEAN_ACCURACY"] == "0.772"
    assert variables["PARAMETER_LEARNING_REFERENCE_MEAN_ACCURACY"] == "0.519"
    assert variables["CROSS_ARCHITECTURE_COUNT"] == "3"
    assert variables["CROSS_ARCHITECTURE_ALL_PASSED"] == "yes"
    assert variables["CROSS_ARCHITECTURE_RESULT"] == "8"
    assert variables["FORMAL_VERIFICATION_STATUS"] == "locally_verified"
    assert variables["FORMAL_THEOREM_COUNT"] == "12"
    assert variables["ACTIVE_INFORMATION_SENSOR_COUNT"] == "4"
    assert variables["ACTIVE_INFORMATION_FRONTIER_ROW_COUNT"] == "16"
    assert variables["ACTIVE_INFORMATION_EXACT"] == "yes"
    assert (
        variables["COVER_IMAGE_ARTIFACT"]
        == "manuscript/assets/active_assembly_cover.png"
    )
    assert math.isfinite(float(variables["RXINFER_FINAL_FREE_ENERGY"]))
    assert (PROJECT_ROOT / "output/data/manuscript_variables.json").exists()
    assert (PROJECT_ROOT / "output/data/manuscript_variable_registry.json").exists()
    assert (PROJECT_ROOT / "output/data/figure_registry.json").exists()
    template_figure_registry = json.loads(
        (PROJECT_ROOT / "output/figures/figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(template_figure_registry["figures"]) == len(FIGURE_CAPTIONS)
    assert all(
        row.get("label", "").startswith("fig:")
        and row.get("filename")
        and row.get("alt")
        for row in template_figure_registry["figures"]
    )
    assert (PROJECT_ROOT / "output/data/build_provenance.json").exists()
    provenance_payload = json.loads(
        (PROJECT_ROOT / "output/data/build_provenance.json").read_text(encoding="utf-8")
    )
    assert (
        "src/assembly_packages/schema.py"
        in provenance_payload["lineage"]["source_hashes"]
    )
    assert provenance_payload["template_repository"]["revision"]
    assert "worktree_status" in provenance_payload["template_repository"]
    assert "worktree_diff_sha256" in provenance_payload["template_repository"]
    assert (
        "infrastructure/rendering/manuscript_injection.py"
        in provenance_payload["template_repository"]["implementation_hashes"]
    )
    assert (
        "infrastructure/rendering/_pdf_title_page_images.py"
        in provenance_payload["template_repository"]["implementation_hashes"]
    )
    assert (
        "output/rxinfer_categorical/categorical_diagnostics.json"
        in (provenance_payload["lineage"]["generated_hashes"])
    )
    assert (PROJECT_ROOT / "output/data/symbolic_execution.json").exists()
    assert (PROJECT_ROOT / "output/data/cfg_evidence.json").exists()
    assert (PROJECT_ROOT / "output/data/memory_evidence.json").exists()
    assert (PROJECT_ROOT / "output/data/action_sensing.json").exists()
    assert (PROJECT_ROOT / "output/data/parameter_learning.json").exists()
    assert (PROJECT_ROOT / "output/data/cross_architecture_fixtures.json").exists()
    assert (PROJECT_ROOT / "output/data/formal_verification.json").exists()
    assert (PROJECT_ROOT / "output/data/active_information_frontier.json").exists()
    assert (PROJECT_ROOT / "output/data/intervention_power.json").exists()
    assert (PROJECT_ROOT / "output/data/held_out_model.json").exists()
    assert (PROJECT_ROOT / "manuscript/assets/active_assembly_cover.png").exists()
    assert variables["ASSEMBLY_CFG_FIGURE"].endswith("assembly_cfg.png")
    assert (PROJECT_ROOT / "output/figures/rxinfer_free_energy.png").exists()
    assert (
        PROJECT_ROOT / "output/figures/rxinfer_categorical_free_energy.png"
    ).exists()
    assert (PROJECT_ROOT / "output/figures/assembly_memory_timeline.png").exists()
    assert (PROJECT_ROOT / "output/figures/trace_feature_matrix.png").exists()
    assert (PROJECT_ROOT / "output/figures/inference_diagnostics.png").exists()
    assert (PROJECT_ROOT / "output/figures/scholarship_source_roles.png").exists()
    assert (PROJECT_ROOT / "output/figures/scholarship_evidence_matrix.png").exists()
    assert (PROJECT_ROOT / "output/figures/claim_evidence_map.png").exists()
    assert (PROJECT_ROOT / "output/figures/artifact_hash_manifest.png").exists()
    assert (PROJECT_ROOT / "output/figures/validation_summary.png").exists()
    assert (PROJECT_ROOT / "output/figures/action_sensing.png").exists()
    assert (PROJECT_ROOT / "output/figures/parameter_learning.png").exists()
    assert (PROJECT_ROOT / "output/figures/cross_architecture_benchmark.png").exists()
    assert (PROJECT_ROOT / "output/figures/formal_verification_receipt.png").exists()
    assert (PROJECT_ROOT / "output/figures/active_information_frontier.png").exists()
    assert (PROJECT_ROOT / "output/figures/intervention_power.png").exists()
    assert variables["ACTION_SENSING_FIGURE"].endswith("action_sensing.png")
    assert variables["PARAMETER_LEARNING_FIGURE"].endswith("parameter_learning.png")
    assert variables["INTERVENTION_POWER_FIGURE"].endswith("intervention_power.png")
    assert variables["TRACE_FEATURE_FIGURE"].endswith("trace_feature_matrix.png")
    assert variables["INFERENCE_DIAGNOSTIC_FIGURE"].endswith(
        "inference_diagnostics.png"
    )
    assert variables["REPRODUCIBILITY_FIGURE"].endswith("artifact_hash_manifest.png")
    assert variables["CALIBRATION_CURVE_FIGURE"].endswith("calibration_curve.png")
    assert variables["DECODER_CALIBRATION_FIGURE"].endswith("decoder_calibration.png")

    records = load_free_energy_records(
        PROJECT_ROOT / "output/rxinfer/free_energy_trace.jsonl"
    )
    validate_free_energy_records(records)
    assert len(records) == 8
    assert {record["source_field"] for record in records} == {"result.free_energy"}


def test_rxinfer_free_energy_validation_rejects_nonfinite_values() -> None:
    records = [
        {
            "run_id": "bad",
            "sample_id": "bad",
            "model_name": "gaussian_event_code_model",
            "mode": "static",
            "iteration": 1,
            "observation_index": None,
            "value": float("nan"),
            "source_field": "result.free_energy",
            "is_finite": False,
            "delta": None,
            "rxinfer_version": "5.4.0",
            "julia_version": "1.12.1",
            "data_hash": "0" * 64,
            "config_hash": "1" * 64,
        }
    ]
    try:
        validate_free_energy_records(records)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("non-finite free energy record was accepted")


def test_scholarship_ledger_matches_bibliography() -> None:
    relative_path = write_scholarship_sources()
    payload = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    references = (PROJECT_ROOT / "manuscript/references.bib").read_text(
        encoding="utf-8"
    )

    keys = [row["key"] for row in payload["sources"]]
    assert len(keys) == payload["source_count"]
    assert len(keys) == len(set(keys))
    assert scholarship_keys() == set(keys)
    for key in keys:
        assert f"{{{key}," in references
    assert all(row["claim_boundary"] for row in payload["sources"])
    assert all(row["url"].startswith("https://") for row in payload["sources"])
    assert all(row["source_type"] for row in payload["sources"])
    assert all(row["evidence_tier"] for row in payload["sources"])
    assert all(row["project_stage"] for row in payload["sources"])
    assert payload["project_stage_counts"]["roadmap"] >= 10
    assert (PROJECT_ROOT / "output/figures/scholarship_source_roles.png").exists()
    assert (PROJECT_ROOT / "output/figures/scholarship_evidence_matrix.png").exists()


def test_scholarship_payload_partitions_evidence_roles() -> None:
    payload = scholarship_payload()

    assert payload["source_count"] == len(scholarship_keys())
    assert payload["project_stage_counts"]["implemented"] >= 4
    assert payload["project_stage_counts"]["comparison"] >= 1
    assert payload["project_stage_counts"]["context"] >= 3
    assert payload["project_stage_counts"]["roadmap"] >= 10
    assert payload["evidence_tier_counts"]["roadmap_adapter"] >= 10
    assert payload["evidence_tier_counts"]["implemented_dependency"] >= 2


def test_claim_register_links_sources_artifacts_and_validation_checks() -> None:
    generate_variables()
    relative_path = write_claim_register()
    payload = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))

    assert payload["claim_count"] == len(CLAIM_RECORDS)
    assert payload["status_counts"]["validated"] == len(CLAIM_RECORDS)
    assert {row["claim_id"] for row in payload["claims"]} == {
        claim.claim_id for claim in CLAIM_RECORDS
    }
    known_source_keys = scholarship_keys()
    for row in payload["claims"]:
        assert set(row["support_keys"]).issubset(known_source_keys)
        assert row["artifact_paths"]
        assert all((PROJECT_ROOT / path).exists() for path in row["artifact_paths"])
        assert row["validation_checks"]
        assert row["boundary"]
    assert (PROJECT_ROOT / "output/figures/claim_evidence_map.png").exists()


def test_claim_register_payload_partitions_claim_types() -> None:
    payload = claim_register_payload()

    assert payload["claim_count"] == len(CLAIM_RECORDS)
    assert payload["type_counts"]["implemented_trace"] == 2
    assert payload["type_counts"]["rxinfer_backend"] == 1
    assert payload["type_counts"]["artifact_reproducibility"] == 1
    assert payload["status_counts"]["validated"] == len(CLAIM_RECORDS)


def test_reproducibility_manifest_records_current_artifact_hashes() -> None:
    generate_variables()
    relative_path = write_reproducibility_manifest()
    payload = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))

    assert payload["artifact_count"] == len(MANIFEST_TARGETS)
    assert payload["all_required_present"] is True
    assert payload["missing_required_count"] == 0
    assert verify_reproducibility_manifest(PROJECT_ROOT) == []
    rows = payload["artifacts"]
    assert {row["path"] for row in rows} == {target.path for target in MANIFEST_TARGETS}
    assert all(
        (row["status"] == "present") == (isinstance(row["sha256"], str)) for row in rows
    )
    assert all(
        row["status"] == "missing" or (len(row["sha256"]) == 64 and row["bytes"] > 0)
        for row in rows
    )
    assert reproducibility_payload()["artifact_count"] == len(MANIFEST_TARGETS)
    assert (PROJECT_ROOT / "output/data/reproducibility_manifest.md").exists()
    assert (PROJECT_ROOT / "output/figures/artifact_hash_manifest.png").exists()


def test_validation_report_checks_outputs_and_nonblank_figures() -> None:
    generate_variables()
    report = write_validation_report()

    assert report.summary["all_passed"] is True
    assert report.summary["failed"] == 0
    assert report.summary["total"] >= 35
    figure_path = PROJECT_ROOT / report.artifact_paths["validation_summary"]
    image = np.asarray(mpimg.imread(figure_path), dtype=float)
    assert image.size > 0
    assert float(np.var(image)) > 0.00001


def test_trace_validator_negative_control_rejects_malformed_bundle(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "trace_bundle.json"
    malformed.write_text(
        json.dumps(
            {
                "schema_version": "bad",
                "sample_id": "not_a_fixture",
                "binary_sha256": "0" * 64,
                "static_instruction_events": [],
                "observations": [],
                "package_provenance": [],
                "basic_blocks": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    checks = validate_trace_file(malformed)
    failed = {check.name for check in checks if not check.passed}
    assert "trace_schema_version" in failed
    assert "trace_binary_hash" in failed
    assert "trace_instruction_observation_alignment" in failed
    assert "trace_required_package_provenance" in failed
    assert "trace_optional_roadmap_provenance" in failed
    assert "trace_observation_semantics" in failed
    assert "trace_dynamic_register_contract" in failed


def test_trace_validator_negative_control_rejects_observation_category_drift(
    tmp_path: Path,
) -> None:
    result = run_experiment()
    payload = json.loads((PROJECT_ROOT / result.data_paths["trace_bundle"]).read_text())
    payload["observations"][0]["category"] = "arithmetic_update"
    malformed = tmp_path / "trace_bundle.json"
    malformed.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    checks = validate_trace_file(malformed)
    failed = {check.name for check in checks if not check.passed}
    assert "trace_observation_semantics" in failed


def test_unicorn_emulation_stops_at_max_instructions() -> None:
    trace = emulate_x86_64(sample_bytes(), max_instructions=2)
    assert trace.instruction_count == 2
    assert trace.addresses == (0x1000, 0x1005)


def _manifest_row(target, **overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "path": target.path,
        "artifact_type": target.artifact_type,
        "sha256": "0" * 64,
        "bytes": 1,
        "required": target.required,
        "generated_by": target.generated_by,
        "claim_ids": list(target.claim_ids),
        "validation_checks": list(target.validation_checks),
        "status": "missing",
    }
    row.update(overrides)
    return row


def _write_fake_manifest(project_root: Path, rows: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {"artifact_count": len(rows), "artifacts": rows}
    data_dir = project_root / "output" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "reproducibility_manifest.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


def test_reproducibility_verify_detects_missing_artifact(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, [_manifest_row(t) for t in MANIFEST_TARGETS])
    errors = verify_reproducibility_manifest(tmp_path)
    assert any("missing_artifact=output/" in error for error in errors)


def test_reproducibility_verify_detects_hash_and_size_mismatch(tmp_path: Path) -> None:
    target = MANIFEST_TARGETS[0]
    artifact = tmp_path / target.path
    artifact.parent.mkdir(parents=True)
    content = b"tampered-bytes"
    artifact.write_bytes(content)
    rows = [
        _manifest_row(
            t,
            sha256="0" * 64,
            bytes=len(content) + 1,
            status="present" if t is target else "missing",
        )
        for t in MANIFEST_TARGETS
    ]
    _write_fake_manifest(tmp_path, rows)
    errors = verify_reproducibility_manifest(tmp_path)
    assert any("hash_mismatch" in error for error in errors)
    assert any("size_mismatch" in error for error in errors)
