"""End-to-end tests for real assembly analysis dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from src.active_inference_bridge import infer_hidden_states
from src.assembly_packages.schema import TRACE_SCHEMA_VERSION, binary_hash
from src.claim_register import CLAIM_RECORDS, claim_register_payload, write_claim_register
from src.disassembly import disassemble_sample
from src.emulation import emulate_sample
from src.experiments import PROJECT_ROOT, run_experiment
from src.features import encode_observations, observation_counts
from src.generative_model import build_generative_model
from src.manuscript_variables import generate_variables
from src.redteam_report import REDTEAM_FINDINGS, write_redteam_report
from src.reproducibility import (
    MANIFEST_TARGETS,
    reproducibility_payload,
    verify_reproducibility_manifest,
    write_reproducibility_manifest,
)
from src.rxinfer_bridge import load_free_energy_records, validate_free_energy_records
from src.scholarship import scholarship_keys, scholarship_payload, write_scholarship_sources
from src.validation import validate_trace_file, write_validation_report


def test_capstone_decodes_expected_mnemonic_sequence() -> None:
    instructions = disassemble_sample()

    assert [event.mnemonic for event in instructions] == ["mov", "add", "shl", "sub"]
    assert [event.address for event in instructions] == [0x1000, 0x1005, 0x1008, 0x100A]
    assert instructions[0].op_str == "eax, 1"


def test_unicorn_bounded_emulation_reaches_expected_register_state() -> None:
    trace = emulate_sample()

    assert trace.instruction_count == 4
    assert trace.addresses == (0x1000, 0x1005, 0x1008, 0x100A)
    assert trace.initial_rax == 0
    assert trace.final_rax == 5
    assert trace.registers["rax"] == 5


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
    assert observation_counts(observations)["arithmetic_update"] == 2


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

    inference_payload = json.loads((PROJECT_ROOT / result.data_paths["inference_result"]).read_text())
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
    provenance = {row["name"]: row for row in payload["package_provenance"]}
    assert provenance["capstone"]["available"] is True
    assert provenance["capstone"]["optional"] is False
    assert provenance["unicorn"]["available"] is True
    assert provenance["angr"]["optional"] is True
    for optional_name in ("ghidra", "bap", "gtirb", "qiling", "miasm", "dyninst", "retdec", "mcsema"):
        assert provenance[optional_name]["optional"] is True
    assert (PROJECT_ROOT / result.figure_paths["package_provenance"]).exists()


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
    assert variables["REDTEAM_FINDING_COUNT"] == str(len(REDTEAM_FINDINGS))
    assert variables["REDTEAM_REMEDIATED_COUNT"] == str(len(REDTEAM_FINDINGS))
    assert variables["REPRODUCIBILITY_ARTIFACT_COUNT"] == str(len(MANIFEST_TARGETS))
    assert variables["VALIDATION_ALL_PASSED"] == "true"
    assert math.isfinite(float(variables["RXINFER_FINAL_FREE_ENERGY"]))
    assert (PROJECT_ROOT / "output/data/manuscript_variables.json").exists()
    assert (PROJECT_ROOT / "output/figures/rxinfer_free_energy.png").exists()
    assert (PROJECT_ROOT / "output/figures/trace_feature_matrix.png").exists()
    assert (PROJECT_ROOT / "output/figures/inference_diagnostics.png").exists()
    assert (PROJECT_ROOT / "output/figures/scholarship_source_roles.png").exists()
    assert (PROJECT_ROOT / "output/figures/scholarship_evidence_matrix.png").exists()
    assert (PROJECT_ROOT / "output/figures/claim_evidence_map.png").exists()
    assert (PROJECT_ROOT / "output/figures/redteam_risk_register.png").exists()
    assert (PROJECT_ROOT / "output/figures/artifact_hash_manifest.png").exists()
    assert (PROJECT_ROOT / "output/figures/validation_summary.png").exists()
    assert variables["TRACE_FEATURE_FIGURE"].endswith("trace_feature_matrix.png")
    assert variables["INFERENCE_DIAGNOSTIC_FIGURE"].endswith("inference_diagnostics.png")
    assert variables["REPRODUCIBILITY_FIGURE"].endswith("artifact_hash_manifest.png")

    records = load_free_energy_records(PROJECT_ROOT / "output/rxinfer/free_energy_trace.jsonl")
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
    references = (PROJECT_ROOT / "manuscript/references.bib").read_text(encoding="utf-8")

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
    assert payload["evidence_tier_counts"]["implemented_dependency"] == 2


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


def test_redteam_report_is_durable_and_remediated() -> None:
    relative_path = write_redteam_report()
    payload = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))

    assert payload["finding_count"] == len(REDTEAM_FINDINGS)
    assert payload["remediated_count"] == len(REDTEAM_FINDINGS)
    assert payload["open_count"] == 0
    assert {row["status"] for row in payload["findings"]} == {"remediated"}
    assert all(row["negative_control"] for row in payload["findings"])
    assert (PROJECT_ROOT / "output/figures/redteam_risk_register.png").exists()


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
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["bytes"] > 0 for row in rows)
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


def test_trace_validator_negative_control_rejects_malformed_bundle(tmp_path: Path) -> None:
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
