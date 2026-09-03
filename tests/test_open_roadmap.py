"""Tests for the source-bound open-roadmap implementation tranche."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import src.formal_correspondence as formal_module
import src.scientific_protocol as scientific_protocol_module
import src.theorem_phase as theorem_module

from src.compiler_transfer import (
    _compile_one,
    _decode_bytes,
    _extract_text_bytes,
    compiler_transfer_payload,
    validate_compiler_transfer_payload,
)
from src.formal_correspondence import (
    FormalState,
    _reference_trace,
    _step,
    formal_correspondence_payload,
    validate_formal_correspondence_payload,
)
from src.disassembly import InstructionEvent
from src.provenance import (
    generator_revision,
    input_hashes,
    sha256_bytes,
    sha256_path,
    toolchain_versions,
)
from src.quotient_classifier import (
    quotient_classifier_payload,
    validate_quotient_classifier_payload,
    write_quotient_classifier_report,
)
from src.replication import (
    run_label_blind_model,
    score_replication_predictions,
    validate_replication_bundle,
    write_replication_bundle,
)
from src.scientific_protocol import (
    _canonical_hash,
    build_scientific_protocol,
    scientific_protocol_payload,
    validate_scientific_protocol_payload,
    write_scientific_protocol,
)
from src.sensor_costs import (
    sensor_cost_payload,
    validate_sensor_cost_payload,
    write_sensor_cost_payload,
)
from src.theorem_phase import (
    theorem_or_countertheorem_payload,
    validate_theorem_or_countertheorem_payload,
)


def test_scientific_protocol_is_hashed_and_label_blind() -> None:
    payload = scientific_protocol_payload()
    assert validate_scientific_protocol_payload(payload) == (True, "ok")
    assert payload["observational_contract"]["label_blind"] is True
    mutated = deepcopy(payload)
    mutated["model_contract"]["co_primary"] = False
    assert not validate_scientific_protocol_payload(mutated)[0]


def test_scientific_protocol_hash_is_independent_of_runtime_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = scientific_protocol_payload()
    monkeypatch.setattr(
        scientific_protocol_module,
        "toolchain_versions",
        lambda _packages: {
            "python": "0.0-test",
            "platform": "test-platform",
            "capstone": "test-capstone",
            "numpy": "test-numpy",
            "unicorn": "test-unicorn",
        },
    )
    second = scientific_protocol_payload()
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["toolchain_versions"] != second["toolchain_versions"]
    assert validate_scientific_protocol_payload(second) == (True, "ok")
    assert first["protocol_sha256"] == _canonical_hash(
        build_scientific_protocol().to_dict()
    )


def test_quotient_classifier_has_disjoint_families_and_three_models() -> None:
    payload = quotient_classifier_payload()
    valid, detail = validate_quotient_classifier_payload(payload)
    assert valid, detail
    assert payload["family_count"] == 64
    assert payload["examples_per_family"] == 64
    assert payload["split_overlap"] is False
    assert set(payload["models"]) == {
        "calibrated_interpretable",
        "temporal_probabilistic",
        "neural_sequence",
        "surface_baseline",
    }
    assert all(
        0.0 <= payload["models"][name]["balanced_accuracy"] <= 1.0
        for name in payload["models"]
    )
    assert payload["neural_gradient_check"]["passed"] is True
    assert all(
        payload["seed_sensitivity"][name]["seed_count"] >= 2
        for name in payload["models"]
    )


def test_quotient_classifier_rejects_label_leakage_and_writes(tmp_path) -> None:
    payload = quotient_classifier_payload()
    payload["label_blind_features"] = False
    assert not validate_quotient_classifier_payload(payload)[0]
    path = write_quotient_classifier_report(tmp_path)
    assert (
        json.loads((tmp_path / path).read_text(encoding="utf-8"))["row_count"] == 4096
    )


def test_formal_correspondence_matches_every_transparent_step() -> None:
    payload = formal_correspondence_payload()
    valid, detail = validate_formal_correspondence_payload(payload)
    assert valid, detail
    assert payload["status"] == "validated_bounded"
    assert payload["fixture_count"] >= 3
    assert all(row["checks"]["matched_at_every_step"] for row in payload["step_rows"])


def test_formal_correspondence_writer_fails_closed_on_invalid_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        formal_module,
        "formal_correspondence_payload",
        lambda project_root: {"status": "invalid"},
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        formal_module.write_formal_correspondence(tmp_path)


def test_formal_correspondence_validator_covers_structural_failures() -> None:
    payload = formal_correspondence_payload()
    mutations = []

    def mutate_status(candidate: dict[str, Any]) -> None:
        candidate["status"] = "failed"

    def mutate_bridge(candidate: dict[str, Any]) -> None:
        cast(dict[str, Any], candidate["cross_language_bridge"])[
            "trace_matches_lean_expected"
        ] = False

    def mutate_subset(candidate: dict[str, Any]) -> None:
        candidate["supported_instruction_subset"] = []

    def mutate_state(candidate: dict[str, Any]) -> None:
        cast(list[str], candidate["state"]).remove("rax")

    def mutate_rows(candidate: dict[str, Any]) -> None:
        candidate["step_rows"] = []

    def mutate_row_match(candidate: dict[str, Any]) -> None:
        cast(
            dict[str, Any],
            cast(list[dict[str, Any]], candidate["step_rows"])[0]["checks"],
        )["matched_at_every_step"] = False

    def mutate_unsupported(candidate: dict[str, Any]) -> None:
        cast(
            dict[str, Any],
            cast(list[dict[str, Any]], candidate["step_rows"])[0]["checks"],
        )["unsupported"] = "unsupported"

    def mutate_missing(candidate: dict[str, Any]) -> None:
        cast(list[dict[str, Any]], candidate["step_rows"])[0].pop("reference_steps")

    def mutate_fixture_count(candidate: dict[str, Any]) -> None:
        candidate["fixture_count"] = 0

    def mutate_intervention_count(candidate: dict[str, Any]) -> None:
        candidate["intervention_count"] = 0

    mutations.extend(
        (
            mutate_status,
            mutate_bridge,
            mutate_subset,
            mutate_state,
            mutate_rows,
            mutate_row_match,
            mutate_unsupported,
            mutate_missing,
            mutate_fixture_count,
            mutate_intervention_count,
        )
    )
    for mutation in mutations:
        candidate = deepcopy(payload)
        mutation(candidate)
        assert not validate_formal_correspondence_payload(candidate)[0]


def test_compiler_transfer_records_unavailable_targets_without_fallback() -> None:
    payload = compiler_transfer_payload()
    valid, detail = validate_compiler_transfer_payload(payload)
    assert valid, detail
    assert payload["execution_performed"] is False
    assert any(row["status"] == "unavailable" for row in payload["rows"])
    assert any(row["status"] == "validated_static_object" for row in payload["rows"])
    assert payload["capability_summary"]["source_family_count"] >= 6
    assert set(payload["capability_summary"]["optimization_levels_validated"]) == {
        "-O0",
        "-O2",
    }


def test_sensor_costs_have_vectors_and_finite_oracle_certificates() -> None:
    payload = sensor_cost_payload()
    valid, detail = validate_sensor_cost_payload(payload)
    assert valid, detail
    assert payload["primary_cost"] == "deterministic_vector"
    assert all(
        row["within_preregistered_tolerance"] for row in payload["policy_certificates"]
    )


def test_replication_bundle_keeps_public_labels_separate(tmp_path) -> None:
    write_replication_bundle(tmp_path)
    valid, detail = validate_replication_bundle(tmp_path)
    assert valid, detail
    public = json.loads(
        (tmp_path / "output/replication/public/public_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert public
    assert all("label" not in row for row in public)
    assert all("family_id" not in row for row in public)


def test_protocol_writer_uses_requested_root(tmp_path) -> None:
    path = write_scientific_protocol(tmp_path)
    assert path == tmp_path / "output/data/research_protocol.json"


def test_theorem_phase_promotes_counterexamples_fail_closed() -> None:
    payload = theorem_or_countertheorem_payload()
    valid, detail = validate_theorem_or_countertheorem_payload(payload)
    assert valid, detail
    assert payload["promotion_status"] == "fail_closed_no_broad_theorem"
    assert payload["counterexample"]["family_id"] == "zeroing_idiom"
    statuses = {row["status"] for row in payload["candidate_theorems"]}
    assert {
        "proved",
        "computationally_validated",
        "falsified_by_counterexample",
        "unresolved",
    }.issubset(statuses)


def _event(
    mnemonic: str,
    op_str: str = "",
    *,
    immediate: tuple[int, ...] = (),
    memory: tuple[dict[str, object], ...] = (),
) -> InstructionEvent:
    return InstructionEvent(
        address=0x1000,
        size=1,
        mnemonic=mnemonic,
        op_str=op_str,
        bytes_hex="90",
        groups=(),
        immediate_operands=immediate,
        memory_operands=memory,
    )


def test_formal_reference_core_exercises_declared_instruction_families() -> None:
    state = FormalState(rax=0xFFFFFFFF, rsp=0x2000, rip=0x1000, eflags=2, memory={})
    _step(_event("nop"), state, 0x1001)  # type: ignore[arg-type]
    _step(_event("mov", "eax, 7", immediate=(7,)), state, 0x1002)  # type: ignore[arg-type]
    _step(
        _event(
            "mov",
            "dword ptr [rsp + 4], 7",
            immediate=(7,),
            memory=({"base": "rsp", "displacement": 4, "size": 4},),
        ),
        state,
        0x1004,
    )
    _step(
        _event(
            "mov",
            "eax, dword ptr [rsp + 4]",
            memory=({"base": "rsp", "displacement": 4, "size": 4},),
        ),
        state,
        0x1005,
    )
    _step(_event("mov", "rcx, rax"), state, 0x1006)  # type: ignore[arg-type]
    _step(_event("add", "rax, rcx"), state, 0x1007)  # type: ignore[arg-type]
    _step(_event("adc", "rax, 1", immediate=(1,)), state, 0x1008)  # type: ignore[arg-type]
    _step(_event("sub", "rax, 1", immediate=(1,)), state, 0x1009)  # type: ignore[arg-type]
    _step(_event("sbb", "rax, 1", immediate=(1,)), state, 0x100A)  # type: ignore[arg-type]
    _step(_event("cmp", "rax, 1", immediate=(1,)), state, 0x100B)  # type: ignore[arg-type]
    _step(_event("inc", "rax"), state, 0x100C)  # type: ignore[arg-type]
    _step(_event("dec", "rax"), state, 0x100D)  # type: ignore[arg-type]
    for mnemonic, op_str, immediate in (
        ("xor", "rax, rax", ()),
        ("or", "rax, rax", ()),
        ("and", "rax, 3", (3,)),
        ("shl", "rax, 1", (1,)),
        ("sal", "rax, 1", (1,)),
        ("shr", "rax, 1", (1,)),
        ("sar", "rax, 1", (1,)),
    ):
        _step(
            _event(mnemonic, op_str, immediate=immediate),
            state,
            state.rip + 1,  # type: ignore[arg-type]
        )
    state.eflags |= 1 << 7
    _step(_event("jl", "0x1100", immediate=(0x1100,)), state, None)  # type: ignore[arg-type]
    _step(_event("jmp", "0x1200", immediate=(0x1200,)), state, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported mnemonic"):
        _step(_event("syscall"), state, 0x1300)  # type: ignore[arg-type]
    rows, unsupported = _reference_trace(bytes.fromhex("0f 0b"), 0)
    assert rows == []
    assert unsupported and "unsupported mnemonic" in unsupported


def test_replication_scoring_accepts_exact_label_separated_predictions(
    tmp_path: Path,
) -> None:
    write_replication_bundle(tmp_path)
    gold = json.loads(
        (tmp_path / "output/replication/private/gold_labels.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = [
        {"record_id": row["record_id"], "probability": float(row["label"])}
        for row in gold
    ]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps({"predictions": predictions}), encoding="utf-8"
    )
    scored = score_replication_predictions(prediction_path, tmp_path)
    assert scored["balanced_accuracy"] == 1.0
    with pytest.raises(ValueError, match="identifiers"):
        prediction_path.write_text(json.dumps(predictions[:-1]), encoding="utf-8")
        score_replication_predictions(prediction_path, tmp_path)


def test_replication_label_blind_model_runs_without_gold_labels(tmp_path: Path) -> None:
    public_path = tmp_path / "public.json"
    public_path.write_text(
        json.dumps(
            [
                {"record_id": "a", "sequence": ["x"], "features": [0.0, 1.0]},
                {"record_id": "b", "sequence": ["y", "z"], "features": [1.0, 2.0]},
            ]
        ),
        encoding="utf-8",
    )
    prediction_path = run_label_blind_model(
        public_path,
        tmp_path / "predictions.json",
        model="neural_sequence",
    )
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert payload["label_blind"] is True
    assert [row["record_id"] for row in payload["predictions"]] == ["a", "b"]


def test_open_roadmap_validators_fail_closed_on_receipt_drift() -> None:
    formal = formal_correspondence_payload()
    formal["all_steps_matched"] = False
    assert not validate_formal_correspondence_payload(formal)[0]

    sensor = sensor_cost_payload()
    sensor["sensors"][0]["wall_clock_nanoseconds_per_query"]["median"] = 0.0
    assert not validate_sensor_cost_payload(sensor)[0]

    theorem = theorem_or_countertheorem_payload()
    theorem["promotion_status"] = "promoted"
    assert not validate_theorem_or_countertheorem_payload(theorem)[0]


def test_provenance_helpers_hash_paths_and_record_unavailable_packages(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    inside = project_root / "input.bin"
    outside = tmp_path / "outside.bin"
    inside.write_bytes(b"active-assembly")
    outside.write_bytes(b"external-input")

    assert sha256_bytes(inside.read_bytes()) == sha256_path(inside)
    assert generator_revision(inside) == sha256_path(inside)
    hashes = input_hashes([inside, outside, project_root / "missing.bin"], project_root)
    assert hashes["input.bin"] == sha256_path(inside)
    assert hashes[str(outside)] == sha256_path(outside)
    versions = toolchain_versions(("numpy", "package-that-is-not-installed"))
    assert versions["numpy"]
    assert versions["package-that-is-not-installed"] == "unavailable"


def test_compiler_helpers_decode_both_supported_static_formats_and_fail_closed() -> (
    None
):
    x86_text = "  0: b8 01 00 00 00 \\t mov eax,1"
    arm_text = "  0: 1f2003d5 \\t nop"
    assert _extract_text_bytes(x86_text, "x86_64") == bytes.fromhex("b8 01 00 00 00")
    assert _extract_text_bytes(arm_text, "aarch64") == bytes.fromhex("d5 03 20 1f")
    assert _decode_bytes(bytes.fromhex("90"), "x86_64")[0]["mnemonic"] == "nop"
    assert _decode_bytes(bytes.fromhex("1f2003d5"), "aarch64")[0]["mnemonic"] == "nop"
    with pytest.raises(ValueError, match="no text bytes"):
        _decode_bytes(b"", "x86_64")
    with pytest.raises(ValueError, match="unsupported decoder"):
        _decode_bytes(b"\x90", "riscv64")
    with pytest.raises(ValueError, match="decoded zero"):
        _decode_bytes(b"\x0f", "x86_64")
    unavailable = _compile_one(
        "invalid_target_fixture",
        "int f(void) { return 1; }\n",
        "invalid",
        "definitely-not-a-real-target",
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["execution_performed"] is False


def test_protocol_validator_rejects_each_core_contract_drift() -> None:
    payload = scientific_protocol_payload()
    assert not validate_scientific_protocol_payload(None)[0]

    missing = deepcopy(payload)
    missing.pop("protocol_id")
    assert not validate_scientific_protocol_payload(missing)[0]

    for key, value in (
        ("schema_version", "0.0.0"),
        ("protocol_sha256", "0" * 64),
    ):
        mutated = deepcopy(payload)
        mutated[key] = value
        assert not validate_scientific_protocol_payload(mutated)[0]

    for contract, key, value in (
        ("model_contract", "families", ["leaky_model"]),
        ("statistics_contract", "primary_metrics", ["accuracy"]),
        ("observational_contract", "label_blind", False),
        ("evidence_contract", "arbitrary_external_execution", True),
    ):
        mutated = deepcopy(payload)
        mutated[contract][key] = value
        mutated["protocol_sha256"] = payload["protocol_sha256"]
        assert not validate_scientific_protocol_payload(mutated)[0]


def test_formal_correspondence_validator_rejects_structural_drift() -> None:
    payload = formal_correspondence_payload()
    assert not validate_formal_correspondence_payload(None)[0]

    for key, value in (
        ("schema_version", "0.0.0"),
        ("protocol_sha256", "0" * 64),
        ("generator_revision", None),
        ("input_hashes", None),
        ("toolchain_versions", None),
        ("step_rows", []),
        ("all_steps_matched", False),
    ):
        mutated = deepcopy(payload)
        mutated[key] = value
        assert not validate_formal_correspondence_payload(mutated)[0]

    for row in (
        None,
        {},
        {"checks": None},
        {"checks": {"matched_at_every_step": False}},
    ):
        mutated = deepcopy(payload)
        mutated["step_rows"] = [row]
        assert not validate_formal_correspondence_payload(mutated)[0]
    unsupported = deepcopy(payload)
    unsupported["step_rows"][0]["checks"]["unsupported"] = "unsupported instruction"
    assert not validate_formal_correspondence_payload(unsupported)[0]


def test_sensor_cost_validator_rejects_vector_timing_and_certificate_drift() -> None:
    payload = sensor_cost_payload()
    assert not validate_sensor_cost_payload(None)[0]
    for key, value in (
        ("schema_version", "0.0.0"),
        ("protocol_sha256", "0" * 64),
        ("generator_revision", None),
        ("input_hashes", None),
        ("toolchain_versions", None),
        ("sensors", None),
        ("policy_certificates", None),
    ):
        mutated = deepcopy(payload)
        mutated[key] = value
        assert not validate_sensor_cost_payload(mutated)[0]

    mutated = deepcopy(payload)
    mutated["sensors"] = []
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["sensors"] = [None]
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["sensors"][0]["deterministic_cost_vector"] = {}
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    first_component = next(iter(mutated["sensors"][0]["deterministic_cost_vector"]))
    mutated["sensors"][0]["deterministic_cost_vector"][first_component] = -1
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["sensors"][0]["wall_clock_nanoseconds_per_query"]["median"] = 0.0
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["policy_certificates"] = []
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["policy_certificates"][0]["optimality_gap"] = 1.0
    assert not validate_sensor_cost_payload(mutated)[0]
    mutated = deepcopy(payload)
    mutated["policy_certificates"][0]["within_preregistered_tolerance"] = False
    assert not validate_sensor_cost_payload(mutated)[0]


def test_sensor_cost_writer_reuses_matching_calibration_and_can_recalibrate(
    tmp_path: Path,
) -> None:
    first_path = write_sensor_cost_payload(tmp_path)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second_path = write_sensor_cost_payload(tmp_path)
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first == second
    recalibrated_path = write_sensor_cost_payload(tmp_path, recalibrate=True)
    recalibrated = json.loads(recalibrated_path.read_text(encoding="utf-8"))
    assert validate_sensor_cost_payload(recalibrated) == (True, "ok")


def test_replication_validator_rejects_manifest_and_label_separation_drift(
    tmp_path: Path,
) -> None:
    write_replication_bundle(tmp_path)
    manifest_path = tmp_path / "output/data/replication_manifest.json"
    public_path = tmp_path / "output/replication/public/public_manifest.json"
    gold_path = tmp_path / "output/replication/private/gold_labels.json"
    originals = {
        path: path.read_text(encoding="utf-8")
        for path in (manifest_path, public_path, gold_path)
    }

    def check(
        mutator: Callable[
            [dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]], None
        ],
    ) -> None:
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public = json.loads(public_path.read_text(encoding="utf-8"))
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        mutator(manifest, public, gold)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        public_path.write_text(json.dumps(public), encoding="utf-8")
        gold_path.write_text(json.dumps(gold), encoding="utf-8")
        assert not validate_replication_bundle(tmp_path)[0]

    check(lambda manifest, public, gold: manifest.update(schema_version="0.0.0"))
    check(lambda manifest, public, gold: manifest.update(protocol_sha256="0" * 64))
    check(lambda manifest, public, gold: manifest.update(generator_revision=None))
    check(lambda manifest, public, gold: public.clear())
    check(lambda manifest, public, gold: public[0].update(label=1))
    check(lambda manifest, public, gold: manifest.update(public_row_count=0))
    check(
        lambda manifest, public, gold: public[1].update(
            record_id=public[0]["record_id"]
        )
    )
    check(
        lambda manifest, public, gold: manifest.update(public_manifest_sha256="0" * 64)
    )
    check(lambda manifest, public, gold: manifest.update(gold_labels_sha256="0" * 64))
    check(lambda manifest, public, gold: gold[0].update(record_id="different"))

    public_path.unlink()
    assert not validate_replication_bundle(tmp_path)[0]
    write_replication_bundle(tmp_path)
    public_path.write_text("{", encoding="utf-8")
    assert not validate_replication_bundle(tmp_path)[0]

    write_replication_bundle(tmp_path)
    public_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert not validate_replication_bundle(tmp_path)[0]


def test_theorem_validator_rejects_missing_or_nonseparating_counterexamples() -> None:
    payload = theorem_or_countertheorem_payload()
    assert not validate_theorem_or_countertheorem_payload(None)[0]
    for key, value in (
        ("schema_version", "0.0.0"),
        ("protocol_sha256", "0" * 64),
        ("status", "validated"),
        ("promotion_status", "promoted"),
        ("counterexample", None),
    ):
        mutated = deepcopy(payload)
        mutated[key] = value
        assert not validate_theorem_or_countertheorem_payload(mutated)[0]

    for candidates in (None, [], [{"status": "unresolved"}]):
        mutated = deepcopy(payload)
        mutated["candidate_theorems"] = candidates
        assert not validate_theorem_or_countertheorem_payload(mutated)[0]

    for interventions in (None, [], [{}]):
        mutated = deepcopy(payload)
        mutated["counterexample"]["interventions"] = interventions
        assert not validate_theorem_or_countertheorem_payload(mutated)[0]
    zero_not_equal = deepcopy(payload)
    zero_not_equal["counterexample"]["interventions"][0]["final_rax_equal"] = False
    assert not validate_theorem_or_countertheorem_payload(zero_not_equal)[0]
    all_equal = deepcopy(payload)
    for item in all_equal["counterexample"]["interventions"]:
        item["final_rax_equal"] = True
    assert not validate_theorem_or_countertheorem_payload(all_equal)[0]


def test_theorem_producer_fails_closed_when_counterexample_search_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(theorem_module, "_family_specs", lambda: ())
    with pytest.raises(RuntimeError, match="counterexample was not found"):
        theorem_module.theorem_or_countertheorem_payload()


def test_theorem_writer_rejects_an_invalid_producer_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        theorem_module,
        "theorem_or_countertheorem_payload",
        lambda project_root: {"schema_version": "invalid"},
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        theorem_module.write_theorem_or_countertheorem(tmp_path)


def test_roadmap_closes_delivered_work_and_scopes_future_tiers() -> None:
    roadmap = (Path(__file__).parents[1] / "docs" / "roadmap.md").read_text(
        encoding="utf-8"
    )
    assert "Completed historical work: no open TODO remains" in roadmap
    assert "## Open evidence gates" in roadmap
    assert "### Minor: maintenance, clarity, and regression hardening" in roadmap
    assert "### Medium: bounded scientific extensions" in roadmap
    assert "### Major: promotion-gated research programs" in roadmap
    for identifier in ("MIN-01", "MED-01", "MED-08", "MAJ-01", "MAJ-08"):
        assert f"| {identifier} |" in roadmap
    assert "The next Phase B tranche should" not in roadmap
    assert "## The active program: P3" not in roadmap
    assert "planned CFG and symbolic facts" not in roadmap
