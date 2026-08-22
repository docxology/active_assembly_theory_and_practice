"""Tests for the Phase A typed observation and factorized model surface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.corpus import build_corpus
from src.disassembly import disassemble_bytes, disassemble_sample, sample_bytes
from src.emulation import emulate_sample, emulate_x86_64
from src.factorized_model import (
    FACTOR_MODEL_MODES,
    FACTOR_MODEL_SCHEMA_VERSION,
    FactorizedObservationModel,
    build_typed_observation_corpus,
    decode_typed_observations,
    fit_factorized_observation_model,
    leave_one_fixture_out_factorized_evaluation,
)
from src.typed_observations import (
    MISSING_VALUE,
    STATIC_TYPED_OBSERVATION_FIELDS,
    TYPED_OBSERVATION_FIELDS,
    TypedObservation,
    encode_typed_observations,
    mask_typed_field,
    typed_field_value,
    validate_typed_observations,
)


def test_typed_encoder_uses_real_capstone_and_unicorn_fields() -> None:
    instructions = disassemble_sample()
    trace = emulate_sample()
    observations = encode_typed_observations(instructions, trace)

    assert [row.instruction_family for row in observations] == [
        "data_transfer",
        "arithmetic",
        "bitwise",
        "arithmetic",
    ]
    assert observations[0].operand_class == "mixed"
    assert observations[1].register_read_set == ("eax",)
    assert observations[1].flag_effect == "write"
    assert all(row.dynamic_status == "executed" for row in observations)
    valid, detail = validate_typed_observations(observations)
    assert valid, detail
    assert "phase_labels" not in observations[0].to_dict()


def test_typed_encoder_distinguishes_memory_and_external_branch() -> None:
    corpus = build_corpus()
    guard = next(fixture for fixture in corpus if fixture.name == "x86_guard")
    memory_rows = encode_typed_observations(
        disassemble_sample("x86_memory"), emulate_sample("x86_memory")
    )
    guard_rows = encode_typed_observations(
        disassemble_bytes(guard.code), emulate_x86_64(guard.code)
    )

    assert memory_rows[0].memory_addressing_mode == "stack_relative"
    assert memory_rows[1].memory_addressing_mode == "stack_relative"
    assert guard_rows[2].branch_target_class == "external"
    assert guard_rows[2].flag_effect == "read"


def test_missing_dynamic_trace_is_explicit_and_not_guessed() -> None:
    observations = encode_typed_observations(disassemble_sample(), trace=None)
    assert all(row.dynamic_status is None for row in observations)
    assert validate_typed_observations(observations) == (True, "ok")


def test_typed_validation_rejects_malformed_sequence() -> None:
    observations = encode_typed_observations(disassemble_sample(), emulate_sample())
    with pytest.raises(ValueError, match="unknown typed observation field"):
        mask_typed_field(observations, "phase_label")
    invalid = list(observations)
    invalid[1] = invalid[1].__class__(
        **{
            **invalid[1].to_dict(),
            "index": 4,
            "register_read_set": ["eax"],
            "register_write_set": ["eax"],
        }
    )
    assert validate_typed_observations(invalid)[0] is False


def test_factorized_model_supports_unknown_and_missing_values() -> None:
    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    training = tuple(fixture for fixture in corpus if fixture.name != "x86_guard")
    model = fit_factorized_observation_model(training, typed)

    assert model.schema_version == FACTOR_MODEL_SCHEMA_VERSION
    assert set(model.fields) == set(TYPED_OBSERVATION_FIELDS)
    assert set(model.field_support["dynamic_status"]) >= {"executed", MISSING_VALUE}
    masked = mask_typed_field(typed["x86_guard"], "dynamic_status")
    emissions = model.emission_matrix(masked, mode="factorized")
    assert emissions.shape == (4, 5)
    assert np.allclose(emissions.sum(axis=0), 1.0)
    assert np.all(np.isfinite(emissions))


def test_factorized_and_hierarchical_decoders_are_functional() -> None:
    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    training = tuple(corpus[:-1])
    model = fit_factorized_observation_model(training, typed)
    held_out = corpus[-1]

    factorized = decode_typed_observations(
        typed[held_out.name], model, mode="factorized"
    )
    hierarchical = decode_typed_observations(
        typed[held_out.name], model, mode="hierarchical"
    )
    assert len(factorized[0].most_likely_states) == len(held_out.phase_labels)
    assert len(hierarchical[1]) == len(held_out.phase_labels)
    assert np.isfinite(factorized[2])
    assert np.isfinite(hierarchical[2])


def test_factorized_model_rejects_empty_field_subset() -> None:
    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    with pytest.raises(ValueError, match="at least one"):
        fit_factorized_observation_model(corpus[:2], typed, fields=())


def test_leave_one_out_factorized_payload_records_boundaries_and_ablations() -> None:
    payload = leave_one_fixture_out_factorized_evaluation(build_corpus())
    assert payload["schema_version"] == FACTOR_MODEL_SCHEMA_VERSION
    assert payload["leakage_checks"]["held_out_excluded_from_training"] is True
    assert (
        payload["leakage_checks"]["phase_label_keys_absent_from_typed_records"] is True
    )
    assert set(payload["field_group_ablation"]) == {
        "instruction_semantics",
        "register_context",
        "control_flow",
        "dynamic_context",
    }
    assert len(payload["per_fixture"]) == len(build_corpus())
    for row in payload["per_fixture"]:
        assert row["held_out_fixture"] not in row["training_fixtures"]
        for mode in ("collapsed", "static_factorized", "factorized", "hierarchical"):
            assert 0.0 <= row[mode]["accuracy"] <= 1.0


def test_static_field_contract_excludes_only_dynamic_status() -> None:
    assert "dynamic_status" not in STATIC_TYPED_OBSERVATION_FIELDS
    assert set(STATIC_TYPED_OBSERVATION_FIELDS) | {"dynamic_status"} == set(
        TYPED_OBSERVATION_FIELDS
    )


def test_typed_encoder_covers_assembly_taxonomy_and_missing_dynamic_rows() -> None:
    """Every public taxonomy branch is driven by a real Capstone decode."""

    cases = {
        "comparison": (bytes.fromhex("83 f8 00"), "comparison", "flag_effect", "write"),
        "flags_read_write": (
            bytes.fromhex("83 d0 01"),
            "arithmetic",
            "flag_effect",
            "read_write",
        ),
        "internal_branch": (
            bytes.fromhex("75 00 90"),
            "control_transfer",
            "branch_target_class",
            "internal",
        ),
        "external_branch": (
            bytes.fromhex("75 7f"),
            "control_transfer",
            "branch_target_class",
            "external",
        ),
        "indirect_branch": (
            bytes.fromhex("ff e0"),
            "control_transfer",
            "branch_target_class",
            "indirect",
        ),
        "return": (
            bytes.fromhex("c3"),
            "control_transfer",
            "branch_target_class",
            "return",
        ),
        "other": (bytes.fromhex("90"), "other", "branch_target_class", "none"),
        "indexed_memory": (
            bytes.fromhex("8b 44 48 04"),
            "data_transfer",
            "memory_addressing_mode",
            "indexed",
        ),
        "base_displacement": (
            bytes.fromhex("8b 40 04"),
            "data_transfer",
            "memory_addressing_mode",
            "base_plus_displacement",
        ),
        "register_indirect": (
            bytes.fromhex("8b 00"),
            "data_transfer",
            "memory_addressing_mode",
            "register_indirect",
        ),
        "absolute_memory": (
            bytes.fromhex("8b 04 25 00 20 00 00"),
            "data_transfer",
            "memory_addressing_mode",
            "absolute",
        ),
    }

    for code, expected_family, detail_field, expected_detail in cases.values():
        row = encode_typed_observations(disassemble_bytes(code))[0]
        assert row.instruction_family == expected_family
        assert getattr(row, detail_field) == expected_detail

    # A bounded Unicorn run can legitimately leave statically decoded rows
    # unexecuted; the encoder must preserve that fact rather than imputing it.
    partial_trace = emulate_x86_64(sample_bytes(), max_instructions=1)
    rows = encode_typed_observations(disassemble_sample(), partial_trace)
    assert rows[0].dynamic_status == "executed"
    assert all(row.dynamic_status == "not_executed" for row in rows[1:])


def test_typed_encoder_rejects_empty_and_duplicate_instruction_addresses() -> None:
    instructions = disassemble_sample()
    with pytest.raises(ValueError, match="must not be empty"):
        encode_typed_observations([])
    duplicate = [
        instructions[0],
        replace(instructions[1], address=instructions[0].address),
    ]
    with pytest.raises(ValueError, match="addresses must be unique"):
        encode_typed_observations(duplicate)


def test_typed_value_and_validation_policies_are_explicit() -> None:
    row = encode_typed_observations(disassemble_sample())[0]
    empty_sets = replace(
        row, register_read_set=(), register_write_set=(), dynamic_status=None
    )
    assert typed_field_value(empty_sets, "register_read_set") == "<none>"
    assert typed_field_value(empty_sets, "dynamic_status") == MISSING_VALUE
    with pytest.raises(ValueError, match="unknown typed observation field"):
        typed_field_value(row, "truth")
    assert validate_typed_observations([]) == (
        False,
        "typed observation sequence is empty",
    )
    assert (
        validate_typed_observations([replace(row, index=1)])[1]
        == "typed observation indexes are not contiguous"
    )
    assert (
        validate_typed_observations([replace(row, mnemonic="")])[1]
        == "typed observation mnemonic is empty"
    )
    assert (
        validate_typed_observations([replace(row, dynamic_status="unknown")])[1]
        == "invalid dynamic status"
    )

    class LeakyObservation(TypedObservation):
        def to_dict(self) -> dict[str, object]:
            return {**super().to_dict(), "phase_label": "guard_check"}

    leaky = LeakyObservation(**row.to_dict())
    assert (
        validate_typed_observations([leaky])[1]
        == "phase-label field leaked into typed observation"
    )


def test_factorized_model_validation_rejects_corrupted_contracts() -> None:
    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    model = fit_factorized_observation_model(corpus[:-1], typed)

    def expect_invalid(candidate: FactorizedObservationModel, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            candidate.validate()

    expect_invalid(replace(model, schema_version="0.0.0"), "schema version")
    expect_invalid(replace(model, B=model.B[:, :-1]), "B shape")
    expect_invalid(replace(model, D=model.D[:-1]), "D shape")
    bad_positive = model.B.copy()
    bad_positive[0, 0] = 0.0
    expect_invalid(replace(model, B=bad_positive), "strictly positive")
    bad_columns = model.B.copy()
    bad_columns[0, 0] *= 2.0
    expect_invalid(replace(model, B=bad_columns), "columns must normalize")
    bad_prior = model.D.copy()
    bad_prior[0] *= 2.0
    expect_invalid(replace(model, D=bad_prior), "D must normalize")

    missing_field = deepcopy(model.field_support)
    missing_field.pop("instruction_family")
    expect_invalid(
        replace(model, field_support=missing_field), "missing factorized field"
    )

    no_sentinels = deepcopy(model.field_support)
    no_sentinels["instruction_family"] = tuple(
        value for value in no_sentinels["instruction_family"] if value != MISSING_VALUE
    )
    expect_invalid(replace(model, field_support=no_sentinels), "lacks missing/unknown")

    wrong_states = deepcopy(model.probabilities)
    wrong_states["instruction_family"].pop(model.hidden_state_labels[0])
    expect_invalid(replace(model, probabilities=wrong_states), "wrong state support")

    wrong_values = deepcopy(model.probabilities)
    wrong_values["instruction_family"][model.hidden_state_labels[0]].pop(
        next(iter(wrong_values["instruction_family"][model.hidden_state_labels[0]]))
    )
    expect_invalid(replace(model, probabilities=wrong_values), "wrong value support")

    invalid_probabilities = deepcopy(model.probabilities)
    first_state = model.hidden_state_labels[0]
    first_value = next(iter(invalid_probabilities["instruction_family"][first_state]))
    invalid_probabilities["instruction_family"][first_state][first_value] = np.nan
    expect_invalid(
        replace(model, probabilities=invalid_probabilities), "invalid probabilities"
    )

    unnormalized = deepcopy(model.probabilities)
    unnormalized["instruction_family"][first_state][first_value] *= 2.0
    expect_invalid(replace(model, probabilities=unnormalized), "do not normalize")


def test_factorized_model_rejects_bad_inputs_and_preserves_json_contract() -> None:
    corpus = build_corpus()
    typed = build_typed_observation_corpus(corpus)
    model = fit_factorized_observation_model(
        corpus[:-1], typed, fields=("instruction_family",)
    )

    assert set(FACTOR_MODEL_MODES) == {"factorized", "hierarchical"}
    assert model.to_dict()["missing_field_policy"]["missing"] == MISSING_VALUE
    with pytest.raises(ValueError, match="fixtures must not be empty"):
        fit_factorized_observation_model([], typed)
    with pytest.raises(ValueError, match="finite and positive"):
        fit_factorized_observation_model(corpus[:1], typed, alpha=0.0)
    with pytest.raises(ValueError, match="unique"):
        fit_factorized_observation_model(
            corpus[:1], typed, fields=("instruction_family", "instruction_family")
        )
    with pytest.raises(ValueError, match="unknown typed observation fields"):
        fit_factorized_observation_model(corpus[:1], typed, fields=("not_a_field",))
    with pytest.raises(ValueError, match="missing typed observations"):
        fit_factorized_observation_model(corpus[:1], {})
    with pytest.raises(ValueError, match="differ"):
        fit_factorized_observation_model(
            corpus[:1], {corpus[0].name: typed[corpus[0].name][:-1]}
        )
    with pytest.raises(ValueError, match="invalid typed observations"):
        fit_factorized_observation_model(
            corpus[:1],
            {corpus[0].name: [replace(typed[corpus[0].name][0], dynamic_status="bad")]},
        )

    with pytest.raises(ValueError, match="unknown factorized model mode"):
        model.log_scores(typed[corpus[-1].name], mode="invalid")
    with pytest.raises(ValueError, match="must not be empty"):
        model.log_scores([], mode="factorized")
    with pytest.raises(ValueError, match="unknown factorized model mode"):
        decode_typed_observations(typed[corpus[-1].name], model, mode="invalid")
    with pytest.raises(ValueError, match="must not be empty"):
        decode_typed_observations([], model)
    with pytest.raises(ValueError, match="corpus must not be empty"):
        build_typed_observation_corpus([])
    with pytest.raises(ValueError, match="at least two"):
        leave_one_fixture_out_factorized_evaluation(corpus[:1])

    no_groups = replace(model, fields=())
    with pytest.raises(ValueError, match="at least one"):
        no_groups.validate()
    with pytest.raises(ValueError, match="no active field groups"):
        no_groups.log_scores(typed[corpus[-1].name], mode="hierarchical")
