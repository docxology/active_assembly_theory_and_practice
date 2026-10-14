"""Tests for the labelled multi-fixture corpus (no mocks; real decode/emulate)."""

from __future__ import annotations

import json
from copy import deepcopy

from src.corpus import (
    CORPUS_FAMILY_SPLIT,
    SPLITS,
    CorpusFixture,
    build_corpus,
    corpus_family_split,
    fixture_family,
    fixture_observation_ids,
    fixture_split,
)
from src.corpus_coverage import (
    CORPUS_COVERAGE_SCHEMA_VERSION,
    corpus_coverage_payload,
    validate_corpus_coverage_payload,
)
from src.disassembly import disassemble_bytes
from src.features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from src.scientific_protocol import (
    PROTOCOL_ID,
    PROTOCOL_SCHEMA_VERSION,
    build_scientific_protocol,
)


def test_build_corpus_has_expanded_fixture_surface() -> None:
    corpus = build_corpus()
    assert len(corpus) >= 4
    assert all(isinstance(fixture, CorpusFixture) for fixture in corpus)


def test_fixture_names_are_unique() -> None:
    names = [fixture.name for fixture in build_corpus()]
    assert len(names) == len(set(names))


def test_accumulator_fixture_present_with_exact_bytes() -> None:
    """The canonical accumulator program must be in the corpus verbatim."""

    expected = bytes.fromhex("b801000000 83c002 d1e0 83e801".replace(" ", ""))
    matches = [f for f in build_corpus() if f.code == expected]
    assert len(matches) == 1
    fixture = matches[0]
    instructions = disassemble_bytes(fixture.code)
    mnemonics = [event.mnemonic for event in instructions]
    assert mnemonics == ["mov", "add", "shl", "sub"]


def test_isc25_label_length_equals_instruction_count() -> None:
    """ISC-25: every fixture has exactly one phase label per decoded instruction."""

    for fixture in build_corpus():
        instructions = disassemble_bytes(fixture.code)
        assert len(fixture.phase_labels) == len(instructions), fixture.name


def test_all_phase_labels_are_valid_hidden_states() -> None:
    valid = set(HIDDEN_STATE_LABELS)
    for fixture in build_corpus():
        assert set(fixture.phase_labels) <= valid, fixture.name


def test_each_fixture_uses_setup_first_and_exit_last() -> None:
    for fixture in build_corpus():
        assert fixture.phase_labels[0] == "setup", fixture.name
        assert fixture.phase_labels[-1] == "exit", fixture.name


def test_corpus_exercises_multiple_phases() -> None:
    """The corpus as a whole must touch accumulator, transform, and guard."""

    seen: set[str] = set()
    for fixture in build_corpus():
        seen.update(fixture.phase_labels)
    assert {"accumulator_update", "representation_transform", "guard_check"} <= seen


def test_isc24_fixture_observation_ids_real_decode_and_emulate() -> None:
    """ISC-24: observation ids come from real decode + emulate + encode."""

    for fixture in build_corpus():
        instructions = disassemble_bytes(fixture.code)
        ids = fixture_observation_ids(fixture)
        assert len(ids) == len(instructions), fixture.name
        assert all(0 <= obs_id < len(OBSERVATION_LABELS) for obs_id in ids)


def test_accumulator_observation_ids_hand_checked() -> None:
    """mov,add,shl,sub -> load_immediate, arithmetic, bitwise, arithmetic."""

    fixture = next(f for f in build_corpus() if f.name == "x86_accumulator")
    ids = fixture_observation_ids(fixture)
    # load_immediate=0, arithmetic_update=1, bitwise_transform=2
    assert ids == (0, 1, 2, 1)


def test_guard_fixture_observation_ids_hand_checked() -> None:
    """mov,cmp,jl,add -> load_immediate, comparison, control_transfer, arithmetic."""

    fixture = next(f for f in build_corpus() if f.name == "x86_guard")
    ids = fixture_observation_ids(fixture)
    # comparison=3, control_transfer=4
    assert ids == (0, 3, 4, 1)


def test_expanded_corpus_contains_memory_and_taken_branch_variants() -> None:
    corpus = build_corpus()
    memory = next(fixture for fixture in corpus if fixture.name == "x86_memory")
    taken = next(fixture for fixture in corpus if fixture.name == "x86_guard_taken")
    assert len(disassemble_bytes(memory.code)) == 3
    assert len(disassemble_bytes(taken.code)) == 5
    assert fixture_observation_ids(memory) == (0, 5, 1)
    assert fixture_observation_ids(taken) == (0, 3, 4, 1, 1)


def test_corpus_coverage_is_source_bound_and_contractual() -> None:
    payload = corpus_coverage_payload()
    assert payload["summary"]["fixture_count"] == len(build_corpus())
    assert payload["summary"]["total_static_instruction_count"] == 56
    assert payload["summary"]["total_executed_instruction_count"] == 55
    assert payload["summary"]["memory_fixture_count"] == 2
    assert payload["summary"]["alternate_path_fixture_count"] == 1
    assert validate_corpus_coverage_payload(payload)[0]


def test_call_return_fixture_present_with_exact_bytes() -> None:
    """MED-01: the call/ret fixture bytes must be in the corpus verbatim."""

    expected = bytes.fromhex("b801000000 50 e803000000 58 eb01 c3".replace(" ", ""))
    matches = [f for f in build_corpus() if f.name == "x86_call_return"]
    assert len(matches) == 1
    fixture = matches[0]
    assert fixture.code == expected
    mnemonics = [event.mnemonic for event in disassemble_bytes(fixture.code)]
    assert mnemonics == ["mov", "push", "call", "pop", "jmp", "ret"]


def test_branch_not_taken_fixture_present_with_exact_bytes() -> None:
    """MED-01: the double not-taken guard fixture bytes must be verbatim."""

    expected = bytes.fromhex(
        "b803000000 83f805 7f03 83c001 83f802 7c03 83e801".replace(" ", "")
    )
    matches = [f for f in build_corpus() if f.name == "x86_branch_not_taken"]
    assert len(matches) == 1
    fixture = matches[0]
    assert fixture.code == expected
    mnemonics = [event.mnemonic for event in disassemble_bytes(fixture.code)]
    assert mnemonics == ["mov", "cmp", "jg", "add", "cmp", "jl", "sub"]


def test_stack_local_fixture_present_with_exact_bytes() -> None:
    """MED-01: the stack-local fixture bytes must be in the corpus verbatim."""

    expected = bytes.fromhex(
        "b807000000 50 b903000000 83c003 83e802 59 d1e0 83c002 83e808".replace(" ", "")
    )
    matches = [f for f in build_corpus() if f.name == "x86_stack_local"]
    assert len(matches) == 1
    fixture = matches[0]
    assert fixture.code == expected
    mnemonics = [event.mnemonic for event in disassemble_bytes(fixture.code)]
    assert mnemonics == [
        "mov",
        "push",
        "mov",
        "add",
        "sub",
        "pop",
        "shl",
        "add",
        "sub",
    ]


def test_optimization_rewrite_fixture_present_with_exact_bytes() -> None:
    """MED-01: the optimization-rewrite motif pair bytes must be verbatim."""

    expected = bytes.fromhex(
        "b801000000 83c001 ffc0 8d4001 d1e0 83c001 ffc0 83e805".replace(" ", "")
    )
    matches = [f for f in build_corpus() if f.name == "x86_optimization_rewrite"]
    assert len(matches) == 1
    fixture = matches[0]
    assert fixture.code == expected
    mnemonics = [event.mnemonic for event in disassemble_bytes(fixture.code)]
    assert mnemonics == [
        "mov",
        "add",
        "inc",
        "lea",
        "shl",
        "add",
        "inc",
        "sub",
    ]


def test_call_return_observation_ids_hand_checked() -> None:
    """mov,push,call,pop,jmp,ret -> load_immediate,other,control x3,other.

    Hand-verified with real Capstone decode and Unicorn execution: the ret
    lands on the in-span return address and every instruction executes, so
    all six observations are recorded even though the final ret is the exit.
    """

    fixture = next(f for f in build_corpus() if f.name == "x86_call_return")
    assert fixture_observation_ids(fixture) == (0, 5, 4, 5, 4, 4)


def test_branch_not_taken_observation_ids_hand_checked() -> None:
    """mov,cmp,jg,add,cmp,jl,sub -> imm,comparison,control,arith,comparison,control,arith."""

    fixture = next(f for f in build_corpus() if f.name == "x86_branch_not_taken")
    assert fixture_observation_ids(fixture) == (0, 3, 4, 1, 3, 4, 1)


def test_stack_local_observation_ids_hand_checked() -> None:
    """mov,push,mov,add,sub,pop,shl,add,sub -> imm,other,imm,arith,arith,other,bitwise,arith,arith."""

    fixture = next(f for f in build_corpus() if f.name == "x86_stack_local")
    assert fixture_observation_ids(fixture) == (0, 5, 0, 1, 1, 5, 2, 1, 1)


def test_optimization_rewrite_observation_ids_hand_checked() -> None:
    """mov,add,inc,lea,shl,add,inc,sub -> imm,arith,arith,other,bitwise,arith,arith,arith.

    The lea rewrite lands in the residual ``other`` category although its
    authored latent phase is accumulator_update: ambiguity is data (MED-01).
    """

    fixture = next(f for f in build_corpus() if f.name == "x86_optimization_rewrite")
    assert fixture_observation_ids(fixture) == (0, 1, 1, 5, 2, 1, 1, 1)


def test_every_fixture_has_a_frozen_family_and_split() -> None:
    for fixture in build_corpus():
        assert fixture_family(fixture.name) == fixture.name
        assert fixture_split(fixture.name) in SPLITS
        assert CORPUS_FAMILY_SPLIT[fixture.name] == fixture_split(fixture.name)


def test_frozen_family_split_is_disjoint_and_holdout_critical() -> None:
    frozen = corpus_family_split()
    assert frozen == dict(CORPUS_FAMILY_SPLIT)
    assert set(frozen) == {fixture.name for fixture in build_corpus()}
    by_split = {
        split: {name for name, value in frozen.items() if value == split}
        for split in SPLITS
    }
    # The three splits are pairwise disjoint and cover the corpus exactly.
    union: set[str] = set()
    for names in by_split.values():
        assert not union & names
        union |= names
    assert union == set(frozen)
    assert by_split["held_out"]  # held_out must be non-empty
    # The control-flow-sensitive guard fixtures must never be fitted on.
    assert {"x86_guard", "x86_guard_taken"} <= by_split["held_out"]


def test_frozen_family_split_is_reimport_stable() -> None:
    first = corpus_family_split()
    second = dict(CORPUS_FAMILY_SPLIT)
    assert first == second
    assert PROTOCOL_SCHEMA_VERSION == "2.2.0"
    protocol = build_scientific_protocol()
    block = protocol.split_contract["temporal_corpus_split"]
    assert block["fixture_split"] == second
    assert block["frozen_before_scoring"] is True
    assert block["holdout_scoring"] == ("leave_one_fixture_out_over_complete_fixtures")
    assert set(block["fixture_bytes_sha256"]) == set(second)


def test_corpus_coverage_payload_validates_with_frozen_split() -> None:
    payload = corpus_coverage_payload()
    assert payload["schema_version"] == CORPUS_COVERAGE_SCHEMA_VERSION
    assert CORPUS_COVERAGE_SCHEMA_VERSION == "1.1.0"
    valid, detail = validate_corpus_coverage_payload(payload)
    assert valid, detail
    frozen = payload["frozen_split"]
    assert frozen["unit"] == "source_program_family"
    assert frozen["frozen_before_scoring"] is True
    assert frozen["protocol_ref"] == PROTOCOL_ID
    assert frozen["families_per_split"] == {
        "training": 4,
        "development": 2,
        "held_out": 4,
    }
    for row in payload["per_fixture"]:
        assert row["family"] == row["fixture"]
        assert row["split"] == CORPUS_FAMILY_SPLIT[row["fixture"]]


def test_corpus_coverage_validator_fails_closed_on_split_drift() -> None:
    payload = corpus_coverage_payload()
    missing_family = deepcopy(payload)
    del missing_family["per_fixture"][0]["family"]
    assert not validate_corpus_coverage_payload(missing_family)[0]
    row_drift = deepcopy(payload)
    row_drift["per_fixture"][0]["split"] = "held_out"
    assert not validate_corpus_coverage_payload(row_drift)[0]
    assignment_drift = deepcopy(payload)
    assignment_drift["frozen_split"]["assignment"]["x86_accumulator"] = "held_out"
    assert not validate_corpus_coverage_payload(assignment_drift)[0]
    missing_frozen = deepcopy(payload)
    del missing_frozen["frozen_split"]
    assert not validate_corpus_coverage_payload(missing_frozen)[0]
    stale_schema = deepcopy(payload)
    stale_schema["schema_version"] = "1.0.0"
    assert not validate_corpus_coverage_payload(stale_schema)[0]


def test_to_dict_is_json_safe_with_hex_code() -> None:
    for fixture in build_corpus():
        payload = fixture.to_dict()
        assert payload["code"] == fixture.code.hex()
        assert payload["phase_labels"] == list(fixture.phase_labels)
        # Round-trips through JSON without error.
        json.dumps(payload, sort_keys=True)


def test_to_dict_hex_round_trips_back_to_code() -> None:
    for fixture in build_corpus():
        payload = fixture.to_dict()
        assert bytes.fromhex(str(payload["code"])) == fixture.code
