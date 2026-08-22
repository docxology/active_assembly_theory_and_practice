"""Tests for the labelled multi-fixture corpus (no mocks; real decode/emulate)."""

from __future__ import annotations

import json

from src.corpus import CorpusFixture, build_corpus, fixture_observation_ids
from src.disassembly import disassemble_bytes
from src.features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from src.corpus_coverage import corpus_coverage_payload


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
    assert payload["summary"]["total_static_instruction_count"] == 26
    assert payload["summary"]["total_executed_instruction_count"] == 25
    assert payload["summary"]["memory_fixture_count"] == 1
    assert payload["summary"]["alternate_path_fixture_count"] == 1


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
