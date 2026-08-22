"""Functional tests for the bounded transfer-function interpreter."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.corpus import build_corpus
from src.disassembly import disassemble_bytes
from src.emulation import emulate_x86_64
from src.symbolic_semantics import symbolic_execute, validate_symbolic_report


@pytest.mark.parametrize("fixture", build_corpus(), ids=lambda fixture: fixture.name)
def test_symbolic_execution_agrees_with_real_unicorn_for_each_fixture(fixture) -> None:
    """Every transparent fixture is decoded, emulated, and semantically replayed."""

    instructions = disassemble_bytes(fixture.code)
    trace = emulate_x86_64(fixture.code)
    report = symbolic_execute(instructions, trace)
    valid, detail = validate_symbolic_report(report)

    assert valid, detail
    assert report.final_rax == trace.final_rax
    assert report.final_rax_matches is True
    assert report.unsupported_instruction_count == 0
    assert report.stepwise_rax_matches is True
    assert report.stepwise_rax_agreement_count == report.stepwise_rax_comparable_count
    assert report.stepwise_rax_mismatch_count == 0
    assert report.modeled_flags_match is True
    assert report.modeled_flag_agreements == report.modeled_flag_comparisons
    assert all(
        event.rax_pre_matches is True and event.rax_post_matches is True
        for event in report.events
        if event.executed
    )
    assert len(report.events) == len(instructions)
    assert len(report.executed_addresses) == trace.instruction_count
    assert json.loads(json.dumps(report.to_dict()))["final_rax_matches"] is True


def test_symbolic_execution_preserves_unexecuted_static_branch_targets() -> None:
    """The report distinguishes decoded-but-not-dynamically-executed code."""

    # mov eax,1; jmp +3; add eax,100; add eax,2
    code = bytes.fromhex("b801000000 eb03 83c064 83c002")
    instructions = disassemble_bytes(code)
    trace = emulate_x86_64(code)
    report = symbolic_execute(instructions, trace)

    assert any(not event.executed for event in report.events)
    assert all(event.supported for event in report.events if not event.executed)
    assert report.final_rax == 3
    assert report.final_rax_matches is True


def test_symbolic_execution_rejects_empty_instruction_stream() -> None:
    """A semantic report without a decoded stream is not meaningful."""

    fixture = build_corpus()[0]
    trace = emulate_x86_64(fixture.code)
    with pytest.raises(ValueError, match="instructions"):
        symbolic_execute([], trace)


def test_symbolic_execution_is_width_correct_and_masks_x86_shift_counts() -> None:
    """RAX arithmetic and the effective shift count agree with Unicorn."""

    # movabs rax,-1; add rax,1; cmp rax,0; test rax,rax; shl rax,0x40.
    # In 64-bit mode 0x40 masks to an effective shift count of zero, so the
    # final shift preserves both the value and the preceding flag state.
    code = bytes.fromhex(
        "48 b8 ff ff ff ff ff ff ff ff 48 83 c0 01 48 83 f8 00 48 85 c0 48 c1 e0 40"
    )
    instructions = tuple(disassemble_bytes(code))
    trace = emulate_x86_64(code)
    report = symbolic_execute(instructions, trace)
    valid, detail = validate_symbolic_report(report)

    assert valid, detail
    assert report.final_rax == 0
    add_event = next(event for event in report.events if event.mnemonic == "add")
    assert add_event.flags["zero"] is True
    assert add_event.flags["carry"] is True
    cmp_event = next(event for event in report.events if event.mnemonic == "cmp")
    assert cmp_event.flags["zero"] is True
    shift_event = next(event for event in report.events if event.mnemonic == "shl")
    assert shift_event.flags["zero"] is True


def test_symbolic_execution_uses_64_bit_signed_arithmetic_for_sar() -> None:
    """Arithmetic right shift must use RAX width rather than truncating to EAX."""

    # movabs rax,0x80000000; sar rax,1.  Treating RAX as a signed 32-bit value
    # would incorrectly sign-extend the result; the architectural 64-bit result
    # is the positive value 0x40000000.
    code = bytes.fromhex("48 b8 00 00 00 80 00 00 00 00 48 c1 f8 01")
    instructions = tuple(disassemble_bytes(code))
    trace = emulate_x86_64(code)
    report = symbolic_execute(instructions, trace)
    valid, detail = validate_symbolic_report(report)

    assert valid, detail
    assert report.final_rax == 0x40000000
    assert report.final_rax_matches is True


def test_symbolic_validation_rejects_intermediate_trace_drift_even_if_terminal_value_matches() -> (
    None
):
    """Terminal equality cannot hide a tampered stepwise architectural row."""

    fixture = build_corpus()[0]
    instructions = tuple(disassemble_bytes(fixture.code))
    trace = emulate_x86_64(fixture.code)
    first = trace.instruction_states[0]
    tampered_first = replace(first, post_rax=first.post_rax + 1)
    tampered_trace = replace(
        trace,
        instruction_states=(tampered_first, *trace.instruction_states[1:]),
    )

    report = symbolic_execute(instructions, tampered_trace)
    valid, detail = validate_symbolic_report(report)

    assert report.final_rax_matches is True
    assert report.stepwise_rax_matches is False
    assert report.stepwise_rax_mismatch_count == 1
    assert valid is False
    assert "stepwise RAX mismatch" in detail
