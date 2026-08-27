"""Boundary and negative-control tests for the normalized machine state."""

from __future__ import annotations

from src.disassembly import disassemble_bytes
from src.emulation import GENERAL_REGISTER_NAMES, emulate_x86_64
from src.formal_correspondence import (
    U64_MASK,
    FormalState,
    StepResult,
    _condition_holds,
    _flag_reads_and_writes,
    _memory_address,
    _memory_width,
    _next_rip,
    _parity_even,
    _parse_immediate,
    _read_memory,
    _read_operand,
    _register_ref,
    _set_arithmetic_flags,
    _set_flag,
    _set_logic_flags,
    _sign_extend,
    _step,
    _split_operands,
    _token_width,
    _write_memory,
    _write_register_value,
    _write_operand,
    _reference_trace,
)
import pytest


def _state(
    *,
    rax: int = 0,
    eflags: int = 2,
    registers: dict[str, int] | None = None,
    flags_defined: dict[str, bool] | None = None,
) -> FormalState:
    return FormalState(
        rax=rax,
        rsp=0x101000,
        rip=0x1000,
        eflags=eflags,
        memory={},
        registers=registers or {},
        flags_defined=flags_defined or {},
    )


def _run_one(code: bytes, state: FormalState) -> FormalState:
    event = disassemble_bytes(code)[0]
    _step(event, state, None)
    return state


def test_machine_state_has_all_gprs_and_round_trips_without_synthetic_memory() -> None:
    state = _state(rax=0x1234, registers={"rax": 0x1234, "rcx": 7})
    event = disassemble_bytes(bytes.fromhex("48 89 c1"))[0]  # mov rcx, rax
    _step(event, state, None)

    assert tuple(state.registers) == GENERAL_REGISTER_NAMES
    assert state.registers["rcx"] == 0x1234
    assert all(address >= 0 for address in state.memory)
    restored = FormalState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert state.rsp == 0x101000


def test_high_byte_alias_and_carry_overflow_use_architectural_width() -> None:
    state = _state(rax=0x1234, registers={"rcx": 0x1234})
    state.set_register("ch", 0xAB)
    assert state.registers["rcx"] == 0xAB34
    assert state.get_register("ch") == 0xAB

    # 0x7f + 0x7f + CF is 0xff: signed 127 + 127 + 1 overflows an 8-bit
    # signed result even though the ordinary ADD formula would be reused.
    state = _state(eflags=2 | 1)
    _set_arithmetic_flags(state, 0x7F, 0x7F, 0xFF, width=8, subtract=False, carry_in=1)
    assert state.flags_defined["OF"] is True
    assert state.eflags & (1 << 11)

    # 0x80 - 0x00 - CF is 0x7f: -128 - 0 - 1 overflows an 8-bit result.
    _set_arithmetic_flags(state, 0x80, 0x00, 0x7F, width=8, subtract=True, carry_in=1)
    assert state.flags_defined["OF"] is True
    assert state.eflags & (1 << 11)


def test_64_bit_add_wraps_and_32_bit_write_zero_extends() -> None:
    state = _run_one(bytes.fromhex("48 83 c0 01"), _state(rax=U64_MASK))
    assert state.rax == 0
    assert state.eflags & 1

    state = _run_one(bytes.fromhex("b8 01 00 00 00"), _state(rax=U64_MASK))
    assert state.rax == 1


def test_inc_and_dec_preserve_carry_flag() -> None:
    state = _run_one(bytes.fromhex("48 ff c0"), _state(rax=U64_MASK, eflags=2 | 1))
    assert state.rax == 0
    assert state.eflags & 1

    state = _run_one(bytes.fromhex("48 ff c8"), _state(rax=0, eflags=2 | 1))
    assert state.rax == U64_MASK
    assert state.eflags & 1


def test_adc_and_sbb_consume_incoming_carry() -> None:
    state = _run_one(bytes.fromhex("48 83 d0 00"), _state(rax=0, eflags=2 | 1))
    assert state.rax == 1

    state = _run_one(bytes.fromhex("48 83 d8 00"), _state(rax=0, eflags=2 | 1))
    assert state.rax == U64_MASK
    assert state.eflags & 1


def test_inc_preserves_undefined_carry_without_inventing_a_value() -> None:
    state = _state(
        rax=1,
        flags_defined={
            "CF": False,
            "ZF": True,
            "SF": True,
            "OF": True,
            "PF": True,
            "AF": True,
        },
    )
    _run_one(bytes.fromhex("48 ff c0"), state)
    assert state.rax == 2
    assert state.flags_defined["CF"] is False
    with pytest.raises(ValueError, match="undefined flag CF"):
        _run_one(bytes.fromhex("72 00"), state)  # jb, which reads CF


def test_extension_lea_and_logical_width_semantics() -> None:
    state = _state(
        rax=0xFFFFFFFFFFFFFF80,
        registers={"rax": 0xFFFFFFFFFFFFFF80, "rcx": 3},
    )
    _run_one(bytes.fromhex("48 0f be c0"), state)  # movsx rax, al
    assert state.rax == 0xFFFFFFFFFFFFFF80
    _run_one(bytes.fromhex("48 8d 4c 88 08"), state)  # lea rcx,[rax+rcx*4+8]
    assert state.registers["rcx"] == (state.rax + 3 * 4 + 8) & U64_MASK
    _run_one(bytes.fromhex("48 f7 d0"), state)  # not rax
    assert state.rax == 0x7F
    _run_one(bytes.fromhex("0f b6 c0"), state)  # movzx eax, al
    assert state.rax == 0x7F


def test_real_branch_path_matches_unicorn_for_taken_and_not_taken_cases() -> None:
    code = bytes.fromhex("48 83 f8 00 74 04 48 83 c0 01 90")
    taken_rows, taken_error = _reference_trace(code, 0)
    not_taken_rows, not_taken_error = _reference_trace(code, 1)
    assert taken_error is None and not_taken_error is None
    assert [row["address"] for row in taken_rows] == [0x1000, 0x1004, 0x100A]
    assert [row["address"] for row in not_taken_rows] == [
        0x1000,
        0x1004,
        0x1006,
        0x100A,
    ]
    assert int(taken_rows[-1]["post_rax"]) == 0
    assert int(not_taken_rows[-1]["post_rax"]) == 2
    taken_unicorn = emulate_x86_64(code, initial_rax=0)
    not_taken_unicorn = emulate_x86_64(code, initial_rax=1)
    assert taken_unicorn.addresses == tuple(row["address"] for row in taken_rows)
    assert not_taken_unicorn.addresses == tuple(
        row["address"] for row in not_taken_rows
    )
    assert taken_unicorn.final_rax == 0
    assert not_taken_unicorn.final_rax == 2


def test_push_pop_call_ret_and_memory_events_stay_in_declared_bounds() -> None:
    state = _state(rax=0x1234)
    _step(disassemble_bytes(bytes.fromhex("50"))[0], state, 0x1001)
    pushed_rsp = state.rsp
    assert pushed_rsp == 0x100FF8
    assert all(address >= 0 for address in state.memory)
    _step(disassemble_bytes(bytes.fromhex("58"))[0], state, 0x1002)
    assert state.rax == 0x1234
    assert state.rsp == 0x101000

    call_state = _state(rax=7)
    _step(
        disassemble_bytes(bytes.fromhex("e8 fb 00 00 00"))[0],
        call_state,
        None,
    )
    assert call_state.rsp == 0x100FF8
    assert call_state.rip == 0x1100
    _step(disassemble_bytes(bytes.fromhex("c3"))[0], call_state, 0x1005)
    assert call_state.rsp == 0x101000
    assert call_state.rip == 0x1005


def test_indexed_memory_addressing_is_bounded_and_uses_register_file() -> None:
    state = _state(
        rax=0xCAFE,
        registers={"rax": 0xCAFE, "rcx": 3, "rsp": 0x101000},
    )
    event = disassemble_bytes(bytes.fromhex("48 89 44 8c 08"))[0]
    _step(event, state, None)
    address = 0x101000 + 3 * 4 + 8
    assert (
        int.from_bytes(bytes(state.memory[address + i] for i in range(8)), "little")
        == 0xCAFE
    )


def test_memory_and_stack_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="bounded byte addresses"):
        FormalState(
            rax=0,
            rsp=0x101000,
            rip=0x1000,
            eflags=2,
            memory={0x300000: 1},
        )
    state = _state()
    state.set_register("rsp", state.stack_bounds[0])
    with pytest.raises(ValueError, match="bounded stack"):
        _run_one(bytes.fromhex("50"), state)  # push rax below the stack window


def test_unsupported_instruction_is_visible_and_never_invented() -> None:
    rows, reason = _reference_trace(bytes.fromhex("0f 05"), 0)  # syscall
    assert rows == []
    assert reason and "unsupported mnemonic" in reason


def test_unicorn_snapshots_all_declared_registers() -> None:
    trace = emulate_x86_64(bytes.fromhex("48 89 c1"))
    assert set(GENERAL_REGISTER_NAMES).issubset(trace.registers)
    assert set(GENERAL_REGISTER_NAMES) == set(
        trace.instruction_states[0].pre_registers or {}
    )


def test_state_and_serialization_fail_closed_on_invalid_bounds_and_aliases() -> None:
    with pytest.raises(ValueError, match="memory bounds"):
        FormalState(
            rax=0,
            rsp=0x101000,
            rip=0x1000,
            eflags=2,
            memory={},
            memory_bounds=(4, 4),
        )
    with pytest.raises(ValueError, match="stack bounds"):
        FormalState(
            rax=0,
            rsp=0x101000,
            rip=0x1000,
            eflags=2,
            memory={},
            stack_bounds=(4, 4),
        )
    with pytest.raises(ValueError, match="bounded byte addresses"):
        FormalState(
            rax=0,
            rsp=0x101000,
            rip=0x1000,
            eflags=2,
            memory={0x1000: 256},
        )
    with pytest.raises(ValueError, match="serialized state bounds"):
        FormalState.from_dict({"registers": {}, "stack_bounds": [0]})
    assert _register_ref("%eax") == ("rax", 32, 0)
    with pytest.raises(ValueError, match="unsupported register"):
        _register_ref("bogus")
    with pytest.raises(ValueError, match="unsupported register width"):
        _write_register_value(0, 1, 7)


def test_normalized_helpers_cover_flags_operands_memory_and_conditions() -> None:
    assert _split_operands("qword ptr [rax+rcx*4+8], rdx") == (
        "qword ptr [rax+rcx*4+8]",
        "rdx",
    )
    assert _token_width("dword ptr [rax]") == 32
    assert _token_width("not-a-register") == 64
    assert _flag_reads_and_writes("adc")[0] == ("CF",)
    assert "CF" not in _flag_reads_and_writes("inc")[1]
    assert _flag_reads_and_writes("jle")[0] == ("SF", "OF", "ZF")
    state = _state(rax=0x80, eflags=2)
    _set_flag(state, "CF", True)
    assert _condition_holds(state, "jb") is True
    assert _condition_holds(state, "jae") is False
    assert _condition_holds(state, "jmp") is True
    for mnemonic in (
        "jl",
        "jge",
        "jle",
        "jg",
        "je",
        "jne",
        "jbe",
        "ja",
        "jo",
        "jno",
        "js",
        "jns",
        "jp",
        "jnp",
    ):
        _condition_holds(state, mnemonic)
    with pytest.raises(ValueError, match="unsupported branch condition"):
        _condition_holds(state, "jcxz")
    assert _parity_even(0) is True
    assert _parity_even(1) is False
    _set_logic_flags(state, 0, 8)
    assert state.flags_defined["AF"] is False
    _set_arithmetic_flags(state, 0xFF, 1, 0, width=8, subtract=False)
    assert state.flags_defined["CF"] is True
    _set_arithmetic_flags(state, 0x7F, 1, 0x80, width=8, subtract=False)
    assert state.flags_defined["OF"] is True
    assert _sign_extend(0x80, 8, 64) == U64_MASK - 127

    memory_event = disassemble_bytes(bytes.fromhex("48 89 44 8c 08"))[0]
    state.registers["rcx"] = 3
    address = _memory_address(memory_event, state)
    assert address == 0x101000 + 3 * 4 + 8
    assert _memory_width(memory_event) == 8
    assert _memory_width(memory_event, 3, default=4) == 4
    _write_memory(state, address, 8, 0xCAFE)
    assert _read_memory(state, address, 8) == 0xCAFE
    with pytest.raises(ValueError, match="unsupported memory width"):
        _read_memory(state, address, 3)
    with pytest.raises(ValueError, match="memory access"):
        _read_memory(state, state.memory_bounds[1] - 1, 8)
    assert _next_rip(state, memory_event, None) == state.rip + memory_event.size


def test_normalized_operand_paths_and_extended_integer_operations() -> None:
    state = _state(
        rax=0x80,
        registers={"rax": 0x80, "rcx": 2, "rsp": 0x101000},
    )
    for code in (
        "48 09 c8",  # or rax, rcx
        "48 21 c8",  # and rax, rcx
        "48 31 c8",  # xor rax, rcx
        "48 85 c0",  # test rax, rax
        "48 f7 d8",  # neg rax
        "48 c1 e0 01",  # shl rax, 1
        "48 d1 e8",  # shr rax, 1
        "48 d1 f8",  # sar rax, 1
    ):
        _run_one(bytes.fromhex(code), state)
    memory_store = disassemble_bytes(bytes.fromhex("48 89 44 24 08"))[0]
    memory_load = disassemble_bytes(bytes.fromhex("48 8b 44 24 08"))[0]
    stored_value = state.rax
    _step(memory_store, state, None)
    state.set_register("rax", 0)
    _step(memory_load, state, None)
    assert state.rax == stored_value
    source_value, source_width = _read_operand(
        disassemble_bytes(bytes.fromhex("48 83 c0 01"))[0],
        state,
        "1",
        memory_index=0,
    )
    assert (source_value, source_width) == (1, 8)
    _write_operand(memory_store, state, "rax", 3, memory_index=0, width=8)
    assert state.rax == 3
    with pytest.raises(ValueError, match="unsupported immediate"):
        _parse_immediate("symbolic", memory_store)
    assert _parse_immediate("0x10", memory_store) == 16
    result = _step(memory_store, state, None)
    assert isinstance(result, StepResult)
    assert result.to_dict()["status"] == "ok"
