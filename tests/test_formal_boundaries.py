"""MIN-04 formal boundary regression pack.

Objective: lock the declared x86-64 boundary semantics of the real reference
interpreter in ``src/formal_correspondence.py`` (FormalState, _step,
normalize_instruction and their helpers) with hand-computed expected values.
Primary endpoint: ``uv run pytest tests/test_formal_boundaries.py -q`` green,
complementing (not repeating) tests/test_formal_state_v2.py.

Boundary matrix covered:
* register width updates  (al/ah/ax/eax/rax, r8b/r8w/r8d/r8; 32-bit
  zero-extend; 8/16-bit and high-byte writes preserve upper bits)
* INC/DEC carry preservation (CF bit value AND flags_defined entry)
* ADC/SBB carry consumption (CF set / cleared / undefined -> ValueError)
* memory bounds fail-closed (below BASE_ADDRESS, past the 2 MiB upper bound,
  little-endian byte order for 2/4/8-byte accesses)
* stack bounds for push/pop/call/ret (lower and upper edges, positive and
  negative)
* branch outcomes for every _condition_holds family, both outcomes where
  reachable, plus undefined-flag reads raising
* serialization round-trip (to_dict -> from_dict -> to_dict equality,
  clone independence, partial-definedness normalization)
* unsupported instructions (normalize_instruction raises ValueError naming
  the mnemonic; SIMD and floating-point representatives)

All expected values below are hand-computed; each assertion group carries the
arithmetic justification in a comment.
"""

from __future__ import annotations

import re

import pytest
from src.disassembly import BASE_ADDRESS, disassemble_bytes
from src.formal_correspondence import (
    MEMORY_UPPER_BOUND,
    MODELED_FLAGS,
    U64_MASK,
    FormalState,
    StepResult,
    _condition_holds,
    _get_flag,
    _read_memory,
    _reference_trace,
    _set_flag,
    _step,
    _write_memory,
    normalize_instruction,
)

RAX_START = 0x1122334455667788
R8_START = 0x1122334455667788
STACK_LOWER = BASE_ADDRESS + 0x100000 - 4096  # 0x100000
STACK_UPPER = BASE_ADDRESS + 0x100000 + 4096  # 0x102000


def _state(
    *,
    rax: int = 0,
    rsp: int = 0x101000,
    eflags: int = 2,
    registers: dict[str, int] | None = None,
    flags_defined: dict[str, bool] | None = None,
    memory: dict[int, int] | None = None,
) -> FormalState:
    base_registers = dict(registers or {})
    base_registers.setdefault("rax", rax)
    return FormalState(
        rax=rax,
        rsp=rsp,
        rip=0x1000,
        eflags=eflags,
        memory=memory or {},
        registers=base_registers,
        flags_defined=flags_defined or {},
    )


def _run_one(code: str, state: FormalState) -> StepResult:
    event = disassemble_bytes(bytes.fromhex(code.replace(" ", "")))[0]
    return _step(event, state, None)


# ---------------------------------------------------------------------------
# Register width updates (8/16/32/64 + high-byte aliases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "value", "expected"),
    [
        # 8-bit low-byte write: only bits [7:0] change (0x88 -> 0xAB), all
        # upper 56 bits of 0x1122334455667788 are preserved.
        ("al", 0xAB, 0x11223344556677AB),
        # 8-bit high-byte write: bits [15:8] change (0x77 -> 0xCD); byte 0
        # keeps 0x88 and bits [63:16] are untouched.
        ("ah", 0xCD, 0x112233445566CD88),
        # 16-bit write: bits [15:0] change (0x7788 -> 0xEF01), bytes 2..7
        # (0x55 44 33 22 11) are preserved.
        ("ax", 0xEF01, 0x112233445566EF01),
        # 32-bit offset-0 write ZERO-EXTENDS: bits [63:32] of the previous
        # 0x11223344... are discarded, result is exactly 0xDEADBEEF.
        ("eax", 0xDEADBEEF, 0x00000000DEADBEEF),
        # 64-bit write replaces the whole register.
        ("rax", 0x0102030405060708, 0x0102030405060708),
    ],
)
def test_rax_family_width_update_boundaries(
    alias: str, value: int, expected: int
) -> None:
    state = _state(rax=RAX_START, registers={"rax": RAX_START})
    state.set_register(alias, value)
    assert state.get_register(alias) == value
    assert state.registers["rax"] == expected
    assert state.rax == expected  # compatibility scalar tracks the file


@pytest.mark.parametrize(
    ("alias", "value", "expected"),
    [
        # r8 mirrors the rax family: 8/16-bit writes preserve upper bits,
        # 32-bit writes zero-extend, 64-bit writes replace.
        ("r8b", 0xAB, 0x11223344556677AB),
        ("r8w", 0xEF01, 0x112233445566EF01),
        ("r8d", 0xDEADBEEF, 0x00000000DEADBEEF),
        ("r8", 0x0102030405060708, 0x0102030405060708),
    ],
)
def test_r8_family_width_update_boundaries(
    alias: str, value: int, expected: int
) -> None:
    state = _state(registers={"r8": R8_START})
    state.set_register(alias, value)
    assert state.registers["r8"] == expected


def test_step_level_mov_honours_width_and_preserve_rules() -> None:
    # mov al, 0xAB on rax=0x1122334455667788: low byte 0x88 -> 0xAB, the
    # remaining 7 bytes are untouched.
    state = _state(rax=RAX_START, registers={"rax": RAX_START})
    _run_one("b0 ab", state)  # mov al, 0xab
    assert state.registers["rax"] == 0x11223344556677AB
    # mov ah, 0xCD: bits [15:8] 0x77 -> 0xCD, low byte keeps 0xAB.
    _run_one("b4 cd", state)  # mov ah, 0xcd
    assert state.registers["rax"] == 0x112233445566CDAB
    # mov eax, 0xDEADBEEF zero-extends: rax becomes exactly 0xDEADBEEF and the
    # old upper dword 0x11223344 is gone.
    state = _state(rax=RAX_START, registers={"rax": RAX_START})
    _run_one("b8 ef be ad de", state)  # mov eax, 0xdeadbeef
    assert state.registers["rax"] == 0xDEADBEEF


# ---------------------------------------------------------------------------
# INC/DEC carry preservation (value AND definedness)
# ---------------------------------------------------------------------------


def test_inc_preserves_carry_value_and_defined_entry() -> None:
    # inc rax on 0xFFFFFFFFFFFFFFFF with CF=1 (eflags=0b11):
    #   result = 2^64 mod 2^64 = 0 -> ZF=1 (bit 6 = 64)
    #   AF=1 because (0xF & 0xF) + 1 > 0xF (bit 4 = 16)
    #   PF=1 because 0x00 has even parity (bit 2 = 4)
    #   SF=0, OF=0 because signed (-1) + 1 = 0 fits in 64 bits
    #   CF is PRESERVED at 1 (bit 0), not recomputed (no carry out of +1 here)
    #   plus the reserved bit 1 (2): 64 + 16 + 4 + 2 + 1 = 87 = 0x57.
    state = _state(rax=U64_MASK, registers={"rax": U64_MASK}, eflags=3)
    _run_one("48 ff c0", state)  # inc rax
    assert state.registers["rax"] == 0
    assert state.eflags == 0x57
    assert state.eflags & 1, "CF bit must stay set"
    assert state.flags_defined["CF"] is True, "CF definedness must stay defined"


def test_dec_preserves_carry_value_against_recomputation() -> None:
    # dec rax on 5 with CF=1: result = 4.  Recomputing the borrow would give
    # CF=0 (5 >= 1), but INC/DEC never touch CF: it must remain 1.
    # Other flags: AF=0 (5 & 0xF = 5 >= 1 + 0), PF=0 (4 = 0b100 has odd
    # parity), ZF=0, SF=0, OF=0 (5 - 1 fits).  eflags = bit1(2) + CF(1) = 3.
    state = _state(rax=5, registers={"rax": 5}, eflags=3)
    _run_one("48 ff c8", state)  # dec rax
    assert state.registers["rax"] == 4
    assert state.eflags == 3
    assert state.flags_defined["CF"] is True


def test_inc_preserves_clear_carry_without_setting_it() -> None:
    # inc rax on 0 with CF=0: result = 1.  CF stays 0 AND defined.
    # PF=0 (1 has odd parity), AF=0, ZF=0, SF=0, OF=0 -> only bit1: eflags=2.
    state = _state(eflags=2)
    _run_one("48 ff c0", state)  # inc rax
    assert state.registers["rax"] == 1
    assert state.eflags == 2
    assert state.flags_defined["CF"] is True
    assert not state.eflags & 1


def test_dec_preserves_undefined_carry_and_later_adc_raises() -> None:
    # Same contract as inc (covered in test_formal_state_v2 for inc): a DEC
    # with CF undefined must leave CF undefined, and a later ADC that reads
    # CF fails closed instead of consuming an invented carry.
    partial = {"CF": False, "ZF": True, "SF": True, "OF": True, "PF": True, "AF": True}
    state = _state(rax=5, registers={"rax": 5}, flags_defined=partial)
    _run_one("48 ff c8", state)  # dec rax
    assert state.registers["rax"] == 4
    assert state.flags_defined["CF"] is False
    with pytest.raises(ValueError, match="undefined flag CF"):
        _run_one("48 83 d0 00", state)  # adc rax, 0 reads CF


# ---------------------------------------------------------------------------
# ADC/SBB carry consumption (CF set, cleared, undefined)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "rax", "eflags", "expected_rax", "expected_eflags"),
    [
        # adc rax, 0 with CF=1: 1 + 0 + 1 = 2, no carry out (2 <= mask).
        # Flags: ZF=0, SF=0, OF=0, AF=0 ((1&0xF)+0+1 = 2 <= 0xF),
        # PF=0 (2 has odd parity) -> only reserved bit1: eflags = 0x2.
        ("48 83 d0 00", 1, 3, 2, 0x2),
        # adc rax, 1 with CF=0 on 0xFFFFFFFFFFFFFFFF:
        # mask + 0 + 1 = 2^64 -> result 0 with carry OUT -> CF=1 (bit 0 = 1);
        # ZF=1 (64), AF=1 (0xF + 1 + 0 > 0xF, 16), PF=1 (parity of 0x00, 4),
        # SF=0, OF=0 -> 64 + 16 + 4 + 2 + 1 = 0x57.
        ("48 83 d0 01", U64_MASK, 2, 0, 0x57),
        # sbb rax, 1 with CF=1 on 5: 5 - 1 - 1 = 3.  Borrow check:
        # 5 < (1 + 1) = 2 is false -> recomputed CF=0 but the INCOMING CF=1
        # is consumed, and the outgoing CF=0 clears the stale bit.
        # AF: (5&0xF)=5 < (1&0xF)+1=2 false -> 0; PF=1 (3 = 0b11, even);
        # ZF/SF/OF=0 -> eflags = bit1(2) + PF(4) = 6.
        ("48 83 d8 01", 5, 3, 3, 0x6),
        # sbb rax, 1 with CF=0 on 0: 0 - 1 - 0 wraps to 0xFFFFFFFFFFFFFFFF.
        # CF=1 (0 < 1), SF=1 (bit 7 = 128), ZF=0, OF=0 (0 - 1 = -1 in range),
        # AF=1 (0 < 1 + 0), PF=1 (0xFF has eight set bits, even) ->
        # 128 + 16 + 4 + 2 + 1 = 151 = 0x97.
        ("48 83 d8 01", 0, 2, U64_MASK, 0x97),
    ],
)
def test_adc_sbb_consume_incoming_carry_value(
    code: str,
    rax: int,
    eflags: int,
    expected_rax: int,
    expected_eflags: int,
) -> None:
    state = _state(rax=rax, registers={"rax": rax}, eflags=eflags)
    _run_one(code, state)
    assert state.registers["rax"] == expected_rax
    assert state.eflags == expected_eflags


@pytest.mark.parametrize("code", ["48 83 d0 00", "48 83 d8 01"])  # adc / sbb
def test_adc_sbb_fail_closed_on_undefined_carry(code: str) -> None:
    # With CF undefined, ADC/SBB must refuse to invent a carry-in.
    partial = {"CF": False, "ZF": True, "SF": True, "OF": True, "PF": True, "AF": True}
    state = _state(rax=9, registers={"rax": 9}, flags_defined=partial)
    with pytest.raises(ValueError, match="undefined flag CF"):
        _run_one(code, state)
    assert state.registers["rax"] == 9  # fail-closed: state value untouched


# ---------------------------------------------------------------------------
# Memory bounds fail-closed + little-endian byte order
# ---------------------------------------------------------------------------


def test_memory_read_write_below_base_address_fails_closed() -> None:
    state = _state()

    with pytest.raises(ValueError, match=r"memory access \[0xfff, 0x1000\)"):
        _read_memory(state, BASE_ADDRESS - 1, 1)
    with pytest.raises(ValueError, match=r"memory access \[0xff8, 0x1000\)"):
        _write_memory(state, BASE_ADDRESS - 8, 8, 0)


def test_memory_write_past_upper_bound_fails_closed_with_inclusive_edge() -> None:
    state = _state()
    upper = MEMORY_UPPER_BOUND  # BASE_ADDRESS + 2 MiB = 0x201000
    # The bound is inclusive at the top: a dword ending exactly at `upper`
    # (address + width == upper) is allowed.
    _write_memory(state, upper - 4, 4, 0xDEADBEEF)
    assert state.memory[upper - 4] == 0xEF
    assert state.memory[upper - 1] == 0xDE
    # One byte further right and the same dword crosses the 2 MiB bound.
    with pytest.raises(ValueError, match=r"memory access \[0x200ffd, 0x201001\)"):
        _write_memory(state, upper - 3, 4, 0)
    with pytest.raises(ValueError, match=r"memory access \[0x200fff, 0x201007\)"):
        _read_memory(state, upper - 1, 8)


def test_memory_byte_order_is_little_endian_for_2_4_8_byte_accesses() -> None:
    state = _state()
    base = BASE_ADDRESS
    # 2-byte 0x1234 -> bytes [0x34, 0x12].
    _write_memory(state, base, 2, 0x1234)
    assert [state.memory[base], state.memory[base + 1]] == [0x34, 0x12]
    # 4-byte 0xDEADBEEF -> bytes [0xEF, 0xBE, 0xAD, 0xDE].
    _write_memory(state, base + 2, 4, 0xDEADBEEF)
    assert [state.memory[base + 2 + i] for i in range(4)] == [0xEF, 0xBE, 0xAD, 0xDE]
    # 8-byte 0x0123456789ABCDEF -> bytes [0xEF, 0xCD, ..., 0x01].
    _write_memory(state, base + 8, 8, 0x0123456789ABCDEF)
    assert [state.memory[base + 8 + i] for i in range(8)] == [
        0xEF,
        0xCD,
        0xAB,
        0x89,
        0x67,
        0x45,
        0x23,
        0x01,
    ]
    assert _read_memory(state, base, 2) == 0x1234
    assert _read_memory(state, base + 2, 4) == 0xDEADBEEF
    assert _read_memory(state, base + 8, 8) == 0x0123456789ABCDEF


def test_step_level_word_and_dword_stores_are_little_endian() -> None:
    # mov word ptr [rax], 0x1234 with rax = BASE_ADDRESS: byte 0x34 at the
    # low address, 0x12 one byte up; the memory-effects log records the
    # assembled little-endian value.
    state = _state(rax=BASE_ADDRESS, registers={"rax": BASE_ADDRESS})
    _run_one("66 c7 00 34 12", state)
    assert state.memory[BASE_ADDRESS] == 0x34
    assert state.memory[BASE_ADDRESS + 1] == 0x12
    assert state.memory_effects == [
        {"access": "write", "address": BASE_ADDRESS, "size": 2, "value": 0x1234}
    ]
    # mov dword ptr [rax], 0xDEADBEEF -> bytes EF BE AD DE.
    state = _state(rax=BASE_ADDRESS, registers={"rax": BASE_ADDRESS})
    _run_one("c7 00 ef be ad de", state)
    assert [state.memory[BASE_ADDRESS + i] for i in range(4)] == [
        0xEF,
        0xBE,
        0xAD,
        0xDE,
    ]


# ---------------------------------------------------------------------------
# Stack bounds for push/pop/call/ret
# ---------------------------------------------------------------------------


def test_push_and_pop_succeed_at_exact_stack_boundaries() -> None:
    # push with rsp exactly at the stack upper bound is legal: the new rsp
    # 0x102000 - 8 = 0x101FF8 plus 8 bytes ends at 0x102000 (inclusive edge).
    state = _state(rax=0xCAFEBABE, registers={"rax": 0xCAFEBABE}, rsp=STACK_UPPER)
    _run_one("50", state)  # push rax
    assert state.rsp == STACK_UPPER - 8
    # 0xCAFEBABE as a little-endian 8-byte value is BE BA FE CA 00 00 00 00:
    # the four high bytes at the top of the window are zero.
    assert [state.memory[STACK_UPPER - 8 + i] for i in range(8)] == [
        0xBE,
        0xBA,
        0xFE,
        0xCA,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
    # pop with rsp exactly at the stack lower bound is legal: the qword read
    # [0x100000, 0x100008) stays inside the window (the memory reads zeros).
    state = _state(rsp=STACK_LOWER)
    _run_one("58", state)  # pop rax
    assert state.rsp == STACK_LOWER + 8
    assert state.registers["rax"] == 0


@pytest.mark.parametrize(
    ("code", "rsp"),
    [
        # push decrements first: rsp = 0x100000 -> access [0xffff8, 0x100000)
        # starts below the lower bound.
        ("50", STACK_LOWER),
        # push with rsp = 0x102008 -> access [0x102000, 0x102008) crosses the
        # upper bound.
        ("50", STACK_UPPER + 8),
        ("58", STACK_LOWER - 8),
        # call pushes the return address, same lower/upper edges as push.
        ("e8 fb 00 00 00", STACK_LOWER),
        ("e8 fb 00 00 00", STACK_UPPER + 8),
        # ret pops from [rsp, rsp+8).
        ("c3", STACK_LOWER - 8),
        ("c3", STACK_UPPER),
    ],
)
def test_stack_operations_fail_closed_outside_bounded_window(
    code: str, rsp: int
) -> None:
    state = _state(rsp=rsp)
    with pytest.raises(ValueError, match="exceeds bounded stack"):
        _run_one(code, state)
    assert state.rsp == rsp  # failed step leaves the stack pointer untouched


# ---------------------------------------------------------------------------
# Branch outcomes for every _condition_holds family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mnemonic", "flags", "expected"),
    [
        # signed-less: SF != OF.  Both outcomes reachable.
        ("jl", {"SF": True, "OF": False}, True),
        ("jl", {"SF": True, "OF": True}, False),
        # signed-greater-or-equal: SF == OF (complement of jl).
        ("jge", {"SF": True, "OF": True}, True),
        ("jge", {"SF": True, "OF": False}, False),
        # less-or-equal: ZF OR (SF != OF).  Both outcomes reachable.
        ("jle", {"ZF": True}, True),
        ("jle", {"ZF": False}, False),
        ("jle", {"ZF": False, "SF": True}, True),  # SF != OF branch of the OR
        # greater: NOT ZF AND (SF == OF).  Both outcomes reachable.
        ("jg", {"ZF": False, "SF": True, "OF": True}, True),
        ("jg", {"ZF": True}, False),
        ("jg", {"ZF": False, "SF": True, "OF": False}, False),
        # zero / not-zero.
        ("je", {"ZF": True}, True),
        ("je", {"ZF": False}, False),
        ("jne", {"ZF": False}, True),
        ("jne", {"ZF": True}, False),
        # below / above-or-equal on CF.
        ("jb", {"CF": True}, True),
        ("jb", {"CF": False}, False),
        ("jae", {"CF": False}, True),
        ("jae", {"CF": True}, False),
        # below-or-equal: CF OR ZF.  Both outcomes reachable.
        ("jbe", {"CF": True, "ZF": False}, True),
        ("jbe", {"CF": False, "ZF": False}, False),
        ("jbe", {"CF": False, "ZF": True}, True),
        # above: NOT CF AND NOT ZF.  Both outcomes reachable.
        ("ja", {"CF": False, "ZF": False}, True),
        ("ja", {"CF": True, "ZF": False}, False),
        ("ja", {"CF": False, "ZF": True}, False),
        # overflow / no overflow.
        ("jo", {"OF": True}, True),
        ("jo", {"OF": False}, False),
        ("jno", {"OF": False}, True),
        ("jno", {"OF": True}, False),
        # sign / not sign.
        ("js", {"SF": True}, True),
        ("js", {"SF": False}, False),
        ("jns", {"SF": False}, True),
        ("jns", {"SF": True}, False),
        # parity / not parity.
        ("jp", {"PF": True}, True),
        ("jp", {"PF": False}, False),
        ("jnp", {"PF": False}, True),
        ("jnp", {"PF": True}, False),
        # jmp is unconditional: only the taken outcome is reachable.
        ("jmp", {}, True),
    ],
)
def test_condition_holds_every_family_both_outcomes(
    mnemonic: str, flags: dict[str, bool], expected: bool
) -> None:
    state = _state()
    for flag, value in flags.items():
        _set_flag(state, flag, value)
    assert _condition_holds(state, mnemonic) is expected


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("jnge", "jl"),
        ("jnl", "jge"),
        ("jng", "jle"),
        ("jnle", "jg"),
        ("jz", "je"),
        ("jnz", "jne"),
        ("jc", "jb"),
        ("jnae", "jb"),
        ("jnb", "jae"),
        ("jnc", "jae"),
        ("jna", "jbe"),
        ("jnbe", "ja"),
        ("jpe", "jp"),
        ("jpo", "jnp"),
    ],
)
def test_branch_aliases_agree_with_canonical_mnemonics(
    alias: str, canonical: str
) -> None:
    mixed = {"CF": True, "ZF": True, "SF": True, "OF": False, "PF": False}
    alias_state = _state()
    canonical_state = _state()
    for flag, value in mixed.items():
        _set_flag(alias_state, flag, value)
        _set_flag(canonical_state, flag, value)
    assert _condition_holds(alias_state, alias) is _condition_holds(
        canonical_state, canonical
    )


def test_condition_holds_fallthrough_contract() -> None:
    # A non-branch mnemonic never satisfies a condition: _condition_holds is
    # only reached for j-mnemonics in _step, and non-j mnemonics fall through
    # to False rather than raising.
    state = _state()
    for flag in MODELED_FLAGS:
        _set_flag(state, flag, True)
    assert _condition_holds(state, "add") is False


@pytest.mark.parametrize(
    ("code", "flag", "taken_target", "not_taken_target"),
    [
        # Short jumps at rip=0x1000 with displacement 4: taken target is the
        # Capstone-rendered absolute 0x1006 (0x1000 + 2 + 4); not-taken falls
        # through to rip + size = 0x1002.
        ("70 04", "OF", 0x1006, 0x1002),  # jo
        ("72 04", "CF", 0x1006, 0x1002),  # jb
        ("76 04", "CF", 0x1006, 0x1002),  # jbe
        ("78 04", "SF", 0x1006, 0x1002),  # js
        ("7a 04", "PF", 0x1006, 0x1002),  # jp
        ("7c 04", "SF", 0x1006, 0x1002),  # jl (with OF cleared)
    ],
)
def test_branch_step_outcomes_update_rip_and_control_target(
    code: str, flag: str, taken_target: int, not_taken_target: int
) -> None:
    taken_state = _state()
    _set_flag(taken_state, flag, True)
    if code == "7c 04":
        _set_flag(taken_state, "OF", False)
    taken = _run_one(code, taken_state)
    assert taken.control_target == taken_target
    assert taken_state.rip == taken_target
    assert taken_state.step_index == 1

    not_taken_state = _state()
    _set_flag(not_taken_state, flag, False)
    _set_flag(not_taken_state, "ZF", False)
    _set_flag(not_taken_state, "OF", False)
    not_taken = _run_one(code, not_taken_state)
    assert not_taken.control_target == not_taken_target
    assert not_taken_state.rip == not_taken_target
    assert not_taken_state.step_index == 1


@pytest.mark.parametrize(
    ("code", "first_flag_read"),
    [
        ("70 04", "OF"),  # jo
        ("72 04", "CF"),  # jb
        ("74 04", "ZF"),  # je
        ("78 04", "SF"),  # js
        ("7a 04", "PF"),  # jp
        ("7c 04", "SF"),  # jl reads SF before OF
    ],
)
def test_branch_on_undefined_flag_fails_closed(code: str, first_flag_read: str) -> None:
    undefined = {flag: False for flag in MODELED_FLAGS}
    state = _state(flags_defined=undefined)
    with pytest.raises(ValueError, match=f"undefined flag {first_flag_read}"):
        _run_one(code, state)


@pytest.mark.parametrize("flag", MODELED_FLAGS)
def test_direct_read_of_undefined_flag_raises(flag: str) -> None:
    state = _state(flags_defined={name: False for name in MODELED_FLAGS})
    with pytest.raises(ValueError, match=f"undefined flag {flag}"):
        _get_flag(state, flag)


# ---------------------------------------------------------------------------
# Serialization round-trip and clone independence
# ---------------------------------------------------------------------------


def _rich_state() -> FormalState:
    state = _state(
        rax=0x1234,
        eflags=0x57,
        registers={"rax": 0x1234, "rcx": 7, "rsp": 0x101000},
        flags_defined={
            "CF": True,
            "ZF": False,
            "SF": True,
            "OF": False,
            "PF": True,
            "AF": False,
        },
        memory={0x1000: 0x34, 0x1001: 0x12},
    )
    state.rip = 0x1008
    state.step_index = 7
    state.status = "halted"
    state.current_instruction = {"mnemonic": "nop", "operands": []}
    state.memory_effects.append(
        {"access": "write", "address": 0x1000, "size": 1, "value": 0x34}
    )
    return state


def test_serialization_round_trip_preserves_every_field() -> None:
    state = _rich_state()
    first = state.to_dict()
    restored = FormalState.from_dict(first)
    second = restored.to_dict()
    assert second == first
    # Explicit field checks so a structural drift in one field cannot hide
    # behind the dict equality above.
    assert restored.flags_defined == {
        "CF": True,
        "PF": True,
        "AF": False,
        "ZF": False,
        "SF": True,
        "OF": False,
    }
    assert restored.memory == {0x1000: 0x34, 0x1001: 0x12}
    assert restored.status == "halted"
    assert restored.step_index == 7
    assert restored.rip == 0x1008
    assert restored.eflags == 0x57
    assert restored.registers["rax"] == 0x1234
    assert restored.rsp == 0x101000
    assert tuple(restored.stack_bounds) == state.stack_bounds
    assert tuple(restored.memory_bounds) == state.memory_bounds
    assert restored.current_instruction == {"mnemonic": "nop", "operands": []}


def test_from_dict_normalizes_partial_flag_definedness_fail_closed() -> None:
    # A serialized state that declares only CF defined must NOT invent
    # definedness for the remaining flags: they normalize to False and a read
    # fails closed.
    payload = _rich_state().to_dict()
    payload["flags_defined"] = {"CF": True}
    restored = FormalState.from_dict(payload)
    assert restored.flags_defined == {
        "CF": True,
        "PF": False,
        "AF": False,
        "ZF": False,
        "SF": False,
        "OF": False,
    }
    with pytest.raises(ValueError, match="undefined flag ZF"):
        _get_flag(restored, "ZF")


def test_clone_is_independent_of_its_origin_in_every_mutable_field() -> None:
    state = _rich_state()
    clone = state.clone()
    clone.set_register("rax", 99)
    clone.registers["rcx"] = 42
    clone.memory[0x1000] = 0xFF
    clone.flags_defined["ZF"] = True
    clone.status = "ok"
    clone.step_index = 0
    clone.current_instruction["mnemonic"] = "mov"  # type: ignore[index]
    clone.memory_effects.append({"access": "read", "address": 1, "size": 1, "value": 1})
    assert state.registers["rax"] == 0x1234
    assert state.registers["rcx"] == 7
    assert state.memory[0x1000] == 0x34
    assert state.flags_defined["ZF"] is False
    assert state.status == "halted"
    assert state.step_index == 7
    assert state.current_instruction is not None
    assert state.current_instruction["mnemonic"] == "nop"
    assert state.memory_effects == [
        {"access": "write", "address": 0x1000, "size": 1, "value": 0x34}
    ]
    assert clone.to_dict() != state.to_dict()


# ---------------------------------------------------------------------------
# Unsupported instructions fail closed, naming the mnemonic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code_hex", "mnemonic", "klass"),
    [
        ("0f 58 c1", "addps", "SIMD"),
        ("f2 0f 58 c1", "addsd", "floating point (SSE2 scalar double)"),
        ("d8 00", "fadd", "floating point (x87)"),
    ],
)
def test_normalize_instruction_rejects_simd_and_floating_point(
    code_hex: str, mnemonic: str, klass: str
) -> None:
    del klass  # classification is documentation; the message names the mnemonic
    event = disassemble_bytes(bytes.fromhex(code_hex.replace(" ", "")))[0]
    assert event.mnemonic == mnemonic
    with pytest.raises(
        ValueError, match=re.escape(f"unsupported mnemonic {mnemonic!r}")
    ):
        normalize_instruction(event)


def test_unsupported_simd_step_and_trace_fail_closed() -> None:
    event = disassemble_bytes(bytes.fromhex("0f58c1"))[0]  # addps xmm0, xmm1
    with pytest.raises(ValueError, match="unsupported mnemonic 'addps'"):
        _step(event, _state(), None)
    rows, reason = _reference_trace(bytes.fromhex("0f58c1"), 0)
    assert rows == []
    assert reason == "unsupported mnemonic 'addps'"


def test_step_result_type_is_preserved_for_successful_boundary_steps() -> None:
    result = _run_one("48 ff c0", _state())  # inc rax
    assert isinstance(result, StepResult)
    assert result.status == "ok"
    assert result.unsupported_reason is None


# ---------------------------------------------------------------------------
# MED-02 expansion families: logic-op flag definedness, cmp/add flag
# exactness, movzx zero-extension.  (push/pop stack bounds, call/ret and the
# jcc undefined-flag raise paths are locked by the test groups above; the
# reference semantics they pin is unchanged by the Lean expansion.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["21 d8", "09 d8", "31 d8", "85 c0"])
def test_logic_ops_mark_af_undefined_and_clear_cf_and_of(code: str) -> None:
    """and/or/xor/test with eax, ebx: AF undefined, CF and OF defined-cleared.

    rax = 0xF0F0F0F0, rbx = 0x0F0F0F0F: and/test -> 0, or/xor -> 0xFFFFFFFF.
    ZF/SF/PF follow the (masked) result; parity of the low byte is even in
    both cases (0x00 and 0xFF each have an even number of set bits).
    """
    state = _state(rax=0xF0F0F0F0, registers={"rax": 0xF0F0F0F0, "rbx": 0x0F0F0F0F})
    _run_one(code, state)
    assert state.flags_defined["AF"] is False
    assert state.flags_defined["CF"] is True
    assert _get_flag(state, "CF") is False
    assert state.flags_defined["OF"] is True
    assert _get_flag(state, "OF") is False
    op_result = 0 if code in ("21 d8", "85 c0") else 0xFFFFFFFF
    assert _get_flag(state, "ZF") is (op_result == 0)
    assert _get_flag(state, "SF") is bool(op_result & (1 << 31))
    assert _get_flag(state, "PF") is True
    if code == "85 c0":
        # test: no writeback
        assert state.registers["rax"] == 0xF0F0F0F0
    else:
        assert state.registers["rax"] == op_result


def test_read_of_af_after_logic_op_fails_closed() -> None:
    state = _state(rax=1, registers={"rax": 1, "rbx": 1})
    _run_one("31 d8", state)  # xor eax, ebx
    with pytest.raises(ValueError, match="undefined flag AF"):
        _get_flag(state, "AF")


def test_cmp_sets_exact_arithmetic_flags_without_writeback() -> None:
    """cmp eax, ebx with eax=3, rbx=5 mirrors the Lean subFlagsN outcome.

    result = 0xFFFFFFFE: CF=1 (3 < 5), AF=1 (3 < 5 in the low nibble),
    ZF=0, SF=1 (bit 31 set), PF=0 (low byte 0xFE has seven set bits),
    OF=0 (-3 fits int32), and eax is untouched.
    """
    state = _state(rax=3, registers={"rax": 3, "rbx": 5})
    _run_one("39 d8", state)
    assert _get_flag(state, "CF") is True
    assert _get_flag(state, "AF") is True
    assert _get_flag(state, "ZF") is False
    assert _get_flag(state, "SF") is True
    assert _get_flag(state, "PF") is False
    assert _get_flag(state, "OF") is False
    assert state.registers["rax"] == 3


def test_add_sets_exact_arithmetic_flags_all_defined() -> None:
    """add eax, ebx with eax=0xFFFFFFFF, rbx=2: result 1.

    CF=1 (carry out), AF=1 (0xF + 2 > 0xF), ZF=0, SF=0, PF=0 (low byte
    0x01 has one set bit), OF=0 (-1 + 2 = 1 fits int32); every modeled flag
    is defined, matching the Lean addFlagsN definedness map.
    """
    state = _state(rax=0xFFFFFFFF, registers={"rax": 0xFFFFFFFF, "rbx": 2})
    _run_one("01 d8", state)
    assert state.registers["rax"] == 1
    assert _get_flag(state, "CF") is True
    assert _get_flag(state, "AF") is True
    assert _get_flag(state, "ZF") is False
    assert _get_flag(state, "SF") is False
    assert _get_flag(state, "PF") is False
    assert _get_flag(state, "OF") is False
    assert all(state.flags_defined[flag] for flag in MODELED_FLAGS)


def test_movzx_eax_ax_zero_extends_full_register() -> None:
    """movzx eax, ax with rax = 0x112233445566BEEF: rax becomes 0xBEEF."""
    state = _state(rax=0x112233445566BEEF, registers={"rax": 0x112233445566BEEF})
    _run_one("0f b7 c0", state)
    assert state.registers["rax"] == 0xBEEF
    # flags are untouched by mov
    assert state.eflags == 2
