"""A labelled multi-fixture corpus of transparent x86-64 byte programs.

Each :class:`CorpusFixture` couples a short, hand-authored x86-64 byte program
with a tuple of ground-truth latent-phase labels — exactly one
:data:`features.HIDDEN_STATE_LABELS` value per *decoded* instruction. The corpus
is the substrate for the quantitative evaluation in :mod:`evaluation`: it lets
us measure how well the static and temporal inference models recover the known
phase sequence of each program.

Phase labelling convention (deterministic, hand-authored):

* ``setup`` for the initial immediate load(s),
* ``accumulator_update`` for ``add`` / ``sub`` arithmetic,
* ``representation_transform`` for shifts and bitwise rewrites,
* ``guard_check`` for ``cmp`` / ``test`` and conditional jumps,
* ``exit`` for the final instruction of each program.

The corpus deliberately contains more than one evidence regime.  In addition
to register-only straight-line programs, it includes a bounded stack-memory
fixture, a branch-taken companion to the external-target guard fixture, and a
MED-01 temporal expansion: a call/return fixture whose landing pad ``ret``
executes inside the bounded harness, a double not-taken guard fixture, a
repeated-phase stack-local fixture, and an optimization-rewrite fixture whose
``add``/``inc``/``lea`` motifs intentionally collapse onto few observation
categories.  These additions are still transparent toy programs, but they make
the Phase A typed-observation evaluation test memory addressing, repeated
phases, and dynamic/static path differences instead of merely counting
near-duplicate register instructions.

All byte fixtures are real machine code: they are decoded with Capstone via
:func:`disassembly.disassemble_bytes` and executed with Unicorn via
:func:`emulation.emulate_x86_64`; no instruction sequence is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .disassembly import disassemble_bytes
from .emulation import emulate_x86_64
from .features import encode_observations


@dataclass(frozen=True)
class CorpusFixture:
    """A labelled x86-64 byte program with one phase label per instruction.

    Attributes:
        name: Stable, human-readable identifier for the fixture.
        code: The raw x86-64 machine code bytes.
        phase_labels: Ground-truth latent-phase labels, one per decoded
            instruction, drawn from :data:`features.HIDDEN_STATE_LABELS`.
    """

    name: str
    code: bytes
    phase_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation (code rendered as hex)."""

        return {
            "name": self.name,
            "code": self.code.hex(),
            "phase_labels": list(self.phase_labels),
        }


def build_corpus() -> tuple[CorpusFixture, ...]:
    """Build the labelled corpus of transparent x86-64 byte fixtures.

    Returns:
        A tuple of :class:`CorpusFixture` instances. Each ``phase_labels`` tuple
        has exactly one label per instruction returned by
        :func:`disassembly.disassemble_bytes` for that fixture's ``code``.
    """

    # mov eax,1; add eax,2; shl eax,1; sub eax,1  -> setup, acc, transform, exit
    accumulator = CorpusFixture(
        name="x86_accumulator",
        code=bytes.fromhex("b801000000 83c002 d1e0 83e801".replace(" ", "")),
        phase_labels=(
            "setup",
            "accumulator_update",
            "representation_transform",
            "exit",
        ),
    )

    # mov eax,0xf; and eax,0xc; shr eax,2; or eax,1 -> setup, transform, transform, exit
    bitwise = CorpusFixture(
        name="x86_bitwise",
        code=bytes.fromhex("b80f000000 83e00c c1e802 83c801".replace(" ", "")),
        phase_labels=(
            "setup",
            "representation_transform",
            "representation_transform",
            "exit",
        ),
    )

    # mov eax,5; cmp eax,3; jl ...; add eax,1 -> setup, guard, guard, exit
    guard = CorpusFixture(
        name="x86_guard",
        code=bytes.fromhex("b805000000 83f803 7c02 83c001".replace(" ", "")),
        phase_labels=(
            "setup",
            "guard_check",
            "guard_check",
            "exit",
        ),
    )

    # mov eax,0; add 0xa; add 0x14; shl 1; sub 3; add 1
    # -> setup, acc, acc, transform, acc, exit
    long_accumulator = CorpusFixture(
        name="x86_long_accumulator",
        code=bytes.fromhex(
            "b800000000 83c00a 83c014 d1e0 83e803 83c001".replace(" ", "")
        ),
        phase_labels=(
            "setup",
            "accumulator_update",
            "accumulator_update",
            "representation_transform",
            "accumulator_update",
            "exit",
        ),
    )

    # mov [rsp+0x20],7; mov eax,[rsp+0x20]; add eax,1
    # -> setup, accumulator update, exit.  The memory operands are real
    # Capstone/Unicorn evidence and are retained in the typed channel.
    memory = CorpusFixture(
        name="x86_memory",
        code=bytes.fromhex("c7 44 24 20 07 00 00 00 8b 44 24 20 83 c0 01"),
        phase_labels=("setup", "accumulator_update", "exit"),
    )

    # mov eax,1; cmp eax,3; jl +3; add eax,1; sub eax,0
    # -> setup, guard, guard, [unexecuted arithmetic], exit.  The branch is
    # taken, so static and bounded dynamic paths intentionally differ.
    guard_taken = CorpusFixture(
        name="x86_guard_taken",
        code=bytes.fromhex("b8 01 00 00 00 83 f8 03 7c 03 83 c0 01 83 e8 00"),
        phase_labels=(
            "setup",
            "guard_check",
            "guard_check",
            "accumulator_update",
            "exit",
        ),
    )

    # x86_call_return (MED-01 temporal expansion).  Real call/ret within the
    # bounded harness: the ``call`` at offset 6 pushes return address
    # base+0x100b, execution lands on the ``ret`` pad, and the ``ret`` pops
    # that in-span address so control returns to the ``pop``.  The ``jmp``
    # then transfers to base + len(code), the bounded emulator's stop
    # address, so the harness terminates cleanly.  Verified with real
    # Capstone decode and Unicorn execution: all six decoded instructions
    # execute and the stack pointer is restored.  Metadata note: the bounded
    # harness (emulation.emulate_x86_64) permits ``ret`` only when the popped
    # return address stays inside the mapped code window; a ``ret`` whose
    # target is not a valid in-span address is an explicit unsupported case,
    # never a silently dropped or mocked fixture.
    #
    # mov eax,1; push rax; call +3; pop rax; jmp +1; ret
    # -> setup, setup, guard_check, setup, guard_check, exit.  ``call`` and
    # ``ret`` carry the control-transfer latent phase (guard_check); the
    # ``pop`` reloads saved state (setup); the final ``ret`` is the exit.
    # Both the setup phase (load_immediate vs other categories) and the
    # guard_check phase (control_transfer here, comparison elsewhere) are
    # observationally ambiguous by design -- ambiguity is data.
    call_return = CorpusFixture(
        name="x86_call_return",
        code=bytes.fromhex("b801000000 50 e803000000 58 eb01 c3".replace(" ", "")),
        phase_labels=(
            "setup",
            "setup",
            "guard_check",
            "setup",
            "guard_check",
            "exit",
        ),
    )

    # x86_branch_not_taken (MED-01 temporal expansion).  Companion to
    # x86_guard: two guard phases whose conditional branches both fall
    # through.  eax=3 is neither > 5 nor < 2, so both ``jg`` and ``jl`` are
    # not taken and every statically decoded instruction executes; both jump
    # targets are in-span instruction boundaries (the second is the stop
    # address itself).
    #
    # mov eax,3; cmp eax,5; jg +3; add eax,1; cmp eax,2; jl +3; sub eax,1
    # -> setup, guard, guard, acc, guard, guard, exit
    branch_not_taken = CorpusFixture(
        name="x86_branch_not_taken",
        code=bytes.fromhex(
            "b803000000 83f805 7f03 83c001 83f802 7c03 83e801".replace(" ", "")
        ),
        phase_labels=(
            "setup",
            "guard_check",
            "guard_check",
            "accumulator_update",
            "guard_check",
            "guard_check",
            "exit",
        ),
    )

    # x86_stack_local (MED-01 temporal expansion).  Bounded stack-local
    # state: the ``push`` writes a scratch slot below rsp and the ``pop``
    # reads it back into a non-accumulator register, so accumulator
    # arithmetic interleaves with real stack memory effects.  The setup and
    # accumulator_update phases each recur in separate contiguous runs,
    # which makes the temporal model's state persistence observable.
    #
    # mov eax,7; push rax; mov ecx,3; add eax,3; sub eax,2; pop rcx;
    # shl eax,1; add eax,2; sub eax,8
    # -> setup x3, acc x2, setup, transform, acc, exit
    stack_local = CorpusFixture(
        name="x86_stack_local",
        code=bytes.fromhex(
            "b807000000 50 b903000000 83c003 83e802 59 d1e0 83c002 83e808".replace(
                " ", ""
            )
        ),
        phase_labels=(
            "setup",
            "setup",
            "setup",
            "accumulator_update",
            "accumulator_update",
            "setup",
            "representation_transform",
            "accumulator_update",
            "exit",
        ),
    )

    # x86_optimization_rewrite (MED-01 temporal expansion).  One program
    # holding the classic optimization-sensitive motif pair: ``add eax,1`` vs
    # ``inc eax`` and the ``lea eax,[rax+1]`` rewrite, in distinct phases.
    # ``add`` and ``inc`` share the arithmetic_update observation category
    # while differing in flag effects, and ``lea`` lands in the residual
    # ``other`` category even though its latent phase is accumulator_update
    # -- exactly the ambiguity-as-data regime MED-01 requires.
    #
    # mov eax,1; add eax,1; inc eax; lea eax,[rax+1]; shl eax,1;
    # add eax,1; inc eax; sub eax,5
    # -> setup, acc x3, transform, acc x2, exit
    optimization_rewrite = CorpusFixture(
        name="x86_optimization_rewrite",
        code=bytes.fromhex(
            "b801000000 83c001 ffc0 8d4001 d1e0 83c001 ffc0 83e805".replace(" ", "")
        ),
        phase_labels=(
            "setup",
            "accumulator_update",
            "accumulator_update",
            "accumulator_update",
            "representation_transform",
            "accumulator_update",
            "accumulator_update",
            "exit",
        ),
    )

    return (
        accumulator,
        bitwise,
        guard,
        long_accumulator,
        memory,
        guard_taken,
        call_return,
        branch_not_taken,
        stack_local,
        optimization_rewrite,
    )


def fixture_observation_ids(fixture: CorpusFixture) -> tuple[int, ...]:
    """Decode, emulate, and encode a fixture into observation-category ids.

    Args:
        fixture: The corpus fixture to encode.

    Returns:
        A tuple of integer observation ids (rows of the likelihood matrix),
        one per decoded instruction.
    """

    instructions = disassemble_bytes(fixture.code)
    trace = emulate_x86_64(fixture.code)
    observations = encode_observations(instructions, trace)
    return tuple(int(observation.observation_id) for observation in observations)


SPLITS = ("training", "development", "held_out")

# Declared fixture-level inventory, retained from protocol 2.1.0.
# Here family means fixture identity, not independent source lineage. Related
# authored variants remain dependent. This inventory does not control the
# current leave-one-fixture-out estimators: each fold excludes its scored
# fixture and trains on the complete complement, across inventory partitions.
# Do not describe the inventory as permanently held-out evaluation.
CORPUS_FAMILY_SPLIT: dict[str, str] = {
    "x86_accumulator": "training",
    "x86_bitwise": "training",
    "x86_call_return": "training",
    "x86_optimization_rewrite": "training",
    "x86_long_accumulator": "development",
    "x86_memory": "development",
    "x86_guard": "held_out",
    "x86_guard_taken": "held_out",
    "x86_branch_not_taken": "held_out",
    "x86_stack_local": "held_out",
}


def corpus_family_split() -> dict[str, str]:
    """Return the frozen fixture-to-split map (defensive copy)."""

    return dict(CORPUS_FAMILY_SPLIT)


def fixture_family(fixture_name: str) -> str:
    """Return the frozen family id for one fixture.

    The declared grouping is one family per fixture, so the family id is the
    fixture name itself.
    """

    if fixture_name not in CORPUS_FAMILY_SPLIT:
        raise KeyError(f"unknown fixture family: {fixture_name}")
    return fixture_name


def fixture_split(fixture_name: str) -> str:
    """Return the frozen split assignment for one fixture."""

    try:
        return CORPUS_FAMILY_SPLIT[fixture_name]
    except KeyError as exc:
        raise KeyError(f"unknown fixture split: {fixture_name}") from exc
