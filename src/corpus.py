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

    return (accumulator, bitwise, guard, long_accumulator)


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
