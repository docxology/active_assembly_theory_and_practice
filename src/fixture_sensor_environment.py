"""Action-gated native sensing of exactly two transparent owned fixtures."""

from __future__ import annotations

from .corpus import build_corpus
from .disassembly import disassemble_bytes
from .emulation import emulate_x86_64
from .features import categorize_instruction


class FixtureSensorEnvironment:
    """The Agent receives observations only; provenance is evaluator-only."""

    available_actions = (0, 1, 2)

    def __init__(self, fixture: str):
        if fixture not in ("x86_accumulator", "x86_bitwise"):
            raise ValueError("SENSOR_UNSUPPORTED_FIXTURE")
        self._fixture = next(f for f in build_corpus() if f.name == fixture)
        self.acquisition_log: list[str] = []

    def reset(self) -> tuple[int, int]:
        self.acquisition_log.clear()
        return (0, 0)

    def step(self, action: int) -> tuple[int, int]:
        if type(action) is not int or action not in self.available_actions:
            raise ValueError("SENSOR_INVALID_ACTION")
        if action == 0:
            return (0, 0)
        if len(self.acquisition_log) >= 100:
            raise ValueError("SENSOR_ACQUISITION_LIMIT")
        if action == 1:
            self.acquisition_log.append("static")
            first = disassemble_bytes(self._fixture.code[:5])
            category = categorize_instruction(first[0])
            return (1 if category == "load_immediate" else 4, 1)
        self.acquisition_log.append("dynamic")
        trace = emulate_x86_64(self._fixture.code, max_instructions=32)
        eax = trace.registers["rax"] & 0xFFFFFFFF
        return ({3: 2, 5: 3}.get(eax, 4), 2)

    def evaluation_provenance(self) -> dict[str, object]:
        """Call after decisions for receipts, never pass this to policy inference."""
        import hashlib

        return {
            "fixture": self._fixture.name,
            "fixture_sha256": hashlib.sha256(self._fixture.code).hexdigest(),
            "acquisition_log": list(self.acquisition_log),
        }
