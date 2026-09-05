# ActiveAssembly (Lean library)

The four Lean 4 source files of the `ActiveAssembly` library declared by
[`../lakefile.toml`](../lakefile.toml). Namespace: `ActiveAssembly`.

## Contents

| File | Lines | Contents |
| --- | --- | --- |
| `IR.lean` | 103 | Closed-world normalized instruction vocabulary and its decidable supported/unsupported boundary. |
| `NormalizedMachine.lean` | 89 | Bounded 16-register machine state, flags, memory bounds, and status. |
| `Semantics.lean` | 101 | Small-step `Word64` accumulator core, `run`, observational equivalence, and its theorems. |
| `GeneratedCorrespondence.lean` | 84 | Generated Python-to-Lean boundary fixture — do not hand edit. |

`ActiveAssembly.Semantics` is the Lake default target; it imports
`NormalizedMachine` and `IR`. `GeneratedCorrespondence` imports `Semantics` and
`NormalizedMachine`.

## Quick Start

```bash
# from the project root — builds via lake and writes the receipt
uv run python scripts/run_formal_verification.py
```

The receipt is written to `output/data/formal_verification.json`.

## Key Points

- `Semantics.lean` models one 64-bit accumulator with `mov`, `add`, `sub`,
  `inc`, and `nop`. That is the whole operational core; it is not x86-64.
- `NormalizedMachine.lean` is a schema for the promotion tranche's bounded
  register/memory interface, mirrored by the Python reference interpreter.
- `IR.lean` keeps `unsupported` an explicit constructor so an unmodelled effect
  can never silently become a no-op.
- `GeneratedCorrespondence.lean` is emitted by
  [`../../src/lean_bridge.py`](../../src/lean_bridge.py). Regenerate it; never
  edit it.

## More Information

See [AGENTS.md](AGENTS.md) for the declaration inventory and claim boundary,
and [`../README.md`](../README.md) for the fragment's scope.
