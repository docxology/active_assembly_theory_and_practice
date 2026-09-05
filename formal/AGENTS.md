# formal — Agent Guide

## Overview

`formal/` is the Lean 4 project holding the deliberately small kernel-checked
fragment of the research program: a closed-world, bounded `Word64` machine with
modular arithmetic, a normalized instruction IR, and a generated
Python-to-Lean boundary fixture. It exists so that the quotient/equivalence
witnesses used by the local protocol have an independently checked algebra
behind them.

It is not a complete x86-64 semantics and not a replacement for the real
Capstone/Unicorn trace producers in [`../src`](../src).

## File Organization

```
formal/
├── AGENTS.md                          this file
├── README.md                          scope and claim boundary of the fragment
├── ActiveAssembly/                    the Lean library (see its AGENTS.md)
├── lake-manifest.json                 Lake manifest; packages list is empty
├── lakefile.toml                      package `active_assembly_formal`, lib `ActiveAssembly`
└── lean-toolchain                     leanprover/lean4:v4.28.0
```

`.lake/` is a build directory and is git-ignored.

## Build and receipt

The toolchain is pinned by `lean-toolchain` (`leanprover/lean4:v4.28.0`) and
resolved through `elan`. `lakefile.toml` declares the package
`active_assembly_formal`, the `ActiveAssembly` Lean library, and
`defaultTargets = ["ActiveAssembly.Semantics"]`. `lake-manifest.json` has an
empty `packages` list: the fragment depends on the Lean core and standard
library only, with no external Lean dependency and no external solver.

The Python side runs `lake build` and writes the receipt:

```bash
# from the project root
uv run python scripts/run_formal_verification.py
```

The receipt lands at `output/data/formal_verification.json`. The driver is
[`../src/formal_verification.py`](../src/formal_verification.py); the pinned
toolchain constant is `FORMAL_TOOLCHAIN` in that module.

Related producers:

- [`../src/lean_bridge.py`](../src/lean_bridge.py) generates
  `ActiveAssembly/GeneratedCorrespondence.lean`.
- [`../src/counterexample_search.py`](../src/counterexample_search.py) runs the
  smallest-first exhaustive counterexample search over declared candidate
  theorems.
- [`../src/formal_correspondence.py`](../src/formal_correspondence.py) checks
  the bounded Python/Capstone/Unicorn correspondence for the declared subset.
- [`../src/theorem_phase.py`](../src/theorem_phase.py) writes the fail-closed
  theorem/countertheorem ledger.

```bash
# from the project root
uv run python scripts/run_scientific_program.py formal
uv run python scripts/run_scientific_program.py counterexamples
uv run python scripts/run_scientific_program.py theorem
uv run pytest tests/test_formal_verification.py tests/test_formal_boundaries.py \
  tests/test_formal_state_v2.py tests/test_lean_bridge.py -q
```

## Claim boundary

A successful receipt means the declared Lean project compiled under the Lean
kernel with the imported core/standard library. It does **not** establish
correspondence with the Python transfer functions, Capstone decoding, Unicorn
execution, x86 flags, memory, exceptions, compiler correctness, or all assembly
programs. No external solver output is accepted as a proof certificate.

## Best Practices

- Do keep the fragment small and the boundary stated in prose next to each new
  declaration.
- Do regenerate `GeneratedCorrespondence.lean` from `src/lean_bridge.py`; it
  carries a do-not-hand-edit marker.
- Do bump `lean-toolchain` only as a deliberate change with a fresh receipt.
- Do not add an external Lean dependency without updating `lake-manifest.json`
  and the claim boundary in [`README.md`](README.md).

## See Also

- [`README.md`](README.md) — the fragment's scope and stated limits
- [`ActiveAssembly/AGENTS.md`](ActiveAssembly/AGENTS.md) — per-file Lean guide
- [`../docs/assembly_packages.md`](../docs/assembly_packages.md) — the pinned formal fragment in context
- [`../docs/scientific_program.md`](../docs/scientific_program.md) — theorem and counterexample gates
