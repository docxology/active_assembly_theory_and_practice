# Kernel-checked formal fragment

This directory contains the deliberately small formal-methods tranche of the
Active Assembly research program. It is a closed-world modular `Word64`
accumulator semantics, not a complete x86-64 model and not a replacement for
the real Capstone/Unicorn trace producers.

`ActiveAssembly/Semantics.lean` defines:

- 64-bit modular words and `mov`, `add`, `sub`, `inc`, and `nop` operations;
- finite-program execution by a small-step fold;
- universal observational equivalence over every formal initial word;
- equivalent witnesses for constant addition and neutral `nop` insertion;
- an explicit subtraction near miss; and
- a run-preservation theorem for appending `nop`.

The checked receipt is written to
`../output/data/formal_verification.json` by:

```bash
uv run python scripts/run_formal_verification.py
```

The command uses the pinned toolchain in `lean-toolchain` through `elan` and
runs `lake build`. A successful receipt means that the declared Lean project
compiled under the Lean kernel with the imported core/standard library. It does
not establish correspondence with Python transfer functions, Capstone
decoding, Unicorn execution, x86 flags, memory, exceptions, compiler
correctness, or all assembly programs. No external solver output is accepted
as a proof certificate.

The next formal milestone is a separately proved correspondence between this
small-step state and the bounded executable trace, with exhaustive bounded
state tests and explicit counterexamples for every unsupported effect. Until
that work is complete, the receipt should be cited as a local formalization
prerequisite rather than as a complete semantics theorem.
