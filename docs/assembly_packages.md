# Assembly Package Adapter Contract

The project now treats assembly-analysis packages as evidence adapters. Each
adapter must write structured rows into the trace bundle and must attach package
provenance instead of only returning informal Python objects.

## Required Adapters

- `capstone_static_disassembly`: decodes fixture bytes into static instruction
  events with address, size, mnemonic, operands, byte span, groups, and
  Capstone-derived register read/write sets and memory-operand metadata.
- `unicorn_bounded_emulation`: executes the same fixture under a bounded
  instruction count and records dynamic register evidence plus read/write
  memory-hook events when the fixture accesses mapped memory.

## Implemented bounded semantics

The package also includes two deliberately closed-world extensions that do not
depend on optional reverse-engineering suites:

- `src/assembly_packages/schema.py` partitions decoded direct-branch fixtures
  into basic blocks and emits taken/fall-through CFG edges. Targets outside the
  decoded byte span are represented as `external_0x...` edges rather than
  receiving fabricated blocks.
- `src/symbolic_semantics.py` applies auditable transfer functions to immediate
  `EAX`/`RAX` moves, arithmetic/bitwise updates, shifts, comparisons, and the
  supported conditional branches. The concrete pre/post `RAX` state is checked
  at every executed step against independent Unicorn snapshots, and only the
  explicitly modeled `EFLAGS` bits are compared. Unsupported executed
  instructions are counted and fail the semantics validation contract; they are
  never silently assigned a guessed meaning. The result is trace refinement for
  the declared fixture subset, not complete x86-64 semantics. Seven corpus
  fixtures pass this interpreter; call/return, stack-local and optimization
  rewrite fixtures explicitly fail because their instructions exceed its
  subset. Broader formal correspondence is checked in a separate lane.

## Optional Adapter Slots

Optional adapters are probed and recorded even when unavailable. They are
context-only extension candidates, not current dependencies or unstarted
implementation obligations:

- `angr_cfg_symbolic_facts`: CFG and symbolic-fact comparison candidate.
- `rizin_static_baseline` and `radare2_static_baseline`: static baseline
  comparison candidates.
- `llvm_objdump_static_baseline`: optional objdump-compatible baseline; the
  current compiler-transfer producer uses the locally available `objdump`
  command for static object inspection and records its capability receipt.
- `keystone_fixture_assembly` and `nasm_fixture_assembly`: assembly-source to
  fixture-byte generation candidates.
- `ghidra_headless_analysis`: full-suite reverse-engineering baseline candidate.
- `bap_ir_analysis` and `gtirb_interchange`: IR-centered trace-export
  candidates.
- `qiling_os_emulation`: higher-level emulation candidate above Unicorn.
- `miasm_python_ir`, `dyninst_instrumentation`, `retdec_decompiler`, and
  `mcsema_lifting`: richer static, dynamic, decompilation, and lifting
  observation candidates.

These packages are not required for the current fixture gate. They must enter as
optional extras or command probes with explicit unavailable status, not as
hidden runtime requirements. If one is selected for implementation, scope it
as a new bounded tranche or under `MAJ-05` in [the roadmap](roadmap.md) and add a
source-bound receipt before referring to it as an evidence adapter.

## Trace Bundle Fields

`output/data/trace_bundle.json` is the stable interchange schema for downstream
inference:

- `schema_version`
- `sample_id`
- `architecture`
- `base_address`
- `binary_sha256`
- `static_instruction_events`
- `basic_blocks`
- `cfg_edges`
- `dynamic_register_trace`
- `memory_events`
- `symbolic_facts`
- `package_provenance`
- `observations`
- `typed_observations`
- `observation_labels`
- `hidden_state_labels`

The `x86_accumulator` fixture has one linear basic block; the branch-bearing
guard fixtures demonstrate multiple static blocks, an external target, and a
taken path that omits one statically decoded instruction. The labelled corpus
now includes `x86_memory`, whose two static memory operands and real store/read
pair are retained in `output/data/memory_evidence.json`; its figure is
`output/figures/assembly_memory_timeline.png`. Memory events in the accumulator
bundle remain empty because that fixture does not access mapped data memory.
`output/data/corpus_coverage.json` records the per-fixture static/dynamic path
and evidence-regime inventory. `symbolic_facts` contains the bounded
transfer-function rows for its declared register/branch subset. The complete
comparison report is emitted as
`output/data/symbolic_execution.json`. The dynamic register trace additionally
contains one `instruction` row per executed step with pre/post `RAX`, `RIP`,
`RSP`, and `EFLAGS`; its `instruction_index` is execution order, not a static
instruction index when a branch skips decoded code. Future adapters should
extend these channels without changing the meaning of existing keys.

`output/figures/package_provenance.png` visualizes the required and optional
adapter states for the rendered manuscript.

`output/figures/trace_feature_matrix.png` visualizes the categorical observation
contract: executed status, observation ID, immediate operand cue, accumulator
operand cue, and category channel for every decoded instruction. The validator
checks the corresponding machine-readable contract in `trace_bundle.json`.

`output/data/typed_observations.json` is the separate Phase A schema projection.
It retains instruction family, operand class, register read/write sets, flag
effect, memory-addressing mode, branch-target class, and bounded dynamic status
without phase labels. `output/data/factorized_observation_model.json` fits and
evaluates field-wise and group-balanced composite scores with fixture-level
holdouts. Missing fields map to `__MISSING__` and unseen training values to
`__UNKNOWN__`; neither token is silently treated as zero. The typed schema is a
companion to the collapsed categorical channel and does not claim complete ISA
semantics.

## Formal and multi-architecture expansion

`formal/ActiveAssembly/Semantics.lean` is a deliberately closed-world Lean
fragment over modular `Word64` state. It defines `mov`, `add`, `sub`, `inc`, and
`nop`, a finite-program run relation, universal observational equivalence over
all formal initial words, and explicit equivalent/near-miss theorems. The
generated `formal/ActiveAssembly/GeneratedCorrespondence.lean` adds a
source-bound fixture with all 16 architectural general-purpose registers,
`RIP`, modeled flag-definedness, bounded memory and stack intervals, a step
counter, and an explicit running status. Its checked invariants cover the
64-bit `RAX` increment, preservation of the other registers and `CF`, step
advancement, and bound preservation. The receipt
`output/data/formal_verification.json` is generated by
`scripts/run_formal_verification.py`; it records the pinned Lean toolchain,
source hashes, theorem manifest, and trusted base. A locally successful build
proves only the declared Lean definitions and generated fixture theorems under
the Lean kernel. It does not yet prove the Python transfer functions,
Capstone decoding, Unicorn behavior, all x86 flags, memory effects, compiler
correctness, or complete assembly semantics. The formal fragment is therefore
a P4/P11 research tranche and a quotient-witness foundation, not a complete
semantics backend.

`src/multi_arch.py` provides three transparent real-adapter fixtures. Capstone
decodes x86-64, AArch64, and RISC-V64 byte sequences; Unicorn executes exactly
two non-returning instructions; and the source-bound receipt
`output/data/cross_architecture_fixtures.json` records the common numeric
outcome plus the explicit API provenance. This is an adapter regression and
falsification surface. It deliberately does not claim cross-ISA semantic
equivalence, compiler invariance, ABI equivalence, exception equivalence,
memory equivalence, or external validity. The P5 local tranche now adds
source-family-disjoint portable C families, independent target probes, exact
compiler/object/disassembly hashes, and explicit unavailable rows in
`output/data/compiler_transfer.json`. It remains static-only and does not
execute emitted objects. A portable semantic relation, multiple compiler and
optimization conditions, licensed real-program labels, and independently
audited external provenance remain required before a cross-compiler or
application-facing transfer claim is made.

`output/data/active_information_frontier.json` is a separate exact finite-
horizon policy receipt. It projects label-blind typed fields into finite sensor
channels and solves the declared sensor/outcome tree by memoized dynamic
programming. Its costs are model-derived proxy units, so it is a mechanism
analysis rather than a runtime cost law. The artifact and
`output/figures/active_information_frontier.png` must not be described as a
deployment policy, an infinite-horizon optimum, or a theorem about active
binary analysis.
