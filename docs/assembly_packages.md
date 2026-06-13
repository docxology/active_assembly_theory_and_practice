# Assembly Package Adapter Contract

The project now treats assembly-analysis packages as evidence adapters. Each
adapter must write structured rows into the trace bundle and must attach package
provenance instead of only returning informal Python objects.

## Required Adapters

- `capstone_static_disassembly`: decodes fixture bytes into static instruction
  events with address, size, mnemonic, operands, byte span, and groups.
- `unicorn_bounded_emulation`: executes the same fixture under a bounded
  instruction count and records dynamic register evidence.

## Optional Adapter Slots

Optional adapters are probed and recorded even when unavailable:

- `angr_cfg_symbolic_facts`: planned CFG and symbolic facts.
- `rizin_static_baseline` and `radare2_static_baseline`: planned static
  baseline comparisons.
- `llvm_objdump_static_baseline`: planned objdump-compatible static baseline.
- `keystone_fixture_assembly` and `nasm_fixture_assembly`: planned assembly
  source to fixture-byte generation.
- `ghidra_headless_analysis`: planned full-suite reverse-engineering baseline.
- `bap_ir_analysis` and `gtirb_interchange`: planned IR-centered trace export
  routes.
- `qiling_os_emulation`: planned higher-level emulation route above Unicorn.
- `miasm_python_ir`, `dyninst_instrumentation`, `retdec_decompiler`, and
  `mcsema_lifting`: planned richer static, dynamic, decompilation, and lifting
  observation channels.

These packages are not required for the v0.1 gate. They must enter as optional
extras or command probes with explicit unavailable status, not as hidden runtime
requirements.

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
- `observation_labels`
- `hidden_state_labels`

The v0.1 fixture has one linear basic block, no memory events, and no symbolic
facts. Future adapters should fill those channels without changing the meaning
of existing keys.

`output/figures/package_provenance.png` visualizes the required and optional
adapter states for the rendered manuscript.

`output/figures/trace_feature_matrix.png` visualizes the categorical observation
contract: executed status, observation ID, immediate operand cue, accumulator
operand cue, and category channel for every decoded instruction. The validator
checks the corresponding machine-readable contract in `trace_bundle.json`.
