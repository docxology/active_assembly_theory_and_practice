# src — Agent Guide

## Overview

`src/` is the project's only home for domain logic. Every producer that decodes
bytes, emulates them, fits a model, scores a protocol, writes a receipt, or
builds a figure lives here. [`../scripts`](../scripts/AGENTS.md) hold no logic:
they import from this package, call it, and write artifacts. Tests in
[`../tests`](../tests/AGENTS.md) import these modules directly.

The package contains 58 modules plus `__init__.py` and the
[`assembly_packages/`](assembly_packages/AGENTS.md) subpackage.

## Import boundary

- `src` never imports from `scripts` or `tests`.
- `src` never reads [`../data`](../data/AGENTS.md) or
  [`../figures`](../figures/AGENTS.md); those are committed snapshots. Producers
  write to `output/data/` and `output/figures/`.
- `src` reads pinned contracts from [`../configs`](../configs/AGENTS.md) and
  fails closed on a missing field, digest, or source.
- `src/__init__.py` maps public names to `(submodule, attribute)` pairs in
  `_LAZY_EXPORTS` and resolves them through `__getattr__`. Add a new public name
  there rather than adding a top-level eager import; eager imports would drag
  the optional Active-Inference/JAX stack into every consumer.

## Module map

### Observation layer

| Module | Purpose |
| --- | --- |
| `corpus.py` | The six transparent x86-64 byte fixtures (`x86_accumulator`, `x86_bitwise`, `x86_guard`, `x86_long_accumulator`, `x86_memory`, `x86_guard_taken`). |
| `disassembly.py` | Capstone-backed static decoding with register and memory-operand metadata. |
| `emulation.py` | Bounded Unicorn execution with read/write memory hooks. |
| `features.py` | Categorical observation features derived from instruction and trace events. |
| `symbolic_semantics.py` | Auditable transfer functions checked step-by-step against Unicorn snapshots. |
| `assembly_formalism.py` | Source-bound assembly listings for the manuscript and audit artifacts. |
| `multi_arch.py` | Real bounded Capstone/Unicorn fixtures for x86-64, AArch64, and RISC-V64. |
| `assembly_packages/` | Adapter layer, package provenance, and the stable JSON trace schema. |

### Inference layer

| Module | Purpose |
| --- | --- |
| `generative_model.py` | Discrete generative model for assembly observations. |
| `active_inference_bridge.py` | pymdp-backed variational updates. |
| `temporal_model.py` | Temporal HMM over latent program phases. |
| `phase_segmentation.py` | Contiguous phase segmentation of belief traces. |
| `active_selection.py` | Expected Free Energy and active observation selection. |
| `advanced_inference.py` | Composes the temporal, active, evaluation, and semantics layers. |
| `model_fitting.py` | Leakage-safe categorical HMM fitting, leave-one-fixture-out. |
| `parameter_learning.py` | Observation-only Baum–Welch parameter learning with fixed starts. |
| `factorized_model.py` | Factorized/hierarchical companion scores over typed evidence. |
| `typed_observations.py` | Typed, provenance-preserving observations (missing vs unseen kept distinct). |
| `action_sensing.py` | Bounded action-conditioned sensing over typed evidence. |
| `information_policy.py` | Exact finite-horizon information-per-cost dynamic program. |
| `sensor_costs.py` | Deterministic sensor-cost vectors and finite policy certificates. |
| `sensing_stress.py` | Active-sensing stress grid with exact DP references. |
| `rxinfer_bridge.py` | Python bridge to the native Julia package in [`../julia`](../julia/AGENTS.md); holds `EXPECTED_JULIA_VERSION` and `EXPECTED_RXINFER_VERSION`. |
| `experiments.py` | End-to-end observation and inference experiment. |

### Protocol and evaluation

| Module | Purpose |
| --- | --- |
| `evaluation.py` | Decoder metrics, negative controls, generative adequacy. |
| `corpus_coverage.py` | Source-bound coverage inventory for the fixture corpus. |
| `quotient_invariance.py` | Bounded operational-equivalence witnesses (P3). |
| `representation_invariance.py` | Family-level representation-invariance protocol. |
| `quotient_classifier.py` | Leakage-safe classifiers for formal-quotient membership. |
| `intervention_analysis.py` | Exact finite intervention-power enumeration. |
| `compiler_transfer.py` | Compiler/source/architecture transfer receipts, static-only decoding. |
| `replication.py` | Frozen local CLI/manifest replication bundle. |
| `replay_vm.py` | Lima VM orchestration for the label-blind replay surface; refuses any VM root outside `/Volumes/blue`. |
| `scientific_program.py` | Machine-readable long-horizon program and promotion gates. |
| `scientific_protocol.py` | Versioned protocol shared by the open-roadmap producers. |

### Formal drivers

`formal_verification.py` (runs `lake build`, holds `FORMAL_TOOLCHAIN`),
`formal_correspondence.py` (bounded Python/Capstone/Unicorn correspondence),
`lean_bridge.py` (generates `ActiveAssembly/GeneratedCorrespondence.lean`),
`counterexample_search.py` (smallest-first exhaustive search),
`theorem_phase.py` (fail-closed theorem/countertheorem ledger). See
[`../formal/AGENTS.md`](../formal/AGENTS.md).

### External sources

`external_sources.py` (pinned, static-only acquisition and validation),
`external_labeling.py` (writes `configs/external_label_contracts.yaml`),
`external_corpus.py` (source-bound contract validation). Acquisition is the
only network-enabled path and always requires `--allow-network`.

### Receipts and registries

`receipt.py` (shared metadata and the fail-closed status vocabulary),
`artifact_contract.py` (protocol-v2 closure over generated JSON),
`provenance.py`, `reproducibility.py` (hash/size manifest),
`build_provenance.py` (source/generated/hydrated/rendered lineage),
`claim_register.py`, `scholarship.py`, `validation.py` (the analysis
validation oracle).

### Figures and manuscript

`figures.py` and `advanced_figures.py` produce the figure set;
`figure_contract.py` declares canonical source-bound specs and the offline
atlas; `visual_atlas.py` produces the narrative atlas figures.
`manuscript_variables.py` and `manuscript_registry.py` hydrate the
`{{TOKEN}}` surface for [`../manuscript`](../manuscript/AGENTS.md);
`docx_validation.py` and `pdf_accessibility.py` audit the rendered artifacts
after the sibling template renders them.

## Boundaries to preserve

- Use real Capstone, Unicorn, and pymdp APIs. Do not mock them in tests.
- Transparent toy byte fixtures only: never download, execute, or analyze an
  arbitrary binary in the canonical fixture demonstrator.
- The Gaussian RxInfer run is the authoritative spine; the categorical
  phase-HMM is a namespaced companion and its free-energy scale must not be
  compared with the Gaussian trajectory.
- The typed observation schema is label-blind and must preserve missing versus
  unseen explicitly. Factorized and hierarchical scores are companion
  diagnostics, not learned semantics.
- `blocked_external`, `underpowered`, `unavailable`, `falsified`, and
  `unresolved` are first-class results. A producer must surface them, never
  round them up to a positive claim.

## Commands

```bash
# from the project root
uv run python scripts/run_assembly_inference.py
uv run python scripts/run_scientific_program.py all
uv run python scripts/validate_analysis_outputs.py
uv run pytest tests --cov=src --cov-fail-under=90
uv run ruff check src tests scripts
uv run mypy
```

## Best Practices

- Do keep new logic here and add only a thin entry point under `scripts/`.
- Do add a public name to `_LAZY_EXPORTS` rather than an eager import.
- Do write a protocol-v2 receipt for any new JSON artifact
  (`receipt.py`, `artifact_contract.py`).
- Do add the figure spec, caption, manuscript token, and manifest target in the
  same change when introducing a figure.
- Do not add a module that reads `../data/` or `../figures/`.

## See Also

- [`README.md`](README.md) — layer table and quick start
- [`assembly_packages/AGENTS.md`](assembly_packages/AGENTS.md) — adapter subpackage
- [`../docs/architecture.md`](../docs/architecture.md) — pipeline shape
- [`../docs/assembly_packages.md`](../docs/assembly_packages.md) — adapter contract
- [`../scripts/AGENTS.md`](../scripts/AGENTS.md) — the orchestrators that call this package
- [`../AGENTS.md`](../AGENTS.md) — project-level boundaries
