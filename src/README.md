# src

The Python package holding all domain logic for the project. Everything that
decodes, emulates, models, evaluates, formalizes, or receipts lives here;
[`../scripts`](../scripts) are thin orchestrators that import from this package
and write artifacts to the untracked `output/` tree.

The package includes the adapter subpackage,
[`assembly_packages/`](assembly_packages/README.md).

## Layers

| Layer | Modules | Role |
| --- | --- | --- |
| Observation | `corpus`, `disassembly`, `emulation`, `features`, `symbolic_semantics`, `assembly_formalism`, `multi_arch`, `assembly_packages/` | Real Capstone decoding and bounded Unicorn execution over transparent byte fixtures. |
| Inference | `generative_model`, `active_inference_bridge`, `temporal_model`, `phase_segmentation`, `active_selection`, `advanced_inference`, `model_fitting`, `parameter_learning`, `factorized_model`, `typed_observations`, `action_sensing`, `information_policy`, `sensor_costs`, `sensing_stress`, `rxinfer_bridge`, `experiments` | Discrete generative models, the temporal HMM, EFE selection, and the native Julia/RxInfer bridge. |
| Protocol & evaluation | `evaluation`, `corpus_coverage`, `quotient_invariance`, `representation_invariance`, `quotient_classifier`, `intervention_analysis`, `compiler_transfer`, `replication`, `replay_vm`, `scientific_program`, `scientific_protocol` | Decoder metrics, negative controls, the P3 invariance benchmark, and the promotion-gate ledger. |
| Formal | `formal_verification`, `formal_correspondence`, `lean_bridge`, `counterexample_search`, `theorem_phase` | Drivers for the Lean fragment in [`../formal`](../formal/README.md). |
| External sources | `external_sources`, `external_labeling`, `external_corpus` | Pinned, static-only licensed-source acquisition and label contracts. |
| Receipts & registries | `receipt`, `artifact_contract`, `provenance`, `reproducibility`, `build_provenance`, `claim_register`, `scholarship`, `docs_validation`, `validation` | Protocol-v2 artifact envelopes, hash manifests, and the validation oracle. |
| Figures | `figures`, `advanced_figures`, `figure_contract`, `visual_atlas` | Source-bound figure specifications, the offline atlas, and the figure registry. |
| Manuscript & delivery | `manuscript_variables`, `manuscript_registry`, `document_rendering`, `docx_validation`, `pdf_accessibility`, `pdf_ua` | Token hydration for [`../manuscript`](../manuscript) and post-render DOCX/PDF audits. |

## Quick Start

```bash
# from the project root
uv run python scripts/run_assembly_inference.py     # end-to-end experiment
uv run python scripts/run_scientific_program.py all # canonical orchestration
uv run pytest tests --cov=src --cov-fail-under=90
```

## Key Points

- `src/__init__.py` exposes a lazy convenience API: names are resolved on first
  attribute access so importing a focused module never pulls in the optional
  Active-Inference/JAX stack.
- Modules write into `output/`, never into [`../data`](../data/README.md) or
  [`../figures`](../figures/README.md), which are committed snapshots only.
- Coverage is gated at 90% by `pyproject.toml`; `mypy` runs over `src` only.
- Capstone, Unicorn, and pymdp are used for real in tests. Do not mock them.

## More Information

See [AGENTS.md](AGENTS.md) for the module-by-module map, dependency direction,
and editing rules; [`../docs/architecture.md`](../docs/architecture.md) for the
pipeline shape.
