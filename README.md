# Active Assembly Theory and Practice

`active_assembly_theory_and_practice` is a private sidecar project that turns
assembly-language traces into categorical observations for Active Inference
experiments. It uses real Capstone disassembly, Unicorn bounded emulation, native
Julia/RxInfer Bethe Free Energy tracking, and a pymdp-backed comparison baseline
on a transparent x86-64 byte fixture. On top of that observation bridge it adds a
temporal hidden Markov model (explicit phase dynamics), an expected-free-energy
active-disassembly policy, and a labelled corpus evaluation — the v0.2 layer.

The project is intentionally narrow: it demonstrates an explicit bridge between
machine-code observations and a discrete generative model. It does not claim
general reverse-engineering automation or malware classification.

## Run

From `/Users/4d/Documents/GitHub/projects`:

```bash
uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/run_assembly_inference.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/run_rxinfer_backend.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/validate_analysis_outputs.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/z_generate_manuscript_variables.py

uv run --project working/active_assembly_theory_and_practice \
  pytest working/active_assembly_theory_and_practice/tests \
  --cov=working/active_assembly_theory_and_practice/src \
  --cov-fail-under=90

julia --project=working/active_assembly_theory_and_practice/julia/ActiveAssemblyRxInfer \
  -e 'using Pkg; Pkg.test()'
```

Render through the sibling template engine:

```bash
cd /Users/4d/Documents/GitHub/template
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/03_render_pdf.py --project working/active_assembly_theory_and_practice
```

## Outputs

- `output/data/disassembly_trace.json`
- `output/data/emulation_trace.json`
- `output/data/inference_result.json`
- `output/data/trace_bundle.json`
- `output/data/scholarship_sources.json`
- `output/data/claim_register.json`
- `output/data/redteam_report.json`
- `output/data/reproducibility_manifest.json`
- `output/data/validation_report.json`
- `output/data/manuscript_variables.json`
- `output/rxinfer/free_energy_trace.jsonl`
- `output/rxinfer/posteriors.json`
- `output/rxinfer/diagnostics.json`
- `output/rxinfer/model_config.json`
- `output/figures/observation_pipeline.png`
- `output/figures/posterior_beliefs.png`
- `output/figures/package_provenance.png`
- `output/figures/trace_feature_matrix.png`
- `output/figures/scholarship_source_roles.png`
- `output/figures/rxinfer_free_energy.png`
- `output/figures/inference_diagnostics.png`
- `output/figures/redteam_risk_register.png`
- `output/figures/scholarship_evidence_matrix.png`
- `output/figures/claim_evidence_map.png`
- `output/figures/artifact_hash_manifest.png`
- `output/figures/temporal_beliefs_x86_accumulator.png`
- `output/figures/model_comparison.png`
- `output/figures/active_efe.png`
- `output/figures/validation_summary.png`
- `output/figures/README.md` (generated caption ledger for every figure)

## Reproducibility notes

- Every figure is regenerated from project artifacts before rendering, and
  `output/figures/README.md` records a caption for each one.
- The native Julia/RxInfer backend checks its toolchain version against the
  pinned `EXPECTED_JULIA_VERSION` in `src/rxinfer_bridge.py`. The generic
  `julia` launcher can auto-advance on juliaup's `release` channel; when the
  pinned version is installed, `resolve_julia_executable()` prefers that exact
  binary so regeneration stays reproducible. Bumping the pin requires a
  deliberate change plus a fresh backend run.
- The validation gate (`scripts/validate_analysis_outputs.py`) recomputes each
  tracked artifact's SHA-256 and byte size and asserts the full 52-check surface.
The pinned Julia/RxInfer toolchain (see `EXPECTED_JULIA_VERSION` /
`EXPECTED_RXINFER_VERSION` in `src/rxinfer_bridge.py`) must be present; the gate
fails loudly on toolchain drift, so a fresh clone should install the pinned
Julia first (e.g. `juliaup add 1.12.6`).
