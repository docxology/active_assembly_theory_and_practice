# Active Assembly Theory and Practice

`active_assembly_theory_and_practice` is a private sidecar project that turns
assembly-language traces into categorical observations for Active Inference
experiments. The v0.1 scaffold uses real Capstone disassembly, Unicorn bounded
emulation, native Julia/RxInfer Bethe Free Energy tracking, and a pymdp-backed
comparison baseline on a transparent x86-64 byte fixture.

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
- `output/figures/validation_summary.png`
