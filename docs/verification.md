# Verification

Run from `/Users/4d/Documents/GitHub/projects`:

```bash
uv run --project working/active_assembly_theory_and_practice \
  pytest working/active_assembly_theory_and_practice/tests \
  --cov=working/active_assembly_theory_and_practice/src \
  --cov-fail-under=90
```

Run scripts:

```bash
uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/run_assembly_inference.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/run_rxinfer_backend.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/validate_analysis_outputs.py

uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/z_generate_manuscript_variables.py
```

Run the Julia integration gate:

```bash
julia --project=working/active_assembly_theory_and_practice/julia/ActiveAssemblyRxInfer \
  -e 'using Pkg; Pkg.test()'
```

Render from `/Users/4d/Documents/GitHub/template`:

```bash
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/03_render_pdf.py --project working/active_assembly_theory_and_practice
```

Post-render sanity checks:

```bash
rg -n "\\{\\{[A-Z0-9_]+\\}\\}" \
  /Users/4d/Documents/GitHub/projects/working/active_assembly_theory_and_practice/output/manuscript \
  /Users/4d/Documents/GitHub/projects/working/active_assembly_theory_and_practice/output/web
```

Artifact freshness sanity check:

```bash
cd /Users/4d/Documents/GitHub/projects/working/active_assembly_theory_and_practice
uv run python - <<'PY'
from src.reproducibility import verify_reproducibility_manifest

errors = verify_reproducibility_manifest()
print(errors)
raise SystemExit(bool(errors))
PY
```
