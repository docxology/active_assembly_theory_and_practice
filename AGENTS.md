# AGENTS.md — Active Assembly Theory and Practice

This project follows the private sidecar conventions in
`/Users/4d/Documents/GitHub/projects/AGENTS.md` and the computational research
shape of `../template/projects/templates/template_code_project/`.

## Boundaries

- Keep domain logic in `src/`.
- Keep `scripts/` as thin orchestrators that call `src/` and write artifacts.
- Use real Capstone, Unicorn, and pymdp APIs in tests; do not mock them.
- Use transparent toy byte fixtures only. Do not download, execute, or analyze
  arbitrary binaries in the v0.1 scaffold.
- Keep manuscript claims cautious: this is a demonstrator linking assembly
  observations to Active Inference/FEP, not a full reverse-engineering system.

## Verification

From `/Users/4d/Documents/GitHub/projects`:

```bash
uv run --project working/active_assembly_theory_and_practice \
  pytest working/active_assembly_theory_and_practice/tests \
  --cov=working/active_assembly_theory_and_practice/src \
  --cov-fail-under=90
```

From `/Users/4d/Documents/GitHub/template`:

```bash
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/03_render_pdf.py --project working/active_assembly_theory_and_practice
```

