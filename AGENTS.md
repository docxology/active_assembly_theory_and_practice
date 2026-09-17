# AGENTS.md — Active Assembly Theory and Practice

This project follows the configured private sidecar conventions when that
sidecar is available, and the computational research shape of the sibling
`template/projects/templates/template_code_project/`.

## Boundaries

- Keep domain logic in `src/`.
- Keep `scripts/` as thin orchestrators that call `src/` and write artifacts.
- Use real Capstone, Unicorn, and pymdp APIs in tests; do not mock them.
- Use transparent toy byte fixtures only. Do not download, execute, or analyze
  arbitrary binaries in the canonical fixture demonstrator.
- Static memory operands may be joined to bounded Unicorn read/write hooks only
  on transparent fixtures; this is provenance instrumentation, not heap or
  alias analysis.
- The typed observation schema is label-blind and must preserve missing versus
  unseen values explicitly. Factorized and hierarchical scores are companion
  diagnostics to the collapsed categorical HMM; do not present them as learned
  semantics or as automatically superior models.
- The Gaussian RxInfer run remains the authoritative spine; the categorical
  phase-HMM is a namespaced, validated companion and its free-energy scale must
  not be compared with the Gaussian trajectory.
- Keep manuscript claims cautious: this is a demonstrator linking assembly
  observations to Active Inference/FEP, not a full reverse-engineering system.

## Verification

From the project checkout (`/Volumes/external_drive/Git/projects/working/active_assembly_theory_and_practice`):

```bash
uv run pytest tests \
  --cov=src \
  --cov-fail-under=90
```

From the configured sibling template checkout (`/Volumes/external_drive/Git/template`):

```bash
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_assembly_theory_and_practice \
  --skip-manuscript-hydration
```

The project-side manuscript generator honors `TEMPLATE_REPO_ROOT`; set it when
the template checkout is not the configured sibling. It fails if hydration
omits a source section or leaves a `{{TOKEN}}` unresolved.

Always render with `--skip-manuscript-hydration`; the README records the
hydration rule (project-side hydration is authoritative).
