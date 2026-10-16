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
- Canonical categorical models live in `configs/gnn/` and execute through
  the actual configured GNN parser and real pymdp Agent. Follow
  [the executable contract](docs/gnn_models.md) for source admission, tensor
  meaning, acquisition order and native parity. Gaussian, fitted, factorized
  and hierarchical diagnostics retain separate scales and limitations.
- Every inference adapter must consume an actual upstream-parsed numeric GNN
  binding. Keep the twelve profiles in `configs/gnn/inference/`, runtime source
  capsules and rendered catalog synchronized. A prototype is not runtime
  coverage; malformed distributions fail without repair. Preserve native
  categorical iteration traces and distinguish prefix evidence from convergence.
- Keep manuscript claims cautious: this is a demonstrator linking assembly
  observations to Active Inference/FEP, not a full reverse-engineering system.

## Verification

From the project checkout:

```bash
uv run pytest tests \
  --cov=src \
  --cov-fail-under=90
```

Set `GNN_ROOT` and `TEMPLATE_REPO_ROOT` to their canonical external checkouts.
Render this explicit project root with the prepared template runtime:

```bash
uv run python scripts/render_manuscript.py \
  --project-root "$PWD" \
  --skip-manuscript-hydration
```

The project-side manuscript generator honors `TEMPLATE_REPO_ROOT`; set it when
the template checkout is not the configured sibling. It fails if hydration
omits a source section or leaves a `{{TOKEN}}` unresolved.

Preserve shared template links; explicit-root rendering selects this checkout.
Always render with `--skip-manuscript-hydration`; the README records the
hydration rule (project-side hydration is authoritative).
