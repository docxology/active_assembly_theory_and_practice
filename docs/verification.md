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

## What the validation gate enforces

`scripts/validate_analysis_outputs.py` regenerates the full analysis surface and
then asserts a 70-oracle contract. The oracles span trace contracts, RxInfer
companion-artifact consistency, the scholarship implemented/comparison/roadmap/context
partition, claim-register closure, RedTeam durability, the SHA-256/byte-size
reproducibility manifest, and a **figure-set oracle** that requires the expected
`output/figures/*.png` set to exist, be non-empty, and pass a pixel-variance check
(so a blank or zero-byte PNG cannot satisfy the gate). The figure set includes the
temporal-belief heatmap, the decoder-comparison bars, and the active-EFE
trajectory.

All deterministic gates (pytest with the 90% coverage gate, `ruff check`,
`ruff format --check`, `mypy`, and this validation oracle) can be run together
with `scripts/run_full_verification.sh`; add `--julia` to also run the Julia
backend `Pkg.test()`. The script prints a pass/fail table and exits non-zero if
any step fails.

The gate also enforces **toolchain drift**: the native Julia/RxInfer backend
asserts its version against the pinned `EXPECTED_JULIA_VERSION` in
`src/rxinfer_bridge.py`. When the pinned version is installed,
`resolve_julia_executable()` prefers that exact binary, so an auto-advancing
juliaup `release` channel cannot silently change the backend under the reported
numbers. A genuine version change requires a deliberate pin bump plus a fresh
backend run.

A generated caption ledger at `output/figures/README.md` documents every figure,
and `output/data/claim_register.md` ties each bounded claim to its artifacts and
checks.
