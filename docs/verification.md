# Verification

Run from the project checkout (`/Users/mini/Documents/GitHub/active_assembly_theory_and_practice`):

```bash
uv run pytest tests \
  --cov=src \
  --cov-fail-under=90
```

Run scripts:

```bash
uv run python scripts/run_assembly_inference.py

uv run python scripts/run_formal_verification.py

uv run python scripts/run_replication.py generate
uv run python scripts/run_replication.py verify

uv run python scripts/run_rxinfer_backend.py

uv run python scripts/validate_analysis_outputs.py

uv run python scripts/z_generate_manuscript_variables.py
```

Run the Julia integration gate:

```bash
uv run python scripts/run_julia_tests.py
```

Render from `/Users/mini/Documents/GitHub/template`:

```bash
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_assembly_theory_and_practice
```

Post-render sanity checks:

```bash
rg -n "\\{\\{[A-Z0-9_]+\\}\\}" \
  /Users/mini/Documents/GitHub/active_assembly_theory_and_practice/output/manuscript
```

Artifact freshness sanity check:

```bash
cd /Users/mini/Documents/GitHub/active_assembly_theory_and_practice
uv run python - <<'PY'
from src.reproducibility import verify_reproducibility_manifest

errors = verify_reproducibility_manifest()
print(errors)
raise SystemExit(bool(errors))
PY
```

PDF accessibility checks for the rendered manuscript:

```bash
uv run python scripts/validate_pdf_accessibility.py
uv run python scripts/validate_docx_render.py
pdfinfo output/pdf/active_assembly_theory_and_practice_combined.pdf | rg "Tagged|PDF version"
qpdf --check output/pdf/active_assembly_theory_and_practice_combined.pdf
verapdf --format text --flavour ua2 output/pdf/active_assembly_theory_and_practice_combined.pdf
```

The source `manuscript/config.yaml` opts into the template renderer's tagged
PDF path. The final artifact should report `Tagged: yes`, pass the VeraPDF
PDF/UA-2 profile, and pass `scripts/validate_pdf_accessibility.py`. The
project-side audit reads the actual `StructTreeRoot` and `ParentTree` and
checks the configured document language, marked structure, standard heading
role mappings and hierarchy, exact figure-registry alternatives, cover
alternative text, caption coverage, MathML/TeX associated files for formulas,
structured link annotations, and parity between source table environments and
PDF `/Table` tags. A visual pass in a PDF reader remains required for layout,
reading order, and figure legibility.

The accessibility report is a post-render gate. Run it only after the template
renderer has regenerated `output/pdf/`; a source edit or template-renderer edit
must be followed by fresh manuscript hydration, rendering, reporting, and a
strict provenance check. The report is written to
`output/data/pdf_accessibility_report.json` and is intentionally separate from
the self-referential reproducibility manifest.

The DOCX receipt is also a post-render gate. Run
`scripts/validate_docx_render.py` only after the sibling template has emitted
`output/docx/active_assembly_theory_and_practice_combined.docx`. It writes
`output/data/docx_render_receipt.json`, checks the actual OOXML package and
figure/caption parity, and fails if the document is stale relative to the
hydrated manuscript or substantive figure registry. The two validator-produced
summary figures are checked independently because including them in the
receipt digest would be self-referential. A passing DOCX receipt does not
certify PDF/UA tags or reading order.

## What the validation gate enforces

`scripts/validate_analysis_outputs.py` regenerates the full analysis surface and
then asserts the current generated oracle contract. The oracles span trace contracts, RxInfer
companion-artifact consistency, the scholarship implemented/comparison/roadmap/context
partition, claim-register closure, fixed-parameter generative adequacy, the SHA-256/byte-size
reproducibility manifest, manuscript-variable/registry closure, caption-ledger
and figure-registry closure, build provenance,
dependency-closed validation of `scientific_program.json`, and the real-API
equivalence/near-miss contract in `quotient_invariance_protocol.json`,
the source-family split and endpoint contract in
`representation_invariance_benchmark.json`,
and a **figure-set oracle** that requires the expected
`output/figures/*.png` set to exist, be non-empty, and pass a pixel-variance check
(so a blank or zero-byte PNG cannot satisfy the gate). The figure set includes the
temporal-belief heatmap, the decoder-comparison bars, the active-EFE trajectory,
    the held-out action-conditioned sensing comparison, the observation-only
    ordered-HMM parameter-learning comparison, and the source-family
representation-invariance benchmark with its negative control.
The current figure set also includes the real multi-architecture adapter
benchmark, the kernel-checked formal theorem manifest, and the exact finite-
horizon information frontier. Their data receipts are rebuilt and compared for
freshness; a hand-edited summary cannot clear the contract.
It also includes the source-family-disjoint quotient classifier portfolio, the
formal Capstone/Unicorn correspondence path, compiler-transfer capability
matrix, sensing-cost calibration, replication manifest, and theorem-or-
countertheorem outcome. Unavailable, unsupported, and underpowered cells remain
explicit rather than being converted to zero or silently removed.
The program, quotient, and representation-invariance receipts are
roadmap/protocol evidence only; they do not promote a local fixture result into
an identifiability theorem, a claim of cross-architecture semantic
generalization, or a measured information-per-cost law. The formal receipt is
only a closed Word64 proof fragment, and the multi-architecture receipt is only
one operation per architecture. The representation receipt additionally requires its
single-zero-state negative control to retain an unresolved near miss; this
guards the separation diagnostic against an intervention-catalog blind spot.

All deterministic gates (pytest with the 90% coverage gate, `ruff check`,
`ruff format --check`, `mypy`, and this validation oracle) can be run together
with `scripts/run_full_verification.sh`; add `--julia` to also run the Julia
backend `Pkg.test()`. The script prints a pass/fail table and exits non-zero if
any step fails.

The gate also enforces **toolchain drift** at two independent boundaries. The
bridge first parses the committed `Manifest.toml` and requires its declared
Julia and RxInfer entries to match `EXPECTED_JULIA_VERSION` and
`EXPECTED_RXINFER_VERSION`. It then resolves the executable, checks
`julia --version`, and validates the backend diagnostics after each producer
run. When the pinned version is installed, `resolve_julia_executable()` prefers
that exact binary, so an auto-advancing juliaup `release` channel cannot
silently change the backend under the reported numbers. A genuine version
change requires a deliberate pin bump, a regenerated manifest, and a fresh
backend run.

A generated caption ledger at `output/figures/README.md` documents every figure,
while `output/data/figure_registry.json` binds each visual to its generator,
inputs, statistics, claims, alt text, and machine QA. The template consumes the
generated `output/figures/figure_registry.json` boundary projection, and the
validation gate checks that its exact `alt` values match the canonical registry.
The variable registry and build receipt provide the corresponding
manuscript/render lineage.
`output/data/claim_register.md` ties each bounded claim to its artifacts and
checks.
