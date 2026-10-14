# Verification

Run from the project checkout:

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

uv run python scripts/run_scientific_program.py all
uv run python scripts/run_scientific_program.py audit-artifacts

uv run python scripts/run_scientific_program.py probe-capabilities
uv run python scripts/run_scientific_program.py prepare-external
uv run python scripts/run_scientific_program.py validate-external
uv run python scripts/run_scientific_program.py audit-external-preparation
uv run python scripts/run_scientific_program.py generate-public
uv run python scripts/run_scientific_program.py replay --offline
uv run python scripts/run_scientific_program.py verify-public
uv run python scripts/run_scientific_program.py promote-evidence

uv run python scripts/run_scientific_program.py generate-figures \
  --formats png svg pdf atlas
uv run python scripts/run_scientific_program.py audit-figures
uv run python scripts/run_scientific_program.py render-manuscript

uv run python scripts/run_rxinfer_backend.py

uv run python scripts/validate_analysis_outputs.py

uv run python scripts/z_generate_manuscript_variables.py
```

The only network-enabled operation is explicit acquisition of the pinned
manifest:

```bash
uv run python scripts/run_scientific_program.py acquire-external \
  --manifest configs/external_sources.yaml \
  --destination output/external_sources \
  --allow-network
```

Acquisition validates archive and license digests, performs static compilation
and disassembly only, and retains unavailable/failed rows. Replay, scoring,
manuscript generation, and document rendering must remain offline. A local
rerun cannot produce an independent-replay status; an external operator must
submit a pending receipt with explicit separate-control attestation for the
protected ingestion command.
`scripts/run_replication.py score --predictions <file>` is the protected
scoring mode: it validates the replication bundle and then scores a
label-blind prediction file against the sealed gold labels, which remain
private to this ingestion boundary.

The active research protocol is `output/data/research_protocol.json` at
schema 2.2.0 (`active-assembly-open-roadmap-v2-2026-10-13`), which freezes the
temporal-corpus split. All protocol-bound receipts rebind on regeneration: a
protocol revision deliberately invalidates previously generated receipts
rather than being silently absorbed by them.

Run the Julia integration gate:

```bash
uv run python scripts/run_julia_tests.py
```

Render from the configured `TEMPLATE_REPO_ROOT` checkout:

```bash
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_assembly_theory_and_practice \
  --skip-manuscript-hydration
```

Hydration is intentionally a two-stage boundary. Run the project-side
`z_generate_manuscript_variables.py` command first; it is the authoritative
producer and validates the generated source tree. Then pass
`--skip-manuscript-hydration` to the sibling template renderer so it consumes
that validated tree without silently regenerating it under a different
timeout. Any source, protocol, or generator change requires fresh project-side
hydration before rendering.

The figure gates require all 68 substantive specifications to have unique
labels, current input hashes, nonblank PNGs, parseable vector companions, and
complete caption/alternative/long-description metadata. The offline atlas must
contain one semantic figure card per registry row and no remote dependency.
Unavailable, unsupported, excluded, and not-observed cells must remain textual
statuses rather than zero-valued measurements. A passing atlas gate is a
source-bound document property, not a promotion of the scientific claim shown
by a figure.

The current template DOCX exporter omits the configured paper cover. Complete
that edition through the same native Pandoc renderer before its integrity gate:

```bash
uv run python scripts/render_docx_cover.py
uv run python scripts/validate_docx_render.py
```

The project-owned final filter adds the source-configured raster and alternative
after scientific figure numbering. It also resolves scientific image references to the canonical output folder, so committed documentation snapshots cannot shadow them through Pandoc resource paths. It preserves the shared manuscript and its
citations; a failed conversion preserves the previous DOCX.


The DOCX 1.1 integrity lane also compares embedded media bytes with all 66
substantive canonical PNGs and binds the active Python cover adapter and Lua
filter. The two validator-summary figures remain outside this digest comparison
to avoid a self-referential receipt; their count, caption, alternative and
nonblankness checks remain active. The DOCX core metadata stores its hydrated-source SHA-256 at conversion time;
receipt writing rejects stale text instead of rebinding it to changed sources.
Counts alone cannot certify figure freshness. The adapter also requires the
canonical composition receipt to match every current numbered hydrated section
and the combined Markdown bytes, rejecting stale text before conversion.

After document export, run `uv run python scripts/validate_analysis_outputs.py --existing-only` with the pre-render environment unset. This validates the existing scientific inputs and images; the default command regenerates them and belongs before rendering. Both modes refresh the build receipt and inspect the actual outputs.

Post-render sanity checks:

```bash
rg -n "\\{\\{[A-Z0-9_]+\\}\\}" \
  output/manuscript
```

Artifact freshness sanity check:

```bash
# From the project checkout
uv run python - <<'PY'
from src.reproducibility import verify_reproducibility_manifest

errors = verify_reproducibility_manifest()
print(errors)
raise SystemExit(bool(errors))
PY
```

Claim-map drift guard:

```bash
uv run python scripts/check_claim_map.py
```

The guard parses the `CLM-xxx` identifier column of `docs/claim_map.md` and
asserts set equality (in both directions) with the claim IDs in
`output/data/claim_register.json`; a claim added, removed, renamed, or
duplicated on either surface exits 1. It is a documentation-closure check on
top of the claim-register closure oracle, not a scientific validity gate.

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

The abstract's Pandoc metadata declares PDF 2.0, tagging and the UA-2 target
before the document class. The source preamble requests display of the document
title. These declarations do not prove compliance: qpdf, the structural audit,
VeraPDF and visual inspection remain required. The native LuaLaTeX environment
also needs `seqsplit` for breakable source paths and `selnolig` for Pandoc's
generated preamble. See the [LaTeX tagging instructions](https://tagging-project.latex-project.org/documentation/usage-instructions).

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

The DOCX lane additionally expects the configured cover asset to be embedded
beside the substantive figures (media count = registry figures + 1). That
embedding is completed by `scripts/render_docx_cover.py`, which uses the
sibling template's native renderer with the project-owned cover filter. Missing
media or alternative text remains a diagnosed receipt failure. The PDF and EPUB lanes embed the cover
independent of this parity gate.

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
`representation_invariance_benchmark.json`, the exact finite
intervention-subset enumeration in `intervention_power.json` (including its
one-state negative control and fail-closed promotion status),
the protocol-v2 receipt envelope on every project-owned JSON artifact,
and a **figure-set oracle** that requires the expected
`output/figures/*.png` set to exist, be non-empty, and pass a pixel-variance check
(so a blank or zero-byte PNG cannot satisfy the gate). The figure set includes the
temporal-belief heatmap, the decoder-comparison bars, the active-EFE trajectory,
    the held-out action-conditioned sensing comparison, the observation-only
    ordered-HMM parameter-learning comparison, the source-family
representation-invariance benchmark with its negative control, and the exact
intervention-power figure. The parameter-learning receipt additionally checks
all declared multi-start rows, observation-only fit contracts, seed ledger,
held-out score summaries, and pairwise predicted-sequence agreement; weak or
unstable starts remain visible.
The current figure set also includes the real multi-architecture adapter
benchmark, the kernel-checked formal theorem manifest, and the exact finite-
horizon information frontier. Their data receipts are rebuilt and compared for
freshness; a hand-edited summary cannot clear the contract.
It also includes the source-family-disjoint quotient classifier portfolio, the
formal Capstone/Unicorn correspondence path, compiler-transfer capability
matrix, sensing-cost calibration, replication manifest, and theorem-or-
countertheorem outcome, the smallest-first counterexample-search receipt
(freshness-recomputed with re-verified witnesses), and the active-sensing
stress receipt (exact reference cross-checks, seeded rollouts, and repeated
calibration-stability rows). Unavailable, unsupported, and underpowered cells remain
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

## Fresh-checkout verification order

The full gate installs the locked development extra and regenerates canonical
analysis inputs before running generated-artifact integration tests. It stops
at the first failed prerequisite and prints the end of the failing log. Use
`--analysis-only` for the explicit pre-render boundary; that option does not
certify a PDF or DOCX. The default retains the rendered-document gates and
requires the sibling template render first. `--julia` runs the real native
package tests with `allow_reresolve=false`, preserving the committed Julia
manifest rather than selecting newer dependencies during test setup.

Set `JULIA` to an installed Julia 1.12.7 executable when the generic launcher
selects another version. If a stale General registry cannot resolve the pinned
RxInfer version, refresh the registry and retry with the original manifest.
Do not replace a missing pin with a different version or a synthetic receipt.
