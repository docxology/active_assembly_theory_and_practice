# Verification

Choose the boundary that matches the task. [Getting started](getting_started.md)
records environments and template registration; [document rendering](document_rendering.md)
owns the edition workflow; [analysis validation](analysis_validation.md)
owns the oracle. [Statistics](statistics.md) defines the limits of measured
results. A passing local gate is not independent or external scientific evidence.

## Canonical modes

Configure `GNN_ROOT` first; see [executable GNN models](gnn_models.md).
The workflow executes upstream parsing, real Agent controls and shared-tensor
native smoothing before the diagnostic producers. Its final GNN audit is
read-only. Stage logs, including failures, are retained under
`output/verification/`.

Run from the project checkout:

```bash
# Regenerate, test, hydrate, export and strictly audit the new editions.
scripts/run_full_verification.sh --julia

# Regenerate and test up to authoritative hydration; no document certification.
scripts/run_full_verification.sh --analysis-only --julia

# Audit an already exported template edition without refitting or rehydrating.
scripts/run_full_verification.sh --post-render-only
```

`--julia` adds the pinned native Julia package tests to a producer build.
`--analysis-only` and `--post-render-only` are mutually exclusive; the latter
also rejects `--julia`. The full mode invokes the native template rendering APIs
between hydration and the DOCX projection, so changed scientific inputs cannot
be compared against an obsolete combined manuscript. Each failed prerequisite
stops the run, prints the useful log tail and exits nonzero.

The producer modes force `ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=1` while inputs are
being regenerated. Full and post-render modes force it to `0` before final
document validation. Inherited pre-render settings cannot silently defer final
DOCX counts. The post-render mode can complete the configured DOCX projection
and refresh receipts; it leaves scientific data, figure bytes, registries and
hydrated sources unchanged.

The pre-render suite selects `-m "not post_render"`. Tests marked
`post_render` run in an explicit stage after the current native export; the
rendered-PDF test requires the PDF to exist and applies the same strict
semantic audit as the standalone receipt. This ordering prevents a previous
edition from being tested against a newly generated figure registry. Running
the unfiltered suite directly therefore requires the current export.

The rendered-text regression also checks the combined PDF and all eight slide
editions with Poppler word rectangles. It rejects text endings outside the paper
and visible unresolved definition, proposition or section identifiers. Real PDF
controls exercise all four page edges and unresolved references; manual image
inspection remains necessary for overlaps, font size and figure interpretation.

## Gate responsibilities

| Gate | Purpose and limitation |
| --- | --- |
| Locked development sync | Installs the committed Python environment before producer tests; does not replace native Julia/Lean/template preparation. |
| Canonical `all` | Produces protocol, formal/benchmark/transfer/sensing/theorem, stress, counterexample and offline external-preparation receipts before artifact-dependent tests. |
| Pre-render pytest with combined line/branch coverage >=90% | Exercises real APIs and source modules after scientific generation; current rendered-artifact tests run after export. |
| Ruff / format / mypy | Code syntax/style/type checks; no scientific validity claim. |
| Authored docs / claim map | Offline Markdown structural/index/citation closure and claim-ID equality against current generated output. |
| Formal / replication / RxInfer | Pinned Lean receipt, public/private bundle verification and real native Gaussian/categorical backends. |
| Hydration | Source-token closure and exact generated manuscript tree. |
| Native template export | Current hydrated composition, configured document formats, native Pandoc and LaTeX renderers. |
| Post-render pytest | Requires the current PDF and exercises its source-bound semantic accessibility audit. |
| DOCX / PDF / qpdf / VeraPDF | Actual exported media/source/structure, PDF syntax and reported UA-2 compliance. |
| Existing-output oracle | Current source-bound scientific and document checks without rewriting their consumed inputs. |
| Optional Julia tests | Real `Pkg.test()` with `allow_reresolve=false`. |

Machine checks leave a manual visual review obligation for the final editions.
The DOCX integrity lane does not certify PDF/UA; a UA-2 report does not prove
scientific correctness. Oracle and figure counts are read from current receipts,
not copied into this guide as permanent totals.

## Individual checks

```bash
uv run --locked pytest tests --cov=src --cov-fail-under=90
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked mypy
uv run --locked python scripts/check_docs.py
uv run --locked python scripts/check_claim_map.py
uv run --locked python scripts/run_julia_tests.py
```

Regeneration and hydration are producer operations:

```bash
uv run --locked python scripts/run_scientific_program.py all
uv run --locked python scripts/validate_analysis_outputs.py
uv run --locked python scripts/z_generate_manuscript_variables.py
```

For already exported documents, use the final boundary:

```bash
ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=0 \
  uv run --locked python scripts/validate_analysis_outputs.py --existing-only
```

The default validator regenerates inputs and summary figures. Running it after
export invalidates the stable render handoff. The existing-only mode refreshes
validation JSON and build lineage and validates actual current outputs.

## Artifact and claim closure

```bash
uv run --locked python scripts/run_scientific_program.py audit-artifacts
uv run --locked python scripts/run_scientific_program.py audit-figures
uv run --locked python - <<'PYTHON'
from src.reproducibility import verify_reproducibility_manifest
errors = verify_reproducibility_manifest()
print(errors)
raise SystemExit(bool(errors))
PYTHON
```

`audit-artifacts` checks protocol envelopes without repairing them;
`seal-artifacts` is an explicit legacy migration, never a validation shortcut.
The claim-map guard compares identifiers in both directions and rejects missing,
renamed or duplicated claims. Figure QA requires current inputs, nonblank
rasters, parseable companions and complete registry/atlas metadata.

## External and replay boundaries

The canonical producer and replay commands consume cached inputs offline.
Explicit pinned source acquisition is separate:

```bash
uv run --locked python scripts/run_scientific_program.py acquire-external \
  --manifest configs/external_sources.yaml \
  --destination output/external_sources \
  --allow-network
```

Acquisition and VM setup are the documented network-capable commands; native
package installation may also require network during preparation. External
compilation/disassembly remains static-only. Replay and protected scoring do
not execute external objects or convert a local run into independent evidence.
Use the [operator runbook](replay_operator_runbook.md) for the exact public-only
handoff and the [VM guide](replay_vm.md) for the narrower same-host boundary.

## Toolchain drift

The RxInfer bridge checks the committed Julia/RxInfer manifest, the selected
Julia executable and returned backend diagnostics. Julia 1.12.7, RxInfer 5.4.0
and Lean v4.35.0 are scientific pins; optional compiler version rows describe
the measured producer environment. A pin change requires a deliberate source
change, regenerated receipt and native run. Never replace an unavailable pin
with synthetic data or an unrecorded compatible version.
