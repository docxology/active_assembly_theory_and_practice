# Analysis Validation

`src/validation.py` is the project oracle for generated analysis outputs. It
checks the artifacts that a normal test suite can miss:

- trace bundle required keys and schema version;
- sample byte SHA-256 hash;
- alignment between decoded instructions and observations;
- observation IDs, labels, categories, and instruction-address alignment;
- dynamic-register trace initial/final shape and instruction count;
- Capstone and Unicorn provenance rows;
- optional roadmap package provenance rows for the expanded package surface;
- RxInfer JSONL free-energy records, finiteness, source field, and trace hash;
- RxInfer companion artifacts: posteriors, model configuration, stdout log, and stderr log;
- figure existence and nonblank image variance;
- source-shape quality for DOI, HTTPS URL, source type, role, and boundary text;
- source-stage and evidence-tier partitioning for implemented, comparison,
  roadmap, and context citations;
- generated claim-register rows with source keys, artifact paths, validation
  checks, status, and boundaries;
- reproducibility manifest rows with file sizes, SHA-256 hashes, and current
  artifact-byte verification;
- bibliography coverage for the scholarship source ledger;
- citation closure for every manuscript citation key;
- manuscript citation usage for source-ledger keys;
- expected manuscript sections and expected figure-reference tokens;
- fixed-parameter generative adequacy contract and freshness;
- action-conditioned sensing contract, held-out freshness, and no-peeking guard;
- observation-only ordered-HMM parameter-learning contract, held-out freshness,
  no-labels-in-fit guard, and complete multi-start/seed-sensitivity rows;
- exact finite intervention-power contract, subset enumeration, negative
  control, fail-closed promotion status, and freshness;
- bounded quotient protocol and source-family representation-invariance
  witnesses, including retained negative-control separation failures;
- real multi-architecture Capstone/Unicorn fixture contract and freshness;
- pinned Lean formal-verification receipt, theorem manifest, source/configuration
  hashes, and local rebuild freshness;
- exact finite-horizon information-policy contract, sensor normalization, and
  frontier freshness;
- expected technical documentation and claim-map boundary language;
- cautious claim-boundary phrases in the manuscript;
- source-to-variable coverage for every renderable manuscript `{{TOKEN}}`;
- unresolved `{{TOKEN}}` patterns in hydrated manuscript output when present;
- one-to-one coverage and present status in the generated figure-caption ledger.
- complete figure metadata, generator/input/claim/check lineage, alt text, long
  descriptions, and current PNG hashes in `output/data/figure_registry.json`;
- exact label/filename/alt-text projection at
  `output/figures/figure_registry.json` for the sibling template renderer;
- exact source/generated manuscript-token closure, typed-value metadata,
  consumer lists, input hashes, and no pending values in
  `output/data/manuscript_variable_registry.json`;
- current target/source/template provenance in `output/data/build_provenance.json`.
- post-render DOCX ZIP integrity, required OOXML parts, embedded-figure and
  caption parity, configured-cover byte/alternative-text binding and page
  break, resolved tokens, hydrated-manuscript hash, and figure-registry hash
  (excluding only the two self-referential validator summary figures) through
  `scripts/validate_docx_render.py` and
  `output/data/docx_render_receipt.json`.

Run the validator from the project checkout (`/Users/mini/Documents/GitHub/active_assembly_theory_and_practice`):

```bash
uv run python scripts/validate_analysis_outputs.py
```

The script regenerates the experiment, runs the native RxInfer bridge, writes
the scholarship source ledger, and emits:

- `output/data/validation_report.json`
- `output/figures/validation_summary.png`
- `output/data/generative_adequacy.json`
- `output/data/reproducibility_manifest.json`
- `output/figures/artifact_hash_manifest.png`
- `output/data/manuscript_variable_registry.json`
- `output/data/figure_registry.json`
- `output/figures/figure_registry.json`
- `output/data/build_provenance.json`
- `output/data/formal_verification.json`
- `output/data/cross_architecture_fixtures.json`
- `output/data/active_information_frontier.json`

The standard DOCX lane is deliberately a post-render gate. The project-side
generator can run before a clean checkout has a DOCX, so source validation
marks that artifact as deferred until the sibling template has rendered it.
After rendering, run:

```bash
uv run python scripts/validate_docx_render.py
```

The command writes `output/data/docx_render_receipt.json` and exits non-zero
for a missing, corrupt, stale, unresolved, figure/caption-inconsistent, or
cover-contract-inconsistent document. When a source-configured cover exists,
the DOCX must contain its exact image bytes and configured alternative text;
the cover is additional to the registered manuscript-figure count. The gate
is independent of the PDF/UA audit: a valid DOCX receipt cannot certify PDF
tags, reading order, or accessibility semantics.

The validator is intentionally stricter than artifact existence. Empty figures,
missing citation keys, malformed DOI/URL fields, nonfinite free-energy records,
stale RxInfer companion artifacts, missing source-stage partitions, mismatched
claim-register links, mismatched observation categories, inadequate or stale
fixed-parameter simulation artifacts, changed manifest hashes, and byte-size
mismatches should fail even if the pipeline exits successfully.
