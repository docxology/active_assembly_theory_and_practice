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
- RedTeam report presence and remediation status;
- expected technical documentation and claim-map boundary language;
- cautious claim-boundary phrases in the manuscript;
- unresolved `{{TOKEN}}` patterns in hydrated manuscript output when present.

Run the validator from `/Users/4d/Documents/GitHub/projects`:

```bash
uv run --project working/active_assembly_theory_and_practice \
  python working/active_assembly_theory_and_practice/scripts/validate_analysis_outputs.py
```

The script regenerates the experiment, runs the native RxInfer bridge, writes
the scholarship source ledger, and emits:

- `output/data/validation_report.json`
- `output/figures/validation_summary.png`
- `output/data/redteam_report.json`
- `output/figures/redteam_risk_register.png`
- `output/data/reproducibility_manifest.json`
- `output/figures/artifact_hash_manifest.png`

The validator is intentionally stricter than artifact existence. Empty figures,
missing citation keys, malformed DOI/URL fields, nonfinite free-energy records,
stale RxInfer companion artifacts, missing source-stage partitions, mismatched
claim-register links, mismatched observation categories, changed manifest
hashes, and byte-size mismatches should fail even if the pipeline exits
successfully.
