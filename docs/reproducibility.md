# Reproducibility Manifest

`src/reproducibility.py` records the generated artifact surface that supports
the current manuscript and validation report. It writes:

- `output/data/reproducibility_manifest.json`
- `output/data/reproducibility_manifest.md`
- `output/figures/artifact_hash_manifest.png`
- `output/figures/README.md` (a caption ledger documenting every figure)

Each manifest row records:

- project-relative artifact path;
- artifact class;
- generator module or function;
- file size in bytes;
- SHA-256 digest;
- linked claim IDs;
- linked validation checks;
- required/present status.

The manifest intentionally excludes self-referential and validator-produced
artifacts: the manifest JSON/Markdown, the manifest figure, the validation
report, the validation summary figure, and manuscript variables. Those files
are generated around the manifest rather than treated as stable inputs to it.

Validation recomputes the current size and SHA-256 digest for every tracked
artifact. A stale, edited, replaced, empty, or missing artifact fails the
`reproducibility_artifact_manifest` check even when the path still exists.

This is a reproducibility and freshness contract, not a scientific-validity
certificate. It can say that a rendered manuscript is pointing to the same
bytes the validator inspected; it cannot say that the model is the best model
for program semantics.
