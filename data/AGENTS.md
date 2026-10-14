# data — Agent Guide

## Overview

`data/` is a tracked snapshot of generated analysis artifacts. It exists so a
reader of the repository (or a reviewer without a Python/Julia toolchain) can
inspect representative receipts without executing the pipeline. It is **not**
an input directory: no module in [`../src`](../src) and no script in
[`../scripts`](../scripts) reads from it, and the code writes exclusively to
`output/data/`.

Each of the 20 snapshot artifacts has a same-named counterpart in `output/data/`.

## Provenance

The canonical producers live in [`../src`](../src) and are orchestrated by the
thin scripts in [`../scripts`](../scripts):

- `src/advanced_inference.py` composes the temporal, active, evaluation, and
  semantics layers and writes `temporal_inference.json`, `phase_segments.json`,
  `active_selection.json`, `evaluation_report.json`,
  `generative_adequacy.json`, `typed_observations.json`,
  `factorized_observation_model.json`, `corpus_coverage.json`,
  `action_sensing.json`, and `parameter_learning.json`. The payloads themselves
  come from `src/temporal_model.py`, `src/phase_segmentation.py`,
  `src/active_selection.py`, `src/evaluation.py`, `src/typed_observations.py`,
  `src/factorized_model.py`, `src/corpus_coverage.py`,
  `src/action_sensing.py`, and `src/parameter_learning.py`.
- `src/scholarship.py` (`write_scholarship_sources`) writes
  `scholarship_sources.json` and `scholarship_matrix.md`.
- `src/claim_register.py` (`write_claim_register`) writes
  `claim_register.json` and `claim_register.md`.
- `src/reproducibility.py` (`write_reproducibility_manifest`) writes
  `reproducibility_manifest.json` and `reproducibility_manifest.md`.
- `src/multi_arch.py` (`write_cross_architecture_payload`) writes
  `cross_architecture_fixtures.json`.
- `src/information_policy.py` (`write_active_information_payload`) writes
  `active_information_frontier.json`.

- `src/scientific_protocol.py` and `src/quotient_classifier.py` write the
  versioned protocol and portfolio/randomization receipts.

## Receipt boundary

Project-owned JSON artifacts carry a protocol-v2 receipt envelope
(`src/receipt.py`, `src/artifact_contract.py`) holding the protocol digest,
producer revision, input hashes, toolchain versions, status, limitations, and
explicit unsupported or underpowered cases. The receipt schema is deliberately
separate from the domain schema, so a snapshot in this folder can be audited
for provenance without interpreting the payload.

Because the envelope records producer revision and input hashes, a snapshot's
bytes will differ from a freshly generated `output/data/` file even when the
scientific content is unchanged. Do not treat that drift as a failure.

## Regenerating and refreshing

```bash
# from the project root
uv run python scripts/run_assembly_inference.py       # end-to-end experiment
uv run python scripts/validate_analysis_outputs.py    # regenerate + validate
uv run python scripts/run_scientific_program.py all   # canonical local orchestration
uv run python scripts/run_scientific_program.py audit-artifacts  # read-only, fail-closed
```

Refreshing the snapshot is a deliberate copy from `output/data/`, not an
automatic step of any gate.

## Best Practices

- Do read these files to understand artifact shape.
- Do run `audit-artifacts` before trusting a receipt envelope.
- Do not hand-edit a payload; it breaks the recorded input hashes.
- Do not point a producer at this directory as an input path.

## See Also

- [`../docs/reproducibility.md`](../docs/reproducibility.md) — hash manifest and registries
- [`../docs/claim_map.md`](../docs/claim_map.md) — claims mapped to these artifacts
- [`../docs/analysis_validation.md`](../docs/analysis_validation.md) — the validation oracle
- [`../figures/AGENTS.md`](../figures/AGENTS.md) — the matching figure snapshot
