# figures — Agent Guide

## Overview

`figures/` is a tracked snapshot of 18 generated PNG figures, kept so a reader
can see representative output without running the pipeline. It is not an input
directory and not the path the manuscript resolves: manuscript sources
reference `../output/figures/<name>.png` after token hydration, per
[`../manuscript/AGENTS.md`](../manuscript/AGENTS.md).

Every file here has a same-named counterpart in `output/figures/`.

## Producers

Figure code lives in [`../src`](../src):

- `src/figures.py` — publication figures for the demonstrator; every file in
  this snapshot is named there.
- `src/advanced_figures.py` — temporal, active, and evaluation-layer figures,
  including `active_efe.png`, `model_comparison.png`, and the ten
  `temporal_beliefs_*` traces.
- `src/figure_contract.py` — canonical source-bound figure specifications and
  the offline figure atlas; it declares the specs behind
  `artifact_hash_manifest.png`, `claim_evidence_map.png`,
  `scholarship_evidence_matrix.png`, and `scholarship_source_roles.png`.
- `src/visual_atlas.py` — source-bound narrative figures for the atlas.

## Figure contract

The project treats figures as registered evidence, not decoration. Each
registered figure carries a generated caption, population, statistic, citation,
alternative text, and machine visual-QA row in `output/figures/README.md` and
`output/data/figure_registry.json`. The sibling template renderer consumes a
boundary projection at `output/figures/figure_registry.json`; validation
requires exact labels, filenames, row count, and alt text.

Adding a figure therefore means adding a spec, a caption, a manuscript token,
and a manifest target together — see
[`../docs/visual_atlas.md`](../docs/visual_atlas.md) and
[`../manuscript/AGENTS.md`](../manuscript/AGENTS.md).

## Commands

```bash
# from the project root
uv run python scripts/run_assembly_inference.py                     # end-to-end run
uv run python scripts/run_scientific_program.py generate-figures    # figure bundle + atlas
uv run python scripts/run_scientific_program.py audit-figures       # specs, hashes, vectors, atlas
uv run python scripts/validate_analysis_outputs.py                  # figure-registry coverage
```

`generate-figures` accepts a format selector with the choices `png`, `svg`,
`pdf`, and `atlas`.

## Best Practices

- Do regenerate into `output/figures/` and refresh this snapshot deliberately.
- Do keep the fixture-named traces aligned with `src/corpus.py`; the ten
  `temporal_beliefs_*` names are the ten corpus fixtures.
- Do not hand-edit or re-crop a PNG here; the audit compares hashes.
- Do not reference this folder from manuscript sources.

## See Also

- [`../docs/visual_atlas.md`](../docs/visual_atlas.md) — the source-bound figure registry
- [`../docs/reproducibility.md`](../docs/reproducibility.md) — hash manifest and registries
- [`../data/AGENTS.md`](../data/AGENTS.md) — the matching data snapshot
- [`../src/AGENTS.md`](../src/AGENTS.md) — producer modules
