# figures

Committed snapshot of 18 generated PNG figures. Every file here has a
same-named counterpart in the untracked `output/figures/` directory, which is
where the figure producers actually write. No module reads from this folder:
this committed snapshot is documentation-only, and `output/figures/` is the
canonical surface that producers, validators, and the manuscript consume.

## Contents

| File | Subject |
| --- | --- |
| `active_efe.png` | Expected Free Energy / active observation selection |
| `artifact_hash_manifest.png` | artifact hash and size manifest |
| `corpus_coverage.png` | static spans, bounded paths and evidence regimes |
| `quotient_classifier.png` | endpoint intervals, paired deltas, calibration and leakage controls |
| `claim_evidence_map.png` | claims mapped to evidence artifacts |
| `model_comparison.png` | model comparison across inference variants |
| `scholarship_evidence_matrix.png` | citation evidence matrix |
| `scholarship_source_roles.png` | citation source roles |
| `temporal_beliefs_x86_accumulator.png` | per-fixture belief trace |
| `temporal_beliefs_x86_bitwise.png` | per-fixture belief trace |
| `temporal_beliefs_x86_guard.png` | per-fixture belief trace |
| `temporal_beliefs_x86_guard_taken.png` | per-fixture belief trace |
| `temporal_beliefs_x86_long_accumulator.png` | per-fixture belief trace |
| `temporal_beliefs_x86_memory.png` | per-fixture belief trace |
| `temporal_beliefs_x86_call_return.png` | per-fixture belief trace |
| `temporal_beliefs_x86_branch_not_taken.png` | per-fixture belief trace |
| `temporal_beliefs_x86_stack_local.png` | per-fixture belief trace |
| `temporal_beliefs_x86_optimization_rewrite.png` | per-fixture belief trace |

The ten `temporal_beliefs_*` files correspond one-to-one to the ten transparent
fixtures declared in [`../src/corpus.py`](../src/corpus.py): `x86_accumulator`,
`x86_bitwise`, `x86_guard`, `x86_long_accumulator`, `x86_memory`, and
`x86_guard_taken`, `x86_call_return`, `x86_branch_not_taken`, `x86_stack_local`,
and `x86_optimization_rewrite`.

## Regenerating

```bash
# from the project root — writes output/figures/, not this folder
uv run python scripts/run_assembly_inference.py
uv run python scripts/run_scientific_program.py generate-figures
uv run python scripts/run_scientific_program.py audit-figures
```

## Key Points

- `output/figures/` is canonical; this folder is a committed snapshot and can
  lag behind it.
- The manuscript references `../output/figures/<name>.png`, never this folder.
- Generated captions, alt text, and accessibility metadata live in
  `output/figures/README.md` and `output/data/figure_registry.json`.

## More Information

See [AGENTS.md](AGENTS.md) for producers and the figure contract.
