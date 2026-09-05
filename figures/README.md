# figures

Committed snapshot of 12 generated PNG figures. Every file here has a
same-named counterpart in the untracked `output/figures/` directory, which is
where the figure producers actually write. No module reads from this folder.

## Contents

| File | Subject |
| --- | --- |
| `active_efe.png` | Expected Free Energy / active observation selection |
| `artifact_hash_manifest.png` | artifact hash and size manifest |
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

The six `temporal_beliefs_*` files correspond one-to-one to the six transparent
fixtures declared in [`../src/corpus.py`](../src/corpus.py): `x86_accumulator`,
`x86_bitwise`, `x86_guard`, `x86_long_accumulator`, `x86_memory`, and
`x86_guard_taken`.

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
