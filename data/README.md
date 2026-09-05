# data

Committed snapshots of generated analysis artifacts. Every file here has a
same-named counterpart under the untracked `output/data/` directory, which is
where the pipeline actually writes. Nothing in this folder is a pipeline input,
and no module reads from it.

## Contents

15 JSON artifacts and 3 Markdown views:

| File | Kind |
| --- | --- |
| `action_sensing.json` | action-conditioned typed-field sensing receipt |
| `active_information_frontier.json` | exact finite-horizon information-per-cost frontier |
| `active_selection.json` | Expected Free Energy / active observation selection |
| `claim_register.json` | machine-readable claim rows |
| `claim_register.md` | Markdown view of the claim register |
| `corpus_coverage.json` | source-bound fixture coverage inventory |
| `cross_architecture_fixtures.json` | x86-64 / AArch64 / RISC-V64 adapter fixtures |
| `evaluation_report.json` | decoder metrics, negative controls, adequacy |
| `factorized_observation_model.json` | factorized/hierarchical companion scores |
| `generative_adequacy.json` | fixed-parameter adequacy checks |
| `parameter_learning.json` | observation-only HMM parameter-learning robustness |
| `phase_segments.json` | contiguous phase segmentation of belief traces |
| `reproducibility_manifest.json` | artifact hash/size manifest |
| `reproducibility_manifest.md` | Markdown view of the manifest |
| `scholarship_matrix.md` | Markdown source/role/boundary matrix |
| `scholarship_sources.json` | citation ledger with roles and boundaries |
| `temporal_inference.json` | temporal HMM filtering result |
| `typed_observations.json` | typed, provenance-preserving observations |

## Regenerating

```bash
# from the project root — writes output/data/, not this folder
uv run python scripts/run_assembly_inference.py
uv run python scripts/validate_analysis_outputs.py
```

## Key Points

- Treat `output/data/` as canonical and this folder as a committed snapshot.
- Snapshots drift from `output/data/` between refreshes; that is expected, and
  the validation gate scores the live `output/` surface, not this copy.
- Do not hand-edit these files. Regenerate and re-copy instead.

## More Information

See [AGENTS.md](AGENTS.md) for provenance and the receipt boundary.
