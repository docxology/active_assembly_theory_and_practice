# docs — Agent Guide

## Overview

`docs/` is the authored project contract: how the disassembly-to-inference
pipeline is structured, what it claims, where each claim stops, and how it is
verified. These documents and the [`../manuscript`](../manuscript) sources are
the two authored interfaces of the project; everything under `output/` is
regenerated from code and must never be hand-edited to agree with prose here.

When code and a document in this folder disagree, the code is the fact and the
document is the defect. Fix the document, or fix the code and then the
document — never leave a claim standing that the pipeline does not produce.

## File Organization

The folder index owns the complete topic list; do not maintain duplicate line
counts or live artifact totals here. Its routes are:

- setup and operations: `getting_started.md`, `verification.md`,
  `document_rendering.md`;
- pipeline: `architecture.md`, `gnn_models.md`, `assembly_packages.md`,
  `rxinfer_roadmap.md`;
- methods and interpretation: `evaluation.md`, `statistics.md`, `scholarship.md`;
- evidence and lineage: `claim_map.md`, `claim_register.md`,
  `analysis_validation.md`, `reproducibility.md`, `visual_atlas.md`;
- replay: `replay_operator_runbook.md`, `replay_vm.md`;
- future evidence: `roadmap.md`, `scientific_program.md`;
- historical observations: dated `review_*.md` files.

## How the documents relate

- **Pipeline shape**: `architecture.md` → `gnn_models.md` →
  `assembly_packages.md` → `rxinfer_roadmap.md`. Read these to understand what is computed and by which
  adapter.
- **What is claimed**: `scholarship.md` (citation roles) → `claim_map.md`
  (claim-to-artifact map) → `claim_register.md` (the generated register's
  contract). The machine-readable register is `output/data/claim_register.json`.
- **Whether it holds**: `evaluation.md` (metrics and negative controls),
  `analysis_validation.md` (the oracle), `verification.md` (the gate list),
  `visual_atlas.md` (figure QA), `reproducibility.md` (hashes and registries).
- **What is next**: `roadmap.md` is the maintainer-facing execution view;
  `scientific_program.md` is the long-horizon program. The machine-readable
  status source is `output/data/scientific_program.json`, not either document.
- **Replay**: `replay_vm.md` (the pinned same-host guest) and
  `replay_operator_runbook.md` (the public-bundle handoff to an operator who is
  not the producer). A local rerun by the repository owner is a rehearsal, not
  independent evidence.

## Status vocabulary

Generated receipts use an explicit fail-closed vocabulary. The statuses
`blocked_external`, `underpowered`, `unavailable`, `falsified`, and
`unresolved` are first-class results and must stay visible. A document in this
folder may not summarize such a status as a positive claim, and a completed
local receipt never silently promotes external validity, independent
reimplementation, complete ISA semantics, universal identifiability, or a
general sensing law.

## Editing rules

- Keep the generated/authored boundary explicit. Reference an `output/...`
  artifact by path rather than restating its numbers as prose.
- Keep counts generated where the project already generates them (for example
  the oracle count in `verification.md`).
- Use relative links between documents and to sibling folders.
- Run `uv run --locked python scripts/check_docs.py` after editing. It checks
  offline structure, local links/anchors, bibliography keys and index closure.
- Keep detailed rendering mechanics in `document_rendering.md`, statistical
  interpretation in `statistics.md`, and environment setup in `getting_started.md`;
  other guides link to those contracts instead of duplicating them.
- Add a new document to the list in [`README.md`](README.md) and to the map
  above in the same change.

## See Also

- [`README.md`](README.md) — the folder index with per-document summaries
- [`../AGENTS.md`](../AGENTS.md) — project-level boundaries
- [`../manuscript/AGENTS.md`](../manuscript/AGENTS.md) — manuscript conventions
- [`../src/AGENTS.md`](../src/AGENTS.md) — the modules these documents describe
