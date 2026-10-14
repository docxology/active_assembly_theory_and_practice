# Documentation index

These guides describe the current code and its evidence boundaries. Source
contracts live in `src/` and `configs/`; current results and statuses live in
regenerated `output/` receipts. Authored prose must agree with those surfaces.
Committed `data/` and `figures/` files are examples and never producer inputs.

Start with [getting started](getting_started.md). Use [verification](verification.md)
for gate selection and [document rendering](document_rendering.md) for the
ordered hydration, export and post-render workflow. Read [statistics](statistics.md)
before interpreting a score, interval, p-value or leakage control.

| Topic | Maintained responsibility |
| --- | --- |
| [Getting started](getting_started.md) | Checkout, locked environments, native tools, template registration, first run and troubleshooting. |
| [Architecture](architecture.md) | Module responsibilities and observation-to-inference data flow. |
| [Assembly packages](assembly_packages.md) | Real adapter provenance, trace schema, bounded semantics and formal/ISA boundaries. |
| [RxInfer](rxinfer_roadmap.md) | Authoritative Gaussian spine, categorical companion, native toolchain and expansion boundary. |
| [Evaluation](evaluation.md) | Temporal, typed, learned and sensing methods, negative results and artifact/check map. |
| [Statistics](statistics.md) | Sampling units, folds, descriptive uncertainty, paired randomization, multiplicity and retrospective limitations. |
| [Scholarship](scholarship.md) | Citation roles, primary-source review history and claims that citations cannot support. |
| [Claim map](claim_map.md) | Authored mapping of scientific claims to evidence and limits. |
| [Claim register](claim_register.md) | Machine-readable claim-row contract and documentation closure. |
| [Reproducibility](reproducibility.md) | Artifact envelopes, source/byte lineage, public/private receipts and producer order. |
| [Visual atlas](visual_atlas.md) | Canonical figure specs, raster/vector metadata, offline atlas and visual QA. |
| [Analysis validation](analysis_validation.md) | Regeneration versus existing-output checks, deferred values and fail-closed oracle. |
| [Document rendering](document_rendering.md) | Native template handoff, DOCX media/source binding, tagged PDF and manual review. |
| [Verification](verification.md) | Pre-render, complete-build and post-render commands, gate scope and failure behavior. |
| [Replay operator runbook](replay_operator_runbook.md) | Public-only operator handoff, predictions, attestation and protected scoring. |
| [Replay VM](replay_vm.md) | Pinned Lima guest, filtered committed runtime archive, network boundary and same-host limits. |
| [Roadmap](roadmap.md) | Completed engineering tranches, actionable maintenance and open promotion gates. |
| [Scientific program](scientific_program.md) | Long-horizon conjectures, falsifiers, future designs and formal-theory ladder. |

Dated reviews are historical observations, not promises about a later checkout:
[October 13 correctness and methods review](review_2026_10_13.md) and
[October 14 documentation and workflow review](review_2026_10_14.md).
[AGENTS.md](AGENTS.md) records maintenance conventions.

Run the offline authored-document check from the repository root:

```bash
uv run --locked python scripts/check_docs.py
uv run --locked python scripts/check_claim_map.py
```

The first checks titles, fenced blocks, inline local links and Markdown heading
anchors, bibliography keys and index coverage. It does not fetch remote links
or establish scientific correctness. The second compares claim identifiers
against the generated claim register and therefore requires canonical outputs.
Scientific, figure and rendered-document validation remain separate gates.
