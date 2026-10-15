# `scripts/` — Agent Guide

## Thin-Orchestrator Contract

Every file in this directory is a thin orchestrator only: path bootstrap
(`PROJECT_ROOT = Path(__file__).resolve().parent.parent` inserted into
`sys.path`), argparse/argument handling, and a single delegated call into a
`src/` entrypoint. Scientific, data, plotting, and analysis logic must live in
`src/`, where it is importable and covered by `tests/`. Generated artifacts
land under `output/` and are disposable; build artifacts such as
`__pycache__/` and `*.pyc` never belong in `scripts/` (gitignored).

## Inventory

| Script | Purpose | Delegates to |
| --- | --- | --- |
| `00_preflight.py` | Dependency/version preflight: imports capstone, pymdp, unicorn, validates the Julia lockfile, and prints versions (always exits `0`; no template checkout involved) | `src/rxinfer_bridge.py` |
| `run_scientific_program.py` | Canonical scientific-program CLI with source-declared subcommands (protocol/formal/benchmark receipts, external sources, VM replay, replication, figures, manuscript, receipts) | `src/scientific_protocol.py`, `src/formal_correspondence.py`, `src/theorem_phase.py`, `src/counterexample_search.py`, `src/quotient_classifier.py`, `src/intervention_analysis.py`, `src/sensor_costs.py`, `src/sensing_stress.py`, `src/compiler_transfer.py`, `src/artifact_contract.py`, `src/external_sources.py`, `src/external_corpus.py`, `src/external_labeling.py`, `src/replay_vm.py`, `src/replication.py` |
| `check_docs.py` | Offline authored Markdown structure, links, citations and index closure | `src/docs_validation.py` |
| `render_manuscript.py` | Exports validated hydrated sources via the registered native template | `src/document_rendering.py` |
| `render_docx_cover.py` | Completes the native DOCX cover and canonical media projection | `src/document_rendering.py` |
| `validate_pdf_ua.py` | Requires actual VeraPDF UA-2 compliance from the JSON report | `src/pdf_ua.py` |
| `run_assembly_inference.py` | Canonical GNN study; explicit `--legacy-diagnostic` retains the observation comparator | `src/gnn_cli.py`, `src/experiments.py` |
| `run_gnn_inference.py` | Upstream qualification, individual runs, study, native parity and read-only audit | `src/gnn_cli.py` |
| `run_formal_verification.py` | Builds `formal/` and writes the pinned Lean correspondence receipt | `src/formal_verification.py` (`write_formal_verification`) |
| `run_full_verification.sh` | Complete producer/test/native-export gate; `--analysis-only` stops before export, `--post-render-only` audits existing editions, and `--julia` adds native tests | the other scripts in this directory |
| `run_julia_tests.py` | Validates Julia/RxInfer pins and runs the native package test suite | `src/rxinfer_bridge.py` |
| `run_replication.py` | Generates, verifies, or scores the frozen local replication bundle | `src/replication.py` |
| `run_rxinfer_backend.py` | Runs the Julia/RxInfer backends on the trace bundle and plots diagnostics | `src/experiments.py`, `src/rxinfer_bridge.py`, `src/figures.py` |
| `validate_analysis_outputs.py` | Regenerates inputs by default; `--existing-only` checks current outputs without changing scientific inputs | `src/manuscript_variables.py`, `src/validation.py` |
| `validate_docx_render.py` | Post-render DOCX structural receipt (fail-closed) | `src/docx_validation.py` |
| `validate_pdf_accessibility.py` | Post-render PDF tag-tree / accessibility audit | `src/pdf_accessibility.py` |
| `z_generate_manuscript_variables.py` | Generates manuscript variables and hydrates the `manuscript/` tree through the template renderer | `src/manuscript_variables.py` + template `infrastructure` package (re-exec) |

## Gotchas

- Run everything from the project root via `uv run python scripts/<name>.py`;
  each script bootstraps `PROJECT_ROOT` itself.
- `00_preflight.py` is a dependency/version preflight only. It never checks
  out or validates a template and does not use exit code `2`.
- Network-reaching subcommands (`acquire-external`, `vm-create`,
  `vm-provision`) require an explicit `--allow-network` flag; see
  [`../configs/AGENTS.md`](../configs/AGENTS.md).
- Figure-producing scripts set `MPLBACKEND=Agg` for deterministic output.
- `run_full_verification.sh` expects `qpdf` and `verapdf` on `PATH` for its
  PDF steps.
- `z_generate_manuscript_variables.py` requires a template checkout; set
  `TEMPLATE_REPO_ROOT` when it is not the configured sibling.
