# Scripts

Thin orchestrators for the assembly Active Inference demonstrator. Every
command delegates to `src/` modules; testable logic lives there, not here.
Run from the project root with `uv run`.

| Script | Purpose | Delegates to | Run command |
| --- | --- | --- | --- |
| `00_preflight.py` | Dependency/version preflight (capstone, pymdp, unicorn, Julia lockfile) | `src/rxinfer_bridge.py` | `uv run python scripts/00_preflight.py` |
| `run_scientific_program.py` | Canonical scientific-program CLI: receipts, external sources, VM replay, replication, figures, hydration (source-declared subcommands) | `src/` producer modules dispatched per subcommand | `uv run python scripts/run_scientific_program.py <subcommand>` |
| `check_docs.py` | Offline authored-document closure | `src/docs_validation.py` | `uv run --locked python scripts/check_docs.py` |
| `render_manuscript.py` | Native export of current validated hydration | `src/document_rendering.py` | `uv run --locked python scripts/render_manuscript.py` |
| `render_docx_cover.py` | Configured DOCX cover and canonical scientific media | `src/document_rendering.py` | `uv run --locked python scripts/render_docx_cover.py` |
| `validate_pdf_ua.py` | Actual reported UA-2 compliance, beyond process exit status | `src/pdf_ua.py` | `uv run --locked python scripts/validate_pdf_ua.py` |
| `run_assembly_inference.py` | End-to-end assembly observation and inference experiment | `src/experiments.py` | `uv run python scripts/run_assembly_inference.py` |
| `run_formal_verification.py` | Builds `formal/` and writes the Lean correspondence receipt | `src/formal_verification.py` | `uv run python scripts/run_formal_verification.py` |
| `run_full_verification.sh` | Full producer/test/native-export gate, or explicit staged boundaries | the other scripts in this directory | `scripts/run_full_verification.sh [--julia] [--analysis-only | --post-render-only]` |
| `run_julia_tests.py` | Validates Julia/RxInfer pins and runs the native test suite | `src/rxinfer_bridge.py` | `uv run python scripts/run_julia_tests.py` |
| `run_replication.py` | Generates, verifies, or scores the replication bundle | `src/replication.py` | `uv run python scripts/run_replication.py {generate,verify,score}` |
| `run_rxinfer_backend.py` | Runs the Julia/RxInfer backends and plots diagnostics | `src/experiments.py`, `src/rxinfer_bridge.py`, `src/figures.py` | `uv run python scripts/run_rxinfer_backend.py` |
| `validate_analysis_outputs.py` | Regenerates by default; `--existing-only` preserves scientific inputs | `src/manuscript_variables.py`, `src/validation.py` | `uv run python scripts/validate_analysis_outputs.py` |
| `validate_docx_render.py` | Post-render DOCX structural receipt (fail-closed) | `src/docx_validation.py` | `uv run python scripts/validate_docx_render.py` |
| `validate_pdf_accessibility.py` | Post-render PDF tag-tree / accessibility audit | `src/pdf_accessibility.py` | `uv run python scripts/validate_pdf_accessibility.py [--pdf PATH] [--report PATH]` |
| `z_generate_manuscript_variables.py` | Generates manuscript variables and hydrates the template tree | `src/manuscript_variables.py`, template `infrastructure` package | `uv run python scripts/z_generate_manuscript_variables.py` |

See [`AGENTS.md`](AGENTS.md) for the thin-orchestrator contract and gotchas.

The complete mode exports after authoritative hydration. The post-render-only
mode forces strict document checking and refreshes receipts without refitting
science. See [verification](../docs/verification.md) and
[document rendering](../docs/document_rendering.md) for prerequisites and scope.
