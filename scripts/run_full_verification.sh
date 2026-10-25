#!/usr/bin/env bash
# Thin orchestrator for the project's verification gate.
#
# Runs every canonical deterministic check in one command and prints a summary
# table. Each failing step surfaces the tail of its log and causes a non-zero
# exit, so this is safe to use as a CI or pre-push gate. It does not replace
# the individual commands -- it simply runs them together.
#
#   scripts/run_full_verification.sh          # complete build and document gates
#   scripts/run_full_verification.sh --julia  # + Julia backend Pkg.test()
#
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep figure-producing tests deterministic and display-server independent.
export MPLBACKEND="${MPLBACKEND:-Agg}"

run_julia=0
analysis_only=0
post_render_only=0
for argument in "$@"; do
  case "$argument" in
    --julia) run_julia=1 ;;
    --analysis-only) analysis_only=1 ;;
    --post-render-only) post_render_only=1 ;;
    --help)
      echo "Usage: scripts/run_full_verification.sh [--julia] [--analysis-only | --post-render-only]"
      echo "Default: regenerate analysis, test, hydrate, export through the native template, and strictly audit documents."
      echo "--analysis-only: run the pre-render gates; does not certify PDF or DOCX."
      echo "--post-render-only: audit a current template export without regenerating scientific inputs."
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
if [ "$post_render_only" -eq 1 ] && { [ "$analysis_only" -eq 1 ] || [ "$run_julia" -eq 1 ]; }; then
  echo "--post-render-only cannot be combined with --analysis-only or --julia" >&2
  exit 2
fi
# Producer gates explicitly defer document-dependent values. Final gates below
# force strictness even if the invoking shell inherited the pre-render flag.
if [ "$post_render_only" -eq 0 ]; then
  export ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=1
else
  export ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=0
fi
steps=0
failures=0
mkdir -p output/verification
log_directory="$(mktemp -d "$PWD/output/verification/run-XXXXXXXX")"
echo "Verification logs: $log_directory"

run_step() {
  local name="$1"
  shift
  local log
  log="$log_directory/step-$steps.log"
  if "$@" >"$log" 2>&1; then
    printf '%-46s PASS\n' "$name"
  else
    printf '%-46s FAIL\n' "$name"
    failures=$((failures + 1))
    tail -n 40 "$log" >&2
    echo "Verification stopped at failed prerequisite: $name" >&2
    exit 1
  fi
  steps=$((steps + 1))
}

# Tests include generated-artifact integration checks. Produce their inputs
# first, so a fresh clone and a previously rendered checkout use the same order.
if [ "$post_render_only" -eq 0 ]; then
run_step "locked development environment" uv sync --locked --extra dev
run_step "upstream GNN models and real Agent controls" \
  uv run --locked python scripts/run_gnn_inference.py study
run_step "shared GNN native smoothing parity" \
  uv run --locked python scripts/run_gnn_inference.py native
run_step "canonical analysis and artifact regeneration" \
  uv run --locked python scripts/run_scientific_program.py all
run_step "pytest (>=90% coverage)" \
  uv run --locked pytest tests -m "not post_render" --cov=src --cov-fail-under=90 -q
run_step "ruff check" \
  uv run --locked ruff check src tests scripts
run_step "ruff format --check" \
  uv run --locked ruff format --check src tests scripts
run_step "mypy" \
  uv run --locked mypy
run_step "authored documentation closure" \
  uv run --locked python scripts/check_docs.py
run_step "claim-map identifier closure" \
  uv run --locked python scripts/check_claim_map.py
run_step "formal correspondence + Lean receipt" \
  uv run --locked python scripts/run_formal_verification.py
run_step "blind replication bundle" \
  uv run --locked python scripts/run_replication.py verify
run_step "RxInfer backend receipt" \
  uv run --locked python scripts/run_rxinfer_backend.py
run_step "manuscript variables + template hydration" \
  uv run --locked python scripts/z_generate_manuscript_variables.py
if [ "$analysis_only" -eq 0 ]; then
run_step "native template manuscript export" \
  uv run --locked python scripts/render_manuscript.py \
    --project-root "$PWD" --skip-manuscript-hydration
fi
fi
if [ "$post_render_only" -eq 1 ]; then
run_step "authored documentation closure" \
  uv run --locked python scripts/check_docs.py
run_step "claim-map identifier closure" \
  uv run --locked python scripts/check_claim_map.py
fi
if [ "$analysis_only" -eq 0 ]; then
export ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=0
run_step "current rendered-artifact regressions" \
  uv run --locked pytest tests -m post_render -q
run_step "DOCX configured cover projection" \
  uv run --locked python scripts/render_docx_cover.py
run_step "DOCX structural receipt" \
  uv run --locked python scripts/validate_docx_render.py
run_step "PDF accessibility receipt" \
  uv run --locked python scripts/validate_pdf_accessibility.py
run_step "PDF syntax (qpdf)" \
  qpdf --check output/pdf/active_assembly_theory_and_practice_combined.pdf
run_step "PDF/UA-2 (VeraPDF)" \
  uv run --locked python scripts/validate_pdf_ua.py

fi

run_step "validation gate (source-bound oracles)" \
  uv run --locked python scripts/validate_analysis_outputs.py --existing-only
run_step "read-only upstream GNN study audit" \
  uv run --locked python scripts/run_gnn_inference.py audit --require-native

if [ "$run_julia" -eq 1 ]; then
  run_step "Julia backend Pkg.test()" \
    uv run --locked python scripts/run_julia_tests.py
fi

echo ""
if [ "$failures" -gt 0 ]; then
  echo "Verification FAILED: ${failures}/${steps} steps failed."
  exit 1
fi
echo "Verification PASSED: ${steps}/${steps} steps."
