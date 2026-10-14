#!/usr/bin/env bash
# Thin orchestrator for the project's verification gate.
#
# Runs every canonical deterministic check in one command and prints a summary
# table. Each failing step surfaces the tail of its log and causes a non-zero
# exit, so this is safe to use as a CI or pre-push gate. It does not replace
# the individual commands -- it simply runs them together.
#
#   scripts/run_full_verification.sh          # deterministic gates
#   scripts/run_full_verification.sh --julia  # + Julia backend Pkg.test()
#
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep figure-producing tests deterministic and display-server independent.
export MPLBACKEND="${MPLBACKEND:-Agg}"

run_julia=0
analysis_only=0
for argument in "$@"; do
  case "$argument" in
    --julia) run_julia=1 ;;
    --analysis-only) analysis_only=1 ;;
    --help)
      echo "Usage: scripts/run_full_verification.sh [--julia] [--analysis-only]"
      echo "Default: regenerate analysis, test, hydrate, and validate existing rendered documents."
      echo "--analysis-only: run the pre-render gates; does not certify PDF or DOCX."
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
if [ "$analysis_only" -eq 1 ]; then
  export ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=1
fi
steps=0
failures=0

run_step() {
  local name="$1"
  shift
  local log
  log="$(mktemp -t aapta_gate)"
  if "$@" >"$log" 2>&1; then
    printf '%-46s PASS\n' "$name"
  else
    printf '%-46s FAIL\n' "$name"
    failures=$((failures + 1))
    tail -n 40 "$log" >&2
    rm -f "$log"
    echo "Verification stopped at failed prerequisite: $name" >&2
    exit 1
  fi
  steps=$((steps + 1))
  rm -f "$log"
}

# Tests include generated-artifact integration checks. Produce their inputs
# first, so a fresh clone and a previously rendered checkout use the same order.
run_step "locked development environment" uv sync --locked --extra dev
run_step "canonical analysis and artifact regeneration" \
  uv run --locked python scripts/run_scientific_program.py all
run_step "pytest (>=90% coverage)" \
  uv run --locked pytest tests --cov=src --cov-fail-under=90 -q
run_step "ruff check" \
  uv run --locked ruff check src tests scripts
run_step "ruff format --check" \
  uv run --locked ruff format --check src tests scripts
run_step "mypy" \
  uv run --locked mypy
run_step "formal correspondence + Lean receipt" \
  uv run --locked python scripts/run_formal_verification.py
run_step "blind replication bundle" \
  uv run --locked python scripts/run_replication.py verify
run_step "RxInfer backend receipt" \
  uv run --locked python scripts/run_rxinfer_backend.py
run_step "manuscript variables + template hydration" \
  uv run --locked python scripts/z_generate_manuscript_variables.py
if [ "$analysis_only" -eq 0 ]; then
run_step "DOCX configured cover projection" \
  uv run --locked python scripts/render_docx_cover.py
run_step "DOCX structural receipt" \
  uv run --locked python scripts/validate_docx_render.py
run_step "PDF accessibility receipt" \
  uv run --locked python scripts/validate_pdf_accessibility.py
run_step "PDF syntax (qpdf)" \
  qpdf --check output/pdf/active_assembly_theory_and_practice_combined.pdf
run_step "PDF/UA-2 (VeraPDF)" \
  verapdf --format text --flavour ua2 output/pdf/active_assembly_theory_and_practice_combined.pdf

fi

run_step "validation gate (source-bound oracles)" \
  uv run --locked python scripts/validate_analysis_outputs.py --existing-only

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
