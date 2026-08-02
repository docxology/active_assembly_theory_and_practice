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
set -u
cd "$(dirname "$0")/.."

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
    sed -n '1,25p' "$log" >&2
    echo "" >&2
  fi
  steps=$((steps + 1))
  rm -f "$log"
}

run_step "pytest (>=90% coverage)" \
  uv run pytest tests --cov=src --cov-fail-under=90 -q
run_step "ruff check" \
  uv run ruff check src tests scripts
run_step "ruff format --check" \
  uv run ruff format --check src tests scripts
run_step "mypy" \
  uv run mypy
run_step "validation gate (63-oracle)" \
  uv run python scripts/validate_analysis_outputs.py

if [ "${1:-}" = "--julia" ]; then
  run_step "Julia backend Pkg.test()" \
    julia --project=julia/ActiveAssemblyRxInfer -e 'using Pkg; Pkg.test()'
fi

echo ""
if [ "$failures" -gt 0 ]; then
  echo "Verification FAILED: ${failures}/${steps} steps failed."
  exit 1
fi
echo "Verification PASSED: ${steps}/${steps} steps."
