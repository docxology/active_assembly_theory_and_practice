---
Owner: Daniel Ari Friedman (daniel@activeinference.institute)
Status: active (v0.2 scope complete; 2026-07-31 review findings remediated)
Last reviewed: 2026-07-31
---

# TODO — Active Assembly Theory and Practice

## Completed/Closed (v0.2 ISA scope)

The v0.2 ISA (ISA.md) defines 61 Implementation Success Criteria (ISC-1..61).
All 61 ISCs were completed and verified at the time of the v0.2 milestone
(June-July 2026). The completed feature set includes:

| Feature | ISCs | Status |
| --- | --- | --- |
| F1: Temporal HMM (temporal_model.py) | ISC-1..10 | 100% coverage, 1e-9 hand-computed tests |
| F2: Active selection / EFE (active_selection.py) | ISC-11..17 | 100% coverage; EFE sign control + anti |
| F3: Phase segmentation (phase_segmentation.py) | ISC-18..22 | 100% coverage; partition + monotone + anti |
| F4: Corpus + evaluation (corpus.py, evaluation.py) | ISC-23..33 | 100%/99% coverage; controls discriminate |
| F5: RxInfer categorical HMM (julia/) | ISC-34..37 | Julia Pkg.test() passes; additive/no-regression |
| F6: Integration (claims, validation, figs, ms, docs) | ISC-38..61 | 15 figures, 13 claims, 47 oracle checks, 130+ tests |

All module coverage >=90% gate was green at v0.2 ship time. All 14 RedTeam
findings (RT-001..014) remediated. Forge cross-vendor audit completed;
over-claims corrected, memoryless baseline added as first-class comparator.

---

## Deepest Hostile Red-Team Review (2026-07-31)

The following findings were discovered during a comprehensive hostile review of
the codebase, tests, docs, and live execution in the current environment.

### Major — real reproducibility/availability defects

1. **Julia version drift blocks the entire validation gate** **[RESOLVED 2026-07-31: pin bumped to installed 1.12.6]**
   - **Affected:** `src/rxinfer_bridge.py`, all tests calling `generate_variables()`,
     `scripts/validate_analysis_outputs.py`, `scripts/z_generate_manuscript_variables.py`
   - **Why it matters:** The project pins `EXPECTED_JULIA_VERSION = "1.12.1"` but the
     current environment has only Homebrew julia 1.12.6. No juliaup-installed pinned
     binary exists at `~/.julia/juliaup/julia-1.12.1*/bin/julia`. The version guard
     (`validate_backend_versions`) raises `RuntimeError`.
   - **Impact:** 6 tests fail (out of 136): 4 integration tests in
     `test_assembly_pipeline.py` and 2 script smoke tests. The validation gate,
     manuscript generation, and full pipeline are blocked. The project's own
     verification docs describe a green gate that is currently red.
   - **Suggested fix:** (a) Install the pinned Julia via `juliaup add 1.12.1` so
     `resolve_julia_executable()` finds the exact binary, OR (b) bump the pin to
     1.12.6 (requires re-running the RxInfer backend and verifying all numbers),
     OR (c) document the bootstrap step and add a preflight check. This is not a
     code defect — the guard correctly prevents silent toolchain drift — but the
     environment cannot currently reproduce the gate.

2. **No project TODO/backlog file exists** **[RESOLVED 2026-07-31: this file created]**
   - No `TODO.md` or `TO-DO.md` in the project root. The ISA.md serves as the
     completion specification, but there's no forward-facing backlog structure.
   - **Suggested fix:** This file resolves this finding.

### Medium — moderate defects, coverage/robustness/design gaps

3. **Mypy reports 44 type errors across 10 source files (not gated, but latent)** **[RESOLVED 2026-07-31: mypy config added; clean]**
   - **Affected:** `src/figures.py` (10 errors), `src/validation.py` (18 errors),
     `src/rxinfer_bridge.py` (2), `src/redteam_report.py` (1), `src/experiments.py` (1)
   - **Why it matters:** Common pattern: JSON payloads typed as `dict[str, Any]`
     where downstream code calls `.keys()`, `.get()`, `len()`, leading to mypy
     errors about `object` having no attribute. Also tuple-vs-list mismatches in
     function signatures (e.g., `tuple[AssemblyObservation, ...]` passed where
     `list[...]` expected). While mypy is not a configured gate, 44 errors
     indicate latent fragility if the typing is ever enforced.
   - **Suggested fix:** Add a `pyproject.toml` mypy config or a `mypy.ini`.
     Type-narrow the `_load_json` return to `dict[str, Any]` (already done) but
     add explicit casts or type-narrowing helpers for known dictionary shapes.
     Fix the tuple/list signature mismatches.

4. **21 of 27 source files would be reformatted by `ruff format`** **[RESOLVED 2026-07-31: formatted; `ruff format --check` clean]**
   - **Affected:** Most `src/` modules
   - **Why it matters:** The project doesn't enforce a formatter; style is
     inconsistent across files. Formatting is not a gate, so this is not a
     defect — but the presence of a `.ruff_cache` directory and a `ruff` import
     in `pyproject.toml` suggests ruff is used for lint but not format.
   - **Suggested fix:** Run `ruff format src/ tests/ scripts/` once.

5. **Validation check `scholarship_claim_boundaries` uses fragile exact string matching** **[RESOLVED 2026-07-31: now tolerant regex]**
   - **Affected:** `src/validation.py` lines 356-357
   - **Why it matters:** The check requires the exact substrings "not a general
     reverse-engineering" and "not yet the planned categorical hidden Markov model"
     in the concatenated manuscript. Prose rewording (the only kind of edit a
     manuscript expects) silently breaks this gate. This is a maintainability
     trap: any future prose refinements must remember to keep the exact strings.
   - **Suggested fix:** Match a regex or a semantic marker (e.g., YAML frontmatter
     flag), or at minimum document the fragility prominently.

6. **`_figure_checks` validates only 1 of 4 temporal-belief variant figures** **[RESOLVED 2026-07-31: all 4 validated + manifest-tracked]**
   - **Affected:** `src/validation.py` lines 231-262
   - **Why it matters:** The expected figure tuple includes only
     `temporal_beliefs_x86_accumulator.png`. Three additional temporal-belief
     variant figures (`temporal_beliefs_x86_guard.png`,
     `temporal_beliefs_x86_bitwise.png`, `temporal_beliefs_x86_long_accumulator.png`)
     are generated by `advanced_figures.write_temporal_belief_figures()` but are
     NOT validated for non-blankness/presence by the validation gate. They also
     are not tracked in the reproducibility manifest (`MANIFEST_TARGETS`).
   - **Suggested fix:** Add the three variant figures to `_figure_checks` expected
     tuple and to `MANIFEST_TARGETS` (or document why they're excluded).

7. **Figure caption ledger has a stale entry for `temporal_beliefs.png`** **[RESOLVED 2026-07-31: 4 per-fixture caption entries, all present]**
   - **Affected:** `output/figures/README.md`
   - **Why it matters:** The caption ledger lists `temporal_beliefs.png` as
     missing, but the actual files are `temporal_beliefs_x86_accumulator.png` and
     three other per-fixture variants. The old figure name was replaced but the
     caption ledger was not updated. The validation gate's `_figure_checks` uses
     the current names (no gap there), but the caption ledger is a user-facing
     document that lists a missing entry.
   - **Suggested fix:** Update `write_figure_captions` or the caption data in
     `src/figures.py` to reference the actual temporal-belief variant filenames.

8. **`plot_observation_pipeline` uses fragile first-occurrence category lookup** **[RESOLVED 2026-07-31: deterministic id->category map]**
   - **Affected:** `src/figures.py` line 32:
     `observations[y.index(value)].category`
   - **Why it matters:** `list.index()` returns the first occurrence, which ties
     the y-axis tick label to which observation first exhibited that ID. With
     the current fixture this is deterministic, but the code assumes no duplicate
     observation IDs with different categories (an invariant that observations
     with the same ID always have the same category). If that assumption ever
     breaks (e.g., an encoding bug), the figure silently shows wrong labels.
   - **Suggested fix:** Build a deterministic map from observation_id to category
     using the first observation with that ID, or verify the invariant.

### Minor — low-impact polish, doc typos, small refactors

9. **`optional_adapters.py` fallback probe for llvm-objdump includes `"objdump"`** **[RESOLVED 2026-07-31: restricted to `llvm-objdump`]**
   - **Affected:** `src/assembly_packages/optional_adapters.py` line 53
   - **Why it matters:** The command_names tuple includes `"objdump"` as a
     fallback. On macOS with Xcode, `objdump` IS LLVM's objdump, so this is
     usually correct — but on a Linux system with GNU binutils, `objdump` is
     GNU objdump, which would make the "llvm-objdump" provenance claim inaccurate.
   - **Suggested fix:** Remove `"objdump"` from the fallback list, or rename the
     adapter to "objdump-compatible" and accept either.

10. **`_check` helper uses `not missing` (list) for truthiness** **[RESOLVED 2026-07-31: `len(missing) == 0`]**
    - **Affected:** `src/validation.py` lines 133, etc.
    - **Why it matters:** Style: `_check("name", not missing, ...)` where `missing`
      is a list — works correctly (`not []` → `True`) but less readable than
      `len(missing) == 0`. Minor readability concern only.
    - **Suggested fix:** Replace `not missing` with `len(missing) == 0` for
      clarity (at least where `missing` is a list).

11. **`shuffled_label_control` allows `max_attempts` to limit unique scrambles** **[RESOLVED 2026-07-31: `_PER_ROUND_DRAW_ATTEMPTS` constant]**
    - **Affected:** `src/evaluation.py` lines 338-341
    - **Why it matters:** `max_attempts = max(rounds * 64, 64)` = 1280 for 20
      rounds. With n=18 labels the number of distinct permutations is 18! ≈
      6.4e15, so the limit will never be hit in practice. The guard is correct
      but the constant `64` is documented only in a comment ("64 attempts per
      requested round is ample headroom") and is not unit-tested as a floor.
    - **Suggested fix:** Extract `max_attempts` to a named constant with a
      rationale comment, or add a unit test that verifies the limit behavior.

12. **Manuscript and verification docs describe a green gate that is not reproducible without juliaup pinned binary** **[RESOLVED 2026-07-31: bootstrap note added; oracle count updated to 50]**
    - **Affected:** `README.md`, `docs/verification.md`
    - **Why it matters:** Both documents state the validation gate passes with
      47 checks / all_passed, which is true only when the pinned Julia 1.12.1
      is available. A reader setting up a fresh environment sees 6 test failures
      and cannot reproduce the claimed green gate.
    - **Suggested fix:** Add an "Environment prerequisites" section to README.md
      and/or docs/verification.md that states: "Install Julia 1.12.1 via
      `juliaup add 1.12.1` before running the full validation suite."

13. **`uniform_belief_control` balanced accuracy is 0.20 when all 5 labels appear (as expected)**
    - Correct by construction; verified by live recompute. **[RESOLVED 2026-07-31: no code change required; recorded as documented invariant]**
    - This is correct behavior, verified by live recompute. No change needed.
    - Recorded as a documented invariant: uniform control = 0.200.

---

## Verification

From `/Users/4d/Documents/GitHub/projects`:

```bash
uv run --project working/active_assembly_theory_and_practice \
  pytest working/active_assembly_theory_and_practice/tests \
  --cov=working/active_assembly_theory_and_practice/src \
  --cov-fail-under=90
```

**Current state (2026-08-01):** 172 passed, 0 failed. Coverage 95.02% (gate
`--cov-fail-under=90` is GREEN). Validation gate `validate_analysis_outputs.py`
reports **55/55** oracles all passed. `ruff check`, `ruff format --check`, and
`mypy` are clean. All findings below marked **RESOLVED 2026-07-31** have been
implemented and verified.

## Completeness & Hardening Pass (2026-08-01)

- **Tests**: new `tests/test_optional_adapters.py` covers the availability-probe
  logic with real modules (numpy resolves, stdlib `os` -> "unknown",
  absent module -> None, absent command -> optional/unavailable);
  `test_rxinfer_bridge` adds the missing-file and unknown-source-field branches.
  Suite 164->172, coverage 94.68%->95.02% (weakest modules rxinfer_bridge and
  optional_adapters both raised).
- **Orchestration**: `scripts/run_full_verification.sh` runs every deterministic
  gate (pytest+coverage, ruff check, ruff format --check, mypy, 55-oracle
  validation) as one command with a pass/fail table and non-zero exit on any
  failure; `--julia` adds the backend `Pkg.test()`. Documented in README and
  docs/verification.md; syntax-checked by a new test_scripts_smoke test.
- **Manuscript**: 06_redteam_limits updated to reflect the mechanically-explained
  temporal result (CLM-014) and measured segmentation (CLM-015), with precise
  future work (categorical-HMM RxInfer backend, larger/ambiguous corpus,
  parameter learning).

## Research Advances (2026-07-31)

- **Ambiguity ablation (CLM-014)** — the head-to-head "temporal has no net
  advantage over memoryless" is now explained *mechanistically*, not left as a
  bare null. `ambiguity_ablation` sweeps the likelihood `A` toward uniform
  (holding corpus, `B`, `D` fixed): at the authored, near-diagonal likelihood
  there is no advantage (gap -0.06), but as observations become ambiguous the
  memoryless/sticky decoders collapse while the temporal forward filter retains
  accuracy (at blend 1.0: temporal 0.604 vs memoryless 0.229; crossover at 0.8).
  New figure `ambiguity_ablation.png`; validation check
  `advanced_ambiguity_ablation_shows_advantage` binds both directions.
- **Measured segmentation (CLM-015)** — the F3 segmentation module is now
  *measured* against ground truth (`evaluate_segmentation`): pooled boundary
  F1 0.64 and step accuracy 0.67 (per fixture in the evaluation report).
  Validation check `advanced_segmentation_measured`.
- Net effect: figures 20->21, oracle 52->55, claims 13->15 (CLM-014, CLM-015);
  suite 157->164, coverage 94.56%->94.68%.
- The two honest non-results remain, now explained rather than removed.

## Docs / Tests / Visualization Pass (2026-07-31)

- **Documentation**: added the missing `evaluation.md` entry to `docs/README.md`
  and expanded its index intro; completed the root `README.md` figure list with
  the per-fixture temporal-belief variants and the new calibration/confusion
  figures.
- **Tests**: `+9` real, no-mock tests. RxInfer free-energy-record validation
  (missing fields, non-increasing iterations, data-hash mismatch, empty trace,
  companion presence/round-trip via `validate_rxinfer_outputs`), reproducibility
  manifest tamper detection (missing artifact, hash + size mismatch), and
  bounded-emulation early stop. Suite 148 -> 157; coverage 93.75% -> 94.56%.
  Sub-90% modules raised: rxinfer_bridge 78->83%, reproducibility 87->92%,
  emulation 88->92%.
- **Visualization**: `model_comparison.png` now has a second panel showing pooled
  accuracy with 95% Wilson confidence intervals (visualizes the new uncertainty
  statistics); its caption was updated to describe both panels.
- **Captions**: model_comparison caption reworded; calibration/confusion captions
  already added in the previous pass.

## Scientific Upgrade (2026-07-31)

Adds measured uncertainty and two visualizations to the quantitative layer, all
deterministic and recomputed by the validation gate:

- **Statistics** (`src/evaluation.py`): `wilson_ci` (95% Wilson scores), pooled
  (micro) accuracies + `*_ci95` in the evaluation summary, pooled confusion
  matrix, pooled calibration bins, per-bin calibration data, shuffled-control
  `std`/`se`, and a `temporal_memoryless_diff`. Every new number is emitted as a
  manuscript variable and recomputed from disk by the freshness oracle.
- **Figures** (`src/figures.py`): `calibration_curve.png` (reliability diagram)
  and `confusion_matrix.png` (pooled annotated heatmap); captions added to the
  caption ledger; both registered in `MANIFEST_TARGETS` and the figure oracle.
  Figure count 18 -> 20; oracle count 50 -> 52.
- **Manuscript** (`manuscript/03_results.md`): reports pooled accuracy with its
  Wilson CI and the temporal-vs-memoryless difference, and references the two
  new figures via `{{CALIBRATION_CURVE_FIGURE}}` / `{{CONFUSION_MATRIX_FIGURE}}`.
- **Tests**: +12 (Wilson CI, calibration bins, pooled stats, figure generation);
  suite 136 -> 148. Coverage 93.53% -> 93.75%.