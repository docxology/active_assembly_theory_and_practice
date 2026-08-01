# Quantitative Evaluation (v0.2)

The v0.2 layer turns the labelled corpus into the project's first *measured*
numbers. All metrics are computed from the live modules and recomputed from disk
by the validation gate; nothing is hand-stamped.

## Corpus

`src/corpus.py` defines four transparent x86-64 byte fixtures
(`x86_accumulator`, `x86_bitwise`, `x86_guard`, `x86_long_accumulator`). Each
fixture carries a hand-authored ground-truth phase-label sequence aligned 1:1 to
its decoded instructions over the five hidden states (`setup`,
`accumulator_update`, `representation_transform`, `guard_check`, `exit`). The
fixtures are decoded with real Capstone and emulated with real Unicorn — no
mocks.

## Decoders compared

| Decoder | Definition |
| --- | --- |
| `static` (sticky) | The v0.1 sticky filter with an implicit identity transition: `posterior_t ∝ posterior_{t-1} · A[obs_t]`. |
| `memoryless` | A per-step MAP decoder `argmax_s D[s]·A[obs_t,s]` that resets to the prior each step — the strongest *honest* non-temporal baseline. |
| `temporal` | The MAP labels of the HMM forward filter (`B` predict + likelihood update). |
| `viterbi` | The most-probable joint phase path. |

## Metrics

- **Decoding accuracy / balanced accuracy / confusion** — `evaluate_decoding`.
- **Calibration** — Brier score and Expected Calibration Error (10 bins on the
  max-posterior confidence), `calibration`.
- **Free-energy convergence** — monotone-decrease fraction and final−initial
  delta of the per-step variational free energy, `free_energy_convergence`.
- **Model comparison** — per-fixture static vs temporal vs Viterbi accuracy,
  `compare_models`.

## Negative controls (anti-vacuity)

A metric that cannot fail proves nothing. Two deterministic (RNG-free) controls
keep the headline honest:

- **Shuffled-label control** — temporal predictions scored against rotated
  ground truth; must sit near chance, far below the true-label accuracy.
- **Uniform-belief control** — a constant single-label predictor; the temporal
  decoder must strictly beat its balanced accuracy.

## Headline result (corrected after cross-vendor audit)

A cross-vendor (GPT-5.4) audit caught an over-claim in an earlier draft and the
narrative was corrected. The honest result is two-sided:

1. The temporal HMM substantially beats the v0.1 **sticky** filter — the sticky
   filter is pathological because a peaked prior keeps it stuck on `setup`.
2. A **memoryless** per-step decoder beats the temporal HMM on this corpus, so
   temporal structure earns **no net advantage** here. The programs are short
   enough that each observation is individually informative, and the terminal
   `exit` phase is observation-underdetermined.

So the defensible claim is the narrow one — "a left-to-right HMM outperforms a
sticky identity-transition filter" — not "temporal beats static" in general. The
validation check `advanced_memoryless_baseline_honest` binds the memoryless
baseline into the artifact so the comparison can never silently drop it. The
exact numbers are emitted as manuscript variables (`STATIC_MEAN_ACCURACY`,
`MEMORYLESS_MEAN_ACCURACY`, `TEMPORAL_MEAN_ACCURACY`, `SHUFFLE_CONTROL_ACCURACY`,
`UNIFORM_CONTROL_ACCURACY`) and written to
`output/data/evaluation_report.json`. Demonstrating a *positive* temporal
advantage requires longer or more ambiguous fixtures and is future work.

## Active selection — reported as a non-result at corpus scale

The expected-free-energy active-disassembly policy (`src/active_selection.py`)
is implemented and its *mechanism* is unit-verified (an informative observation
placed late in the natural order is selected earlier). On the current
short-program corpus, however, the first observation already saturates the phase
belief, so the active ordering **ties** the fixed left-to-right scan while never
underperforming it. This non-result is recorded explicitly in
`output/data/active_selection.json`, asserted by the
`advanced_active_never_worse` validation check, and bounded in claim CLM-013 —
it is not laundered into a corpus-scale advantage.

## Where temporal dynamics matter (ambiguity ablation)

`ambiguity_ablation` blends the authored likelihood `A` toward uniform (0 =
authored likelihood, 1 = fully ambiguous) while holding the corpus, transition
matrix `B`, and prior `D` fixed, and recomputes every decoder at each blend.
Because `A` is near-diagonal (each observation peaks sharply on one phase),
per-step decoding is already near-saturated, so the temporal model adds no net
advantage at blend 0. As observations become ambiguous, the memoryless and
sticky decoders collapse while the temporal forward filter — using the
left-to-right transition structure — retains accuracy. The crossover blend
(first value where temporal beats memoryless) is recorded, so the head-to-head
"no net advantage" result is explained *mechanistically* rather than left as a
bare null. It is visualized as `output/figures/ambiguity_ablation.png` and bound
by claim CLM-014. The ablation changes only the likelihood conditioning; it is a
mechanism demonstration on the same transparent fixtures, not a claim about
real-world observation noise.

## Measured phase segmentation

`segmentation_evaluation` decodes the smoothed forward-backward beliefs into
per-step phases and compares the detected phase boundaries against the
hand-authored ground truth, reporting boundary precision/recall/F1 and step
accuracy per fixture and pooled. This makes the F3 segmentation module
*measured*, not merely self-consistent (claim CLM-015).

## Validation binding

| Artifact | Claim | Validation check |
| --- | --- | --- |
| `evaluation_report.json` | CLM-010, CLM-011 | `advanced_temporal_beats_static`, `advanced_negative_controls_discriminate`, `advanced_evaluation_freshness` |
| `temporal_inference.json` | CLM-010 | `advanced_temporal_artifacts_present` |
| `phase_segments.json` | CLM-012 | `advanced_phase_segmentation_partition` |
| `active_selection.json` | CLM-013 | `advanced_active_never_worse` |
| `evaluation_report.json` (ambiguity_ablation) | CLM-014 | `advanced_ambiguity_ablation_shows_advantage` |
| `evaluation_report.json` (segmentation) | CLM-015 | `advanced_segmentation_measured` |

The `advanced_evaluation_freshness` check re-derives the headline summary from
the live generator and compares it to the on-disk artifact, so a stale or
hand-edited number fails the gate.
