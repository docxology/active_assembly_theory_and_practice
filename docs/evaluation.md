# Quantitative Evaluation (current)

The current layer turns the labelled corpus into the project's first *measured*
numbers. All metrics are computed from the live modules and recomputed from disk
by the validation gate; nothing is hand-stamped.

## Corpus

`src/corpus.py` defines six transparent x86-64 byte fixtures
(`x86_accumulator`, `x86_bitwise`, `x86_guard`, `x86_long_accumulator`,
`x86_memory`, and `x86_guard_taken`). Each fixture carries a hand-authored
phase-label sequence aligned 1:1 to its decoded instructions over the five
hidden states (`setup`, `accumulator_update`, `representation_transform`,
`guard_check`, `exit`). The fixtures are decoded with real Capstone and
emulated with real Unicorn — no mocks. The final two fixtures add bounded
stack-memory operands and a taken branch that skips a statically decoded
instruction, so the typed-model evaluation has observable static/dynamic
variation.

## Decoders compared

| Decoder | Definition |
| --- | --- |
| `identity-transition` | An explicit identity-transition comparator with `B = I`: `posterior_t ∝ posterior_{t-1} · A[obs_t]`. |
| `memoryless` | A per-step MAP decoder `argmax_s D[s]·A[obs_t,s]` that resets to the prior each step — the strongest *honest* non-temporal baseline. |
| `temporal` | The MAP labels of the HMM forward filter (`B` predict + likelihood update). |
| `viterbi` | The most-probable joint phase path. |

## Metrics

- **Decoding accuracy / balanced accuracy / confusion** — `evaluate_decoding`.
- **Calibration** — Brier score and Expected Calibration Error (the fixed
  `CALIBRATION_BIN_COUNT` equal-width bins on max-posterior confidence),
  `calibration`. The reliability diagram follows the calibration framing of
  Guo et al. (2017); it is descriptive for this labelled fixture corpus.
- **Free-energy convergence** — monotone-decrease fraction and final−initial
  delta of the per-step variational free energy, `free_energy_convergence`.
- **Model comparison** — per-fixture identity-transition vs temporal vs Viterbi accuracy,
  `compare_models`.
- **Leakage-safe estimation** — `fit_temporal_model` estimates categorical
  emission, transition, and initial-state probabilities with positive
  Dirichlet smoothing. `leave_one_fixture_out_evaluation` fits on five
  fixtures and scores the sixth, repeating for every fixture; the training and
  held-out names are written into `held_out_model.json`.
- **Typed observation expansion** — `src.typed_observations` retains static
  Capstone fields and bounded dynamic status without phase labels, while
  `src.factorized_model` compares collapsed, static typed, factorized, and
  group-balanced hierarchical composite scores under the same fixture-level
  holdout boundary. `src.corpus_coverage` separately records whether the
  fixture design exercises memory modes, repeated phases, and alternate paths.
- **Observation-only parameter learning** — `src.parameter_learning` performs
  scaled Baum--Welch EM using only training-fixture observation IDs. A soft
  self/next-state support prior keeps the ordered scaffold inspectable while
  positive floors preserve the log-space model contract. Every complete-fixture
  holdout is scored against supervised MAP and the fixed reference; phase labels
  are explicit scoring-only data. The implementation follows the incomplete-data
  EM formulation [@dempster1977em] and standard HMM recursions
  [@rabiner1989tutorial].

## Negative controls (anti-vacuity)

A metric that cannot fail proves nothing. Two deterministic controls keep the
headline honest:

- **Shuffled-label control** — temporal predictions scored against distinct full
  permutations of the ground truth, drawn from an explicit seeded local RNG;
  must sit near chance, far below the true-label accuracy.
- **Uniform-belief control** — a constant single-label predictor; the temporal
  decoder must strictly beat its balanced accuracy.

The controls are reproducible because the permutation seed is explicit; the
implementation is not RNG-free and does not consume ambient/global RNG state.

## Headline result

The source-bound result is two-sided:

1. The temporal HMM substantially beats the **identity-transition** filter — the identity-transition
   filter is pathological because a peaked prior keeps it stuck on `setup`.
2. A **memoryless** per-step decoder beats the temporal HMM on this corpus, so
   temporal structure earns **no net advantage** here. The programs are short
   enough that each observation is individually informative, and the terminal
   `exit` phase is observation-underdetermined.

So the defensible claim is the narrow one — “a left-to-right HMM outperforms an
identity-transition filter on these fixtures” — not “temporal dynamics always
improve decoding.” The
validation check `advanced_memoryless_baseline_honest` binds the memoryless
baseline into the artifact so the comparison can never silently drop it. The
exact numbers are emitted as manuscript variables (`IDENTITY_TRANSITION_MEAN_ACCURACY`,
`MEMORYLESS_MEAN_ACCURACY`, `TEMPORAL_MEAN_ACCURACY`, `SHUFFLE_CONTROL_ACCURACY`,
`UNIFORM_CONTROL_ACCURACY`) and written to
`output/data/evaluation_report.json`. Wilson intervals in the report are
descriptive per-step score intervals; within-fixture observations are dependent
and the toy corpus is not an inferential sample of binaries. Demonstrating a *positive* temporal
advantage requires longer or more ambiguous fixtures and is future work.

## Generative adequacy

The decoder scores are complemented by a fixed-parameter simulation check. For
each fixture, `src.evaluation.generative_adequacy` draws 256 same-length latent
state/observation sequences from the declared `A`, `B`, and `D` tensors. It
compares the observed phase-transition rate, run count, phase occupancy, and
temporal recovery accuracy with central 95% simulation intervals. The generated
`output/data/generative_adequacy.json` artifact stores the seed, replicate count,
all per-fixture statistics, interval-membership indicators, and interpretation.
The figure `output/figures/generative_adequacy.png` shows the two scalar metrics
most directly relevant to the model's temporal behavior.

This check is intentionally not called a posterior predictive check: no
parameters are fitted and no posterior is sampled. It is a descriptive
fixed-model adequacy diagnostic in the model-checking sense [@gelman2013bda].
Interval coverage is not a p-value, a confidence statement about a population
of binaries, or evidence that the hand-authored phase labels are true semantics.

## Typed and factorized observations

The Phase A observation expansion is intentionally adjacent to, rather than a
replacement for, the six-category model. `src.typed_observations` encodes
instruction family, operand class, register read/write sets, flag effect,
memory mode, branch-target class, and dynamic status from the real Capstone and
Unicorn records. The encoder does not accept a phase label, and the generated
record carries an empty `phase_label_keys` declaration as a machine-readable
leakage boundary.

`src.factorized_model` fits one smoothed categorical factor per retained field.
The training support is built from the training fixtures only; an absent field
maps to `__MISSING__`, and a held-out value outside that support maps to
`__UNKNOWN__`. The factorized score sums field-wise log probabilities. The
hierarchical companion averages within the configured instruction-semantics,
register-context, control-flow, and dynamic-context groups before normalizing
the state scores for HMM decoding. The latter is a composite score, not a
claim of a normalized joint distribution.

The generated report includes six complete-fixture held-out rows, training
membership, fitted supports, removal of each field family, and a deliberate
masked-memory run. A null or negative typed-model delta remains scientifically
useful: the expanded corpus adds one memory fixture and one alternate dynamic
path, but does not create independent samples of binaries. The package
therefore does not select the most elaborate score by default; it records the
comparison and its design coverage before any richer model is promoted.

## Fixture-design coverage

The source-bound coverage inventory is written to
`output/data/corpus_coverage.json` and plotted in
`output/figures/corpus_coverage.png`. It reports static decoded length,
bounded executed length, contiguous authored phase runs, memory addressing
modes, direct branch-target classes, and typed instruction-family support per
fixture. The artifact documents memory, repeated-phase, and static/dynamic
path variation, but it does not turn six hand-authored programs into
independent samples of binaries.

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

The active module now reports two related epistemic quantities. The existing
decoded-category score uses the entropy reduction for the category already
revealed. The predictive companion multiplies that reduction by the category's
prior predictive probability; summing those contributions over the observation
vocabulary equals the one-step mutual information of the categorical channel.
This is a proper expected-information diagnostic while remaining honest about
the current candidate contract: candidate instruction categories are decoded
before selection, so the project does not claim a hidden sensor model for
instruction acquisition.

## Action-conditioned sensing

`src.action_sensing` advances the active layer to a bounded sensor-choice
experiment. An action identifies one decoded instruction and one typed field;
the policy scores the candidate from a training-only field likelihood, chooses
it, and only then receives the held-out field value. This ordering is enforced
in the implementation by separating predictive scoring (`observation=None`)
from post-choice revelation. The experiment uses the same fixture-level
leave-one-out boundary as the typed model and records the training complement
for every row.

The generated `output/data/action_sensing.json` compares four policies under a
common action budget and equal normalized action cost: predictive active
selection, source-order fixed selection, a seeded-random permutation, and an
explicitly inadmissible oracle that is allowed to inspect realized values
before selecting. The oracle is an upper-bound reference, not a deployable
method. `output/figures/action_sensing.png` retains policies that fail to reach
the threshold at the budget rather than dropping them from the summary. The
current result is a mechanism test: active choices change the acquired typed
evidence and remain leakage-free, while their budget advantage is not assumed
and is reported against the fixed and random baselines.

Costs in this action-conditioned experiment are deliberately normalized action
units, not runtime or analyst-time measurements. A separate
`output/data/sensor_cost_receipt.json` records the deterministic eight-component
cost vector and optional local wall-clock calibration, so the normalized action
experiment and measured-cost receipt are not conflated. The experiment therefore satisfies the roadmap's action-change
and no-peeking requirements for a transparent demonstrator, but it does not
establish autonomous sensing over arbitrary binaries or real acquisition-cost
optimality.

## Observation-only parameter learning

`output/data/parameter_learning.json` is a separate estimation artifact, not a
replacement for the hand-authored reference model. For each held-out fixture,
the fit receives only the observation-category ID sequences from the other
fixtures. The held-out phase labels are retained under an explicit
`truth_labels_for_scoring_only` field and are never passed to the EM routine.
The artifact records the ordered-state support, positive transition floor,
Dirichlet concentrations, scaled log-evidence history, parameter deltas, and
the fitted `A`, `B`, and `D` tensors. `output/figures/parameter_learning.png`
shows the per-fixture held-out accuracy comparison for ordered EM, supervised
MAP, the fixed reference, and a fixed-penalty Dirichlet-MAP comparator, together
with training evidence and five-start sensitivity. The receipt also contains
label-free posterior-predictive checks and a training-only model-selection
ledger; none of these diagnostics must be read as held-out performance or
population uncertainty intervals.

The current method is intentionally conservative. The reference initialization
and ordered self/next-state scaffold address latent-state label switching, but
they also prevent a claim of unconstrained unsupervised phase discovery. The
current corpus result is therefore allowed to be negative: the observation-only
fit is a functional research method and a leakage-boundary test, while its
held-out decoder accuracy need not exceed supervised or fixed comparators.

## Bounded instruction semantics and CFG evidence

The static trace now includes Capstone register read/write sets, direct-target
basic-block partitions, and CFG edges. The small transfer-function interpreter
in `src/symbolic_semantics.py` supports the immediate register/shift/compare
subset exercised by the transparent fixtures. Its concrete pre/post `RAX`
values are compared at every dynamically executed step with independent
Unicorn snapshots, and the subset of `EFLAGS` bits that the interpreter
defines (`CF`, `ZF`, `SF`, and `OF` when defined by the operation) is compared
against post-step hardware flags. The terminal `RAX` check remains as a useful
summary, but it is no longer sufficient to pass the semantics contract: a
stepwise mismatch fails even if a later operation happens to restore the final
value. The refinement figure visualizes the exact overlays and the modeled-bit
counts. This is a checked explanation layer for a declared instruction subset,
not a complete x86 lifter, symbolic solver, alias analysis, or decompiler
[@intel_sdm;@dasgupta2019x86semantics;@unicorn].

The direct-target CFG evidence is generated separately for `x86_guard`, where
the conditional branch points outside the decoded fixture span. The artifact
keeps the static blocks, edge labels, external endpoint, and bounded dynamic
addresses together so a reader can inspect control-flow scope without treating
the diagram as whole-binary reachability.

## Formal verification receipt

The formal tranche is intentionally a separate evidence class from executable
trace agreement. `formal/ActiveAssembly/Semantics.lean` defines a modular
Word64 accumulator semantics and proves a finite manifest of equivalent,
neutral, and separating programs. `scripts/run_formal_verification.py` invokes
the pinned Lean toolchain and writes `output/data/formal_verification.json` with
the source hashes, theorem names, build command, and trusted base. The
validation gate rebuilds the project and compares the deterministic receipt;
the formal status therefore cannot be cleared by editing a JSON file. The
generated `formal/ActiveAssembly/GeneratedCorrespondence.lean` bridge checks
one shared Python/Lean Word64 boundary fixture. The receipt still does not
establish a general correspondence theorem between Lean and Capstone/Unicorn,
and it does not cover flags, memory, exceptions, ABI behavior, compiler
correctness, or the x86 ISA as a whole.

## Cross-architecture adapter benchmark

`output/data/cross_architecture_fixtures.json` is a real API benchmark over
three transparent fixtures. Each row must decode exactly three instructions
with Capstone, execute exactly two non-returning instructions with Unicorn,
and reach the declared accumulator result. The figure
`output/figures/cross_architecture_benchmark.png` shows both static/executed
coverage and the endpoint result. The benchmark is useful as an adapter drift
check and software-integration falsifier. It is not an external-validity study:
the fixtures are one hand-authored operation per ISA and do not test compiler
families, optimization, memory, flags, exceptions, calling conventions, or
semantics-preserving source transformations.

## Exact information-per-cost frontier

`src.information_policy` solves a finite decision problem exactly by enumerating
all positive-probability sensor outcomes, recursively evaluating the remaining
sensor set, and memoizing normalized beliefs. The receipt
`output/data/active_information_frontier.json` records the field channels,
finite horizons, cost weights, policy counts, expected entropy, information
gain, and objective. The plot `output/figures/active_information_frontier.png`
is consequently stronger than a single greedy-policy trace within the stated
finite mechanism. Its sensor likelihoods come from the label-blind factorized
model and its costs are unit model-channel costs. The separate measured-cost
receipt records deterministic vectors and local calibration; neither artifact
is a measured cross-machine runtime law,
an infinite-horizon optimum, an autonomous binary-analysis result, or a
theorem about all active sensing systems.

## Where temporal dynamics matter (ambiguity ablation)

`ambiguity_ablation` blends the authored likelihood `A` toward uniform (0 =
authored likelihood, 1 = fully ambiguous) while holding the corpus, transition
matrix `B`, and prior `D` fixed, and recomputes every decoder at each blend.
Because `A` is near-diagonal (each observation peaks sharply on one phase),
per-step decoding is already near-saturated, so the temporal model adds no net
advantage at blend 0. As observations become ambiguous, the memoryless and
identity-transition decoders collapse while the temporal forward filter — using the
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
| `evaluation_report.json` | CLM-010, CLM-011 | `advanced_temporal_beats_identity_transition`, `advanced_negative_controls_discriminate`, `advanced_evaluation_freshness` |
| `temporal_inference.json` | CLM-010 | `advanced_temporal_artifacts_present` |
| `phase_segments.json` | CLM-012 | `advanced_phase_segmentation_partition` |
| `active_selection.json` | CLM-013 | `advanced_active_never_worse` |
| `held_out_model.json` | CLM-018 | `held_out_model_contract`, `held_out_model_freshness` |
| `symbolic_execution.json` and `trace_refinement.png` | CLM-019 | `symbolic_semantics_contract`, `trace_symbolic_facts_contract`, `symbolic_stepwise_trace_refinement`, `figure_trace_refinement` |
| `cfg_evidence.json` and `assembly_cfg.png` | CLM-001, CLM-019 | `cfg_evidence_contract`, `cfg_evidence_freshness`, `figure_assembly_cfg` |
| `memory_evidence.json` and `assembly_memory_timeline.png` | CLM-001, CLM-002, CLM-019 | `memory_evidence_contract`, `memory_evidence_freshness`, `figure_assembly_memory_timeline` |
| `evaluation_report.json` (ambiguity_ablation) | CLM-014 | `advanced_ambiguity_ablation_shows_advantage` |
| `evaluation_report.json` (segmentation) | CLM-015 | `advanced_segmentation_measured` |
| `generative_adequacy.json` and `generative_adequacy.png` | CLM-020 | `generative_adequacy_contract`, `generative_adequacy_freshness`, `figure_generative_adequacy` |
| `typed_observations.json` and `trace_bundle.json` | CLM-021 | `trace_typed_observation_alignment`, `typed_observation_schema_contract`, `typed_observation_schema_freshness` |
| `factorized_observation_model.json` | CLM-021 | `factorized_observation_contract`, `factorized_observation_freshness`, `factorized_observation_leakage_guard` |
| `typed_observation_matrix.png` and `factorized_observation_comparison.png` | CLM-021 | `figure_typed_observation_matrix`, `figure_factorized_observation_comparison` |
| `action_sensing.json` and `action_sensing.png` | CLM-023 | `action_sensing_contract`, `action_sensing_freshness`, `action_sensing_leakage_guard`, `figure_action_sensing` |
| `parameter_learning.json` and `parameter_learning.png` | CLM-024 | `parameter_learning_contract`, `parameter_learning_freshness`, `parameter_learning_leakage_guard`, `figure_parameter_learning` |

The `advanced_evaluation_freshness` check re-derives the headline summary from
the live generator and compares it to the on-disk artifact, so a stale or
hand-edited number fails the gate.
