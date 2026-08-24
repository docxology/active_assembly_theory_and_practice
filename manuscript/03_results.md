# Results {#sec:results}

The canonical fixture produces `{{INSTRUCTION_COUNT}}` decoded instructions and a
bounded Unicorn trace with final `RAX = {{FINAL_RAX}}`. The inference stream
contains `{{FREE_ENERGY_STEPS}}` finite pymdp comparison diagnostics, one per
observation. The native RxInfer run records `{{RXINFER_FREE_ENERGY_COUNT}}`
finite Bethe Free Energy values with final value
`{{RXINFER_FINAL_FREE_ENERGY}}` on Julia `{{JULIA_VERSION}}` and RxInfer
`{{RXINFER_VERSION}}`. The final most likely pymdp latent class is
`{{FINAL_STATE}}`, reflecting the terminal observation history rather than an
externally asserted semantic label.

The native package also executes the categorical phase-HMM companion
`{{RXINFER_CATEGORICAL_MODEL_NAME}}`, recording `{{RXINFER_CATEGORICAL_FREE_ENERGY_COUNT}}`
finite iterations and final Bethe Free Energy `{{RXINFER_CATEGORICAL_FINAL_FREE_ENERGY}}`
in `{{RXINFER_CATEGORICAL_ARTIFACT}}`. This is a separate, fixed-parameter
factor graph; its free-energy scale is not compared with the Gaussian spine.

The independent bounded-semantics check supports the trace interpretation at
the concrete level: {{SYMBOLIC_SUPPORTED_COUNT}} decoded instructions are
covered, {{SYMBOLIC_UNSUPPORTED_COUNT}} executed instructions fall outside the
declared subset, and the symbolic terminal RAX agrees with Unicorn
({{SYMBOLIC_FINAL_RAX_MATCH}}). The static control-flow projection contains
{{CFG_BLOCK_COUNT}} basic blocks and {{CFG_EDGE_COUNT}} edges. These are
audited fixture facts, not a claim that a general x86 binary has been lifted or
understood.

The stepwise comparison agrees for {{SYMBOLIC_STEPWISE_RAX_AGREEMENT_COUNT}} of
{{SYMBOLIC_STEPWISE_RAX_COMPARABLE_COUNT}} executed steps, while the explicitly
modeled flag bits agree for {{SYMBOLIC_STEPWISE_FLAG_AGREEMENT_COUNT}} of
{{SYMBOLIC_STEPWISE_FLAG_COMPARISON_COUNT}} comparisons. This tighter result is
still a local instrument check: it prevents a later operation from hiding an
intermediate transfer-function error, but it does not establish complete
x86-64 semantics or semantic understanding.

The corresponding trace-refinement visualization is reported in
@fig:trace-refinement; it is kept as a single registry-owned figure so that
the manuscript, HTML, and tagged-PDF outputs cannot assign competing labels to
the same source-bound image.

The new formal and adapter receipts extend the evidence boundary without
changing that interpretation. The Lean fragment is locally verified at status
`{{FORMAL_VERIFICATION_STATUS}}`, while the cross-architecture runtime fixture
set reports `{{CROSS_ARCHITECTURE_COUNT}}` architectures, a common bounded
result of `{{CROSS_ARCHITECTURE_RESULT}}`, and an all-passed execution flag of
`{{CROSS_ARCHITECTURE_ALL_PASSED}}`. These are independent software and
algebra checks on small declared surfaces. They do not turn the hand-authored
phase labels into semantic states, and they do not establish x86/AArch64/RISC-V
equivalence, compiler invariance, or external validity.

The memory instrumentation test is independently positive: the static
`x86_memory` fixture contains `{{MEMORY_STATIC_INSTRUCTION_COUNT}}` memory
instructions, and bounded Unicorn emits `{{MEMORY_DYNAMIC_EVENT_COUNT}}`
events comprising `{{MEMORY_WRITE_COUNT}}` write and `{{MEMORY_READ_COUNT}}`
read at `{{MEMORY_DISTINCT_ADDRESS_COUNT}}` distinct effective address(es).
The observed store-to-load value and terminal result are retained in
`{{MEMORY_EVIDENCE_ARTIFACT}}`; this is evidence that the instrumentation path
functions on the transparent fixture, not a claim of general memory analysis.

![Memoryless posterior beliefs over the `{{HIDDEN_STATE_COUNT}}` hand-authored latent phase labels at each of the `{{INSTRUCTION_COUNT}}` decoded steps. Each line is a normalized probability from the fixed comparison path; the figure shows the belief update and does not supply independent phase ground truth.]({{POSTERIOR_FIGURE}}){#fig:posterior-beliefs width=88%}

![Bethe Free Energy values emitted by the native Julia/RxInfer backend for the implemented Gaussian event-code model. Points are backend iterations from the canonical trace bundle; this objective is distinct from the pymdp comparison diagnostic and is not a semantic-understanding score.]({{RXINFER_FREE_ENERGY_FIGURE}}){#fig:rxinfer-free-energy width=88%}

![Two inference diagnostics kept on separate axes: the pymdp comparison variational diagnostic above and native RxInfer Bethe Free Energy below. The traces share input lineage but are distinct objectives and are not numerically interchangeable.]({{INFERENCE_DIAGNOSTIC_FIGURE}}){#fig:inference-diagnostics width=92%}

The exact active-information analysis provides a second, mechanism-level
result. The declared finite sensor family contains
`{{ACTIVE_INFORMATION_SENSOR_COUNT}}` typed-field channels and
`{{ACTIVE_INFORMATION_FRONTIER_ROW_COUNT}}` horizon/cost rows; the receipt
marks exhaustive branch enumeration as `{{ACTIVE_INFORMATION_EXACT}}` in
`{{ACTIVE_INFORMATION_ARTIFACT}}`. This is stronger than reporting one greedy
policy score because the bounded policy space is solved exactly, but it remains
weaker than an information-per-cost law: sensor costs are proxy units, the
channel is model-derived, and the horizon and sensor family are finite.

These results are useful as a working contract: every number in this manuscript
is generated by the project code, and every figure—from the observation
pipeline, temporal beliefs, model comparison, active-disassembly EFE, and
inference diagnostics to the scholarship, claim, generative-adequacy,
reproducibility, and validation panels—is written to `output/figures/` before
rendering. The generated caption ledger (`output/figures/README.md`) and figure
registry (`output/data/figure_registry.json`) document each figure's inputs,
statistics, citations, and accessibility metadata.

## Measured phase decoding {#sec:measured-decoding}

We compare three decoders that share the *same* likelihood matrix and prior over
`{{HIDDEN_STATE_COUNT}}` phase classes (chance accuracy `{{CHANCE_ACCURACY}}`): the **identity-transition** filter (an
implicit identity transition that carries the previous posterior forward), a
**memoryless** per-step decoder (`argmax_s D[s]·A[obs,s]`, the strongest honest
non-temporal baseline), and the current temporal HMM **forward filter** (the
pre-declared headline decoder; Viterbi and smoothing are reported as secondary).
Across the `{{CORPUS_FIXTURE_COUNT}}`-fixture labelled corpus the mean decoding
accuracies are `{{IDENTITY_TRANSITION_MEAN_ACCURACY}}` (identity-transition), `{{MEMORYLESS_MEAN_ACCURACY}}`
(memoryless), and `{{TEMPORAL_MEAN_ACCURACY}}` (temporal), over
`{{CORPUS_OBSERVATION_COUNT}}` labelled observation steps.

Two honest conclusions follow, neither of which is the naive "temporal wins."
First, the temporal HMM substantially beats the *identity-transition* filter — adding a
left-to-right transition prior fixes the pathology where a peaked prior keeps the
identity-transition filter stuck on `setup`. Second, and importantly, the *memoryless* decoder
beats the temporal HMM on this corpus: temporal structure earns no net advantage
here (temporal minus memoryless = `{{TEMPORAL_MEMORYLESS_DIFF}}`), because the
programs are short enough that each observation alone is strongly informative
and the terminal `exit` phase is observation-underdetermined (it shares
categories with earlier phases). Across the pooled corpus the temporal decoder
recovers `{{POOLED_TEMPORAL_ACCURACY}}` of phases (descriptive Wilson
`{{CI_LEVEL_PERCENT}}%` score interval
`{{TEMPORAL_ACC_CI}}`) versus `{{POOLED_MEMORYLESS_ACCURACY}}` (CI
`{{MEMORYLESS_ACC_CI}}`) for the memoryless baseline and the pooled
identity-transition baseline (CI `{{IDENTITY_TRANSITION_ACC_CI}}`). The right reading is
conditional — temporal modelling helps relative to an identity-transition filter, but a
memoryless decoder is sufficient until programs are long or ambiguous enough
that carrying belief forward pays off. The headline comparison uses
hand-authored parameters and therefore measures model-structure value; the
separate leakage-safe fit below is the only result presented as a
generalization diagnostic. The
accuracy metric is discriminative rather than green-by-construction: a
shuffled-label control scores `{{SHUFFLE_CONTROL_ACCURACY}}` and a uniform-belief
control `{{UNIFORM_CONTROL_ACCURACY}}`, both near chance. Per-fixture accuracy,
calibration, free-energy convergence, and the full `{{DECODER_COUNT}}`-decoder comparison
are written to the evaluation report and recomputed from disk by the validation
gate. The Wilson intervals are descriptive per-step intervals, not independent
fixture-level or population-level uncertainty estimates: steps within each
fixture are dependent and the corpus contains only hand-authored toy programs.

### Leakage-safe held-out estimation {#sec:held-out-estimation}

The fixed-parameter comparison is complemented by a supervised estimation
experiment with an explicit exclusion boundary. For each fixture, the smoothed
categorical HMM estimates its emission matrix A, transition matrix B, and
initial prior D from the other {{CORPUS_FIXTURE_COUNT}} - 1 fixtures only.
The excluded fixture is then decoded without any parameter update. Across the
six held-out runs, mean temporal accuracy is {{HELD_OUT_MEAN_ACCURACY}}
(range {{HELD_OUT_ACCURACY_MIN}}--{{HELD_OUT_ACCURACY_MAX}}) versus
{{HELD_OUT_FIXED_ACCURACY}} for the fixed full-corpus reference. The lower
held-out score is not a failure of the code; it is evidence that the tiny
fixture set and phase-label distribution do not support a strong
generalization claim. The exact training/held-out membership and fitted
probability tensors are in {{HELD_OUT_MODEL_ARTIFACT}}. No claim of
population performance or unsupervised phase discovery is made.

### Observation-only ordered-HMM learning {#sec:parameter-learning-results}

The next research tranche removes the need to use phase labels while estimating
the categorical parameters. The leave-one-complete-fixture-out experiment fits
an ordered Baum--Welch model on the observation IDs from
`{{PARAMETER_LEARNING_HOLDOUT_COUNT}}` complementary training sets, with a cap of
`{{PARAMETER_LEARNING_MAX_ITERATIONS}}` iterations per fit. The leakage contract
is `{{PARAMETER_LEARNING_LEAKAGE_FREE}}`: labels appear only in the held-out
scoring rows, not in the fit input. `{{PARAMETER_LEARNING_CONVERGED_HOLDOUT_COUNT}}`
of the fits reached the parameter-delta tolerance before the cap.

The observation-only EM fit reaches mean held-out accuracy
`{{PARAMETER_LEARNING_EM_MEAN_ACCURACY}}`, compared with
`{{PARAMETER_LEARNING_SUPERVISED_MEAN_ACCURACY}}` for supervised MAP fitting and
`{{PARAMETER_LEARNING_REFERENCE_MEAN_ACCURACY}}` for the fixed reference. Its
mean difference is `{{PARAMETER_LEARNING_EM_MINUS_SUPERVISED}}` against supervised
MAP and `{{PARAMETER_LEARNING_EM_MINUS_REFERENCE}}` against the fixed reference.
The negative EM delta is the scientifically important outcome here: the
observation-only estimator is real, reproducible, and leakage-safe, but it is
not yet a competitive phase decoder on this short, highly constrained corpus.
The ordered scaffold and reference initialization stabilize the latent-state
coordinate system, while also limiting what can be called independent discovery.
Training evidence histories and per-fixture predictions are retained in
`{{PARAMETER_LEARNING_ARTIFACT}}`.

The multi-start extension makes the remaining optimization ambiguity measurable.
It retains `{{PARAMETER_LEARNING_MULTI_START_COUNT}}` fits across the five
declared starts for each holdout. Pooled across those fits, mean held-out
accuracy is `{{PARAMETER_LEARNING_MULTI_START_MEAN_ACCURACY}}`, the mean
within-holdout accuracy range is `{{PARAMETER_LEARNING_MULTI_START_MEAN_RANGE}}`,
and mean pairwise predicted-sequence agreement is
`{{PARAMETER_LEARNING_MULTI_START_MEAN_AGREEMENT}}`. These values are seed
sensitivity summaries, not confidence intervals or a new sampling unit. Their
spread is a limitation: the existence of a functioning observation-only EM
implementation does not establish stable latent parameters, and no start is
selected post hoc using the held-out labels.

The fixed-penalty Dirichlet-MAP comparator and posterior-predictive checks add a
second robustness axis without changing the primary endpoint. The comparator is
fit from the same label-free training sequences, while the predictive check
reports observation-level structure rather than phase-label recovery. The
receipt's selection ledger records training-evidence decisions only; it does
not turn a favorable predictive interval into evidence of semantic adequacy. If
the ordered model remains initialization-sensitive, that instability is
retained as a negative result.

![Observation-only ordered-HMM parameter learning under complete-fixture holdouts. The left panel shows per-fixture held-out temporal accuracy for ordered Baum--Welch EM, supervised MAP, the fixed reference, and the fixed-penalty Dirichlet-MAP comparator; lines are not pooled pseudo-replicates. The middle panel shows training log evidence across EM iterations for each holdout, with the mean trajectory overlaid. The right panel shows held-out accuracy for the five frozen initialization families, with dots for holdouts and bars for within-start means. The receipt additionally contains label-free posterior-predictive checks and a training-only model-selection ledger. Labels are reserved for scoring, and the evidence, predictive checks, and initialization panels are optimization/robustness diagnostics rather than population uncertainty intervals.]({{PARAMETER_LEARNING_FIGURE}}){#fig:parameter-learning width=100%}

### Typed observation expansion {#sec:typed-results}

The first roadmap expansion is now executable rather than aspirational. The
source-bound typed schema retains `{{TYPED_FIELD_COUNT}}` fields across
`{{TYPED_FIELD_GROUP_COUNT}}` evidence families, with `{{TYPED_STATIC_FIELD_COUNT}}`
static fields and one explicit dynamic-status field. The records are derived
from Capstone detail and bounded Unicorn addresses, while the hand-authored
phase sequence is consumed only by the supervised training/evaluation layer.
The typed record artifact is `{{TYPED_OBSERVATION_ARTIFACT}}`; its leakage guard
reports `{{TYPED_LEAKAGE_FREE}}` across `{{TYPED_FIXTURE_COUNT}}` fixtures. The
comparison reports `{{FACTORIZED_ABLATION_GROUP_COUNT}}` field-family ablations,
so the result is not being inferred from a single arbitrary feature drop.

The held-out comparison is deliberately more informative than a claim of
improvement. The collapsed categorical reference reaches mean accuracy
`{{FACTORIZED_COLLAPSED_MEAN_ACCURACY}}`; dropping dynamic status but retaining
the static typed fields reaches `{{FACTORIZED_STATIC_MEAN_ACCURACY}}`; the full
factorized composite reaches `{{FACTORIZED_MEAN_ACCURACY}}`; and the balanced
hierarchical score reaches `{{HIERARCHICAL_MEAN_ACCURACY}}`. The full typed score
therefore differs from the collapsed reference by only
`{{FACTORIZED_MINUS_COLLAPSED}}` and from the static typed ablation by
`{{FACTORIZED_MINUS_STATIC}}`. The experiment is behaving as a useful
measurement, including a negative result: the additional fields do not
automatically create signal merely because the schema is larger. On the
expanded corpus, the full field-wise score is `{{FACTORIZED_MINUS_COLLAPSED}}`
relative to the collapsed reference, and the balanced hierarchical score is
not promoted as a replacement. This is not evidence that typed observations
are unhelpful in general; it is evidence that this particular composite
scoring rule is not yet adequate for the available fixture diversity. The
initial register-dominated fixtures remain near-duplicate in some fields,
while the expanded corpus adds one memory and one taken-path contrast. The
field-group ablation and explicit masked-memory sensitivity are reported in
[@fig:factorized-observation-comparison], with every training membership
listed in `{{FACTORIZED_OBSERVATION_ARTIFACT}}`.

This result sharpens the scientific next step. A typed schema prevents
information loss, but it cannot manufacture independent samples or justify
conditional independence. The expanded corpus has a bounded memory-mode
contrast and a static/dynamic path contrast, while still lacking the breadth
needed for external validity. Masking the memory field changes mean accuracy to
`{{MISSING_FIELD_MEAN_ACCURACY}}` (`{{MISSING_FIELD_MEAN_DELTA}}` versus full
factorization), which is a sensitivity diagnostic rather than a causal effect
estimate. The result supports continuing to expand the fixture design with
longer repeated phases, multiple memory addressing modes, and additional
branch outcomes before attempting larger models; it does not support selecting
the hierarchical score as a preferred model. The source-bound design inventory
is reported in `{{CORPUS_COVERAGE_ARTIFACT}}` and [@fig:corpus-coverage].

### Statistical reporting contract {#sec:statistical-reporting}

For decoder $m$, the primary accuracy is the pooled hit proportion

$$
\widehat{p}_m = \frac{H_m}{N},
$$ {#eq:pooled-accuracy}

where $H_m$ is the number of correctly decoded labelled steps and
$N={{CORPUS_OBSERVATION_COUNT}}$ is the common pooled step denominator. We
use pooled accuracy for the headline comparison and report fixture-level means
separately; neither should be read as an independent sample of binaries. The
reported interval is the descriptive Wilson score interval

$$
\frac{\widehat{p}_m + z^2/(2N) \pm z\sqrt{\widehat{p}_m(1-\widehat{p}_m)/N + z^2/(4N^2)}}{1+z^2/N},
$$ {#eq:wilson-interval}

with $z$ set by the configured `{{CI_LEVEL_PERCENT}}%` confidence level
[@brown2001binomial].
Calibration is reported with equal-width reliability bins, expected
calibration error, and Brier score; the bin counts remain visible in
[@fig:calibration-curve]. We do not attach p-values or claim external
uncertainty because the corpus is hand-labelled, the headline comparison uses
fixed parameters, and instruction steps within a fixture are correlated. The statistical
purpose is auditability of a mechanism demonstration, not a population-level
benchmark.

![Reliability diagram for the temporal forward filter over `{{CORPUS_OBSERVATION_COUNT}}` labelled steps. The plot shows `{{CALIBRATION_NONEMPTY_BIN_COUNT}}` non-empty bins from `{{CALIBRATION_BIN_COUNT}}` configured equal-width confidence bins; each point plots mean maximum posterior against observed accuracy, `n` gives the bin denominator, vertical bars are 95% Wilson intervals around bin hit counts, and the dashed diagonal is perfect calibration. Points below the diagonal indicate overconfidence for this fixture corpus [@guo2017calibration;@brown2001binomial].]({{CALIBRATION_CURVE_FIGURE}}){#fig:calibration-curve width=82%}

Calibration is also compared across decoders. Reporting each decoder's own
per-step posterior ([@fig:decoder-calibration]) shows ECE tracking accuracy
on this corpus: the `{{DECODER_BEST_CALIBRATED}}` per-step decoder is the best calibrated (ECE
`{{MEMORYLESS_ECE}}`), the temporal model is middle (`{{TEMPORAL_ECE}}`), and the
identity-transition filter is worst (`{{IDENTITY_TRANSITION_ECE}}`). This is stated honestly: the
temporal model is neither the most accurate nor the best calibrated.

![Decoder calibration on the labelled corpus. Separate panels report expected calibration error and Brier score for memoryless, temporal, and identity-transition decoders; both are dimensionless and lower is better. The ECE ordering identifies `{{DECODER_BEST_CALIBRATED}}` as best for these hand-authored traces, not as a deployment-reliability guarantee.]({{DECODER_CALIBRATION_FIGURE}}){#fig:decoder-calibration width=88%}

![Pooled phase-decoding confusion matrix for the temporal forward filter across the labelled fixtures. Each cell shows the integer count followed by the percentage of its true-phase row; diagonal cells are correct recoveries and off-diagonal mass identifies confusable phase labels.]({{CONFUSION_MATRIX_FIGURE}}){#fig:confusion-matrix width=78%}

The decoding results above are classification metrics. A complementary,
principled *generative* comparison — the log marginal likelihood of the observed
traces under each model ([@def:model-evidence]; [@fig:model-evidence]) — shows
`{{BEST_MODEL}}` is the best *explainer* of the data: its pooled evidence
(`{{TEMPORAL_LOG_EVIDENCE}}`)
is higher than the identity-transition (`{{IDENTITY_TRANSITION_LOG_EVIDENCE}}`) and memoryless
(`{{MEMORYLESS_LOG_EVIDENCE}}`) alternatives, a margin of
`{{TEMPORAL_EVIDENCE_MARGIN}}` over its best alternative. So the temporal model
best explains the observed sequences even though the memoryless decoder is the
better per-step classifier: the two comparison criteria (discriminative accuracy
vs generative evidence) do not pick the same model, and we report both honestly.

![Pooled log marginal likelihood under temporal-HMM, identity-transition, and memoryless assumptions. Higher values indicate better in-sample predictive normalization for these fixed parameters; the highlighted model is `{{BEST_MODEL}}`. This is a descriptive model-evidence comparison in the Bayes-factor tradition [@kass1995bayes], not a population-level model-selection claim.]({{MODEL_EVIDENCE_FIGURE}}){#fig:model-evidence width=82%}

## Where temporal dynamics matter {#sec:ambiguity-ablation}

The head-to-head result — temporal beats the identity-transition filter but not the memoryless
decoder — has a precise, measured mechanism. The authored likelihood is nearly
diagonal: each observation category peaks sharply on one phase, so a single
observation already reveals the phase and transition structure adds little. To
surface this, we blend the likelihood toward uniform (observation ambiguity),
holding the corpus, transition matrix, and prior fixed. As ambiguity rises,
per-step decoding collapses — at blend `{{AMBIGUITY_HIGH_BLEND}}` the memoryless
decoder falls to `{{AMBIGUITY_HIGH_MEMORYLESS_ACC}}`, while the temporal forward
filter, using the left-to-right transition structure, retains
`{{AMBIGUITY_HIGH_TEMPORAL_ACC}}`. The temporal advantage over the memoryless
decoder first emerges at blend `{{AMBIGUITY_CROSSOVER_BLEND}}`. The result is
honest in both directions: with the authored, informative likelihood there is no
net temporal advantage, and a positive advantage appears exactly when
observations become ambiguous. This is a mechanism demonstration on the same
transparent fixtures (only the likelihood conditioning `A` is swept; `B` and `D`
are fixed), not a claim about real-world observation noise.

The same mechanism appears in the *segmentation* dimension (middle panel of
[@fig:ambiguity-ablation]): at the authored likelihood the memoryless
segmenter already recovers phase boundaries well (boundary F1
`{{SEG_MEMORYLESS_F1}}`), but under full ambiguity its boundary recovery
collapses to `{{SEG_HIGH_MEMORYLESS_F1}}` while the smoothed temporal
segmentation retains a boundary F1 of `{{SEG_HIGH_TEMPORAL_F1}}`. Temporal
structure rescues segmentation exactly when per-step decoding cannot.

The *active* layer behaves the same way (right panel of
[@fig:ambiguity-ablation]): the realized-information-gain EFE heuristic reaches the confidence
threshold in fewer steps than a fixed scan on `{{ACTIVE_ADV_MAX_FRACTION}}` of
fixtures, within an intermediate-ambiguity band (`{{ACTIVE_ADV_BAND_MIN}}` to
`{{ACTIVE_ADV_BAND_MAX}}`), and the machine-checked never-worse result is
`{{ACTIVE_NEVER_WORSE}}` (see `{{ACTIVE_SELECTION_ARTIFACT}}`). The EFE ordering
carries real value only when observations are *differentially* informative —
neither when they are all equally informative (the authored likelihood) nor when
none are (fully uniform).

The same artifact records a predictive-information policy for comparison. On
the selected fixture it reaches the threshold in
{{PREDICTIVE_ACTIVE_STEPS}} steps under the predictive category contribution;
this is reported as a policy diagnostic, not as a corpus-scale gain. The
one-step expected information of the authored observation channel is
{{EXPECTED_OBSERVATION_INFO_GAIN}} nats, and the upper panel of
[@fig:active-efe] keeps the realized and predictive EFE traces visibly
separate.

![Three linked ablations on the same `{{CORPUS_FIXTURE_COUNT}}` transparent fixtures. Left: mean per-step decoder accuracy as the fixed categorical likelihood is blended toward uniform. Middle: pooled phase-boundary F1 for smoothed temporal and memoryless segmentations. Right: fixtures in which the realized-EFE policy reaches its confidence threshold sooner than a fixed scan. Blend 0 is authored and blend 1 is uniform, so this is a mechanism demonstration under synthetic ambiguity rather than a measurement of real-world binary noise.]({{AMBIGUITY_ABLATION_FIGURE}}){#fig:ambiguity-ablation width=100%}

## How well we segment phases {#sec:segmentation-evaluation}

Segmentation is not only self-consistent but measured. Decoding the smoothed
forward-backward beliefs into per-step phases and comparing the detected phase
boundaries against the hand-authored ground truth yields a pooled boundary
F1 of `{{SEGMENTATION_BOUNDARY_F1}}` and a per-step accuracy of
`{{SEGMENTATION_STEP_ACC}}`, available per fixture in `{{PHASE_SEGMENTS_ARTIFACT}}`
and the evaluation report (`{{EVALUATION_REPORT_ARTIFACT}}`). The temporal
filter and smoothing outputs are retained in `{{TEMPORAL_INFERENCE_ARTIFACT}}`.
[@fig:segmentation-ribbons] makes this visual: for each fixture the top
ribbon is the ground-truth phase sequence and the bottom ribbon is the sequence
recovered from the smoothed beliefs, so a mis-segmentation shows up as a
boundary mismatch between the two rows.

![Per-fixture phase-segmentation ribbons. For each fixture, the upper row is the hand-authored phase sequence and the lower row is the sequence decoded from smoothed forward-backward beliefs. Contiguous runs share a color, so a boundary disagreement is a displaced color transition; the pooled boundary F1 is reported in the figure subtitle.]({{SEGMENTATION_RIBBONS_FIGURE}}){#fig:segmentation-ribbons width=100%}

A second ablation isolates the contribution of the transition structure itself
([@fig:transition-ablation]; [@prop:transition-collapse]): as the transition matrix `B` is blended toward
a uniform matrix, the temporal decoder converges *exactly* to the per-step
memoryless decoder — with `B @ posterior` uniform, the forward filter no longer
carries state history — reaching equivalence by blend `{{TRANSITION_CONVERGENCE_BLEND}}`.
The entire temporal-vs-memoryless difference is therefore attributable to the
transition structure `B`. On this small corpus the peak mean accuracy occurs at a
mild blend (`{{TRANSITION_PEAK_ACC}}` at blend `{{TRANSITION_PEAK_BLEND}}`), an
honest reminder that brute accuracy and the authored transition structure need
not coincide.

![Mean temporal accuracy as transition matrix `B` is blended toward uniform, with constant memoryless and identity-transition references. At the uniform endpoint, `B` removes state persistence and the temporal update converges to memoryless, isolating the transition contribution; the marked peak records the corpus-specific accuracy optimum.]({{TRANSITION_ABLATION_FIGURE}}){#fig:transition-ablation width=88%}

## Which phases are recoverable {#sec:phase-recovery}

Collapsing the pooled confusion matrix into per-phase metrics quantifies the
recovery limitation that the manuscript cites qualitatively: the `setup` and
`accumulator_update` phases are recovered at recall `{{SETUP_RECALL}}` and
`{{ACCUMULATOR_RECALL}}`, the `representation_transform` phase has F1
`{{REPRESENTATION_F1}}`, while the `exit` phase (F1 `{{EXIT_F1}}`) and — hardest
of all — the `guard_check` phase (F1 `{{GUARD_F1}}`) are
observation-underdetermined. The macro F1 is `{{PHASE_MACRO_F1}}` and the hardest
phase to recover is `{{HARDEST_PHASE}}`. These are forward-filter decoding limits
of the demonstrator, not claims about a complete binary-analysis system.

Likelihood information is uneven ([@fig:observation-informativeness]):
`{{MOST_INFORMATIVE_OBS}}` is most informative and `{{LEAST_INFORMATIVE_OBS}}`
least, with an information range of `{{OBS_INFO_RANGE}}` nats.

![Realized information gain for each decoded-instruction category over prior `D`, sorted from largest to smallest and colored by the phase receiving the largest posterior contribution. The companion predictive contribution weights each category by its prior predictive probability, and the weighted contributions sum to {{EXPECTED_OBSERVATION_INFO_GAIN}} nats of one-step channel information.]({{OBSERVATION_INFO_FIGURE}}){#fig:observation-informativeness width=92%}

![Per-phase recall and F1 for the temporal forward filter, ordered by F1. Bars report recovery of the `{{HIDDEN_STATE_COUNT}}` hand-authored phase labels; the hardest phase and macro F1 are stated in the figure subtitle. These are corpus-specific label-recovery metrics, not evidence of universal phase discovery.]({{PHASE_RECOVERY_FIGURE}}){#fig:phase-recovery width=88%}

## Robustness to the small corpus {#sec:corpus-stability}

Because the corpus has only `{{CORPUS_FIXTURE_COUNT}}` fixtures, pooled
accuracies could in principle be driven by a single fixture. A leave-one-out
analysis holds out each fixture in turn and recomputes the pooled accuracies over
the remaining corpus: the pooled temporal accuracy ranges from
`{{CORPUS_TEMPORAL_MIN}}` to `{{CORPUS_TEMPORAL_MAX}}` (span
`{{CORPUS_TEMPORAL_SPAN}}`), stays above the pooled static (identity-transition) accuracy
under every single-fixture removal, and remains well above the `{{CHANCE_ACCURACY}}`
chance floor.
This makes the small-corpus fragility explicit rather than hidden: the headline
temporal result is not the artifact of any one fixture, and the span quantifies
how much a single fixture moves the estimate.

![Leave-one-fixture-out stability analysis. Bars show pooled temporal accuracy on the full corpus and after removing each fixture; the dashed line is the pooled identity-transition reference and the dotted line is the configured chance floor. The span in the title is the observed range across holdouts, making small-corpus sensitivity explicit.]({{CORPUS_STABILITY_FIGURE}}){#fig:corpus-stability width=95%}

![Phase-decoder accuracy across the `{{CORPUS_FIXTURE_COUNT}}`-fixture labelled corpus. The left panel shows fixture-level identity-transition, memoryless, and temporal accuracy; the right panel shows pooled per-step accuracy with descriptive Wilson intervals over `{{CORPUS_OBSERVATION_COUNT}}` labelled steps. The dashed reference is chance (`{{CHANCE_ACCURACY}}`); dependent steps and the small corpus make the intervals descriptive rather than population-level uncertainty.]({{MODEL_COMPARISON_FIGURE}}){#fig:model-comparison width=100%}

The active-disassembly policy is reported honestly as a non-result at corpus
scale. On programs this short the first observation already pushes the phase
belief past the confidence threshold, so the expected-free-energy ordering ties
the fixed left-to-right scan on all `{{ACTIVE_FIXED_TIE_COUNT}}` fixtures while
never underperforming it. The active *mechanism* — that an informative
observation placed late in the natural order is selected earlier — is verified
in the unit tests; the absence of a corpus-scale advantage is recorded as a
measured fact rather than hidden. The complete evaluation surface is regenerated
from `{{EVALUATION_REPORT_ARTIFACT}}` before template hydration.

![Greedy active-disassembly trajectory for the `{{SAMPLE_NAME}}` fixture. The upper panel compares decoded-category EFE with the predictive information-contribution EFE; the lower panel reports posterior entropy on a separate axis. Lower EFE is preferred, and the corpus-level result is a tie on the short fixtures rather than an asserted active-policy gain.]({{ACTIVE_EFE_FIGURE}}){#fig:active-efe width=88%}

## Action-conditioned sensing

The action-conditioned experiment changes the evidence contract in a bounded
way: the policy chooses a typed field query before the held-out field value is
revealed. Across the six fixture holdouts, the active policy uses a mean of
`{{ACTION_SENSING_ACTIVE_MEAN_ACTIONS}}` actions under the common censored
budget, compared with `{{ACTION_SENSING_FIXED_MEAN_ACTIONS}}` for source-order
fixed sensing. The active-minus-fixed contrast is
`{{ACTION_SENSING_ACTIVE_MINUS_FIXED_ACTIONS}}` actions, so this tranche does
not assert a universal active-policy budget advantage. The seeded-random
baseline reaches the threshold on `{{ACTION_SENSING_RANDOM_CONFIDENCE_RATE}}`
of holdouts; the inadmissible oracle is shown only as an upper-bound reference.
The action experiment is leakage-free by construction and by artifact check
(`{{ACTION_SENSING_LEAKAGE_FREE}}`): each fitted field model excludes the
displayed fixture, and predictive ranking receives no realized field value.
The complete policy records and revealed values are available in
`{{ACTION_SENSING_ARTIFACT}}`.

![Held-out action-conditioned sensing over typed assembly fields. The left panel reports the mean number of actions used before the configured confidence threshold, retaining failures at the common budget; the right panel reports mean final posterior confidence. Active choices use predictive training-only field likelihoods, fixed and seeded-random policies use the same candidate pool, and the oracle is explicitly inadmissible. Equal normalized action cost is a mechanism contract, not a wall-clock measurement.]({{ACTION_SENSING_FIGURE}}){#fig:action-sensing width=95%}

## Open-roadmap evidence tranche {#sec:open-roadmap-results}

The new co-primary quotient endpoint is a source-family-held-out portfolio, not
a post-hoc winner selection. The benchmark contains
`{{QUOTIENT_CLASSIFIER_FAMILY_COUNT}}` families and
`{{QUOTIENT_CLASSIFIER_ROW_COUNT}}` pair rows under the training/development/
held-out split. The interpretable classifier reaches balanced accuracy
`{{QUOTIENT_CLASSIFIER_INTERPRETABLE_BALANCED_ACCURACY}}` and macro-F1
`{{QUOTIENT_CLASSIFIER_INTERPRETABLE_MACRO_F1}}`; the temporal probabilistic
classifier reaches `{{QUOTIENT_CLASSIFIER_TEMPORAL_BALANCED_ACCURACY}}` and
`{{QUOTIENT_CLASSIFIER_TEMPORAL_MACRO_F1}}`; the fixed neural sequence model
reaches `{{QUOTIENT_CLASSIFIER_NEURAL_BALANCED_ACCURACY}}` and
`{{QUOTIENT_CLASSIFIER_NEURAL_MACRO_F1}}`; and the typed-trace surface baseline reaches
`{{QUOTIENT_CLASSIFIER_BASELINE_BALANCED_ACCURACY}}` and
`{{QUOTIENT_CLASSIFIER_BASELINE_MACRO_F1}}`. Clustered intervals, Brier/ECE,
confusion counts, and false-equivalence/false-distinction rates are retained in
the source receipt. The receipt also fits
`{{QUOTIENT_CLASSIFIER_RAW_BASELINE_COUNT}}` independent raw surface baselines
and records `{{QUOTIENT_CLASSIFIER_RAW_COMPARISON_COUNT}}` paired model-by-baseline
comparison rows under one Holm family. With
`{{QUOTIENT_CLASSIFIER_HELD_OUT_FAMILY_COUNT}}` held-out families and development scores
that need not track test scores, these results do not establish superiority or
learned semantics. They establish a reproducible, label-source-controlled
endpoint with visible negative evidence, including a neural model that does not
recover the endpoint reliably.

The intervention-power analysis is a separate identifiability diagnostic rather
than another model score. Exact enumeration of
`{{INTERVENTION_POWER_SUBSET_COUNT}}` non-empty subsets over
`{{INTERVENTION_POWER_PAIR_COUNT}}` pair outcomes finds a minimum fully
separating cardinality of `{{INTERVENTION_POWER_MINIMAL_CARDINALITY}}`, with
`{{INTERVENTION_POWER_MINIMAL_SET_COUNT}}` tied minimum sets. The one-state
negative control separates `{{INTERVENTION_POWER_SINGLE_STATE_SEPARATION}}` of
near-misses and leaves `{{INTERVENTION_POWER_SINGLE_STATE_UNRESOLVED_COUNT}}`
families unresolved. This is an exact result for the declared finite outcome
matrix, not evidence that the intervention catalog is complete or that semantic
identifiability holds outside the benchmark. The source-bound receipt is
`{{INTERVENTION_POWER_ARTIFACT}}` and the subset curve is shown in
[@fig:intervention-power].

The real transfer and correspondence results are similarly bounded. The
Python/Capstone/Unicorn receipt reports status
`{{FORMAL_CORRESPONDENCE_STATUS}}` for
`{{FORMAL_CORRESPONDENCE_FIXTURE_COUNT}}` transparent fixtures and
`{{FORMAL_CORRESPONDENCE_STEP_COUNT}}` compared reference steps. The compiler
receipt validates `{{COMPILER_TRANSFER_VALIDATED_COUNT}}` source-by-optimization
static rows and retains `{{COMPILER_TRANSFER_UNAVAILABLE_COUNT}}` unavailable
rows; validated
architecture labels are limited to `{{COMPILER_TRANSFER_ARCHITECTURES}}`.
Because emitted objects were not executed and the external-source manifest is
`{{COMPILER_EXTERNAL_STATUS}}`, no cross-compiler, application, or compiler-
correctness claim follows. The figures make those boundaries visible rather
than collapsing unavailable capability into a zero.

The independent external-source receipt is `{{EXTERNAL_SOURCE_STATUS}}`: it
records `{{EXTERNAL_SOURCE_PROJECT_COUNT}}` manifested project families and
`{{EXTERNAL_SOURCE_LABELABLE_EXAMPLE_COUNT}}` labelable examples. The empty
manifest keeps this track blocked; it is not a failed accuracy result and does
not support application-facing effectiveness.

![Licensed-source provenance and application gate. The left panel shows the pinned manifest, explicit acquisition, static-only compilation, and disassembly path; the right panel lists the five promotion requirements and preserves the current `blocked_external` status. No unavailable project or labelable example is rendered as a zero-performance cell, and no arbitrary external binary is executed.]({{EXTERNAL_SOURCE_PROVENANCE_FIGURE}}){#fig:external-source-provenance width=96%}

The measured sensing receipt reports `{{SENSOR_COST_SENSOR_COUNT}}` deterministic
sensor vectors and `{{SENSOR_CERTIFICATE_COUNT}}` exact finite certificates.
The maximum optimality gap is `{{SENSOR_CERTIFICATE_MAX_GAP}}`, but this is a
certificate for the enumerated finite state/action space because the evaluated
policy is the dynamic-programming oracle itself. Local wall-clock calibration
is included for auditability and is explicitly machine-specific. It does not
replace the deterministic cost vector or imply a cross-machine speed law.

The replication bundle has `{{REPLICATION_PUBLIC_COUNT}}` public rows and
`{{REPLICATION_GOLD_COUNT}}` separated gold rows, with label separation
`{{REPLICATION_LABEL_SEPARATED}}`. The current status is a frozen local,
no-network harness; independent replay is not inferred from the existence of a
CLI. A changed protocol/source hash invalidates the receipt, preserving the
distinction between rerunning the same code and independently implementing the
method.

## Theorem-or-countertheorem outcome {#sec:theorem-countertheorem}

The strongest candidate tested in this tranche was intentionally vulnerable:
that a single zero-state intervention would separate every declared near-miss
family. The automatically generated counterexample is
`{{THEOREM_COUNTEREXAMPLE_FAMILY}}`: under `initial_rax=0`, the near-miss pair
has equal terminal output, while another declared intervention separates it.
The result is recorded in `{{THEOREM_COUNTERTHEOREM_ARTIFACT}}` with promotion
status `{{THEOREM_COUNTERTHEOREM_STATUS}}`. This is a useful scientific result:
the intervention family has separating power only when it contains states that
activate the relevant semantic difference. The single-state claim is
falsified, not softened into a theorem-shaped narrative.

The receipt separates `{{THEOREM_PROVED_COUNT}}` kernel-checked theorem group(s)
from `{{THEOREM_COMPUTATIONAL_ONLY_COUNT}}` computationally validated
finite-policy result(s); broad trace-refinement, compiler-transfer, and
independent-replay claims remain unresolved. No broader theorem or complete
x86-64 claim is promoted.
