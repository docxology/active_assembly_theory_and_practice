# Limits, validity, and scientific roadmap {#sec:limits-roadmap}

The evidence supports a narrow methodological conclusion. A real assembly
trace can be decomposed into auditable static and bounded dynamic records,
projected into an explicit categorical observation channel, and evaluated with
generative and discriminative diagnostics whose assumptions are visible. The
package does not establish that the latent labels are intrinsic program
semantics. They are hand-authored phase hypotheses used to make the inference
contract measurable.

The most important validity boundary is between a decoder and a generative
model. A decoder can score well on a short labelled sequence because an
individual observation is highly informative. A generative model must also
produce plausible state persistence, transition structure, and observation
frequencies. The fixed-parameter generative adequacy analysis performs that
second check by simulating `{{GENERATIVE_ADEQUACY_REPLICATES}}` same-length
sequences per fixture. It tests `{{GENERATIVE_ADEQUACY_METRIC_COUNT}}` scalar
structural metrics; `{{GENERATIVE_ADEQUACY_INSIDE_COUNT}}` fall inside the
central simulation intervals, for an observed coverage fraction of
`{{GENERATIVE_ADEQUACY_INSIDE_FRACTION}}`. These are descriptive diagnostics
of the declared `A/B/D` model, not posterior uncertainty intervals or evidence
of population-level fit [@gelman2013bda]. The complete per-fixture record is
available in `{{GENERATIVE_ADEQUACY_ARTIFACT}}`.
This fixed-parameter categorical phase-HMM does not establish learned semantic
representations or external validity.

![{{FIGURE_GENERATIVE_ADEQUACY_CAPTION}}]({{FIGURE_GENERATIVE_ADEQUACY_PATH}}){#fig:generative-adequacy width=96%}

Several limitations remain substantive. The corpus contains six transparent
x86-64 programs, so complete fixtures rather than independent instruction steps
are the meaningful sampling units. The coverage inventory records one bounded
stack-memory fixture, repeated authored phases, and one taken-branch path that
leaves a statically decoded instruction unexecuted; these additions improve
design variation but do not create independent binary samples. The labels cover
only five broad phases and are assigned from the known fixture construction. The observation map collapses
flags, effective-address details, calling conventions, indirect targets, and
unsupported effects. The bounded symbolic interpreter therefore explains only
the documented transfer-function subset; agreement with Unicorn on those
fixtures is a cross-check, not a complete-ISA certification.

The supervised leave-one-fixture-out fit is a useful leakage diagnostic because
the excluded fixture does not contribute to its fitted tensors. It is not
unsupervised phase discovery, and six held-out evaluations cannot establish
generalization to arbitrary binaries. The likelihood and transition ablations
are controlled interventions on the declared model matrices. They explain the
mechanism by which temporal structure matters under ambiguity; they do not
measure a real-world noise distribution.

The observation-only ordered-EM tranche makes the next limitation measurable.
It excludes phase labels from fitting and reports a complete training complement
for every holdout, but it still inherits a reference initialization, an ordered
self/next-state scaffold, and a positive transition-support prior. Its held-out
mean is `{{PARAMETER_LEARNING_EM_MEAN_ACCURACY}}`, with a mean difference of
`{{PARAMETER_LEARNING_EM_MINUS_REFERENCE}}` relative to the fixed reference.
That result is an implementation and leakage-boundary advancement, not evidence
that latent phase meanings have been discovered. Future work must compare
multiple state-order constraints and deterministic initializations, report
training-versus-held-out evidence separately, and add longer programs with
ambiguous repeated phases before making any claim about unsupervised semantics.

The typed observation expansion addresses a different limitation: the original
six-category channel was lossy by construction. The new schema retains
`{{TYPED_FIELD_COUNT}}` fields in `{{TYPED_FIELD_GROUP_COUNT}}` evidence
families, and the factorized/hierarchical comparison is fit with the same
fixture-level exclusion boundary. Its mean full-factorized accuracy is
`{{FACTORIZED_MEAN_ACCURACY}}` versus `{{FACTORIZED_COLLAPSED_MEAN_ACCURACY}}`
for the collapsed reference; the difference is `{{FACTORIZED_MINUS_COLLAPSED}}`.
Thus the expanded typed result is a measured negative companion-model result,
not a null that can be rounded into a success claim: the current composite
score is lower than the collapsed reference on these held-out fixtures. The
masked-memory check changes the mean to `{{MISSING_FIELD_MEAN_ACCURACY}}`,
which is `{{MISSING_FIELD_MEAN_DELTA}}` relative to the full typed score. These
results are intentionally modest: most labelled fixtures execute every
instruction and contain no memory operand, so the new fields are not yet
independent sources of variation. The typed model is a supervised composite
likelihood diagnostic, not a semantic representation, and the hierarchical
score is not assumed to be a normalized joint probability.

The active layer has the same requirement, and the first action-conditioned
tranche now makes that boundary explicit. `action_sensing.json` chooses typed
field queries before revealing held-out values and compares active, fixed,
seeded-random, and inadmissible-oracle policies under one normalized budget.
This is a genuine sensor-choice mechanism test, but it remains a small
fixture-level result: the normalized action budget is not the primary measured
cost vector, local timing calibration is machine-specific, the oracle is not
deployable, and no autonomous sensing claim is made.

Three further tranches make the roadmap more rigorous without inflating the
claim. First, the formal receipt `{{FORMAL_VERIFICATION_ARTIFACT}}` reports a
locally checked modular Word64 fragment under toolchain `{{FORMAL_TOOLCHAIN}}`,
with `{{FORMAL_THEOREM_COUNT}}` named theorems and status
`{{FORMAL_VERIFICATION_STATUS}}`. The generated bridge also records all 16
general-purpose registers, `RIP`, modeled flag-definedness, bounded memory and
stack bounds, step index, and running status, with kernel-checked fixture
invariants for register preservation, carry preservation, step advancement, and
bound preservation. This earns a bounded kernel-checked state foundation for
quotient witnesses, but not a correspondence theorem between Lean and the
Python/Capstone/Unicorn trace. Second, the real adapter receipt
`{{CROSS_ARCHITECTURE_ARTIFACT}}` exercises `{{CROSS_ARCHITECTURE_COUNT}}`
architectures, all with bounded result `{{CROSS_ARCHITECTURE_RESULT}}` and
execution flag `{{CROSS_ARCHITECTURE_ALL_PASSED}}`. That is an integration and
adapter-drift benchmark, not cross-ISA semantic equivalence or compiler
invariance. Third, the exact finite sensor analysis
`{{ACTIVE_INFORMATION_ARTIFACT}}` contains `{{ACTIVE_INFORMATION_SENSOR_COUNT}}`
channels and `{{ACTIVE_INFORMATION_FRONTIER_ROW_COUNT}}` frontier rows, with
exhaustive enumeration recorded as `{{ACTIVE_INFORMATION_EXACT}}`. The
companion measured-cost receipt records deterministic sensor vectors and
optional local calibration; both analyses remain finite and therefore preclude
a general information-per-cost law.

The research program now proceeds through coupled measurement, formal,
identifiability, transfer, active-sensing, and replication expansions. The
typed, observation-only, action-conditioned, formal-fragment, adapter, and
finite-policy first tranches are implemented as local source-bound surfaces;
the correspondence theorem, source/compiler-disjoint external benchmark,
measured query costs, and independent replication remain active or unearned.
Each expansion must add an artifact, tests, negative controls, uncertainty
definitions, and manuscript language stating what the evidence cannot show.
This keeps the project ambitious in method while preventing a larger toolchain
or more elaborate figure from being mistaken for stronger scientific evidence.

## Long-horizon scientific program {#sec:scientific-program}

The machine-readable successor program is `{{SCIENTIFIC_PROGRAM_ARTIFACT}}`.
Of its `{{SCIENTIFIC_PROGRAM_MILESTONE_COUNT}}` milestones,
`{{SCIENTIFIC_PROGRAM_IMPLEMENTED_COUNT}}` are implemented only on the local
source-bound surface, `{{SCIENTIFIC_PROGRAM_ACTIVE_COUNT}}` is active, and
`{{SCIENTIFIC_PROGRAM_PLANNED_COUNT}}` remain unearned; these counts describe
program state, not scientific success.

Next, define an executable operational equivalence relation and separating
intervention family, then test quotient-state invariance under
semantics-preserving representations. Freeze source-family adjusted mutual
information, posterior equivalence-consistency, calibration, invariance gap,
and near-miss separation before scoring; source and compiler lineages must be
disjoint so a model cannot define its own semantic labels.

Promotion requires counterexamples, multiple initializations, strong baselines,
and independent reimplementation. A mechanically checked theorem or decisive
countertheorem is separate: benchmark performance cannot substitute for it.
The conjecture, falsifiers, gates, and stop conditions live in
`docs/scientific_program.md` and its generated receipt.

The first P3 prerequisite is the bounded equivalence/near-miss protocol
`{{QUOTIENT_PROTOCOL_ARTIFACT}}`: finite initial-state interventions run through
real Capstone and bounded Unicorn, requiring equivalent pairs to agree on final
`RAX` and a near miss to separate. It is an operational protocol, not evidence
of latent identifiability, posterior invariance, complete-ISA equivalence, or
cross-architecture transfer.

The protocol now retains a richer diagnostic surface for every intervention:
terminal `RAX`/`RIP`/`RSP`/`EFLAGS` projections, stepwise `RAX` and modeled flag
states, memory-event counts, observation categories, and a deterministic
surface digest. These projections are deliberately reported beside the
declared final-`RAX` relation. An equivalent pair may differ on representation
surface or intermediate state while remaining equivalent under the declared
output relation; conversely, a final-state relation cannot certify unmodeled
memory, flags, exceptions, or calling-convention behavior.

Benchmark receipt `{{REPRESENTATION_BENCHMARK_ARTIFACT}}`: `{{REPRESENTATION_BENCHMARK_FAMILY_COUNT}}`/`{{REPRESENTATION_BENCHMARK_PAIR_COUNT}}`
families/pairs; primary consistency/separation `{{REPRESENTATION_BENCHMARK_EQUIVALENCE_CONSISTENCY}}`/`{{REPRESENTATION_BENCHMARK_NEAR_MISS_SEPARATION}}`;
`{{REPRESENTATION_BENCHMARK_SURFACE_BLIND_NEAR_MISS_COUNT}}` surface-blind controls; negative-control separation
`{{REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_SEPARATION}}` (`{{REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_UNSEPARATED_COUNT}}` unresolved).
The accompanying benchmark plot reports the same endpoints with descriptive
Wilson intervals whose unit is the source-program family. The intervals make the
small number of independent families visible; they are not posterior or
population uncertainty, and the negative-control panel is an intervention
sensitivity check rather than a competing fitted model.

![{{FIGURE_REPRESENTATION_INVARIANCE_BENCHMARK_CAPTION}}]({{FIGURE_REPRESENTATION_INVARIANCE_BENCHMARK_PATH}}){#fig:representation-invariance-benchmark width=88%}

## Current promotion ledger for P3--P17 {#sec:promotion-ledger}

The open roadmap is now executable in bounded local tranches, but its
promotion state remains deliberately asymmetric:

| Phase | Earned local evidence | What remains unearned |
| --- | --- | --- |
| P3 quotient benchmark | 64-family source-disjoint quotient labels, three-way co-primary portfolio, clustered intervals, calibration, leakage controls, and negative controls | independent implementation and any claim of semantic discovery |
| P4 formal correspondence | explicit 16-register/width-aware Python state, real Capstone/Unicorn step comparison, and a separately kernel-checked Lean fragment | broad Lean/Python correspondence theorem for the full declared user-mode subset |
| P5 transfer | local compiler capability receipt with exact target/source/object hashes and explicit unavailable targets | compiler-disjoint, source-disjoint, project-held-out licensed-source effectiveness validation |
| P6 active sensing | deterministic vectors, overhead-corrected local calibration, regression diagnostics, and exact finite dynamic-program certificates | multi-hardware acquisition model and non-oracle near-optimal policy evidence |
| P7 replication | frozen no-network CLI, digest-ordered public rows, private gold separation, hash validation, replay metadata, and an observed same-host controlled guest run | independent implementation, independent physical/CI replay, and external scientific review |
| P8 theorem phase | proved small algebraic theorem group, automatically generated bounded counterexample, and a smallest-first exhaustive counterexample-search receipt over declared candidates | broad trace theorem, transfer guarantee, or field-changing general result |
| P9 intervention power | exhaustive finite subset enumeration for the separate four-state representation pilot, minimum separating sets, and a one-state negative control | universal identifiability, intervention completeness, or a separation law outside the frozen matrix |
| P10 initialization robustness | five-start observation-only holdout ledger with accuracy spread and sequence agreement | stable latent-parameter recovery or semantic discovery beyond the finite fixtures |
| P11 normalized machine state | all-GPR, width-aware, bounded-memory reference state with boundary tests and Lean normalized-state definitions | complete cross-language proof for the declared broad subset |
| P12 scaled transfer/quotient | 64-family quotient receipt, source/compiler/architecture capability matrix, and protocol-bound splits | external validity, universal invariance, or superiority without adjusted evidence |
| P13 licensed external validation | ten pinned rows across five projects and two revisions, static-only acquisition/compiler path, Capstone decoding, 100 source-bound contract rows, and project-held-out labeling | application-facing effectiveness until source-bound label, project-held-out, build, and external acquisition/build replay gates pass |
| P14 measured policy robustness | deterministic/calibrated cost receipts, finite oracle certificate, stress-grid policy regret and budget-value gaps, and repeated local calibration-stability rows | universal cost law or non-oracle near-optimality claim beyond the declared grid |
| P15 blind replication | public/private label separation, hashed public protocol/toolchain/instructions, protected ingestion boundary, and `{{INDEPENDENT_REPLAY_STATUS}}` in `{{INDEPENDENT_REPLAY_CONTROL_SCOPE}}` | independent implementation or independent physical/CI replay; the observed same-host guest is not independent reimplementation |
| P16 promotion hardening | theorem/countertheorem status map, source-bound claim/figure/scholarship closure, and accessible regeneration path | publication or award-level claim absent independent scientific review |
| P17 protocol/artifact closure | protocol-v2 envelope on the project-owned JSON surface, atomic producer writes, receipt audit, and source-bound manuscript/document closure | semantic identifiability, external validity, independent replication, or theorem promotion |

The theorem/countertheorem receipt is
`{{THEOREM_COUNTERTHEOREM_ARTIFACT}}`. Its fail-closed status is
`{{THEOREM_COUNTERTHEOREM_STATUS}}`; the counterexample family is
`{{THEOREM_COUNTEREXAMPLE_FAMILY}}`. The correct scientific response to that
counterexample is to strengthen the intervention contract and state the
separating assumptions, not to call the failed single-state claim “almost
true.” The local formal proof count is `{{THEOREM_PROVED_COUNT}}`, while
`{{THEOREM_COMPUTATIONAL_ONLY_COUNT}}` result(s) remain computationally
validated rather than kernel proved.

Two bounded falsification-and-robustness receipts extend this ledger. The
counterexample search (`{{COUNTEREXAMPLE_SEARCH_ARTIFACT}}`) resolves
`{{COUNTEREXAMPLE_SEARCH_CANDIDATE_COUNT}}` declared candidates with
`{{COUNTEREXAMPLE_SEARCH_FALSIFIED_COUNT}}` falsification(s); each holds row is
bounded to its enumerated domain and is not promoted to an unrestricted
identity. The sensing-stress receipt (`{{SENSING_STRESS_ARTIFACT}}`) scores
policy controls against exact references on a declared grid and repeats the
local timing calibration; its regret and drift rows are machine-bounded
diagnostics, not cross-machine laws.

Application-facing promotion remains closed: the licensed-source receipt is
`{{EXTERNAL_SOURCE_STATUS}}` with `{{EXTERNAL_SOURCE_PROJECT_COUNT}}` project
families and `{{EXTERNAL_SOURCE_LABELABLE_EXAMPLE_COUNT}}` labelable examples,
the correspondence is `{{FORMAL_CORRESPONDENCE_STATUS}}`, and the replication
bundle reports local label separation `{{REPLICATION_LABEL_SEPARATED}}` plus a
same-host controlled replay status of `{{INDEPENDENT_REPLAY_STATUS}}`. The
licensed-source external gate remains `{{EXTERNAL_SOURCE_STATUS}}` because its
independent acquisition/build replay requirement is not satisfied. Until
licensed provenance, project-held-out labels, static-only safety, reproducible
builds, and the external-validation replay gate pass together, this manuscript uses “demonstrator,”
“bounded endpoint,” and “capability receipt” rather than “general
reverse-engineering system,” “architecture invariant,” or “application
effectiveness.” This rule protects scientific integrity.

## Generated figure atlas {#sec:figure-atlas}

The complete visual inventory is generated alongside the manuscript rather
than maintained as a second hand-edited appendix. The source-bound registry
is `{{FIGURE_REGISTRY_ARTIFACT}}` and the offline accessible atlas is
`{{FIGURE_ATLAS_ARTIFACT}}`. Together they retain all {{SUBSTANTIVE_FIGURE_COUNT}} substantive
figures,
including diagnostics that are too dense for the main narrative. Each entry
records its source artifact hashes, evidence status, population grain, split,
uncertainty method, visible caption, short alternative text, and long
description. The cover remains a separate document asset under the existing
cover-media contract.

The atlas is intentionally a scientific audit surface. It makes unavailable,
unsupported, excluded, underpowered, and not-yet-observed states visible;
provides exact tables for dense matrices where a chart alone would be
ambiguous; and keeps deterministic enumerations separate from inferential
intervals. It should therefore be read together with the claim register and
the corresponding receipt, not as a collection of independent results. The
canonical command is:

```bash
uv run python scripts/run_scientific_program.py generate-figures \
  --formats png svg pdf atlas
```

The renderer hydrates figure captions and paths through the same variable
registry used for the rest of the manuscript. No rendered caption, figure
number, or accessibility string is hand-edited after hydration.
