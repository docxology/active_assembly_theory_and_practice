# Scientific roadmap

This project studies a sharply defined interface: real assembly observations are
made explicit, encoded as a categorical channel, and evaluated under a
generative model whose assumptions can be inspected. The roadmap expands that
interface in stages. A feature becomes part of the implemented surface only
when it has a source-bound artifact, a deterministic regeneration path, a
regression test, a figure or table when it carries a scientific result, a
manuscript explanation, and an explicit claim boundary.

For the longer-horizon research program, see
[`scientific_program.md`](scientific_program.md). That program defines the
central conjecture, formal-theory ladder, quotient-state identifiability target,
cross-architecture benchmark, active-sensing bound, independent replication
gate, and explicit stop conditions. Its status is also emitted as the
source-bound artifact `output/data/scientific_program.json`. P0--P2 remain
implemented instrumentation/mechanism milestones. P3--P16 now each have a
source-bound implementation or explicit blocked/planned receipt, but their
broad promotion claims remain gated: P3 has a 64-family classifier benchmark
and a separate four-state representation pilot, P4 has no full-ISA theorem,
P5 has no licensed external-source evaluation, P6 is finite-benchmark
optimality, P7 has no independent replay, and P8 has no broad theorem. A
bounded receipt is an implementation result, not automatic scientific
promotion.

## Implemented research surface

The current package contains the following executable layers:

1. Capstone static decoding and bounded Unicorn execution over transparent
   x86-64 fixtures.
2. A trace contract joining instruction bytes, operands, register effects,
   executed addresses, memory-hook events, observation categories, and package
   provenance.
3. A bounded transfer-function subset whose pre/post `RAX` values and explicitly
   modeled `EFLAGS` bits are checked stepwise against independent Unicorn
   snapshots; the terminal value remains a summary check.
4. Direct-target CFG evidence and a separate static-memory/dynamic-memory join.
5. A discrete generative model with explicit likelihood `A`, transition `B`, and
   initial-state prior `D`.
6. Forward filtering, forward-backward smoothing, Viterbi decoding, phase
   segmentation, calibration, descriptive Wilson intervals, negative controls,
   model evidence, likelihood and transition ablations, and fixture-level
   held-out estimation.
7. An expected-free-energy policy whose realized and predictive information
   quantities are reported separately.
8. A fixed-parameter generative adequacy check that simulates same-length
   sequences and compares transition structure, run counts, phase occupancy, and
   temporal recovery with the observed fixtures.
9. A versioned typed observation schema retaining instruction family, operand
   class, register read/write sets, flag effects, memory mode, branch-target
   class, and dynamic status; a label-blind record artifact; and a
   leave-one-fixture-out comparison of collapsed, factorized, and hierarchical
   composite scores with field-family ablations and explicit missing-value
   handling.
10. Native Julia/RxInfer Gaussian free-energy tracking plus a namespaced
    categorical companion. The Gaussian event-code trajectory remains the
    authoritative Bethe Free Energy surface; the categorical trajectory has its
    own model-specific scale.
11. A bounded action-conditioned sensing experiment in which a policy chooses a
    typed field query before revealing its held-out value, compared with fixed,
    seeded-random, and explicitly inadmissible oracle policies under a common
    normalized budget.
12. An observation-only ordered-HMM Baum--Welch/EM estimator evaluated with
    complete-fixture holdouts against supervised MAP and the fixed reference,
    including a fixed-penalty Dirichlet-MAP comparator, label-free
    posterior-predictive checks, fit-input leakage contracts, and training
    evidence histories.
13. A pinned Lean project and deterministic receipt for a closed Word64
    accumulator fragment with explicit equivalence and counterexample theorems;
    this is a formalization tranche, not complete ISA semantics.
14. Real Capstone/Unicorn adapter fixtures for x86-64, AArch64, and RISC-V64,
    with exact bounded instruction counts, common numeric outcome, API
    provenance, and an explicit non-equivalence scope.
15. An exact finite-horizon information-per-cost dynamic program over
    label-blind typed-field sensors, retaining finite policy counts and proxy
    cost assumptions rather than promoting a general active-sensing law.
16. An exact intervention-power receipt that enumerates every non-empty subset
    of the declared quotient interventions, reports minimum separating sets,
    and retains a deliberately insufficient one-state negative control.
17. A multi-start observation-only parameter-learning ledger that records
    deterministic initialization sensitivity separately from sampling
    uncertainty and never uses held-out labels for fitting or start selection.
18. A normalized 64-bit machine-state correspondence layer with all sixteen
    general-purpose registers, explicit width/flag semantics, bounded memory
    and stack checks, a generated Python-to-Lean boundary fixture, and
    structured unsupported outcomes.
19. A raw surface-baseline portfolio spanning opcode, byte n-gram, mnemonic,
    instruction-length, control-flow, and memory-event descriptors, with
    source-family-paired bootstrap and Holm-adjusted comparisons.
20. A deterministic sensor-cost vector, local timing calibration, and finite
    policy ledger comparing exact, fixed, greedy, seeded-random, and abstaining
    policies.

## Phase A result and next scientific phases

### Phase A — typed observation expansion (implemented first tranche)

The first tranche retains the six-category channel as a deliberately collapsed
reference and adds a typed record beside it. The companion fit uses positive
Dirichlet smoothing, training-only categorical supports, an explicit
`__MISSING__`/`__UNKNOWN__` policy, and fixture-level holdouts. It compares a
field-wise factorized score with a group-balanced hierarchical composite score,
and removes each field family in turn. This is a supervised demonstrator and
does not claim that the typed fields are conditionally independent or that
dynamic status contains semantic information.

The first result is a measured null for improvement on the initial
register-dominated corpus: additional fields do not automatically improve the
collapsed reference. The next tranche is now implemented. It adds a transparent
stack-memory fixture and a taken-branch companion, and writes
`output/data/corpus_coverage.json` so memory modes, repeated phases, and
static-versus-dynamic path differences are counted from the real producer path.
The typed model remains a companion diagnostic; the larger fixture surface is
not treated as external validation. Phase A is promoted only to this expanded
design audit, not to a claim of generalization.

### Phase B — parameter learning and external validity (first parameter-learning tranche implemented)

`src/parameter_learning.py` now fits `A`, `B`, and `D` with observation-only
ordered Baum--Welch expectation-maximization on training fixtures only. It
compares that fit with supervised MAP and the fixed reference, reserves complete
fixtures for evaluation, records scaled evidence histories, and asserts that
phase labels are present only in held-out scoring rows. The estimator uses a
soft self/next-state support prior and a reference initialization so that state
coordinates remain inspectable; it is therefore an ordered latent-state
demonstrator rather than unsupervised phase discovery.

The next Phase B tranche should expand the transparent corpus with longer
programs, repeated phase patterns, branch paths, memory accesses, and deliberately
ambiguous observations. It should compare multiple ordered-state initializations
and constraints, retain training-versus-held-out evidence separately, and reserve
complete fixtures for evaluation.
`output/data/parameter_learning.json` and
`output/figures/parameter_learning.png` are the current source-bound outputs.
External binaries enter only after a safety and provenance contract specifies
architecture, mapping, execution bound, timeout, and non-execution modes.

Promotion criteria: fixture-level or binary-level splits, leakage tests, an
explicit sampling unit for uncertainty, calibration on held-out data, and a
failure analysis that reports where the observation vocabulary is insufficient.

### Phase C — action-conditioned sensing (first tranche implemented)

The first action-conditioned tranche is now executable in
`src/action_sensing.py`. Candidate actions query one typed field at one decoded
instruction. The active policy uses training-only predictive field likelihoods
to choose before revelation; fixed-order and seeded-random policies use the
same candidate pool, while the oracle is explicitly marked inadmissible. Every
policy shares a confidence threshold, action budget, held-out fixture, and
equal normalized proxy cost. `output/data/action_sensing.json` records the
training complement, selected actions, revealed values after selection, and
confidence/cost outcomes.

The tranche satisfies the key promotion contracts—action changes the acquired
evidence, pre-action selection cannot read the answer, costs are recorded, and
results are summarized across fixture holdouts. The current corpus result is
still a mechanism demonstration: active selection is not promoted as a budget
advantage merely because it is implemented, and the oracle is not a deployable
comparison. A fuller Phase C study should add multiple safe query types with
measured acquisition costs and longer held-out programs.

### Phase D — richer semantics and interchange

Add checked adapters for an intermediate representation and richer control-flow
facts only when the adapter emits machine-readable provenance. Indirect targets,
flags, calling-convention state, memory regions, and unsupported operations must
be represented as explicit uncertainty or out-of-scope facts. A larger tool is
not automatically better evidence: every adapter is compared against the
bounded Capstone/Unicorn contract on fixtures where agreement is testable.

### Phase E — native probabilistic model expansion

The Julia/RxInfer package can add a discrete state-space model, a Gaussian
register-delta model, a changepoint model, and eventually a hierarchical model
joining static, dynamic, and symbolic evidence. Each model must keep its own
free-energy units and diagnostic namespace. No trajectory may be called
authoritative merely because it is numerically finite.

### Phase F — quotient-state theory, invariance, and replication

The current package cannot honestly promote hand-authored phases to semantic
discoveries. The ambitious successor therefore begins by defining an executable
operational equivalence relation and a finite separating intervention family,
then tests whether latent quotient states are identifiable up to relabelling.
The first benchmark must include semantics-preserving transformation families
and near-miss counterexamples, with source-program and compiler-family splits.
The source-family receipt now lives at
`output/data/representation_invariance_benchmark.json`, while the co-primary
classifier portfolio is emitted as `output/data/quotient_classifier.json`. The
portfolio uses 64 synthetic families, 64 pair rows per family, and disjoint
training/development/held-out family partitions. It reports a calibrated
interpretable model, a temporal Markov model, a fixed NumPy recurrent model,
and a surface baseline under identical balanced-accuracy, macro-F1, Brier,
calibration, confusion, and clustered-interval contracts. Four held-out
families make the intervals visibly broad; the receipt therefore supports a
reproducible endpoint, not semantic discovery or superiority.

The coordinated P3--P16 tranche is now implemented in bounded form. The
source-bound receipts are `research_protocol.json`,
`formal_trace_correspondence.json`, `compiler_transfer.json`,
`sensor_cost_receipt.json`, `replication_manifest.json`, and
`theorem_or_countertheorem.json`, in addition to the earlier formal and
adapter receipts. The compiler track retains unavailable targets explicitly
and executes no emitted object; the sensing track certifies only finite
dynamic-program policies; the replication track is local and no-network; and
the theorem phase records both checked local results and a generated
counterexample while keeping promotion fail closed. The intervention-power
receipt enumerates the finite outcome matrix rather than learning its labels,
and the multi-start parameter-learning receipt retains weak and unstable starts
as robustness evidence. The licensed-source receipt is currently blocked by an
intentionally empty manifest; independent replay, full-ISA correspondence, and
a broad identifiability or transfer theorem remain future gates. No blocked
status is converted into a zero-valued result.

The promotion rule is conjunctive: quotient labels must be generated outside
the inference model; primary endpoints and test identifiers must be frozen;
uncertainty must be clustered at the source-program family; strong surface
baselines and non-separating intervention controls must be reported; and no
semantic or external-validity claim may be promoted before independent
reimplementation. A theorem or decisive countertheorem is a separate scientific
deliverable, not something inferred from benchmark performance.

## Scientific invariants

The roadmap does not authorize claims of general reverse engineering, malware
classification, semantic decompilation, or autonomous security triage. The
central invariants are:

- implementation evidence is distinct from software or roadmap context;
- hand-authored labels are not discovered semantics;
- descriptive simulation intervals are not posterior or population uncertainty;
- a passing artifact contract is not independent replication;
- a new method is not promoted until its negative controls and failure cases are
  visible;
- public documentation describes maintained entrypoints and current outputs,
  never historical implementation milestones or private review process.

The command of record for the currently implemented surface is documented in
`docs/verification.md`, while this file describes how the next scientific result
must earn entry into that surface.
