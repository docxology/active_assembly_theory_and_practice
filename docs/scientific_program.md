# Long-horizon scientific program

This document turns the ambition of making a field-changing contribution into a
falsifiable research program. It is intentionally more demanding than a
feature roadmap. The program is generated as a machine-readable receipt at
`output/data/scientific_program.json`; the source of that receipt is
`src/scientific_program.py`.

The current repository is not a prize-level result. It is a carefully bounded
measurement and inference demonstrator whose core temporal corpus contains six
transparent x86-64 fixtures, augmented by a 64-family co-primary quotient
benchmark, a separate four-state representation pilot, and bounded P3--P17
research receipts.
Its value is that it supplies a trustworthy starting instrument: the trace
lineage is visible, the observation channel is explicit, negative results are
retained, the manuscript is hydrated from producer artifacts, and the PDF is
tagged and checked. The next program must earn a much deeper scientific claim
on top of that instrument.

## Protocol-v2 artifact closure

The research protocol is now accompanied by a separate receipt schema. Domain
payloads retain their own `schema_version`; the common envelope adds
`receipt_schema_version`, protocol and producer digests, input hashes, toolchain
versions, status, limitations, unsupported cases, and underpowered cases. This
is intentionally a provenance contract rather than a semantic result. Producers
write atomically, legacy migration is explicit, and the validator never repairs
an unsealed artifact while claiming that validation passed.

The canonical local commands are:

```bash
uv run python scripts/run_scientific_program.py all
uv run python scripts/run_scientific_program.py audit-artifacts
```

`all` runs the source-bound orchestration summary and writes a machine-readable
result; `audit-artifacts` is read-only and reports every unsealed or
mixed-protocol JSON artifact. The closure figure and manuscript variables are
generated from this same surface. This design follows provenance and research-
object packaging practice [@w3cprov2013;@rocrate2023;@sandve2013repro] and
keeps local reproducibility distinct from independent reproduction
[@acm2020artifact].

## The scientific center of gravity

The project should not try to become “a larger binary-analysis toolkit.” That
would produce more capability without necessarily producing a new scientific
principle. The ambitious question is instead:

> Can computational motifs be defined as quotient objects of operational
> program semantics and recovered with calibrated uncertainty from active,
> architecture-invariant observations?

The working conjecture is:

> Under a separating family of safe interventions, observationally equivalent
> motifs are identifiable up to quotient-state relabeling, while an active
> policy approaches a non-trivial information-per-cost frontier.

This conjecture has four parts that must remain connected:

1. **Operational semantics.** A motif is grounded in what a program does under
   a declared state and intervention model, not in a mnemonic name or a
   convenient hand-authored label.
2. **Quotient structure.** Semantically equivalent traces are treated as the
   same object for the purpose of the scientific question, while genuinely
   different behaviors must remain distinguishable when the intervention family
   is separating.
3. **Statistical identifiability.** The model must recover the quotient object
   with uncertainty and must state exactly which ambiguities are impossible to
   resolve from the available observations.
4. **Active information acquisition.** If observations have costs, the policy
   should be judged by information gained per measured cost and against a
   certified lower or upper bound, not by an isolated policy score.

This gives the work a possible mathematical core and an empirical testbed. It
also creates a principled way to fail: if the quotient is not identifiable,
the intervention family is insufficient, or the active bound is false, the
program should publish that counterresult rather than widening the claim.

## What “Fields/Nobel territory” actually means

The prize language is useful only as a seriousness test, not as a project
metric.

- A Fields-level trajectory would require a genuinely new and broadly
  important mathematical theorem, proof, or theory. A better implementation,
  a larger benchmark, and a polished manuscript are not substitutes for that
  standard.
- There is no Nobel category for software or computer science. A Nobel-relevant
  trajectory would require the work to make a fundamental contribution that
  genuinely belongs in an eligible scientific field. This project therefore
  makes no Nobel claim and should not use Nobel language in the manuscript's
  empirical conclusions.
- The first credible target is not an award. It is a result that survives
  formal scrutiny, hostile counterexamples, external datasets, an independent
  implementation, and time. Recognition is an external consequence of that
  result, never a validation gate.

## Program phases

The authoritative status, dependencies, falsifiers, and promotion rules are in
the generated program receipt. The abbreviated sequence is:

| ID | Scientific task | Current status | What would count as progress |
| --- | --- | --- | --- |
| P0 | Source-bound assembly observation bridge | Implemented locally | Exact trace lineage and fail-closed bounded semantics |
| P1 | Temporal inference under controlled ambiguity | Implemented locally | A preregistered mechanism result with both null and positive directions |
| P2 | Observation-only parameter learning | Implemented locally | Leakage-safe held-out estimation with negative result preserved |
| P3 | Quotient-state identifiability and representation invariance | Implemented bounded tranche; promotion blocked | 64-family source-disjoint co-primary benchmark, clustered statistics, leakage controls, and a separate four-state representation pilot; no semantic-discovery claim |
| P4 | Mechanically verified operational semantics | Implemented bounded correspondence; theorem unearned | Explicit 16-register, width-aware Python state, real Capstone/Unicorn step matching, and a separate Lean fragment; no full-ISA Lean/Python theorem |
| P5 | Cross-architecture and cross-compiler invariance | Implemented local capability tranche; external gate blocked | Exact local compiler/static-object matrix with unavailable targets retained; no licensed external-source effectiveness result |
| P6 | Information-per-cost law for active observation | Implemented finite measured-cost tranche; general law unearned | Eight-component deterministic costs, local calibration with overhead/regression diagnostics, and exact finite-policy certificates; no infinite-horizon or cross-machine law |
| P7 | Independent implementation and open challenge | Local frozen harness plus an observed same-host controlled guest replay; independent implementation remains unearned | No-network CLI, digest-ordered label-blind public inputs, private gold separation, hash validation, protected scoring, and replay receipt |
| P8 | General theorem or decisive countertheorem | Implemented fail-closed phase; broad theorem unresolved | Checked local theorem group, finite computational certificate, and generated bounded counterexample; no broad theorem promotion |
| P9 | Finite intervention power and separating sets | Implemented finite diagnostic; identifiability promotion blocked | Exhaustive enumeration of the separate four-state pilot catalog, minimum separating sets, and one-state negative control; no universal identifiability claim |
| P10 | Initialization robustness of observation-only learning | Implemented finite robustness diagnostic; stability promotion blocked | Five-start holdout ledger with accuracy spread and sequence agreement; no stable latent-parameter claim |
| P11 | Complete normalized machine-state correspondence | Active bounded implementation; cross-language bridge unresolved | 64-bit all-GPR state, width/flag/memory boundary tests, Lean normalized-state definitions, and explicit unsupported outcomes |
| P12 | Scaled transfer and quotient evaluation | Active promotion candidate; external evidence absent | 64-family classifier receipt, compiler/source/architecture capability matrix, and source-bound split/leakage receipts |
| P13 | Licensed-source external validation | Underpowered after pinned acquisition | Ten pinned rows across five projects and two revisions, 10/10 static rows decoded, 100 source-bound contract rows with project-held-out labeling, and the remaining external acquisition/build replay gate |
| P14 | Measured active-sensing robustness | Active bounded implementation | Deterministic vectors, calibrated local timing, finite oracle regret, non-oracle baselines, and matching calibration receipts |
| P15 | Blind replication and adversarial audit | Same-host separately controlled replay observed; independent implementation and physical/CI replay remain unearned | Public/private separation, hashed protocol/toolchain/instructions, offline replay, protected scoring, controlled replay receipt, and explicit independent-reimplementation boundary |
| P16 | Theorem/countertheorem synthesis and publication promotion | Planned; fail closed | Kernel-checked theorem or preserved counterexample, claim-evidence closure, regenerated accessible package, and owner/reviewer promotion gates |
| P17 | Protocol-v2 artifact and manuscript closure | Implemented locally; integrity-only | Shared receipt envelope, atomic producer writes, fail-closed artifact audit, source-bound figure/variable closure, and post-render document lineage |

The order is deliberate. The project does not jump from six toy fixtures to a
large opaque corpus. It first defines what a semantic object is, then builds
and tests a small formal contract, then exercises real compiler and adapter
APIs, solves a finite active-information mechanism exactly, freezes a local
replication interface, and finally records theorem/countertheorem outcomes.
The remaining external-validity and broad-theorem gates are deliberately not
silently inferred from these bounded receipts.

## Source-bound visual atlas

The visual-atlas tranche adds a canonical, receipt-backed specification layer
over the legacy figure-path interface. The 62 substantive figure rows (the
cover is separate) record the generator, source artifact hashes, population
grain, split and uncertainty semantics, evidence status, claim boundary,
visible caption, short alternative text, long description, and vector/raster
outputs. The canonical producer is `src/figure_contract.py`; the nine central
evidence diagrams are generated by `src/visual_atlas.py`.

Generate and audit the package with:

```bash
uv run python scripts/run_scientific_program.py generate-figures \
  --formats png svg pdf atlas
uv run python scripts/run_scientific_program.py audit-figures
```

The offline atlas is `output/figure_atlas/index.html`. It has semantic
`figure`/`figcaption` structure, image alternatives, expandable long
descriptions, source-hash blocks, and no remote assets or scripts. PNGs under
`output/figures/` remain the DOCX/PDF compatibility surface; SVG/PDF companions
are written under `output/figures/vector/`. Appendix and atlas-only metadata is
still generated and audited even when the main manuscript embeds only the
curated narrative subset. A visual status such as `unavailable`, `unsupported`,
`underpowered`, or `not observed` is textually preserved and is never replaced
with a numeric zero.

## The active program: P3

P3 is the next scientific tranche because it attacks the largest current gap:
the present labels are authored phase hypotheses, not discovered semantic
objects.

### P3.1 Define the quotient before fitting a model

Create a formal, executable equivalence relation over bounded programs. A first
version should be deliberately small:

- architecture: x86-64 only at first;
- state: the declared register/flag/memory slice;
- intervention: bounded initial-state and input perturbations plus allowed
  control-flow probes;
- observation: the typed Capstone/Unicorn record plus bounded semantic effects;
- equivalence: agreement on the declared output relation for every intervention
  in a finite separating catalog.

The equivalence relation must be generated independently of the inference model.
The model may not define its own labels and then call those labels semantics.
Every equivalence class needs executable witnesses and counterexamples showing
which nearby programs are *not* equivalent.

### P3.2 Build a transformation-pair corpus

For each source-level computational motif, generate pairs or families under
semantics-preserving transformations:

- register renaming where the output relation is preserved;
- instruction selection changes;
- neutral instruction insertion;
- basic-block layout changes;
- control-flow structuring changes;
- compiler optimization changes;
- later, architecture-specific encodings of the same source operation.

The corpus must also contain near-miss pairs that differ in exactly one
semantically relevant effect. Near misses are essential: invariance without
separation is just blindness.

Each family is split as a whole. No member of a transformation family may occur
in both training and test sets. Source programs, compiler versions, and
architectures are separate grouping variables rather than ordinary row-level
features.

The first source-family benchmark receipt is
`output/data/representation_invariance_benchmark.json`, and the co-primary
classifier receipt is `output/data/quotient_classifier.json`. The classifier
benchmark freezes 64 transparent x86-64 families with 64 pair rows per family,
disjoint training/development/held-out family partitions, and a 16-family
held-out test. Labels are generated from the declared intervention/output
relation before model fitting; model inputs retain missing versus unseen
values and omit family, transformation, byte, and quotient fields. The
portfolio includes a regularized interpretable classifier, a class-conditional
temporal Markov classifier, a fixed NumPy recurrent classifier, and a surface
baseline. It reports balanced accuracy, macro-F1, Brier score, ten-bin ECE,
confusion counts, false-equivalence/false-distinction rates, and source-family
cluster bootstrap intervals. With four held-out families, the intervals are
an audit of the frozen benchmark rather than population inference, and no
superiority or learned-semantics claim is promoted.

The same protocol retains a deliberately insufficient single-state
intervention catalog as a negative control: it preserves equivalent-pair
consistency but leaves declared near misses unresolved. This is evidence that
separating power depends on the intervention family; it is not a model failure
or a semantic result.

### P3.3 Define the primary statistics before the final test set

The initial preregistered endpoints should be:

- aligned adjusted mutual information between inferred and quotient states;
- pairwise equivalence-consistency of posterior distributions;
- posterior calibration for equivalence and non-equivalence decisions;
- invariance gap between original and transformed representations;
- separation gap between equivalent and near-miss pairs;
- failure-stratified results by unsupported instruction, memory effect, branch
  effect, and intervention coverage.

The sampling unit is the source-program family, not the instruction. Uncertainty
must be clustered at that family level. If there are too few independent
families to support an interval, the result is descriptive and must be labelled
that way.

### P3.4 Use strong baselines and negative controls

The proposed representation must be compared against:

- mnemonic and opcode histograms;
- register/operand surface features;
- compiler and architecture identity controls;
- a fixed-reference HMM;
- a supervised upper-bound model with an explicitly disclosed label contract;
- a permutation or family-shuffle negative control;
- a deliberately non-separating intervention catalog.

The last control is important. The program should demonstrate that it fails when
the intervention family cannot distinguish the classes. A method that succeeds
under every catalog is likely exploiting leakage or an invalid equivalence
definition. The current `single_zero_state_v1` receipt supplies that control
and keeps its unresolved pairs in the generated artifact.

### P3.5 Promotion gate

P3 cannot be promoted to “semantic discovery” unless all of the following hold:

1. The quotient labels are generated from the formal equivalence contract and
   not from model output.
2. All transformations and near misses are source-disjoint across splits.
3. The primary endpoint and test set were frozen before final scoring.
4. Results survive multiple deterministic initializations and state-alignment
   procedures.
5. The proposed method beats strong surface baselines on the preregistered
   invariance and separation endpoints.
6. Counterexamples and unsupported cases are reported in the main result.
7. An independent implementation reproduces the primary endpoint.

Until then, the correct language is “quotient-state benchmark” or
“representation-invariance experiment,” not “learned program semantics.”

## The formal-theory ladder

The project should pursue theory in increasing strength rather than announce a
grand theorem prematurely.

### Level 1: trace-refinement lemma

For the bounded instruction subset, prove that the executable transfer function
refines the declared small-step operational semantics and that the observation
projection preserves the specified fields. This is a correctness theorem for
the instrument.

The current package now has two distinct prerequisites for this level:
real Unicorn emits per-step `RAX`, `RIP`, `RSP`, and `EFLAGS` snapshots, and the
closed-world interpreter checks pre/post `RAX` plus its defined flag bits at each
executed step; independently, Lean compiles a Word64 accumulator semantics with
explicit equivalence and counterexample theorems. The resulting
`trace_refinement.png`, `formal_trace_correspondence.json`, and
`formal_verification.json` are stronger than a single final-state check. The
Python/Capstone/Unicorn correspondence is now checked step by step for the
transparent fixture subset, and a generated Lean bridge kernel-checks one
shared 64-bit boundary fixture. These are distinct receipts: the bridge is not
a proof that the full Python adapter, Capstone, Unicorn, and Lean semantics
correspond for every supported instruction. The shared broad-state theorem and
exhaustive full-subset proof remain open.

### Level 2: quotient soundness

Prove that the declared equivalence relation is reflexive, symmetric, and
transitive over the bounded model class, and characterize the effect of the
intervention catalog. This prevents informal “semantic similarity” from
becoming a label generator.

### Level 3: identifiability theorem

State conditions under which a finite family of interventions separates quotient
classes and the latent model is identifiable up to class permutation. Explicitly
include counterexamples when those conditions fail.

### Level 4: active-sensing bound

Characterize the minimum information or number of actions needed to distinguish
classes under a specified sensor-cost model. Compare the active policy with the
bound, not merely with a weak baseline.

### Level 5: transfer theorem or decisive countertheorem

Determine whether the quotient/inference construction transfers across an
architecture translation or whether there is an unavoidable obstruction. A
rigorous impossibility result could be scientifically important even if the
positive conjecture fails.

Only Levels 3–5 could plausibly support Fields-level mathematical ambition, and
only if the statements are non-trivial, broadly reusable, and independently
checked. The existing implementation is evidence for Level 1 instrumentation,
not for any higher level.

## External-validity program

The current corpus is intentionally transparent and small. The expansion path
must preserve that transparency while increasing independence:

1. **Synthetic generator families.** Many source programs with controlled
   motifs, transformations, and known semantics.
2. **Compiler matrix.** Multiple compiler versions, optimization levels, and
   code-generation paths, split by source family.
3. **Architecture matrix.** x86-64 first, then AArch64 and RISC-V, with a
   declared portable representation where appropriate.
4. **Public real-program evaluation.** Only after the safety, licensing,
   provenance, and non-execution/sandbox contract is reviewed.
5. **Blind challenge set.** A held-out set owned or frozen by an independent
   party, with scoring code that the authors cannot alter after unblinding.

The benchmark should report coverage, not just a single score: memory modes,
indirect control flow, flags, calling conventions, exceptions, loops, recursion,
self-modification exclusions, unsupported operations, and architecture-specific
effects must be visible. Missingness is data, not a reason to silently drop a
hard case.

## Full-integrity operating protocol

Every future tranche follows this protocol:

### Before data access

- Write the hypothesis, primary endpoint, sampling unit, split rule, stopping
  rule, and analysis code contract.
- Freeze the test identifiers and hash the benchmark manifest.
- Define the admissible intervention and sensor set.
- Register the strongest baseline and the negative controls.

### During implementation

- Keep domain logic in `src/` and orchestration in `scripts/`.
- Add source-bound artifacts, tests, captions, variable definitions, and claim
  boundaries together.
- Fail closed on unsupported semantics and missing provenance.
- Keep training, validation, and held-out labels physically and logically
  separated.
- Record convergence failure, missingness, and discarded cases with reasons.

### Before promotion

- Recompute results from source rather than editing receipts.
- Check that all primary endpoints, nulls, and counterexamples are present.
- Run an adversarial review focused on leakage, near duplicates, hidden
  metadata, and alternative explanations.
- Re-run from a clean environment or independent lockfile.
- Obtain independent implementation or proof review for any broad claim.

### If the result fails

The failure is the result. Keep the artifact, figure, code, and manuscript
language that demonstrate the failure. Do not broaden the model, revise the
endpoint, or enlarge the claim until the failed experiment has been preserved
and explained.

## What success would look like

The strongest credible outcome is not “assembly plus Active Inference wins.” It
would be a general principle with three mutually reinforcing forms of evidence:

1. A theorem or countertheorem about operational equivalence,
   identifiability, or active information acquisition.
2. A mechanically checked reference implementation whose assumptions and
   failure cases are explicit.
3. An independent, cross-architecture empirical demonstration that the theorem
   predicts something surface heuristics do not, with calibrated uncertainty and
   a successful blind replication.

If the theorem fails but the counterexamples reveal a fundamental limitation of
program observation, that can still be a major scientific contribution. If the
empirical method improves but no general principle emerges, it should be
reported as a useful research-software contribution rather than inflated into a
prize narrative.

The generated program receipt is the operational source of truth for this
document. Current status is intentionally conservative: P0–P2 are implemented
locally, and P3–P17 have implemented bounded tranches with explicit receipts.
The remaining promotion gates are not implied: P3 remains a bounded benchmark,
P4 lacks a full-ISA theorem, P5 lacks licensed external validity, P6 is
finite-benchmark optimality, P7 has an observed same-host controlled replay but
no independent implementation, P13 has source-bound labels but remains
underpowered until its external acquisition/build replay gate is observed, and
P16 lacks a broad theorem. The source-bound status is therefore “implemented,
bounded, and fail-closed,” not “scientifically complete.”
