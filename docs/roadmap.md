# Scientific roadmap and current backlog

**Roadmap revision:** 2026-09-03

**Repository:** `docxology/active_assembly_theory_and_practice`
**Machine-readable status source:** `output/data/scientific_program.json`

This document is the human-maintained execution view of the research program.
It has two jobs that must not be conflated:

1. close implementation work that has already produced receipts, tests, and
   manuscript surfaces; and
2. scope the next experiments and promotion gates without presenting a plan as
   evidence.

The generated scientific-program receipt is authoritative for milestone fields,
protocol identity, dependencies, and status vocabulary. This file is the
authoritative maintainer view of the actionable backlog. A completed
implementation item is not reopened merely because its broader scientific
promotion gate remains blocked. A defect, stale receipt, or reproducibility
failure is a new maintenance issue with its own evidence.

## Current release state

The repository is a bounded, source-bound technical monograph and replication
package. Its current local engineering surface includes real Capstone and
Unicorn fixture adapters, a full normalized 64-bit register state, a bounded
Lean bridge, temporal and typed observation models, quotient and transfer
diagnostics, finite active-sensing certificates, a label-separated replay
interface, source-bound figures, and tagged document outputs.

The release state is intentionally asymmetric: implementation and document
integrity are strong, while the claims that require external or independent
evidence remain visibly unearned.

| Surface | Implementation state | Scientific/promotion state | Receipt or source of truth |
| --- | --- | --- | --- |
| P0–P2 observation, temporal inference, and observation-only estimation | Closed local tranches | Local demonstrator only | `output/data/trace_bundle.json`, `output/data/evaluation_report.json`, `output/data/parameter_learning.json` |
| P3 quotient and representation-invariance benchmark | Closed bounded tranche | Promotion blocked; no semantic-discovery claim | `output/data/representation_invariance_benchmark.json`, `output/data/quotient_classifier.json` |
| P4/P11 normalized formal correspondence | Closed bounded implementation tranche | `validated_bounded`; no complete x86-64 theorem | `output/data/formal_trace_correspondence.json`, `output/data/formal_verification.json` |
| P5/P12 compiler, architecture, and transfer surface | Closed local capability tranche | Bounded transfer evidence; external/general transfer unearned | `output/data/compiler_transfer.json`, `output/data/cross_architecture_fixtures.json` |
| P6/P14 active sensing and cost accounting | Closed finite measured-cost tranche | `validated_bounded`; no universal information-per-cost law | `output/data/sensor_cost_receipt.json`, `output/data/active_information_frontier.json` |
| P7/P15 public replay interface | Closed local harness and same-host controlled guest run | Replay observed under a distinct guest boundary; independent physical replay and independent reimplementation remain unearned | `output/data/replication_manifest.json`, `output/data/independent_replay_receipt.json` |
| P8/P16 theorem and countertheorem phase | Closed fail-closed bounded phase | Counterexample and unresolved candidates retained; no broad theorem | `output/data/theorem_or_countertheorem.json` |
| P9 intervention power | Closed finite diagnostic | Identifiability promotion blocked outside the declared finite catalog | `output/data/intervention_power.json` |
| P10 initialization robustness | Closed finite robustness diagnostic | Stability promotion blocked by initialization sensitivity | `output/data/parameter_learning.json` |
| P13 licensed-source track | Acquisition, static compilation, and source-bound labels closed locally | External validation remains `underpowered` until the full gate passes | `output/data/external_acquisition.json`, `output/data/external_static_validation.json`, `output/data/external_label_contract.json` |
| P17 protocol, figures, manuscript, and document closure | Closed local integrity tranche | Source-bound local package; not peer review or independent validation | `output/data/reproducibility_manifest.json`, `output/data/figure_registry.json`, `output/data/validation_report.json` |

### Completed historical work: no open TODO remains

The following action items have been completed and are now maintenance
surfaces, not unstarted roadmap tasks:

- the transparent Capstone-to-Unicorn observation bridge and typed observation
  schema;
- temporal filtering, smoothing, Viterbi, calibration, model-evidence, model
  adequacy, and negative-control diagnostics;
- fixture-held-out and observation-only parameter-learning paths, including
  multi-start sensitivity reporting;
- bounded memory, branch, CFG, symbolic-transfer, and cross-architecture
  adapter fixtures;
- protocol-v2 envelopes, atomic producer writes, artifact freshness checks,
  source-bound claim and scholarship registries, and manuscript-token closure;
- the 64-family quotient classifier portfolio, leakage controls, clustered
  statistics, and finite intervention-power diagnostics;
- the 16-register normalized state with explicit width, flag-definedness,
  bounded memory/stack, step, status, and unsupported-state fields;
- deterministic sensing-cost vectors, local timing calibration, exact finite
  policy certificates, and non-oracle baselines;
- the public/private replay bundle, no-network replay path, controlled guest
  replay receipt, and stale-source rejection;
- the 62-substantive-figure source-bound atlas, cover contract, caption and
  alternative-text registry, vector/raster outputs, tagged PDF, and DOCX
  validation surface.

These items should receive regression coverage and freshness audits. They should
not reappear in future documentation as “to do,” “next tranche,” or “not yet
implemented” unless a new defect is demonstrated.

## Open evidence gates

The open work is not a generic feature wishlist. It is the smallest set of
evidence needed to change a current scientific status. The following gates are
still open:

| Gate | Current boundary | Required change |
| --- | --- | --- |
| External validity | P13 remains `underpowered` | A separately controlled acquisition/build replay must reproduce the pinned licensed-source recipe, after which the already-defined project-held-out endpoint may be scored. |
| Independent replay | The observed replay is same-host and guest-controlled | A separately operated machine or CI worker must supply a matching receipt; this still does not equal independent reimplementation. |
| Independent reimplementation | Not observed | A separate implementation must reconstruct the public interface and reproduce or falsify the frozen endpoint without access to protected labels. |
| Broad formal correspondence | Only the declared bounded fragment is checked | A genuine cross-language proof must cover the declared subset and all assumptions, or a preserved counterexample must narrow the claim. |
| General transfer | Local static capability is not behavioral validity | New compiler, architecture, source-family, and project-held-out evidence must pass without substituting unavailable capabilities. |
| General sensing law | Only a finite exact certificate and local calibration exist | A broader result requires a theorem or repeated independently calibrated evidence; one-machine timing cannot establish a universal law. |
| Publication promotion | Local source-bound package is validated | External review, scientific claim review, artifact review, and owner authorization must be recorded separately from engineering checks. |

## Tiered upcoming backlog

The tiers describe scope and risk, not scientific importance. “Minor” means a
bounded maintenance or clarity change with no new claim. “Medium” means a
substantial local experiment or interface extension whose result remains
bounded. “Major” means a promotion-gated evidence program that can change the
scientific interpretation of the work.

### Minor: maintenance, clarity, and regression hardening

Minor work may proceed continuously. It must not change the protocol, endpoint,
label contract, split rule, or promotion language without a protocol revision.

| ID | Deliverable | Definition of done | Claim boundary |
| --- | --- | --- | --- |
| MIN-01 | Receipt-refresh guard | Run `all`, `audit-artifacts`, `audit-figures`, manuscript-variable validation, and post-render checks from a clean producer order; stale source/protocol hashes fail closed. | Freshness is an engineering property, not scientific validity. |
| MIN-02 | Command and runbook consistency | Keep `README.md`, `docs/verification.md`, `docs/reproducibility.md`, `docs/replay_operator_runbook.md`, and CLI help synchronized; every documented command has a smoke test or explicit environment prerequisite. | A runnable command does not imply a promoted result. |
| MIN-03 | Figure-atlas maintenance | For every future figure change, require one canonical spec, source hashes, caption, short alternative, long description, exact status table where needed, and page/contact-sheet visual QA. | Accessibility and source closure do not strengthen the underlying evidence. |
| MIN-04 | Formal boundary regression pack | Add focused boundary fixtures for width updates, `INC`/`DEC` carry preservation, `ADC`/`SBB` carry consumption, memory bounds, branch outcomes, serialization, and unsupported instructions. | More boundary tests strengthen the bounded instrument only. |
| MIN-05 | Claim and scholarship lint | Verify that each changed claim has a receipt, each methodological citation supports the exact statement, and no stale quantitative literal or hard-coded figure number enters the manuscript. | Citation closure is not peer review. |
| MIN-06 | Reproducible developer envelope | Keep the locked Python/Julia/Lean toolchain receipts, supported platform notes, and optional-dependency behavior current; record unavailable capabilities explicitly. | Toolchain parity enables replay but is not independent evidence. |
| MIN-07 | Failure-surface maintenance | Preserve `underpowered`, `unavailable`, `unsupported`, `blocked_external`, `falsified`, and `unresolved` rows in tests, figures, captions, and docs; add a regression whenever a failure mode is found. | Negative evidence remains evidence; it is not converted into a null score. |

### Medium: bounded scientific extensions

Medium work should use the existing protocol and receipt envelope. Each item
must produce an artifact, tests, uncertainty definition, figure or table when
scientifically material, and a manuscript boundary statement.

| ID | Deliverable | Dependencies and acceptance evidence | Promotion effect |
| --- | --- | --- | --- |
| MED-01 | Longer transparent corpus | Add longer programs with repeated phases, ambiguous observations, taken/not-taken branches, bounded memory, stack-local state, calls/returns, and optimization-sensitive rewrites. Freeze family-level splits before scoring; retain all exclusions. | May improve mechanism-level inference evidence; cannot establish external validity. |
| MED-02 | Broader declared-subset semantics | Extend the normalized IR and Lean/Python fixtures for the highest-value supported instruction families, including exact flag definedness and memory effects. Compare Python, Capstone, Unicorn, and Lean field by field; preserve unsupported outcomes. | May strengthen P11 to a larger bounded fragment; no complete-ISA claim. |
| MED-03 | Robust classifier portfolio | Refit all co-primary models over the expanded corpus using development-only selection, multiple deterministic starts, family-clustered bootstrap, calibration, selective risk, label permutation, family-shuffle, compiler/architecture leakage attacks, and near-duplicate checks. | May support a stronger bounded benchmark result; no semantic-identifiability claim without the major gate. |
| MED-04 | Active-sensing stress study | Evaluate multiple finite budgets and horizons, fixed/greedy/random/abstaining controls, exact dynamic-program references, deterministic eight-component costs, repeated local timing runs, calibration residuals, regret, and optimality gaps. | May improve P14 finite robustness; no universal cost law. |
| MED-05 | Separately controlled replay | Run the public bundle on a separately controlled CI worker or machine, disable network access, return the signed/hash-bound receipt and predictions, and ingest them through the protected scorer. | Can change P15 from local/guest evidence to observed external replay, but not to independent reimplementation. |
| MED-06 | Capability-controlled transfer expansion | Probe locally available compilers, target triples, optimization levels, decoders, and object formats. Add only rows with complete source/object/disassembly provenance; leave unavailable targets explicit. | May enlarge bounded P5/P12 coverage; cannot turn capability coverage into behavioral transfer. |
| MED-07 | External corpus preparation audit | Rehearse pinned manifest parsing, license-file checks, archive safety, static-only compilation, build reproducibility, source contracts, and exclusion accounting without using the rows as an efficacy claim. | Reduces P13 operational risk; promotion still requires the major external gate. |
| MED-08 | Formal counterexample search | Search the declared bounded model for the smallest failure of each candidate theorem, store the counterexample and assumptions, and regenerate theorem-status figures and manuscript text. | A falsified or narrowed theorem can itself be a valid result. |

### Major: promotion-gated research programs

Major work must not be represented by a green engineering check alone. Every
major item requires independent evidence, explicit reviewer or operator
identity where applicable, and a claim-register update that is generated from
the resulting receipt.

| ID | Deliverable | Hard gate | Possible status change |
| --- | --- | --- | --- |
| MAJ-01 | P13 licensed-source external validation | At least five accepted projects, two revisions per project where available, at least 100 labelable examples, project-held-out evaluation, complete license/revision/build/label provenance, static-only safety, reproducible acquisition/build, and an independently replayed recipe. | `underpowered` → `validated` only if every condition passes. |
| MAJ-02 | P15 independent public replay | Separate operator/machine receipt, exact protocol/public-bundle hashes, public-only inputs, protected scoring, complete identifiers, network-disabled evidence, and preserved mismatches. | `not_yet_observed`/local boundary → `independent_replay_observed`; never infer this from a local rerun. |
| MAJ-03 | Independent reimplementation | A separate implementation consumes only the public interface, reproduces the frozen split and endpoint, and publishes its own code/output provenance or a falsifying report. | Can support replication; it is distinct from replay and does not automatically prove generality. |
| MAJ-04 | P11 cross-language theorem | Lean proof for the exact declared normalized subset, or a checked countertheorem; all assumptions, axioms, trusted base, memory bounds, flags, exceptions, and unsupported cases are explicit. | `validated_bounded` → `proved` only for the proved subset; a counterexample narrows claims. |
| MAJ-05 | P12 transfer and quotient promotion | Source/compiler/architecture/project/revision-disjoint evidence with frozen endpoints, strong baselines, clustered uncertainty, calibrated predictions, leakage controls, and independent replay. | May promote only the measured bounded transfer statement; never universal identifiability by benchmark score alone. |
| MAJ-06 | P14 multi-environment sensing result | Matching calibration receipts across environments or a reusable theorem; finite-policy certificates remain exact and machine-specific timing remains labeled. | May support robustness across declared environments; no universal information-per-cost law without theory. |
| MAJ-07 | P16 theorem/countertheorem synthesis | Integrate formal, empirical, transfer, sensing, and replication receipts; obtain independent proof or scientific review; preserve falsified and unresolved candidates; regenerate all claims and documents. | Promotes only the narrowest checked theorem or counterresult. |
| MAJ-08 | Publication and archival release | Complete author/reviewer decision, clean source-bound build, artifact archive, license/provenance manifest, accessible PDF/DOCX/HTML, reproducibility instructions, and explicit separation of local, replay, and independent evidence. | Changes publication/release status only; it cannot rewrite scientific results. |

## Execution order and dependencies

The practical order is:

1. Keep the minor maintenance loop green and refresh receipts after every
   source, protocol, or toolchain change.
2. Run MED-01 through MED-04 in parallel only after the protocol, split, and
   observation schemas are unchanged and sealed.
3. Run MED-05 and MED-07 with controlled operators and retain every failure.
4. Use MED-06 to expand only actually available capabilities; do not download
   or silently substitute an architecture or compiler to complete a matrix.
5. Start MAJ-01, MAJ-02, and MAJ-03 only from frozen bundles and receipts.
6. Use the results of MED-02, MED-08, and the major evidence tracks to decide
   whether MAJ-04–MAJ-07 is a theorem, a countertheorem, or an unresolved
   program.
7. Run MAJ-08 only after the claim register, document package, and independent
   evidence boundaries have been reviewed.

No major item should change the endpoint after looking at held-out predictions.
If a capability is unavailable, it remains unavailable. If a theorem fails,
the counterexample is preserved. If a replay fails, the failure receipt remains
part of the result. If the external corpus is too small, the result remains
`underpowered` rather than being re-described as a positive application result.

## Definition of done for every future item

An item is complete only when all applicable conditions hold:

- the objective and primary endpoint were written before final scoring;
- the sampling unit, split, seed, stopping rule, and practical margin are
  recorded;
- the producer writes a source-bound receipt with protocol and input hashes;
- tests cover the normal path, unsupported path, stale-input path, and failure
  path;
- figures and captions state sample hierarchy, uncertainty, units, and status;
- manuscript variables are generated through the existing template injection
  path;
- the result is classified as formal, computational, inferential, descriptive,
  blocked, underpowered, falsified, or unresolved;
- no unavailable, unsupported, excluded, or not-observed cell is rendered as a
  numeric zero;
- the complete relevant verification and rendering ladder passes; and
- the claim register says exactly what the evidence does **not** establish.

## Stop conditions

Stop and preserve the result when any of the following occurs:

- a public input contains a protected label or recoverable metadata channel;
- a source, revision, compiler, target, or object cannot be traced;
- a supported instruction disagrees with the declared semantics;
- a model reads held-out labels or an action reads an unrevealed value;
- a confidence interval treats instruction steps or variants as independent
  source families;
- a calibration is reported without a matching machine/software receipt;
- a theorem relies on an undocumented axiom, `sorry`, `admit`, unsafe execution,
  or an assumption that already encodes its conclusion;
- a renderer bypasses a generated variable, figure registry, or accessibility
  field; or
- an independent result is being inferred from a local rerun.

The scientific result of a stopped item is the preserved failure, not a reason
to broaden the endpoint or remove the hard case.

## What remains deliberately unclaimed

This project does not currently claim complete x86-64 semantics, universal
program identifiability, general reverse engineering, malware classification,
application effectiveness, a universal active-sensing law, cross-architecture
semantic equivalence, independent reimplementation, or a Fields/Nobel-level
result. Those statements could become eligible only through the major gates
above and only in the exact form supported by future receipts.

The goal of the roadmap is therefore ambitious but falsifiable: turn a reliable
bounded instrument into a theorem, countertheorem, or independently reproduced
scientific result without confusing implementation completeness with discovery.
