# Scholarship and Claim-Boundary Contract

The source ledger lives in `src/scholarship.py` and is generated into:

- `output/data/scholarship_sources.json`
- `output/data/scholarship_matrix.md`
- `output/figures/scholarship_source_roles.png`
- `output/figures/scholarship_evidence_matrix.png`

Each source has a citation key, source type, evidence tier, project stage,
project role, and claim boundary. The validator checks that each ledger key
exists in `manuscript/references.bib`, is cited in the manuscript, uses an
approved source type, stage, and tier, has an HTTPS URL, has a valid DOI shape
when a DOI is present, and carries non-empty role and boundary text. This
prevents quiet growth of unused or unsupported scholarship claims.

## Fresh metadata review

On 2026-08-16, the current bibliography was spot-checked against primary or
authoritative landing pages before the source-bound render was accepted. The
review used web search with the following inclusion rule: retain a source when
the landing page or publisher record supports the exact bibliographic metadata
or the narrowly stated software/ISA role; do not infer project results from a
software paper or documentation page. No new source was added solely to make a
claim appear stronger.

| Review target | Query / authority | Decision |
| --- | --- | --- |
| RxInfer | DOI `10.21105/joss.05161`; JOSS/TU Eindhoven record | Retained as the peer-reviewed implementation reference; local JSONL values remain project outputs. |
| pymdp | DOI `10.21105/joss.04098`; JOSS record | Retained as the comparison-software reference; not conflated with RxInfer free energy. |
| x86-64 formal semantics | DOI `10.1145/3314221.3314601`; Illinois Formal Systems Laboratory / PLDI record | Retained as the formal-semantic contrast; corrected author metadata to Sandeep Dasgupta, Daejun Park, Theodoros Kasampalis, Vikram S. Adve, and Grigore Rosu. |
| Binomial intervals | DOI `10.1214/ss/1009213286`; Project Euclid record | Retained for Wilson-interval reporting; the intervals are descriptive for dependent fixture steps, not population-level uncertainty. |
| Capstone and Unicorn | Official project documentation and feature pages | Retained as API and instrumentation authorities; optional adapter names remain roadmap/context evidence only. |
| Ordered-HMM parameter learning | Dempster--Laird--Rubin (1977), JRSS B publisher record | Added as the primary EM-method source; it motivates incomplete-data estimation but does not support the package's fixture results or semantic claims. |
| Clustered uncertainty and multiplicity | Cameron--Gelbach--Miller (2008), MIT Press; Holm (1979), JSTOR record | Added to bind the source-family bootstrap and Holm-adjusted comparison family; neither source supplies population validity for this finite fixture benchmark. |
| Behavioral equivalence and information gain | Sangiorgi (2011), Cambridge University Press; Lindley (1956), Project Euclid | Added to distinguish the finite observation quotient from bisimulation and to ground the model-derived expected-information objective without implying a universal sensing law. |

## 2026-09-17 claim and citation audit

On 2026-09-17, a full closure audit recorded 47/47 ledger keys cited 1:1 with
`manuscript/references.bib`, and 40/40 claim-register entries validated and
spot-checked against their receipts. That historical audit did not cover dedicated Rust compiler documentation
or the updated role of the existing `nasm` manual citation. This audit introduced no new claim and changed no claim
boundary.

## Source Roles

The 2026-10-16 categorical-energy revision uses the RxInfer 5.4.0 source
documentation pinned to commit `b4335792b4c8fbb2cdb5b560491f2443c97e7e35`:
the Bethe-free-energy manual and message-passing concepts chapter. Their role
is to explain the objective and scheduling boundary. They do not supply the
project's numerical results. The stationary fixed-chain trace and native
prefix-evidence values are project outputs, checked against an independent
scaled-forward calculation. Prefix evidence is not presented as convergence,
and one-sweep exactness is restricted to the declared structured tree model.

- Free Energy Principle and active inference theory: Friston 2009 and Friston
  et al. 2017.
- Inference software: RxInfer JOSS paper, RxInfer static-inference docs, RxInfer
  Python integration docs, and pymdp JOSS paper.
- Assembly formalism: Intel's architecture manuals define the ISA reference
  frame used to describe bytes, operands, and register effects; the official
  Capstone and Unicorn technical presentations document the concrete metadata
  and instrumentation interfaces used by the package.
- Formal-semantics boundary: Dasgupta et al.'s peer-reviewed x86-64 semantics
  establish the distinction between a complete ISA semantics and this package's
  deliberately bounded transfer-function subset.
- Quotient semantics: Sangiorgi supplies the behavioral-equivalence and
  bisimulation vocabulary; the package's finite observational quotient is
  explicitly weaker than a transition-system bisimulation.
- Statistical reporting: Brown, Cai, and DasGupta motivate Wilson score
  intervals for small binomial hit counts; the project still treats them as
  descriptive because fixture steps are dependent. Cameron, Gelbach, and
  Miller motivate grouped resampling, Holm supplies the declared step-down
  multiplicity adjustment, and Phipson--Smyth motivates the Monte Carlo
  plus-one rule. None establishes independent source lineages or the
  conditional prediction-exchangeability assumption; see [statistics](statistics.md).
- Active sensing: Lindley's expected-information formulation motivates the
  finite model-derived information objective; it does not calibrate physical
  sensor time or establish a general policy theorem.
- Latent-variable estimation: Dempster, Laird, and Rubin provide the EM
  incomplete-data estimator used by the observation-only ordered-HMM companion;
  the package reports its finite held-out behavior separately from the method's
  general statistical motivation.
- Assembly evidence packages: Capstone, Unicorn, angr, Rizin, radare2,
  LLVM objdump, Keystone, NASM, Ghidra, Qiling, Miasm, Dyninst, RetDec, and
  McSema.
- Binary-analysis context: the angr-adjacent SoK, BAP, and GTIRB.

## Evidence Tiers

- `implemented_backend`: RxInfer artifacts produced by the project.
- `implemented_dependency`: required Capstone and Unicorn trace producers.
- `comparison_baseline`: pymdp diagnostics.
- `roadmap_adapter`: real packages retained as context-only extension
  candidates; no evidence stream is assumed until a producer and receipt exist.
- `binary_analysis_context`: peer-reviewed or technical binary-analysis context.
- `theory`: FEP and active-inference background.
- `integration_guidance`: documentation that constrains how packages are wired.
- `assembly_formalism`: authoritative ISA documentation used to constrain
  assembly notation; it is contextual evidence, not a project-run backend.
- `statistical_method`: peer-reviewed interval-estimation methodology used to
  document a reporting choice; it is not a claim of population uncertainty.
- `formal_verification`: formal-methods documentation used to identify the
  checked proof-assistant toolchain and its trusted-kernel boundary; it does
  not certify the project's executable ISA adapters.

## Boundary Rule

The manuscript may say that assembly traces are observations for an explicit
generative model. It must not say that:

- the toy byte fixture is a general reverse-engineering system;
- finite RxInfer Bethe Free Energy proves semantic understanding;
- optional adapters produced facts when they were only probed;
- Free Energy Principle citations imply a binary is an active biological agent.

## 2026-10-13 targeted primary-source review

The NASM manual's running/format section and the official rustc command-line
and code-generation chapters were checked against the static compiler-transfer
interfaces. The existing `nasm` role now describes optional transfer cells,
and `rustc_manual` is cited at the same boundary. These sources document
interfaces; they do not certify compiler correctness or behavioral transfer.

Phipson and Smyth (2010; DOI `10.2202/1544-6115.1585`) was checked against the
authors' corrected manuscript and added as `phipson2010permutation` for the
inclusive-tail, plus-one Monte Carlo rule. It does not establish the benchmark's
family-exchangeability assumption or retroactively preregister a correction.
All added keys must pass the existing ledger/BibTeX/manuscript closure gate.

The DOI-bearing ledger was also checked through Crossref metadata: 16 of 21
records were retrieved, with title/year agreement after stripping publisher
HTML markup. Five Crossref lookups were unavailable (three rate limits, plus
JSTOR and Zenodo identifiers outside the retrieved Crossref records); these
are retrieval limitations rather than evidence of invalid sources. The RxInfer
JOSS publisher page independently confirmed its title, authors and 2023 date.
This metadata audit does not certify every source's full scientific content.

Four unavailable Crossref lookups were resolved through primary records:
[JOSS](https://joss.theoj.org/papers/10.21105/joss.05161),
[Rabiner's publication list](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/publications.html),
[Cambridge](https://www.cambridge.org/core/books/introduction-to-bisimulation-and-coinduction/8B54001CB763BAE9C4BA602C0A341D60), and
[Zenodo](https://zenodo.org/records/7867028). The Cambridge landing URL is
corrected and the RO-Crate citation identifies the pinned 1.1.3 release.
The JSTOR landing title resolves, but its detailed metadata was not accessible
in this follow-up; the prior record and citation remain rather than inferring
verification from an unavailable page.

## 2026-10-15 executable-model and interpretation review

The implemented GNN software is now cited at its exact source revision
`46459949944e3217f8a6a804d4ff8c52829ef275`, using the upstream
[citation file](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/46459949944e3217f8a6a804d4ff8c52829ef275/CITATION.cff).
The software version is 4.1.0; the model language declaration remains GNN v1.1.
The concept DOI does not identify an archived copy of this exact revision.
The software citation supports parser attribution, while local receipts establish
the stricter admitted profile and actual execution.

The pymdp role now includes the prospective GNN Agent as well as its separate
comparison diagnostic. The implementation's utility-plus-state-information-gain
objective and negative-EFE sign were checked against the
[versioned source](https://github.com/infer-actively/pymdp/blob/v1.0.2/pymdp/control.py).
The Holm ledger role now correctly identifies a retrospective correction;
adjustment supplies neither valid exchangeability assumptions nor prospective
registration. Transformation-family identifiers are distinguished from their
dependent authored template lineages throughout the main interpretation.

The role partition requires complete, unique source records and exact stage and
tier totals. Both pymdp and RxInfer must be attributed as implemented backends.
A comparison-only source is optional: the comparison diagnostic does not erase
pymdp's implemented prospective role. Real corrupted ledger copies exercise
count mismatches, duplicate or missing sources, and incorrect role assignments.

## Current maintenance boundary

Dated metadata reviews above apply to their recorded source sets. Current
ledger/BibTeX/manuscript closure is generated and validated rather than inferred
from historical totals. The offline docs check also rejects unknown citation
keys in authored guides; it does not revisit publisher records or certify a
citation's scientific relevance. New scholarship must identify the precise
method, interface or limitation it supports and obtain primary-source review.
