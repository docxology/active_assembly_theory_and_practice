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

## Source Roles

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
  Miller motivate resampling at the source-family level, and Holm supplies the
  declared step-down multiplicity adjustment.
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
