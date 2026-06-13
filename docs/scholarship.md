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

## Source Roles

- Free Energy Principle and active inference theory: Friston 2009 and Friston
  et al. 2017.
- Inference software: RxInfer JOSS paper, RxInfer static-inference docs, RxInfer
  Python integration docs, and pymdp JOSS paper.
- Assembly evidence packages: Capstone, Unicorn, angr, Rizin, radare2,
  LLVM objdump, Keystone, NASM, Ghidra, Qiling, Miasm, Dyninst, RetDec, and
  McSema.
- Binary-analysis context: the angr-adjacent SoK, BAP, and GTIRB.

## Evidence Tiers

- `implemented_backend`: RxInfer artifacts produced by the project.
- `implemented_dependency`: required Capstone and Unicorn trace producers.
- `comparison_baseline`: pymdp diagnostics.
- `roadmap_adapter`: real packages that are named but not yet evidence streams.
- `binary_analysis_context`: peer-reviewed or technical binary-analysis context.
- `theory`: FEP and active-inference background.
- `integration_guidance`: documentation that constrains how packages are wired.

## Boundary Rule

The manuscript may say that assembly traces are observations for an explicit
generative model. It must not say that:

- the toy byte fixture is a general reverse-engineering system;
- finite RxInfer Bethe Free Energy proves semantic understanding;
- optional adapters produced facts when they were only probed;
- Free Energy Principle citations imply a binary is an active biological agent.
