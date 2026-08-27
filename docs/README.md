# Documentation

This folder records the maintained project contract: the observation bridge,
temporal and active inference layers, quantitative evaluation, scholarship
boundaries, and scientific roadmap. It is the source of truth for how the
pipeline is structured, what it claims, and how it is verified. Generated
artifacts under `output/` are regenerated from the code; the documents in this
folder and the `manuscript/` sources are authored interfaces.

- `architecture.md` describes the disassembly-to-inference pipeline.
- `assembly_packages.md` defines adapter provenance and the trace schema.
- `rxinfer_roadmap.md` defines the native Julia/RxInfer free-energy spine.
- `evaluation.md` defines the quantitative evaluation: decoders, metrics,
  negative controls, adequacy checks, and the bounded headline results.
- `scholarship.md` defines citation roles and claim boundaries.
- `claim_map.md` maps key claims to evidence artifacts and boundaries.
- `claim_register.md` defines generated claim rows, source keys, artifacts, and validation checks.
- `roadmap.md` defines the promotion criteria for the next scientific phases.
- `scientific_program.md` defines the long-horizon conjecture, falsifiers,
  dependencies, promotion gates, replication requirements, and award-reality
  boundary for ambitious research advancement; its machine-readable receipt is
  `output/data/scientific_program.json`.
- The P3 representation-invariance benchmark is emitted at
  `output/data/representation_invariance_benchmark.json`; it freezes
  source-family-disjoint transformation pairs, near-miss controls, and
  predeclared protocol endpoints without claiming fitted semantic recovery. It
  also retains a deliberately insufficient single-state intervention catalog as
  a negative control, so the primary witness rate cannot be read as a universal
  separation law.
- The P9 intervention-power receipt is emitted at
  `output/data/intervention_power.json`; it exactly enumerates all non-empty
  intervention subsets, records minimum separating sets, and retains the
  deliberately underpowered one-state control. It is a finite diagnostic, not
  semantic identifiability.
- The P10 parameter-learning robustness surface is emitted in
  `output/data/parameter_learning.json` and `output/figures/parameter_learning.png`;
  five deterministic starts per complete-fixture holdout are retained, including
  weak or unstable fits.
- The P11--P17 promotion tranche is emitted through
  `scripts/run_scientific_program.py`: the normalized machine-state and
  generated Lean bridge receipts, 64-family quotient portfolio, compiler and
  static-only external-source manifests, measured sensing-cost certificates,
  label-separated offline replication bundle, theorem/countertheorem ledger,
  source-bound manuscript registries, and protocol-v2 artifact-envelope audit
  are all separate evidence surfaces. Use `all` for the canonical local
  orchestration summary, `seal-artifacts` only for an explicit legacy migration,
  and `audit-artifacts` as a read-only fail-closed check.
  `blocked_external`, `underpowered`, `unavailable`, `falsified`, and
  `unresolved` statuses remain visible and are not promoted to positive claims.
- `reproducibility.md` defines the generated artifact hash manifest, build receipt, and figure/variable registries.
- `analysis_validation.md` defines the generated-output validation oracle.
- `assembly_packages.md` also documents the pinned Lean formal fragment, real
  multi-architecture adapter fixtures, and exact finite-horizon information
  frontier, each with a separate claim boundary.
- `verification.md` lists the local and template-side gates and keeps the oracle count generated rather than hard-coded. It also defines the post-render PDF tag-tree audit and VeraPDF gate.

The project emits the figure set declared by the source-bound registry under
`output/figures/`; each has a
generated caption, population, statistic, citation, alternative text, and
machine-QA row in `output/figures/README.md` and
`output/data/figure_registry.json`. The rendered manuscript additionally emits
`output/data/manuscript_variable_registry.json`,
`output/data/build_provenance.json`, and the post-render
`output/data/pdf_accessibility_report.json`. The last report inspects the real
PDF tag tree, not only its metadata flag; it checks language, marked structure,
heading hierarchy, figure alternatives/captions, formula associated files,
structured links, and table-tag parity with the generated TeX surface.
The sibling template consumes the generated boundary projection at
`output/figures/figure_registry.json`; validation checks that projection is an
exact alternative-text view of the canonical data registry.
The same template configuration emits the standard combined DOCX at
`output/docx/active_assembly_theory_and_practice_combined.docx`; its separate
post-render receipt is `output/data/docx_render_receipt.json`. The DOCX gate
checks OOXML integrity, hydration, figure/caption parity, and—when configured—
the exact cover bytes, cover alternative text, and cover/page-break contract.
The cover is additional to the registered manuscript-figure count. This gate
does not stand in for the PDF/UA or visual reading-order audits.
