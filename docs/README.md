# Documentation

This folder records the project contract (the observation bridge plus the v0.2
temporal / active / evaluation layer). It is the source of truth for how the
pipeline is structured, what it claims, and how it is verified; the generated
artifacts under `output/` are regenerated from the code, while these docs (and
the `manuscript/`, `ISA.md`, and `TODO.md`) are authored.

- `architecture.md` describes the disassembly-to-inference pipeline.
- `assembly_packages.md` defines adapter provenance and the trace schema.
- `rxinfer_roadmap.md` defines the native Julia/RxInfer free-energy spine.
- `evaluation.md` defines the quantitative evaluation: decoders, metrics,
  negative controls, and the (corrected) headline result.
- `scholarship.md` defines citation roles and claim boundaries.
- `claim_map.md` maps key claims to evidence artifacts and boundaries.
- `claim_register.md` defines generated claim rows, source keys, artifacts, and validation checks.
- `redteam.md` defines the durable adversarial findings artifact.
- `reproducibility.md` defines the generated artifact hash manifest and the figure caption ledger (`output/figures/README.md`).
- `analysis_validation.md` defines the generated-output validation oracle.
- `verification.md` lists the local and template-side gates (58-oracle contract).

The project emits 23 figures under `output/figures/`; each has a generated
caption in `output/figures/README.md`.
