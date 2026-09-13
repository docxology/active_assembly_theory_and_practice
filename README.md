# Active Assembly Theory and Practice

`active_assembly_theory_and_practice` is a private sidecar project that turns
assembly-language traces into categorical observations for Active Inference
experiments. It uses real Capstone disassembly, bounded Unicorn emulation,
native Julia/RxInfer Bethe Free Energy tracking, and a pymdp comparison path on
transparent x86-64 byte fixtures. The observation bridge is extended with a
temporal hidden Markov model, an expected-free-energy active-disassembly policy,
calibration and model-evidence diagnostics, controlled ambiguity and transition
ablations, fixed-parameter generative adequacy checks, and leakage-safe
fixture-level holdouts. It also exposes Capstone register access, a bounded
transfer-function cross-check against Unicorn, stepwise architectural snapshots,
direct-target CFG evidence, static-to-dynamic memory provenance, and a native categorical RxInfer
companion inference path. The Phase A extension also retains typed
Capstone/Unicorn evidence beside the collapsed channel and evaluates
factorized and hierarchical composite scores with fixture-level holdouts,
field-family ablations, and explicit missing-value handling.
The expanded transparent corpus also records bounded memory and alternate-path
coverage in a source-bound design inventory before richer model promotion.
The current research tranche additionally fits an ordered categorical HMM by
observation-only Baum--Welch EM on complete-fixture training complements,
keeping the fixed reference and supervised MAP fits as explicit comparators.
That estimator is a leakage-safe methodological extension, not an unsupervised
semantic-discovery claim.

The current research extensions also include a pinned Lean proof receipt for a
closed Word64 accumulator fragment, a generated Python/Lean boundary fixture,
real Capstone/Unicorn adapter fixtures for x86-64, AArch64, and RISC-V64, and
an exact finite-horizon information-per-cost dynamic program over label-blind
typed sensors. The normalized Python state now carries all 16 general-purpose
registers, explicit operand widths, modeled flag-definedness, bounded
byte-addressable memory, stack bounds, and structured unsupported states. These
advances strengthen the local falsification and formalization surface; they do
not establish complete ISA semantics, cross-compiler invariance, semantic
identifiability, external validity, or a general active-sensing law.
The formal-semantics surface now also runs a smallest-first exhaustive
counterexample search over declared candidate theorems of the bounded
reference model, preserving re-verified minimal witnesses and explicit
holds-in-domain rows as first-class evidence, and the active-sensing tranche
adds a stress-grid robustness study with exact dynamic-programming references,
fixed-seed policy rollouts, repeated calibration-stability rows, and
budget-constrained value gaps. Both are bounded instruments: they narrow
claims and record machine-specific diagnostics; they do not establish
complete ISA semantics, cross-machine cost laws, or external validity.

The current P3--P17 tranche additionally freezes protocol v2, expands the
co-primary quotient benchmark to 64 source families and eight declared
classifier interventions, uses an explicit 16-register width-aware normalized
state model, records exact finite intervention power for the
separate four-state representation pilot, measures auditable sensing costs,
and preserves the licensed-source, blind-replay, and theorem gates as explicit
underpowered, blocked, or unresolved statuses. The pinned external manifest now
contains ten archive/license-verified rows across five projects and two
revisions; static-only compilation and Capstone decoding are recorded in a
separate receipt, while a source-bound contract now records 100 accepted rows
(40 lexical-equivalence witnesses, 40 contract-directed near misses, and 20
non-separating identity controls) with a project-held-out test split. These are
finite identifiability, formal, transfer, sensing, replication, and robustness
diagnostics: weak starts, unresolved controls, unavailable capabilities, and
the fail-closed promotion boundary remain part of the evidence.
It also seals every project-owned JSON artifact with a receipt layer distinct
from the domain schema and carries that boundary through the template-backed
manuscript and accessible document pipeline.

The package also maintains a machine-readable scientific program: its active
frontier is quotient-state identifiability and representation invariance, with
explicit theorem, counterexample, external-validity, cost, replication, and
stop-condition gates. The accompanying equivalence/near-miss protocol uses real
Capstone and bounded Unicorn execution on transparent fixtures. These artifacts
make the next research program auditable; they are not evidence that the current
demonstrator has already established semantic identifiability or
cross-architecture transfer. The representation benchmark also includes a
deliberately insufficient single-state intervention catalog that leaves some
near misses unresolved, making intervention-separation limits visible rather
than hiding them behind the primary witness rate.

The project is intentionally narrow: it demonstrates an explicit bridge between
machine-code observations and a discrete generative model. It does not claim
general reverse-engineering automation or malware classification.

The implementation tranches through the current bounded package are closed;
the remaining work is organized by evidence gate in
[`docs/roadmap.md`](docs/roadmap.md). That roadmap distinguishes minor
maintenance, medium bounded experiments, and major promotion programs. A
completed local receipt never silently promotes external validity, independent
reimplementation, complete ISA semantics, universal identifiability, or a
general sensing law.

## Run

From the project checkout (`/Volumes/external_drive/Git/projects/working/active_assembly_theory_and_practice`):

```bash
uv run python scripts/run_assembly_inference.py

uv run python scripts/run_formal_verification.py

uv run python scripts/run_rxinfer_backend.py

uv run python scripts/validate_analysis_outputs.py

uv run python scripts/z_generate_manuscript_variables.py

uv run python scripts/run_scientific_program.py all

uv run python scripts/run_scientific_program.py probe-capabilities

uv run python scripts/run_scientific_program.py prepare-external

uv run python scripts/run_scientific_program.py validate-external

uv run python scripts/run_scientific_program.py generate-public

uv run python scripts/run_scientific_program.py replay --offline

uv run python scripts/run_scientific_program.py verify-public

# Acquisition is the only network-enabled command and is always explicit.
uv run python scripts/run_scientific_program.py acquire-external \
  --manifest configs/external_sources.yaml \
  --allow-network

# Operator-side replay consumes only output/replication/public/.
uv run python scripts/run_scientific_program.py verify-public
uv run python scripts/run_scientific_program.py replay --offline
uv run python scripts/run_scientific_program.py run-model \
  --model calibrated_interpretable

# Optional same-host, separately controlled replay in the pinned Lima VM.
# All VM state and receipts are stored on /Volumes/blue.
limactl validate configs/replay_vm.yaml
uv run python scripts/run_scientific_program.py vm-probe
uv run python scripts/run_scientific_program.py vm-create --allow-network
uv run python scripts/run_scientific_program.py vm-provision --allow-network
uv run python scripts/run_scientific_program.py vm-replay \
  --model calibrated_interpretable --seed 0
uv run python scripts/run_scientific_program.py vm-attest-replay
uv run python scripts/run_scientific_program.py vm-stop

uv run python scripts/run_scientific_program.py audit-artifacts

uv run pytest tests \
  --cov=src \
  --cov-fail-under=90

uv run python scripts/run_julia_tests.py
```

The first `generate-public` call freezes the public toolchain receipt. An
intentional protocol or public-bundle refresh must explicitly reseed that
surface; manuscript or figure regeneration does not silently replace it.
Acquisition is the only network-enabled operation, and operator-side replay
consumes only the label-blind `output/replication/public/` directory.

After rendering through the sibling template checkout, validate the standard
DOCX and tagged PDF as separate delivery surfaces:

```bash
uv run python scripts/validate_docx_render.py
uv run python scripts/validate_pdf_accessibility.py
```

Render through the sibling template engine:

```bash
cd /Volumes/external_drive/Git/template
uv run python -m infrastructure.orchestration link-projects --dry-run
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_assembly_theory_and_practice \
  --skip-manuscript-hydration
```

The project-side `z_generate_manuscript_variables.py` command above is the
authoritative hydration step. The render command is deliberately run with
`--skip-manuscript-hydration` so that the template consumes the already
validated, source-bound manuscript tree; this also avoids a second hydration
pass with a shorter renderer timeout. If the source, protocol, or generator
changes, rerun project-side hydration before rendering again.

## Outputs

- `output/data/disassembly_trace.json`
- `output/data/emulation_trace.json`
- `output/data/inference_result.json`
- `output/data/trace_bundle.json`
- `output/data/symbolic_execution.json`
- `output/figures/trace_refinement.png`
- `output/data/cfg_evidence.json`
- `output/data/memory_evidence.json`
- `output/data/held_out_model.json`
- `output/data/generative_adequacy.json`
- `output/data/typed_observations.json`
- `output/data/factorized_observation_model.json`
- `output/data/corpus_coverage.json`
- `output/data/action_sensing.json`
- `output/data/parameter_learning.json`
- `output/data/scientific_program.json`
- `output/data/quotient_invariance_protocol.json`
- `output/data/representation_invariance_benchmark.json`
- `output/data/intervention_power.json`
- `output/data/formal_verification.json`
- `output/data/cross_architecture_fixtures.json`
- `output/data/active_information_frontier.json`
- `output/data/quotient_classifier.json`
- `output/data/formal_trace_correspondence.json`
- `output/data/compiler_transfer.json`
- `output/data/sensor_cost_receipt.json`
- `output/data/replication_manifest.json`
- `output/data/theorem_or_countertheorem.json`
- `output/data/counterexample_search.json`
- `output/data/sensing_stress.json`
- `output/docx/active_assembly_theory_and_practice_combined.docx`
- `output/data/docx_render_receipt.json`
- `output/data/scholarship_sources.json`
- `output/data/claim_register.json`
- `output/data/reproducibility_manifest.json`
- `output/data/validation_report.json`
- `output/data/manuscript_variables.json`
- `output/rxinfer/free_energy_trace.jsonl`
- `output/rxinfer/posteriors.json`
- `output/rxinfer/diagnostics.json`
- `output/rxinfer/model_config.json`
- `output/rxinfer_categorical/categorical_free_energy_trace.jsonl`
- `output/rxinfer_categorical/categorical_posteriors.json`
- `output/rxinfer_categorical/categorical_diagnostics.json`
- `output/rxinfer_categorical/categorical_model_config.json`
- `output/figures/observation_pipeline.png`
- `output/figures/posterior_beliefs.png`
- `output/figures/package_provenance.png`
- `output/figures/trace_feature_matrix.png`
- `output/figures/scholarship_source_roles.png`
- `output/figures/rxinfer_free_energy.png`
- `output/figures/inference_diagnostics.png`
- `output/figures/scholarship_evidence_matrix.png`
- `output/figures/claim_evidence_map.png`
- `output/figures/artifact_hash_manifest.png`
- `output/figures/temporal_beliefs_x86_accumulator.png`
- `output/figures/temporal_beliefs_x86_bitwise.png`
- `output/figures/temporal_beliefs_x86_guard.png`
- `output/figures/temporal_beliefs_x86_long_accumulator.png`
- `output/figures/temporal_beliefs_x86_memory.png`
- `output/figures/temporal_beliefs_x86_guard_taken.png`
- `output/figures/model_comparison.png`
- `output/figures/active_efe.png`
- `output/figures/assembly_cfg.png`
- `output/figures/assembly_memory_timeline.png`
- `output/figures/rxinfer_categorical_free_energy.png`
- `output/figures/calibration_curve.png`
- `output/figures/confusion_matrix.png`
- `output/figures/ambiguity_ablation.png`
- `output/figures/segmentation_ribbons.png`
- `output/figures/corpus_stability.png`
- `output/figures/phase_recovery.png`
- `output/figures/decoder_calibration.png`
- `output/figures/generative_model.png`
- `output/figures/observation_informativeness.png`
- `output/figures/model_evidence.png`
- `output/figures/transition_ablation.png`
- `output/figures/validation_summary.png`
- `output/figures/generative_adequacy.png`
- `output/figures/typed_observation_matrix.png`
- `output/figures/factorized_observation_comparison.png`
- `output/figures/corpus_coverage.png`
- `output/figures/action_sensing.png`
- `output/figures/parameter_learning.png`
- `output/figures/intervention_power.png`
- `output/figures/cross_architecture_benchmark.png`
- `output/figures/formal_verification_receipt.png`
- `output/figures/active_information_frontier.png`
- `output/figures/quotient_classifier.png`
- `output/figures/compiler_transfer.png`
- `output/figures/formal_correspondence.png`
- `output/figures/artifact_receipt_closure.png`
- `output/figures/sensor_costs.png`
- `output/figures/replication_manifest.png`
- `output/figures/counterexample_search.png`
- `output/figures/sensing_stress.png`
- `output/figures/external_label_coverage.png`
- `output/figures/README.md` (generated caption ledger for every figure)
- `output/data/figure_registry.json` (generator, inputs, statistics, claims,
  alt text, long descriptions, and machine visual-QA receipt)
- `output/figures/figure_registry.json` (template-renderer boundary projection;
  validation requires exact labels, filenames, row count, and alt text)
- `output/data/manuscript_variable_registry.json` (typed values, formatting,
  consumers, input hashes, and template-hydration contract)
- `output/data/build_provenance.json` (target/template revisions and lineage)
- `manuscript/assets/active_assembly_cover.png` (config-driven paper cover asset)

## Reproducibility notes

- Every figure is regenerated from project artifacts before rendering, and
  `output/figures/README.md` records its caption and accessibility metadata.
- Every project-owned JSON artifact is atomically written with a protocol-v2
  receipt envelope containing the protocol digest, producer revision, input
  hashes, toolchain versions, status, limitations, and explicit unsupported or
  underpowered cases. `seal-artifacts` is an explicit migration step for old
  payloads; `audit-artifacts` is read-only and fails closed. Domain schema
  versions remain separate from the receipt schema.
- The native Julia/RxInfer backend checks both the committed
  `julia/ActiveAssemblyRxInfer/Manifest.toml` and the runtime diagnostics
  against the pinned `EXPECTED_JULIA_VERSION` and `EXPECTED_RXINFER_VERSION`
  in `src/rxinfer_bridge.py`. The generic `julia` launcher can auto-advance on
  juliaup's `release` channel; when the pinned version is installed,
  `resolve_julia_executable()` prefers that exact binary so regeneration stays
  reproducible. `scripts/run_julia_tests.py` is the canonical integration gate:
  it validates the lockfile, resolves the pinned executable, checks
  `julia --version`, and only then runs the real Julia `Pkg.test()` suite.
  Bumping either pin requires a deliberate manifest change plus a fresh backend
  run.
- The validation gate (`scripts/validate_analysis_outputs.py`) recomputes each
  tracked artifact's SHA-256 and byte size and asserts the current generated
  validation surface, including strict manuscript-token and figure-registry
  coverage.
  All deterministic gates can be run together with `scripts/run_full_verification.sh`
  (add `--julia` to also run the Julia backend `Pkg.test()`); it fails fast and
  prints a pass/fail table.
The pinned Julia/RxInfer toolchain (see `EXPECTED_JULIA_VERSION` /
`EXPECTED_RXINFER_VERSION` in `src/rxinfer_bridge.py`) must be present; the gate
fails loudly on toolchain drift, so a fresh clone should install the pinned
Julia first (e.g. `juliaup add 1.12.7`).
