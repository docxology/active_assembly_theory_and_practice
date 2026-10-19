# Active Assembly Theory and Practice

Version 0.1.0 archival identifier:
[10.5281/zenodo.23484108](https://doi.org/10.5281/zenodo.23484108).

For the versioned review package, start with the
[review and release guide](docs/release.md). It maps the real inference paths,
primary manuscript, figure atlas, provenance receipts and citation metadata.
Project-owned software and documentation use the [MIT license](LICENSE);
third-party dependencies and acquired source materials retain their own licenses.

The canonical categorical models are executable
[GNN specifications](docs/gnn_models.md), consumed through the configured
Generalized Notation Notation parser and the real pymdp Agent. The assembly
entry point runs the passive phase model and action-before-acquisition sensor
POMDP; source-bound controls and shared-tensor native smoothing are separate
verification stages. Set `GNN_ROOT` before the commands below.

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

Use the committed Python 3.13 pin and install the locked development tools:

```bash
uv sync --locked --extra dev
```

The package requires Python 3.11 or newer (`tomllib` is part of the runtime);
the canonical replay environment is pinned to 3.13. Python 3.15 currently
cannot install the locked JAX wheel. Keep the environment, base interpreter,
caches and temporary files on mounted GitVault; verify both `sys.prefix` and
`sys.base_prefix` before running source commands. See
[external runtime preparation](docs/getting_started.md#locked-environments).

Protocol v2.2 corrects paired significance testing with whole-source-family
prediction swaps. Bootstrap intervals remain descriptive, and the correction
withholds superiority claims pending prospective independent evaluation.
Temporal fixture exclusion folds remain distinct from the declared fixed
partition inventory; see [`docs/evaluation.md`](docs/evaluation.md).

From the project checkout:

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

uv run python scripts/run_scientific_program.py audit-external-preparation

uv run python scripts/run_scientific_program.py generate-public

# Pinned external acquisition is network-capable only with explicit permission.
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
External acquisition and VM setup are explicit network-capable operations;
operator-side replay consumes only the label-blind `output/replication/public/` directory.

For the complete producer, test, hydration, native export and document gate:

```bash
scripts/run_full_verification.sh --julia
```

For the staged workflow, use `--analysis-only --julia`, then
`uv run --locked python scripts/render_manuscript.py`, then
`--post-render-only`. The latter forces strict validation and does not regenerate
scientific inputs. Template environment/registration prerequisites are in
[getting started](docs/getting_started.md); [verification](docs/verification.md)
and [document rendering](docs/document_rendering.md) own the exact gate order.
The [docs index](docs/README.md) routes methods, statistics, source lineage,
replay and remaining scientific promotion gates.

## Outputs

Generated artifacts live under `output/`. The authoritative enumerations are
the reproducibility manifest (`output/data/reproducibility_manifest.json`,
with SHA-256 and byte-size rows for every project-owned artifact) and the
figure registry (`output/data/figure_registry.json`, the registered figures with
generators, inputs, statistics, claims, alt text, long descriptions, and the
machine visual-QA receipt). `output/figures/README.md` is the generated
caption ledger for every figure. Representative paths:

- `output/data/trace_bundle.json`
- `output/data/representation_invariance_benchmark.json`
- `output/data/replication_manifest.json`
- `output/data/validation_report.json`
- `output/figures/temporal_beliefs_x86_accumulator.png`
- `output/rxinfer/free_energy_trace.jsonl`
- `output/docx/active_assembly_theory_and_practice_combined.docx`

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
