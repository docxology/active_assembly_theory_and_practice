# Architecture

The project follows the template code-project pattern:

1. `src/assembly_packages/` records adapter provenance and the shared trace
   bundle schema.
2. `src/disassembly.py` decodes transparent byte fixtures with Capstone.
3. `src/emulation.py` runs bounded Unicorn emulation on the same bytes.
4. `src/features.py` maps instruction events to categorical observations and
   stable observation-label vocabularies.
5. `src/typed_observations.py` preserves typed static fields and bounded dynamic
   status in a label-blind schema; `src/factorized_model.py` fits the
   leakage-safe field-wise and hierarchical companion scores.
6. `src/assembly_formalism.py` joins static instruction rows to dynamic
   addresses and categorical labels for the source-bound assembly listing.
7. `src/symbolic_semantics.py` applies the bounded, documented x86 transfer
   functions and checks the concrete terminal state against Unicorn.
8. `src/assembly_packages/schema.py` exposes a direct-target CFG partition;
   `src.advanced_inference.cfg_evidence_payload` preserves the branch-bearing
   fixture's blocks, edge types, external target, and dynamic addresses.
9. `src.advanced_inference.memory_evidence_payload` joins Capstone memory
   operands to Unicorn read/write hooks for a separate transparent store-load
   fixture.
10. `src/generative_model.py` defines normalized likelihood and prior tensors.
11. `src/active_inference_bridge.py` performs pymdp-backed comparison updates.
12. `julia/ActiveAssemblyRxInfer/` runs native Gaussian and categorical RxInfer
   models and records namespaced Bethe Free Energy values.
13. `src/rxinfer_bridge.py` launches both native Julia paths, validates model
   identity and trace hashes, and joins figures and manuscript variables back
   into the Python artifact set.
14. `src/scholarship.py` maintains the source ledger and claim boundaries.
15. `src/claim_register.py` maps bounded claims to source keys, generated
   artifacts, and validation checks.
16. `src/reproducibility.py` writes SHA-256 and byte-size manifests for the
   generated artifact surface.
17. `src/validation.py` validates trace semantics, assembly-listing lineage,
   RxInfer companion artifacts,
   figure content, scholarship, and manuscript artifacts.
18. `src/experiments.py` orchestrates the domain modules and writes artifacts.

### current temporal / active / evaluation layer

19. `src/temporal_model.py` defines the temporal generative model with a
    left-to-right transition matrix `B` and implements HMM forward filtering,
    Viterbi decoding, and forward–backward smoothing. This is the proper fix for
    the identity-transition filter, which omitted state dynamics (implicit `B = I`).
20. `src/active_selection.py` implements the expected-free-energy policy
    (epistemic + pragmatic value) and the greedy active-disassembly ordering —
    the *active* half of Active Inference.
21. `src/phase_segmentation.py` segments a trace into contiguous program phases
    from smoothed beliefs.
22. `src/corpus.py` defines the labelled multi-fixture corpus, and
    `src/evaluation.py` computes decoding accuracy, calibration, free-energy
    convergence, the identity-transition/temporal/Viterbi comparison, and the shuffled-label
    and uniform-belief negative controls.
23. `src/model_fitting.py` fits smoothed categorical HMM parameters on declared
    training fixtures and evaluates them with leave-one-fixture-out holdouts.
24. `src/evaluation.py` also runs fixed-parameter generative adequacy checks:
    same-length HMM simulations are compared with observed transition rates,
    run counts, phase occupancy, and temporal recovery. These intervals are
    descriptive model checks, not population uncertainty.
25. `src.advanced_inference.typed_observation_payload` writes the label-blind
    typed records, and `factorized_observation_payload` writes the complete
    fixture-level comparison, field-family ablation, and missing-field audit.
26. `src/action_sensing.py` adds a leakage-safe action-conditioned sensing
    experiment: typed field queries are scored from training-only predictive
    likelihoods before held-out values are revealed, then compared with fixed,
    seeded-random, and explicitly inadmissible oracle policies under one
    normalized budget. It writes `action_sensing.json`; the figure registry,
    claim register, and validation oracle bind the result separately from the
    already-decoded-category active-selection diagnostic.
27. `src/parameter_learning.py` fits an observation-only ordered categorical HMM
    with scaled forward--backward responsibilities and positive Dirichlet
    updates. Its complete-fixture holdout artifact retains the training
    observation IDs, fit diagnostics, held-out scoring labels, and explicit
    no-labels-in-fit contract; it is separate from the hand-authored reference
    and supervised MAP companion.
28. `src/formal_correspondence.py` defines the normalized 64-bit machine state,
    width-aware instruction IR, bounded memory/stack semantics, modeled flag
    definedness, and stepwise Python reference trace. Real Capstone decoding
    and bounded Unicorn execution are compared on transparent fixtures; the
    path fails closed for unsupported instructions and bounds.
29. `src/formal_verification.py` builds the pinned Lean project, hashes its
    checked sources, and writes a deterministic theorem/build receipt. The
    generated `src.lean_bridge` serializes a full 16-register bounded machine
    state, `RIP`, modeled flags, memory/stack bounds, and step status into a
    generated Lean fixture with kernel-checked state invariants. The `formal/`
    tree is a declared normalized fragment, not complete x86 semantics or a
    proof of the Python/Capstone/Unicorn path.
29. `src/multi_arch.py` runs real Capstone and Unicorn adapters for one
    transparent numeric operation on x86-64, AArch64, and RISC-V64. Its
    receipt is an adapter benchmark with explicit cross-ISA non-claims.
30. `src.information_policy.py` projects the label-blind factorized model into
    finite sensor channels and solves a bounded information-per-cost policy
    problem exactly by dynamic programming. `src.sensor_costs` records an
    auditable deterministic cost vector for each action and, when requested,
    overhead-corrected local timing calibration; cross-machine timing and
    universal cost laws remain out of scope.
31. `src.external_sources.py` accepts only a pinned, license-cleared manifest,
    acquires archives through the explicit network-enabled command, compiles
    locally in static-only mode, hashes source/object/disassembly products, and
    never executes emitted external binaries. `src.external_corpus.py` keeps
    mechanically reviewed source-level label contracts separate from static
    coverage. `src.replication` separates label-blind public inputs from
    private gold labels, hashes the public protocol/toolchain/instructions,
    exposes offline replay, and requires an explicit separate-control
    attestation before protected ingestion can promote a replay receipt.
32. `src.intervention_analysis.py` computes the exact finite intervention-power
    matrix from the source-bound quotient outcomes, enumerating every non-empty
    subset and retaining the minimum separating sets and one-state negative
    control. It never derives labels from model predictions.
33. `src/parameter_learning.py` retains deterministic reference, uniform, and
    three seeded-random starts for each holdout so initialization sensitivity is
    visible in the receipt and figure rather than hidden by selecting one fit.
34. `src.advanced_inference.py` is the thin orchestrator that composes the above
    and writes `temporal_inference.json`, `phase_segments.json`,
    `active_selection.json`, `evaluation_report.json`, `cfg_evidence.json`,
    `memory_evidence.json`, `held_out_model.json`, `generative_adequacy.json`,
    `typed_observations.json`, `factorized_observation_model.json`,
    `action_sensing.json`, `parameter_learning.json`,
    `intervention_power.json`,
    `cross_architecture_fixtures.json`, and `active_information_frontier.json`.
    The claim register, validation checks, and
    reproducibility-manifest rows bind these artifacts to the same source-bound
    discipline as the trace surface. See
    `docs/evaluation.md`.

35. `src.visual_atlas.py` produces the nine narrative visuals from live
    receipts, while `src/figure_contract.py` unifies all 64 substantive figure
    specifications, source hashes, captions, alternatives, long descriptions,
    vector companions, and the offline semantic atlas. Legacy plotting callers
    remain compatible through `FIGURE_CAPTIONS` and the existing path tokens.
36. `src.manuscript_variables.py` projects canonical figure metadata into the
    template variable surface. Embedded figures use registry-backed caption and
    path tokens; unembedded figures are marked explicitly as atlas-only so the
    variable audit does not confuse a complete metadata surface with an unused
    scientific value.
37. `scripts/run_scientific_program.py` exposes `generate-figures`,
    `audit-figures`, and `render-manuscript` as thin orchestration entry points.
    The sibling template remains the renderer of record for DOCX, PDF, HTML,
    and slides.

Scripts are thin entrypoints. Manuscript variables are generated from the live
experiment so prose and figures stay aligned with the code.
