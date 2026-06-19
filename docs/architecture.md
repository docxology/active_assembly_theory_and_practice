# Architecture

The project follows the template code-project pattern:

1. `src/assembly_packages/` records adapter provenance and the shared trace
   bundle schema.
2. `src/disassembly.py` decodes transparent byte fixtures with Capstone.
3. `src/emulation.py` runs bounded Unicorn emulation on the same bytes.
4. `src/features.py` maps instruction events to categorical observations and
   stable observation-label vocabularies.
5. `src/generative_model.py` defines normalized likelihood and prior tensors.
6. `src/active_inference_bridge.py` performs pymdp-backed comparison updates.
7. `julia/ActiveAssemblyRxInfer/` runs native RxInfer and records Bethe Free
   Energy values.
8. `src/rxinfer_bridge.py` launches Julia, validates RxInfer outputs, and joins
   figures and manuscript variables back into the Python artifact set.
9. `src/scholarship.py` maintains the source ledger and claim boundaries.
10. `src/claim_register.py` maps bounded claims to source keys, generated
    artifacts, and validation checks.
11. `src/redteam_report.py` stores verifier-first critique as generated
    artifacts.
12. `src/reproducibility.py` writes SHA-256 and byte-size manifests for the
    generated artifact surface.
13. `src/validation.py` validates trace semantics, RxInfer companion artifacts,
    figure content, scholarship, and manuscript artifacts.
14. `src/experiments.py` orchestrates the domain modules and writes artifacts.

### v0.2 temporal / active / evaluation layer

15. `src/temporal_model.py` defines the temporal generative model with a
    left-to-right transition matrix `B` and implements HMM forward filtering,
    Viterbi decoding, and forward–backward smoothing. This is the proper fix for
    the v0.1 sticky filter, which omitted state dynamics (implicit `B = I`).
16. `src/active_selection.py` implements the expected-free-energy policy
    (epistemic + pragmatic value) and the greedy active-disassembly ordering —
    the *active* half of Active Inference.
17. `src/phase_segmentation.py` segments a trace into contiguous program phases
    from smoothed beliefs.
18. `src/corpus.py` defines the labelled multi-fixture corpus, and
    `src/evaluation.py` computes decoding accuracy, calibration, free-energy
    convergence, the static/temporal/Viterbi comparison, and the shuffled-label
    and uniform-belief negative controls.
19. `src/advanced_inference.py` is the thin orchestrator that composes the above
    and writes `temporal_inference.json`, `phase_segments.json`,
    `active_selection.json`, and `evaluation_report.json`. The new claims
    (CLM-010..013), validation checks (`advanced_*`), reproducibility-manifest
    rows, and RedTeam findings (RT-013, RT-014) bind these artifacts to the same
    verifier-first discipline as the v0.1 surface. See `docs/evaluation.md`.

Scripts are thin entrypoints. Manuscript variables are generated from the live
experiment so prose and figures stay aligned with the code.
