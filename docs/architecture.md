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

Scripts are thin entrypoints. Manuscript variables are generated from the live
experiment so prose and figures stay aligned with the code.
