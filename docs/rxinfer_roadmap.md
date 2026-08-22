# RxInfer Free-Energy Roadmap

Native Julia/RxInfer is the authoritative backend for Bethe Free Energy in this
project. Python remains responsible for assembly package orchestration, artifact
layout, manuscript token generation, and comparison baselines.

## Implemented Spine

The first model lives in `julia/ActiveAssemblyRxInfer/` and is invoked through:

```bash
uv run python scripts/run_rxinfer_backend.py
```

This project-level bridge is the canonical invocation: it regenerates the
trace bundle, validates the committed Julia manifest, resolves the pinned
Julia executable, checks the returned Julia/RxInfer diagnostics, and writes
both the Gaussian and namespaced categorical artifacts. The lower-level Julia
CLI remains available for package development, but it should be run only after
`uv run python scripts/run_julia_tests.py` has established that the selected
environment satisfies the project pins.

It reads observation IDs from the trace bundle as numeric event-code evidence,
runs a conjugate Gaussian observation model with `free_energy=true`, and writes
every exposed `result.free_energy` value to:

```text
output/rxinfer/free_energy_trace.jsonl
```

The categorical companion is invoked against the same bundle with:

```bash
uv run python scripts/run_rxinfer_backend.py
```

The first model is intentionally conservative. It records real RxInfer Bethe
Free Energy values and remains the authoritative project spine. The Julia
package also exposes an additive categorical phase-HMM through
`--model categorical`; the bridge writes it under `output/rxinfer_categorical/`
so its model-specific free-energy scale cannot be mistaken for the Gaussian
trajectory. The pymdp path is retained only as a comparison baseline and does
not stand in for RxInfer Bethe Free Energy.

## Required Free-Energy Fields

Each JSONL record must contain:

- `run_id`
- `sample_id`
- `model_name`
- `mode`
- `iteration`
- `observation_index`
- `value`
- `source_field`
- `is_finite`
- `delta`
- `rxinfer_version`
- `julia_version`
- `data_hash`
- `config_hash`

`source_field` is `result.free_energy` for static inference. Streaming models
must use the corresponding RxInfer engine history field when they are added.

## Model Expansion

> **current status:** Item 1 (a discrete/categorical model over phases) has landed
> as an *additive* model inside the Julia package
> (`julia/ActiveAssemblyRxInfer/`) with its own `Pkg.test()` coverage asserting
> finite Bethe Free Energy output. The Python bridge now executes the categorical
> CLI path, validates its model identity and trace hash, writes a separate
> artifact namespace, and renders a companion figure. The Gaussian spine remains
> authoritative for the project's headline RxInfer value; the categorical model
> is implemented and tested but is not silently substituted into that spine. The
> Python side independently gained a genuine temporal HMM
> (`src/temporal_model.py`, forward filter + Viterbi + smoothing) used for the
> measured phase-decoding evaluation (see `docs/evaluation.md`).

1. Discrete HMM over instruction and trace phases.
2. Gaussian state-space model over register deltas and entropy features.
3. Changepoint model for phase transitions.
4. Hierarchical model joining static evidence, dynamic evidence, and symbolic
   facts.
5. Streaming model that records `engine.free_energy_history`,
   `engine.free_energy_final_only_history`, and `engine.free_energy_raw_history`
   when those fields are available.

Categorical RxInfer models should be added only when their Bethe Free Energy
terms are fully supported and tested. If a custom factor or average-energy rule
is required, that rule belongs in the Julia package with tests. The current
companion uses native `Categorical` and `DiscreteTransition` factors with fixed
left-to-right transition and emission matrices; its finite values are not
evidence that the parameters were learned or that assembly semantics were
recovered.

## Artifact Set

The bridge expects:

- `output/rxinfer/free_energy_trace.jsonl`
- `output/rxinfer/posteriors.json`
- `output/rxinfer/diagnostics.json`
- `output/rxinfer/model_config.json`
- `output/rxinfer/stdout.log`
- `output/rxinfer/stderr.log`
- `output/figures/rxinfer_free_energy.png`
- `output/rxinfer_categorical/categorical_free_energy_trace.jsonl`
- `output/rxinfer_categorical/categorical_posteriors.json`
- `output/rxinfer_categorical/categorical_diagnostics.json`
- `output/rxinfer_categorical/categorical_model_config.json`
- `output/figures/rxinfer_categorical_free_energy.png`
- `output/figures/inference_diagnostics.png`
- `output/data/validation_report.json`
- `output/figures/validation_summary.png`

Python validation rejects missing files, nonfinite free-energy values, wrong
source fields, stale companion artifacts, posterior iteration mismatches, and
trace-hash mismatches. `output/figures/inference_diagnostics.png` is generated
as a two-panel figure so readers can see that pymdp comparison diagnostics and
RxInfer Bethe Free Energy are separate recorded objectives rather than a single
interchangeable series.
