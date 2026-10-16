# Canonical executable models

[`passive_phase.md`](passive_phase.md) declares the authored categorical
phase model. [`fixture_sensor.md`](fixture_sensor.md) declares the bounded
action-before-acquisition sensor POMDP. Both are numeric GNN Markdown models
consumed by the real upstream parser and a strict local compiler.

`inference/` contains twelve numeric architecture profiles for the remaining
comparators, fitted diagnostics, sensing channels and calibrations. Runtime
parameters specialize these profiles into actual upstream-parsed GNN instances.
The generated `output/gnn/inference/README.md` links the fourteen profiles to
their native upstream graphs and larger-label reader exports. These adapters
implement a bounded declared architecture; they do not imply universal GNN
code generation.

See [execution and interpretation](../../docs/gnn_models.md) for the supported
subset, actual Agent/native interfaces, upstream custody and failure controls.
Model tensor edits change generated digests and require regeneration; generated
receipts are never edited to agree with a specification.
