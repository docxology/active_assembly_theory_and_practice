# Canonical executable models

[`passive_phase.md`](passive_phase.md) declares the authored categorical
phase model. [`fixture_sensor.md`](fixture_sensor.md) declares the bounded
action-before-acquisition sensor POMDP. Both are numeric GNN Markdown models
consumed by the real upstream parser and a strict local compiler.

See [execution and interpretation](../../docs/gnn_models.md) for the supported
subset, actual Agent/native interfaces, upstream custody and failure controls.
Model tensor edits change generated digests and require regeneration; generated
receipts are never edited to agree with a specification.
