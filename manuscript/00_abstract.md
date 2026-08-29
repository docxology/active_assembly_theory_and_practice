# Abstract {#sec:abstract}

`{{PROJECT_TITLE}}` is a source-bound demonstrator for connecting assembly-language
evidence to Active Inference and the Free Energy Principle. Its canonical path
decodes a transparent `{{SAMPLE_NAME}}` x86-64 byte fixture, runs bounded emulation,
and turns `{{INSTRUCTION_COUNT}}` decoded instructions into
`{{OBSERVATION_COUNT}}` typed categorical observations. A native Julia/RxInfer
backend records `{{RXINFER_FREE_ENERGY_COUNT}}` Bethe Free Energy values from the
native `{{RXINFER_MODEL_NAME}}` implementation [@bagaev2023rxinfer], while the
Python/pymdp path remains a comparison baseline [@heins2022pymdp]. The shared
assembly layer includes a width-aware normalized machine state, all sixteen
general-purpose registers, explicit flag definedness, bounded byte memory, and
structured unsupported states. A Python reference interpreter, Capstone decoding,
bounded Unicorn traces, and a Lean formal fragment are compared only over the
declared user-mode integer subset; this is a checked boundary, not complete x86-64
semantics.

The empirical program has two linked surfaces. First, a frozen quotient benchmark
contains `{{QUOTIENT_CLASSIFIER_FAMILY_COUNT}}` source families and
`{{QUOTIENT_CLASSIFIER_ROW_COUNT}}` rows, with
`{{QUOTIENT_CLASSIFIER_HELD_OUT_FAMILY_COUNT}}` families held out from model fitting.
The co-primary portfolio includes an interpretable classifier, a temporal
probabilistic model, a deterministic recurrent classifier, and surface baselines;
the temporal endpoint reaches balanced accuracy
`{{QUOTIENT_CLASSIFIER_TEMPORAL_BALANCED_ACCURACY}}` and macro-F1
`{{QUOTIENT_CLASSIFIER_TEMPORAL_MACRO_F1}}` on this frozen bounded benchmark.
Clustered uncertainty, calibration, leakage controls, and negative controls keep
these values as benchmark endpoints rather than claims of semantic identifiability.
Second, a pinned static-only corpus supplies `{{EXTERNAL_LABEL_PROJECT_COUNT}}`
licensed source projects and `{{EXTERNAL_LABEL_ROW_COUNT}}` source-contract rows,
including `{{EXTERNAL_LABEL_EQUIVALENT_COUNT}}` lexical witnesses,
`{{EXTERNAL_LABEL_NEAR_MISS_COUNT}}` contract-directed near misses, and
`{{EXTERNAL_LABEL_NEGATIVE_COUNT}}` identity controls. The corpus establishes
provenance and coverage; it does not execute arbitrary external binaries or support
application-effectiveness claims.

Finite active sensing is certified by exact dynamic programming with auditable
deterministic costs and an optional machine-specific calibration. The local frozen
replication harness separates public inputs from protected labels. A separately
controlled replay is currently `{{INDEPENDENT_REPLAY_STATUS}}` in the recorded
`{{INDEPENDENT_REPLAY_CONTROL_SCOPE}}` scope; this is not an independent physical
machine or independent reimplementation. Every result, figure, caption, manuscript
variable, and document surface is generated from hashed receipts. The contribution
is therefore a reproducible, formally bounded and statistically audited research
substrate for assembly-to-inference experiments, with explicit negative results and
promotion boundaries. It is not a general reverse-engineering system.
