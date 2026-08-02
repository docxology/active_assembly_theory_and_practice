# Methodology {#sec:methodology}

The scaffold starts with a fixed x86-64 fixture whose instruction sequence is
known and auditable. Capstone decodes the bytes into address, mnemonic, operand,
and byte-span records. Unicorn then executes the same bytes under a bounded
instruction count and records the executed addresses and final register state.
These evidence streams are joined into trace schema `{{TRACE_SCHEMA_VERSION}}`
with sample hash `{{BINARY_SHA256}}`. The bundle records
`{{AVAILABLE_PACKAGE_COUNT}}` available package adapters out of
`{{TOTAL_PACKAGE_COUNT}}` total adapter slots, including unavailable optional
packages as explicit provenance rather than silent assumptions.

The feature layer maps instruction mnemonics into six observation categories:
load immediate, arithmetic update, bitwise transform, comparison, control
transfer, and other. These observations are supplied to a discrete likelihood
matrix over five latent program-state classes: setup, accumulator update,
representation transform, guard check, and exit. The posterior update uses
pymdp's softmax routine over log prior and log likelihood terms, producing one
belief vector and one comparison free-energy diagnostic per observation step.

The native RxInfer layer is invoked through `julia --project` and reads the
same trace bundle. The current model maps observation IDs into numeric event
codes and runs a conjugate Gaussian observation model with `free_energy=true`.
Every exposed `result.free_energy` value is written to
`output/rxinfer/free_energy_trace.jsonl` with data and configuration hashes,
following the RxInfer static-inference interface [@rxinfer_static_docs]. The
Python-to-RxInfer integration is a native Julia CLI call because the RxInfer
documentation identifies the server/client route as third-party and not
maintained by the core team [@rxinfer_python_docs]. This is an implemented
Bethe Free Energy spine, not yet the planned categorical hidden Markov model
over instruction phases.

![Assembly observation pipeline generated from decoded bytes.]({{OBSERVATION_FIGURE}}){#fig:observation-pipeline}

![Adapter provenance for required and optional assembly packages.]({{PACKAGE_PROVENANCE_FIGURE}}){#fig:package-provenance}

![Trace feature matrix linking instruction events, execution status, observation IDs, and category channels.]({{TRACE_FEATURE_FIGURE}}){#fig:trace-feature-matrix}

## Temporal, active, and evaluated inference {#sec:advanced-methods}

The v0.1 belief update is a sticky filter that carries the previous posterior
forward unchanged and therefore omits state dynamics — implicitly assuming an
identity transition over latent phases. The v0.2 model adds a left-to-right
transition matrix over the five program phases, yielding a genuine discrete
hidden Markov model. Forward filtering alternates a predict step (the transition
matrix applied to the previous posterior) with a categorical likelihood update;
Viterbi decoding recovers the most probable joint phase path; and
forward–backward smoothing refines the per-step beliefs
[@friston2017activeinference;@rabiner1989tutorial]. Contiguous program phases are then segmented
directly from the smoothed beliefs.

The complete Bayesian graph of this model is drawn in Figure
`fig:generative-model`: the hidden phase states form the left-to-right Markov
backbone, each emits a categorical observation through the likelihood `A`, the
initial belief is fixed by the prior `D`, and the active layer scores candidate
observations through the expected free energy.

![Generative model as a Bayesian graph]({{GENERATIVE_MODEL_FIGURE}}){#fig:generative-model}
*Bayesian-graph (plate) diagram of the generative model. Hidden phase states
form a left-to-right Markov chain (transition `B`) that emits categorical
observations (likelihood `A`); `D` is the initial prior. The active layer scores
candidate observations via the expected free energy `G(pi)`. Rendered from
constants, so it is deterministic.*

![Temporal HMM forward-filter beliefs for the accumulator fixture]({{TEMPORAL_BELIEFS_FIGURE}}){#fig:temporal-beliefs}
*Forward-filter posterior over the five program phases at each observation step
for the `x86_accumulator` fixture. Rows are normalized posteriors; the temporal
model lets the latent phase evolve across steps (via the transition matrix `B`)
rather than being re-estimated independently. Reading the most-probable phase
down the column recovers `setup -> accumulator_update ->
representation_transform -> accumulator_update` once the `shl` is decoded.*

To make the *active* in Active Inference explicit, an expected-free-energy
policy scores each candidate observation by an epistemic term (the reduction in
phase-belief entropy) and a pragmatic term (a log-preference over the
observation categories), and selects the observation that minimises expected
free energy [@friston2017activeinference]. This realises an active-disassembly
ordering in which the agent chooses which instruction to read next rather than
scanning left to right.

We flag one deliberate deviation from the canonical terminology. Each
candidate's observation category is already decoded before selection, so the
epistemic term here is computed as the *realized* information gain of the
observation actually revealed — the entropy reduction from that specific
category — rather than a full expectation over the posterior-predictive
distribution of possible future observations, which is what the canonical
expected-free-energy epistemic term averages over. For a single realized
observation this quantity can even be negative, whereas the true expectation is
guaranteed non-negative. The policy uses the realized gain because it is exact
and cheap given the known category; readers should nonetheless interpret the
"expected free energy" label accordingly.

A labelled corpus of `{{CORPUS_FIXTURE_COUNT}}` transparent byte fixtures, each
carrying a hand-authored ground-truth phase sequence aligned to its decoded
instructions, supports the project's first quantitative evaluation: decoding
accuracy, calibration, a free-energy convergence summary, and a
static-versus-temporal-versus-Viterbi comparison, each guarded by shuffled-label
and uniform-belief negative controls so that a reported gain cannot be vacuous
[@heins2022pymdp].
