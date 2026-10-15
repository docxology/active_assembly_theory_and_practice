# Executable GNN models

The canonical finite categorical models live in
[`configs/gnn/`](../configs/gnn/README.md). The public assembly command
executes their declared A/B/C/D/E tensors with the real pymdp Agent.
The configured
[Generalized Notation Notation repository](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
supplies the actual Markdown parser. The historical identity-transition,
fitted, factorized, hierarchical and Gaussian paths remain explicitly scoped
diagnostics.

## Preparation and commands

Set `GNN_ROOT` to the canonical GNN checkout supplied for this project. Both
that checkout and this project's runtime must reside on the mounted GitVault
volume. The following example uses an already configured volume root:

```bash
export GNN_ROOT="${GITVAULT_ROOT:?set mounted GitVault root}/Username/ActiveInferenceInstitute/Public/Generalized_Notation_Notation"
uv run --locked python scripts/run_gnn_inference.py validate \
  --model configs/gnn/fixture_sensor.md
uv run --locked python scripts/run_assembly_inference.py
uv run --locked python scripts/run_gnn_inference.py native
uv run --locked python scripts/run_gnn_inference.py audit --require-native
```

`--gnn-root` overrides the environment. `run` accepts `--model`,
`--fixture`, `--steps`, `--seed` and `--output-dir` for bounded individual
experiments. Individual runs default to `output/gnn/individual` and use the
model's declared seed unless overridden. `study` generates the canonical runs
and controls. `audit` reads current core outputs without repairing them;
`--require-native` also requires the current Julia comparison. The legacy assembly
diagnostic is available through `run_assembly_inference.py --legacy-diagnostic`.
The complete verification workflow runs the GNN study, native parity and
read-only audit as separate stages.

## Upstream admission and executable subset

The parser runs in an isolated Python subprocess with bytecode writes disabled.
Its imported source files must match both the actual upstream checkout and
their Git blobs at the recorded revision. Dirty or untracked parser sources,
an unexpected origin, parser warnings, changed input bytes and mismatched
tensor shapes or values fail admission. Every upstream-produced tensor is
compared with the strict local contract before it enters the Agent. Receipts
retain the upstream revision, imported-file hashes, model-source SHA-256,
tensor digest and actual package versions.

The supported profile is one UTF-8 Markdown file of at most 64 KiB with ordered
`GNNSection`, `GNNVersionAndFlags`, `ModelName`, `StateSpaceBlock`,
`Connections`, `InitialParameterization`, `Metadata` and optional
`ModelAnnotation` sections. The declarations are `ActInfPOMDP` and
`GNN v1.1`. Numeric tensors use JSON arrays, including explicit singleton
axes. The upstream parser's brace/tuple notation is excluded from this
executable profile because nested values can be misparsed despite a successful
parse result.

The `Metadata` profile extension explicitly declares factor, state, action,
modality and outcome names, A dependencies, horizon, gamma and seed. The local
compiler checks it rather than assuming that upstream interprets this
extension. Connections must declare exactly the D-to-A factor dependencies.
Unsupported sections, expressions, malformed distributions and extra exported
fields fail closed. No expressions or generated Python are evaluated.

A is observation likelihood; B is an action-conditioned transition; C is
finite dimensionless utility; D is an initial prior; E is a policy prior.
Probability columns must sum to one within 1e-10 without normalization.
Tensor rank is at most four, each tensor has at most 4096 entries, horizons
are one to three, and enumeration admits at most 256 policies. The independent
reference additionally bounds joint state allocation. The Agent adapter
rejects zero E support because its supported upstream policy implementation
floors logarithms; the mathematical IR retains structural-zero support.
This is a bounded executable profile, not universal support for every GNN
family or backend.

## Models and interpretation

`passive_phase.md` specifies the authored phase likelihood and temporal
transition used by the categorical companions. Observation zero conditions D;
subsequent observations condition the selected-action prediction. Native
RxInfer consumes the same exported tensors and uses that same timing.
Independent NumPy filtering and smoothing provide the parity reference.

`fixture_sensor.md` factors fixture hypothesis and sensor mode. The Agent
receives only observations, selects idle, static or dynamic sensing, and then
the environment performs the selected Capstone or bounded Unicorn acquisition.
Static acquisition does not distinguish these fixtures. Dynamic evidence
does. Evaluation identity is attached only after decisions. Missing acquisition
and observed-but-unseen evidence have distinct likelihood outcomes.

The policy objective is
`negative_EFE = expected_utility + state_information_gain`, with
`q_pi = softmax(gamma * negative_EFE + log(E))`.
Reported EFE has the opposite sign. Parameter learning and parameter information
gain are disabled. This predictive-policy calculation is distinct from
adaptive Bayes-optimal dynamic programming and from retrospective,
category-informed ordering. Its dimensionless utility/EFE scale is separate
from Gaussian and categorical Bethe free energies.

## Artifacts and controls

`output/gnn/` contains compiled models, runs, trajectories, upstream
qualification, controls, source-bound figure inputs and native comparison.
The audit recomputes priors, posteriors and policy scores, checks the actual
action/acquisition sequence, and binds trajectories back to their run.
It also checks A/B/C/D/E interventions, hidden-fixture invariance, horizon
enumeration, fixed-action and uniform-policy baselines, and native smoothing.
Corrupt and missing receipts remain failures; the audit leaves their bytes
unchanged.

All fixtures are transparent and owned. Their authored phase categories,
illustrative utility values and local acquisition costs establish no recovered
semantics, learned ontology, general reverse-engineering capability or external
validity. The broader manuscript and diagnostics retain their own scientific
and [statistical boundaries](statistics.md).
