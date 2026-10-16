# Discussion {#sec:discussion}

The bridge is rigorous in a deliberately local sense: the observation
vocabulary, likelihood matrix, prior, package provenance, trace hash, assembly
listing, and update rule are explicit and regenerated from the same artifacts.
The assembly listing keeps instruction bytes, decoded operands, dynamic
addresses, and categorical labels visibly distinct. That distinction matters
because the ISA reference defines architectural effects, whereas a mnemonic
string alone is not a proof of dataflow, control-flow intent, or semantic
equivalence [@intel_sdm]. The project consequently treats Capstone and Unicorn
as evidence-producing mechanisms for a bounded fixture, not as an oracle for
program understanding.

The limitation is equally explicit. The canonical phase fixture is small, its
vocabulary is hand-authored, the headline Active Inference comparison uses fixed
likelihood and transition parameters, and steps within a trace are dependent. The
quotient benchmark with disjoint family identifiers increases the number of synthetic
families and tests a co-primary model portfolio, but its held-out scores remain
bounded benchmark endpoints rather than evidence of population-level semantic
identifiability. The supervised leave-one-fixture-out fit estimates tensors
without using the held-out fixture, yet its small training complements remain a
generalization diagnostic. The normalized state now makes flags, widths, memory
bounds, stack effects, and unsupported instructions explicit; it does not resolve
calling conventions, unbounded aliasing, indirect targets, privilege state,
exceptions, or architecture-specific side effects. Binary-analysis platforms such
as BAP and GTIRB show why intermediate representations and tool interchange become
important as analyses scale [@brumley2011bap; @schulte2019gtirb]. Future modules
can add those features through new checked adapters, but each extension should
enter with its own semantics, tests, provenance, and claim boundary.

The quantitative results are honest and explanatory rather than inflated. On this
corpus the temporal HMM beats the identity-transition filter but not
the memoryless per-step decoder — a non-result that the ablation turns into a
mechanism: because the authored likelihood is near-diagonal, a single observation
already reveals the phase, so transition structure adds little. Sweeping the
likelihood toward uniform, the temporal decoder (and the smoothed temporal
segmentation) is exactly where the advantage appears: per-step decoding collapses
while the transition structure carries the information ([@fig:ambiguity-ablation]).
Phase segmentation is now measured against the hand-authored ground truth
([@fig:segmentation-ribbons]), and both the
boundary-recovery result and the mechanism are registered as bounded claims
(CLM-014, CLM-015). This framing keeps the headline faithful: no net temporal
advantage at an informative likelihood, a clear one under ambiguity, and every
number recomputed by the validation oracle rather than asserted.

The robustness machinery added around the quotient portfolio changes how the
endpoint is defended, not what it claims. Hyperparameter selection is
development-only, five deterministic starts are rerun and their seed ranges are
recorded separately from the source-family clustered intervals, and Platt
calibration is reported alongside — the uncalibrated balanced accuracy and
macro-F1 stay primary and the calibrated metrics gate no claim. The
family-shuffle control, the bounded metadata probe (held to its own boundary:
bounded probe only; no architecture-metadata claim), the declared
near-duplicate template siblings, and the cross-encoding consistency check all
qualify the endpoint rather than strengthen it. No superiority claim is gated
by any of this machinery: the retrospective correction withholds all such
claims, and the recurrent model's at-or-below-chance exclusion from the
within-family label-scramble diagnostic is
declared in the receipt rather than absorbed into a favorable summary. On the
temporal side, the optimization-rewrite fixture plays the analogous negative
role: its `lea` rewrite shares the residual `other` observation category while
its latent phase is accumulator update, and that designed ambiguity is retained
as measured evidence that the corpus can hold cases the decoder does not
resolve.

This is the intended relationship to the Free Energy Principle
[@parr2022activeinference]: assembly traces become sensory evidence for a
generative model whose beliefs are updated under an explicit variational
objective. The actual GNN parser supplies the admitted model tensors, pymdp
executes both the comparison diagnostic and prospective sensor Agent, and
RxInfer supplies separately scoped native Bethe free-energy diagnostics
[@friedman2026gnn; @heins2022pymdp; @bagaev2023rxinfer]. The sensor experiment
establishes a causal acquisition order: policy selection precedes the chosen
Capstone or Unicorn observation. It does not establish that the authored
fixture hypotheses are discovered semantics or that the model-derived utility
is a measured physical cost. The source-bound
external track adds a complementary validity surface: pinned permissively licensed
revisions are compiled and decoded locally, source-contract edits are recorded,
and emitted objects remain outside execution. Its accepted rows establish a
documented provenance denominator and a project-held-out allocation; they do not
turn lexical witnesses into universal equivalence proofs or static coverage into
application effectiveness. The model is a reproducible computational substrate for
asking which temporal and active-inference mechanisms are visible once assembly
evidence, formal scope, and external provenance have all been made explicit.

Requiring GNN sources for the fitted and discriminative comparators makes the
parameter boundary inspectable across that substrate. The generated catalog
binds numeric instances, training context, adapter code and rendered topology.
It supports a reproducibility claim: the recorded parameters are the parameters
admitted to inference. It does not show that a chosen architecture is adequate
for program semantics. The categorical energy diagnostic illustrates the same
distinction. A stationary iteration trace is expected for the exact fixed
chain; prefix evidence reveals how each observation changes the model's
evidence, while supplying no learning curve or general convergence guarantee
[@rxinfer54_bethe_docs;@rxinfer54_message_passing].

The additional formal and cross-architecture tranches clarify what a stronger
scientific contribution would require. The Lean receipt gives the quotient
program a genuine kernel-checked algebraic nucleus: equivalence, neutrality,
and a separating subtraction witness are stated over every formal 64-bit
initial word. That nucleus is intentionally not conflated with the Python
adapter. A theorem relating the Lean small-step semantics to the Capstone and
Unicorn trace rows remains a separate obligation, because agreement between
two executable paths is not by itself a proof of correspondence. Similarly,
the three-ISA fixture benchmark demonstrates that real decoder/emulator APIs
can be exercised on a shared numeric toy operation, but it does not compare
flags, memory, exceptions, ABIs, compiler transformations, or semantic
quotients [@lean_reference;@dasgupta2019x86semantics]. These are useful
falsification surfaces precisely because their boundaries are narrower than
the claims they might tempt a reader to make [@capstone;@unicorn].

The exact information frontier is another deliberate separation. Dynamic
programming removes Monte Carlo noise from the finite sensor mechanism and
shows how additional non-repeating typed-field queries trade proxy cost against
expected entropy reduction. It does not turn normalized units into acquisition
time or convert a model-derived channel into a physical sensor. A future result
would need measured costs, source-family-clustered uncertainty, alternative
sensor models, strong fixed/random baselines, and a theorem or counterexample
for the proposed bound [@friston2017activeinference;@parr2022activeinference].
Thus the current advance is methodological: it makes
the next conjecture executable and falsifiable without reporting the local
frontier as a general law.

The replication boundary follows the same logic. The public bundle is label-blind,
hash-bound, and replayable without network access; its protected scorer can compare
predictions without distributing the gold labels. No independent replay has been
observed yet: the recorded status is `{{INDEPENDENT_REPLAY_STATUS}}` in the
`{{INDEPENDENT_REPLAY_CONTROL_SCOPE}}` control scope, while the replay interface
and the protected-scoring endpoint remain frozen and ready. A recorded controlled
replay would demonstrate that the frozen public interface crossed a distinct
control boundary; it would not be an independent physical-machine run, an
independent reimplementation, or independent scientific review, and it would
provide no external confirmation of the inference claim. Keeping these evidence
states separate prevents an engineering rerun from being mistaken for scientific
generalization.

The new preparation-audit receipt extends the same discipline to the
external-corpus path without changing its claim boundary. Its eight steps —
manifest parsing, archive digest verification, archive-safety rehearsal,
license checking, static compilation, build reproducibility, label-contract
conformance, and exclusion accounting — are validated as a rehearsal-only
preparation path. The receipt is not an efficacy claim: it performs no
execution, P13 licensed external validation remains underpowered, and
promotion still requires MAJ-01, an independent replay of the
acquisition/build recipe. The MED-05 separately controlled replay boundary is
likewise unchanged and locally blocked by design (`blocked_external`); no
local rerun, however complete, substitutes for it.

## What the visual atlas adds to the scientific argument

The visual atlas is not an ornamental supplement. It is a second, inspectable
projection of the evidence graph. The architecture and state plates show where
static decoding ends and dynamic state begins; the quotient construction and
statistical hierarchy show why source roles, interventions, and clusters cannot
be collapsed into one row count; the coverage matrix and theorem-status map show
which formal statements are checked and which remain outside the contract; and
the replay audit shows the protected failure boundary rather than hiding it in a
log file. This is especially important for assembly research, where a readable
listing can create an impression of semantic completeness even when the actual
observation channel is narrow.

The atlas also makes absence legible. An unavailable compiler target is not a
zero-valued performance cell; an unsupported instruction is not a failed
semantic transition; a missing observation is not an observed zero; and a
one-run controlled replay is not an independent sample. Text labels, hatching,
direct annotations, and accessible tables reinforce those distinctions without
requiring color perception. The long descriptions and source hashes provide a
route from a visual claim back to the producer receipt, while the visible
captions state the population and uncertainty boundary in the place where a
reader is most likely to encounter the result.

The added observation-semantics plate is useful because it locates these
negative spaces before any model score is interpreted. It treats the typed
record as an evidence object with a vocabulary, not as a bag of numerical
defaults: `__MISSING__` means that an observation was not supplied, whereas
`__UNKNOWN__` means that a categorical value was outside the fitted support.
Unsupported and unavailable statuses describe contract boundaries, while a
zero is retained only when a receipt measured zero. The public/private replay
boundary is shown in the same vocabulary, so protected labels remain a scoring
resource rather than an input feature. This is an important methodological
discipline, but it does not by itself prove that the observation channel is
complete, that a protected boundary has no side channel, or that the finite
quotient relation is unrestricted program equivalence.

The central interpretation therefore remains modest but useful. The package
demonstrates how to make an assembly-to-inference experiment inspectable across
representation, execution, formal, statistical, and document boundaries. It
does not demonstrate that a model has recovered program meaning in the
unrestricted sense. Its scientific value lies in making the stronger claim
harder to assert without the corresponding source-bound evidence, and in
turning the next stronger claim into an explicit, falsifiable research task.
