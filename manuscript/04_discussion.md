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

The limitation is equally explicit. The fixture is small, the phase vocabulary
is hand-authored, the headline model comparison uses fixed likelihood and
transition parameters, and steps within a fixture are dependent. A separate
supervised leave-one-fixture-out fit now estimates those tensors without using
the held-out fixture, but its tiny training complements are a generalization
diagnostic rather than evidence of population performance. The observation map also intentionally
collapses details that matter in serious binary analysis: flags, memory
aliasing, calling conventions, indirect control flow, privilege state, and
architecture-specific side effects. Binary-analysis platforms such as BAP and
GTIRB show why intermediate representations and tool interchange become
important as analyses scale [@brumley2011bap; @schulte2019gtirb]. Future modules
can add those features through new checked adapters, but each extension should
enter with its own semantics, tests, provenance, and claim boundary rather than
as a broader assertion about understanding binaries.

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

This is the intended relationship to the Free Energy Principle
[@parr2022activeinference]: assembly traces become sensory evidence for a
generative model whose beliefs are updated under an explicit variational
objective. RxInfer is the source of Bethe Free Energy values in this version;
pymdp remains useful for the discrete comparison baseline. The model is not
claimed to be a universal cognitive account of reverse engineering. It is a
reproducible computational substrate for asking which temporal and active
inference mechanisms are visible once the assembly evidence has been made
explicit.

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
