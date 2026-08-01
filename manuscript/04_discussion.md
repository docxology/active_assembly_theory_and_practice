# Discussion {#sec:discussion}

The bridge is rigorous because the observation vocabulary, likelihood matrix,
prior, package provenance, trace hash, and update rule are explicit. It is
limited because the fixture is small, the state vocabulary is hand-authored, and
the model does not infer full program semantics. Future modules can add richer
features from control-flow recovery, symbolic execution, dynamic
instrumentation, or generated fixture assembly, but those extensions should
enter as new observation channels with tests and provenance rather than as broad
claims about understanding binaries.

The quantitative results are honest and explanatory rather than inflated. On this
corpus the temporal HMM beats the v0.1 sticky identity-transition filter but not
the memoryless per-step decoder — a non-result that the ablation turns into a
mechanism: because the authored likelihood is near-diagonal, a single observation
already reveals the phase, so transition structure adds little. Sweeping the
likelihood toward uniform, the temporal decoder (and the smoothed temporal
segmentation) is exactly where the advantage appears: per-step decoding collapses
while the transition structure carries the information (Figure
`fig:ambiguity-ablation`). Phase segmentation is now measured against the
hand-authored ground truth (Figure `fig:segmentation-ribbons`), and both the
boundary-recovery result and the mechanism are registered as bounded claims
(CLM-014, CLM-015). This framing keeps the headline faithful: no net temporal
advantage at an informative likelihood, a clear one under ambiguity, and every
number recomputed by the validation oracle rather than asserted.

This is the intended relationship to the Free Energy Principle
[@parr2022activeinference]: assembly traces become sensory evidence for a
generative model whose beliefs are updated to reduce an explicit variational
objective. RxInfer is the source of Bethe Free Energy values in this version;
pymdp remains useful for the discrete comparison baseline. The model is not
claimed to be a universal cognitive account of reverse engineering; it is a
reproducible computational substrate for adding better hypotheses.
