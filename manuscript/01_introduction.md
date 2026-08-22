# Introduction {#sec:introduction}

Assembly-analysis workflows sit between a machine-level instruction set and
the higher-level hypotheses an analyst wants to test. A disassembler recovers a
candidate sequence of variable-length instructions from bytes; an emulator can
then expose a bounded dynamic path and register state; a downstream model may
summarize those observations without pretending that the summary is a complete
program meaning. The x86-64 architecture manuals provide the reference frame
for instruction encodings and architectural effects [@intel_sdm]. Capstone
provides the programmable static decode [@capstone], Unicorn provides
instruction-level emulation [@unicorn], and larger systems such as angr can add
symbolic or control-flow analyses when the research question requires them
[@angr; @shoshitaishvili2016sok].

This project deliberately studies the seam between those layers. Its canonical
fixture experiment uses a Capstone--Unicorn core and keeps the byte program
small enough that every byte, address, mnemonic, operand, category, and
register result can be inspected. Optional adapter slots for Rizin, radare2,
LLVM objdump, Keystone, and NASM are recorded as provenance rather than
silently promoted to executed evidence [@rizin; @radare2; @llvm_objdump;
@keystone; @nasm]. The resulting question is narrow but testable: can an
assembly trace be converted into an explicit categorical observation stream
that supports reproducible variational and temporal inference diagnostics?

The contribution is therefore methodological rather than a new disassembler
or reverse-engineering algorithm. The package provides (i) a source-bound
static/dynamic trace contract, (ii) a formal observation map and discrete
generative model, (iii) a bounded transfer-function and CFG explanation layer
cross-checked against independent Unicorn execution, (iv) a leakage-safe
supervised HMM fit with fixture-level holdouts, and (v) real Python and Julia
execution paths with negative controls, manuscript variables, figures,
captions, and validation receipts regenerated from the same artifacts. The
experiment is intentionally small so that the limitations of hand-authored
states, fixed likelihoods, supervised toy estimation, and dependent
within-fixture observations are visible rather than averaged away.

The broader package roadmap is intentionally larger than the implemented
fixture. Binary-analysis platform work such as BAP and GTIRB motivates
intermediate-representation and tool-interchange questions [@brumley2011bap;
@schulte2019gtirb]. Full-suite and lifting/decompilation packages such as
Ghidra, Qiling, Miasm, Dyninst, RetDec, and McSema are cited as future adapter
surfaces, not as current evidence-producing backends [@ghidra; @qiling; @miasm;
@dyninst; @retdec; @mcsema].

The Active Inference bridge is framed as an observation problem. Machine-code
events are not treated as mental states or semantic ground truth. Instead, each
decoded instruction contributes a categorical observation under an explicit
likelihood matrix. Posterior beliefs over latent program-state classes can then
be updated as the observation stream arrives, and the temporal extension can be
tested against memoryless and identity-transition baselines.

The theoretical connection is deliberately narrow. The Free Energy Principle
motivates variational objectives as a way to relate evidence, model structure,
and belief updating [@friston2009freeenergy]. Active inference provides a
process-theoretic account of belief propagation under generative models
[@friston2017activeinference]. Hidden Markov models supply the standard
forward, Viterbi, and forward--backward machinery used for the temporal layer
[@rabiner1989tutorial], while calibration and fixed-parameter evidence are
reported with their own statistical caveats [@guo2017calibration;
@kass1995bayes]. This project borrows that computational grammar for
transparent assembly traces; it does not claim that a byte fixture is a living
agent or that reverse engineering has been automated.
