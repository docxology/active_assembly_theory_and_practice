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
static/dynamic trace contract and typed observation schema that preserve missing,
unseen, static, and dynamic states; (ii) a width-aware normalized x86-64 machine
state with explicit register, flag, memory, stack, control-flow, and unsupported
states; (iii) a Python/Capstone/Unicorn/Lean correspondence surface for a declared
user-mode integer subset, with theorem-backed and computational checks separated;
(iv) a source-family-disjoint quotient benchmark with a co-primary classifier
portfolio, calibration, clustered uncertainty, and leakage controls; (v) a pinned,
permissively licensed, static-only source-contract track that records source,
compiler, object, and disassembly provenance without executing external binaries;
(vi) measured deterministic costs and exact finite active-sensing certificates; and
(vii) a label-separated local replication harness whose independent-replay state
cannot be fabricated by a local rerun. Real Python, Julia, and Lean execution
paths, negative controls, manuscript variables, figures, captions, and validation
receipts are regenerated from the same source-bound artifacts. This architecture
allows stronger evidence to be added without silently changing the endpoint or
claim boundary.

The experiment remains intentionally bounded. The canonical Active Inference
fixture uses hand-authored phase states and fixed likelihoods; the larger quotient
benchmark is still synthetic and its observations are dependent within traces; the
external track provides source contracts and static coverage rather than behavioral
labels; and the formal layer covers a declared subset rather than the complete
architecture. These constraints are part of the result: the package makes the
assumptions, failure states, and missing evidence inspectable instead of averaging
them away.

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
