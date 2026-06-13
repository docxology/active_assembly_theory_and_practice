# Introduction {#sec:introduction}

Assembly-analysis workflows often combine disassembly, emulation, and higher
level program-analysis tooling. Capstone provides a programmable disassembly
interface [@capstone], Unicorn provides instruction-level emulation [@unicorn],
and heavier systems such as angr can add symbolic or control-flow analyses when
the research question requires them [@angr; @shoshitaishvili2016sok]. This
project uses the smaller Capstone-Unicorn core for a reproducible v0.1
experiment, while documenting optional adapter slots for Rizin, radare2,
LLVM objdump, Keystone, and NASM [@rizin; @radare2; @llvm_objdump; @keystone;
@nasm].

The broader package roadmap is intentionally larger than the implemented v0.1
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
be updated as the observation stream arrives.

The theoretical connection is deliberately narrow. The Free Energy Principle
motivates variational objectives as a way to relate evidence, model structure,
and belief updating [@friston2009freeenergy]. Active inference provides a
process-theoretic account of belief propagation under generative models
[@friston2017activeinference]. This project borrows that computational grammar
for transparent assembly traces; it does not claim that a byte fixture is a
living agent or that reverse engineering has been automated.
