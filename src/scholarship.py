"""Scholarship source ledger for assembly analysis and active inference claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifact_contract import write_json_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ScholarshipSource:
    """One cited source and the bounded role it plays in the project."""

    key: str
    title: str
    authors: str
    year: int
    venue: str
    doi: str | None
    url: str
    source_type: str
    evidence_tier: str
    project_stage: str
    project_role: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe source record."""

        return asdict(self)


SCHOLARSHIP_SOURCES: tuple[ScholarshipSource, ...] = (
    ScholarshipSource(
        key="friston2009freeenergy",
        title="The free-energy principle: a rough guide to the brain?",
        authors="Karl Friston",
        year=2009,
        venue="Trends in Cognitive Sciences",
        doi="10.1016/j.tics.2009.04.005",
        url="https://pubmed.ncbi.nlm.nih.gov/19559644/",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Defines the Free Energy Principle background.",
        claim_boundary="Does not by itself validate assembly-code reverse engineering.",
    ),
    ScholarshipSource(
        key="friston2017activeinference",
        title="Active Inference: A Process Theory",
        authors="Karl Friston, Thomas FitzGerald, Francesco Rigoli, Philipp Schwartenbeck, and Giovanni Pezzulo",
        year=2017,
        venue="Neural Computation",
        doi="10.1162/NECO_a_00912",
        url="https://pubmed.ncbi.nlm.nih.gov/27870614/",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Grounds active inference as belief updating under generative models.",
        claim_boundary="Does not imply the toy byte fixture has cognitive agency.",
    ),
    ScholarshipSource(
        key="bagaev2023rxinfer",
        title="RxInfer: A Julia package for reactive real-time Bayesian inference",
        authors="Dmitry Bagaev, Albert Podusenko, and Bert de Vries",
        year=2023,
        venue="Journal of Open Source Software",
        doi="10.21105/joss.05161",
        url="https://joss.theoj.org/papers/10.21105/joss.05161",
        source_type="software_paper",
        evidence_tier="implemented_backend",
        project_stage="implemented",
        project_role="Cited for the native Julia/RxInfer inference backend.",
        claim_boundary="Only the JSONL values emitted by this project are empirical project outputs.",
    ),
    ScholarshipSource(
        key="rxinfer_static_docs",
        title="RxInfer Static Inference Manual",
        authors="ReactiveBayes",
        year=2026,
        venue="RxInfer.jl Documentation",
        doi=None,
        url="https://docs.rxinfer.com/stable/manuals/inference/static/",
        source_type="software_documentation",
        evidence_tier="implemented_backend",
        project_stage="implemented",
        project_role="Documents `infer(..., free_energy=true)` and `result.free_energy`.",
        claim_boundary="Versioned documentation can change; tests pin behavior against local RxInfer output.",
    ),
    ScholarshipSource(
        key="rxinfer_python_docs",
        title="Using RxInfer from Python",
        authors="ReactiveBayes",
        year=2026,
        venue="RxInfer.jl Documentation",
        doi=None,
        url="https://docs.rxinfer.com/stable/manuals/how-to-use-rxinfer-from-python/",
        source_type="software_documentation",
        evidence_tier="integration_guidance",
        project_stage="implemented",
        project_role="Justifies native Julia CLI integration over RxInferServer as the primary bridge.",
        claim_boundary="The note concerns integration route maintenance, not model correctness.",
    ),
    ScholarshipSource(
        key="heins2022pymdp",
        title="pymdp: A Python library for active inference in discrete state spaces",
        authors="Conor Heins, Beren Millidge, Daphne Demekas, Brennan Klein, Karl Friston, Iain D. Couzin, and Alexander Tschantz",
        year=2022,
        venue="Journal of Open Source Software",
        doi="10.21105/joss.04098",
        url="https://joss.theoj.org/papers/10.21105/joss.04098",
        source_type="software_paper",
        evidence_tier="comparison_baseline",
        project_stage="comparison",
        project_role="Cited for the pymdp comparison baseline.",
        claim_boundary="The pymdp diagnostic is not labeled as RxInfer Bethe Free Energy.",
    ),
    ScholarshipSource(
        key="capstone",
        title="Capstone: The Ultimate Disassembler",
        authors="Capstone Engine Project",
        year=2026,
        venue="Capstone Project Documentation",
        doi=None,
        url="https://www.capstone-engine.org/",
        source_type="software_documentation",
        evidence_tier="implemented_dependency",
        project_stage="implemented",
        project_role="Cited for static instruction decoding.",
        claim_boundary="Disassembly output is fixture-specific and does not infer high-level semantics.",
    ),
    ScholarshipSource(
        key="quynh2014capstone",
        title="Capstone: Next-Gen Disassembly Framework",
        authors="Nguyen Anh Quynh",
        year=2014,
        venue="Black Hat USA 2014 conference slides",
        doi=None,
        url="https://www.capstone-engine.org/BHUSA2014-capstone.pdf",
        source_type="technical_report",
        evidence_tier="assembly_formalism",
        project_stage="context",
        project_role="Provides primary technical context for Capstone detail-mode instruction, operand, and implicit-register metadata.",
        claim_boundary="The presentation documents the decoder's interface and design; it does not validate this project's particular fixture outputs or semantic labels.",
    ),
    ScholarshipSource(
        key="intel_sdm",
        title="Intel 64 and IA-32 Architectures Software Developer's Manuals",
        authors="Intel Corporation",
        year=2026,
        venue="Intel Architecture Documentation",
        doi=None,
        url="https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html",
        source_type="software_documentation",
        evidence_tier="assembly_formalism",
        project_stage="context",
        project_role="Defines the ISA-level reference frame for instruction encodings, operands, and architectural register effects used in the assembly notation.",
        claim_boundary="The manual constrains the notation and semantics of x86-64; it does not validate this project's Capstone/Unicorn run or establish reverse-engineering performance.",
    ),
    ScholarshipSource(
        key="clang_users_manual",
        title="Clang Compiler User's Manual",
        authors="LLVM Project",
        year=2026,
        venue="Clang Documentation",
        doi=None,
        url="https://clang.llvm.org/docs/UsersManual.html",
        source_type="software_documentation",
        evidence_tier="assembly_formalism",
        project_stage="implemented",
        project_role=(
            "Documents the compiler-driver and target-selection contract used by "
            "the local source-to-object capability receipt."
        ),
        claim_boundary=(
            "The manual documents Clang behavior and options; it does not establish "
            "that an unavailable target was compiled or that emitted code is semantically correct."
        ),
    ),
    ScholarshipSource(
        key="dasgupta2019x86semantics",
        title="A Complete Formal Semantics of x86-64 User-Level Instruction Set Architecture",
        authors="Sandeep Dasgupta, Daejun Park, Theodoros Kasampalis, Vikram S. Adve, and Grigore Rosu",
        year=2019,
        venue="Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation",
        doi="10.1145/3314221.3314601",
        url="https://fsl.cs.illinois.edu/publications/dasgupta-park-kasampalis-adve-rosu-2019-pldi.html",
        source_type="peer_reviewed_binary_analysis",
        evidence_tier="assembly_formalism",
        project_stage="context",
        project_role="Supplies a formal-semantics reference point for distinguishing this package's checked transfer-function subset from complete x86-64 ISA semantics.",
        claim_boundary="The published semantics are not executed by this package; they motivate the bounded formalism boundary and do not certify unsupported instructions.",
    ),
    ScholarshipSource(
        key="lean_reference",
        title="The Lean Language Reference",
        authors="Leonardo de Moura and contributors",
        year=2026,
        venue="Lean Documentation",
        doi=None,
        url="https://lean-lang.org/doc/reference/latest/",
        source_type="formal_methods_documentation",
        evidence_tier="formal_verification",
        project_stage="implemented",
        project_role=(
            "Documents the dependent-type-theoretic proof assistant used for the "
            "closed operational fragment."
        ),
        claim_boundary=(
            "Lean kernel checking applies only to the declared formal fragment; "
            "it does not certify the Python pipeline or unsupported ISA effects."
        ),
    ),
    ScholarshipSource(
        key="demoura2021lean4",
        title="The Lean 4 Theorem Prover and Programming Language",
        authors="Leonardo de Moura and Sebastian Ullrich",
        year=2021,
        venue="Automated Deduction -- CADE 28, Lecture Notes in Computer Science",
        doi="10.1007/978-3-030-79876-5_37",
        url="https://doi.org/10.1007/978-3-030-79876-5_37",
        source_type="peer_reviewed_formal_methods",
        evidence_tier="formal_verification",
        project_stage="context",
        project_role=(
            "Provides the primary scholarly account of Lean 4 as the proof "
            "assistant and programming language used by the formal receipt."
        ),
        claim_boundary=(
            "The paper supports the formal-methods context; only the checked local "
            "Lean sources support the project-specific theorem status."
        ),
    ),
    ScholarshipSource(
        key="unicorn",
        title="Unicorn: The Ultimate CPU Emulator",
        authors="Unicorn Engine Project",
        year=2026,
        venue="Unicorn Project Documentation",
        doi=None,
        url="https://www.unicorn-engine.org/",
        source_type="software_documentation",
        evidence_tier="implemented_dependency",
        project_stage="implemented",
        project_role="Cited for bounded CPU emulation.",
        claim_boundary="The bounded fixture trace is not a sandbox for arbitrary binaries.",
    ),
    ScholarshipSource(
        key="quynh2015unicorn",
        title="Unicorn: Next Generation CPU Emulator Framework",
        authors="Nguyen Anh Quynh and Dang Hoang Vu",
        year=2015,
        venue="Black Hat USA 2015 conference slides",
        doi=None,
        url="https://www.unicorn-engine.org/BHUSA2015-unicorn.pdf",
        source_type="technical_report",
        evidence_tier="assembly_formalism",
        project_stage="context",
        project_role="Provides primary technical context for Unicorn's register, memory, and instrumentation-hook execution model.",
        claim_boundary="The presentation documents emulator capabilities; it does not establish arbitrary-binary safety or validate the project's bounded trace independently.",
    ),
    ScholarshipSource(
        key="angr",
        title="angr",
        authors="angr Project",
        year=2026,
        venue="angr Project Documentation",
        doi=None,
        url="https://angr.io/",
        source_type="software_documentation",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional future CFG and symbolic-analysis adapter.",
        claim_boundary="Optional probe status does not mean angr facts are present in current fixture outputs.",
    ),
    ScholarshipSource(
        key="shoshitaishvili2016sok",
        title="SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis",
        authors="Yan Shoshitaishvili, Ruoyu Wang, Christophe Hauser, Christopher Kruegel, and Giovanni Vigna",
        year=2016,
        venue="IEEE Symposium on Security and Privacy",
        doi="10.1109/SP.2016.17",
        url="https://sites.cs.ucsb.edu/~vigna/publications/2016_SP_angrSoK.pdf",
        source_type="peer_reviewed_binary_analysis",
        evidence_tier="binary_analysis_context",
        project_stage="context",
        project_role="Provides binary-analysis context for symbolic execution and angr-adjacent tooling.",
        claim_boundary="Cited as context, not as evidence that the fixture is vulnerability-relevant.",
    ),
    ScholarshipSource(
        key="rizin",
        title="Rizin reverse engineering framework",
        authors="Rizin Project",
        year=2026,
        venue="Rizin Project Documentation",
        doi=None,
        url="https://rizin.re/",
        source_type="software_documentation",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional static-analysis baseline adapter.",
        claim_boundary="Unavailable optional adapters are recorded as provenance, not evidence streams.",
    ),
    ScholarshipSource(
        key="radare2",
        title="radare2",
        authors="radare2 Project",
        year=2026,
        venue="radare2 Project Documentation",
        doi=None,
        url="https://rada.re/",
        source_type="software_documentation",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional static-analysis baseline adapter.",
        claim_boundary="No radare2-derived facts are reported unless the adapter produces them.",
    ),
    ScholarshipSource(
        key="llvm_objdump",
        title="llvm-objdump: LLVM's object file dumper",
        authors="LLVM Project",
        year=2026,
        venue="LLVM Documentation",
        doi=None,
        url="https://llvm.org/docs/CommandGuide/llvm-objdump.html",
        source_type="software_documentation",
        evidence_tier="implemented_dependency",
        project_stage="implemented",
        project_role="Documents the local static object-disassembly command used by the compiler capability receipt.",
        claim_boundary="Static object dumping and decoding are provenance evidence only; they are not a semantic agreement proof or compiler-correctness result.",
    ),
    ScholarshipSource(
        key="necula2000translationvalidation",
        title="Translation Validation for an Optimizing Compiler",
        authors="George C. Necula",
        year=2000,
        venue="Proceedings of the 2000 ACM SIGPLAN Conference on Programming Language Design and Implementation",
        doi="10.1145/349299.349314",
        url="https://doi.org/10.1145/349299.349314",
        source_type="peer_reviewed_compiler_correctness",
        evidence_tier="compiler_formalism",
        project_stage="context",
        project_role=(
            "Provides the primary translation-validation precedent for treating "
            "each compiler output as a separately checkable artifact."
        ),
        claim_boundary=(
            "The precedent motivates the provenance boundary; this package does not "
            "perform translation validation or prove compiler correctness."
        ),
    ),
    ScholarshipSource(
        key="keystone",
        title="Keystone Engine",
        authors="Keystone Engine Project",
        year=2026,
        venue="Keystone Engine Documentation",
        doi=None,
        url="https://www.keystone-engine.org/",
        source_type="software_documentation",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional fixture-generation adapter.",
        claim_boundary="The current fixture is transparent bytes, not a generated assembly corpus.",
    ),
    ScholarshipSource(
        key="nasm",
        title="NASM: The Netwide Assembler",
        authors="NASM Project",
        year=2026,
        venue="NASM Documentation",
        doi=None,
        url="https://www.nasm.us/docs.html",
        source_type="software_documentation",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional x86 assembler for future fixture generation.",
        claim_boundary="No NASM-derived bytes are claimed in the current run.",
    ),
    ScholarshipSource(
        key="brumley2011bap",
        title="BAP: A Binary Analysis Platform",
        authors="David Brumley, Ivan Jager, Thanassis Avgerinos, and Edward J. Schwartz",
        year=2011,
        venue="Computer Aided Verification",
        doi="10.1007/978-3-642-22110-1_37",
        url="https://link.springer.com/chapter/10.1007/978-3-642-22110-1_37",
        source_type="peer_reviewed_binary_analysis",
        evidence_tier="binary_analysis_context",
        project_stage="context",
        project_role="Adds peer-reviewed context for binary-analysis platforms and IR-centered analysis.",
        claim_boundary="Cited for ecosystem context, not as evidence that BAP runs in the current fixture pipeline.",
    ),
    ScholarshipSource(
        key="schulte2019gtirb",
        title="GTIRB: Intermediate Representation for Binaries",
        authors="Eric Schulte, Jonathan Dorn, Antonio Flores-Montoya, Aaron Ballman, and Tom Johnson",
        year=2019,
        venue="arXiv",
        doi=None,
        url="https://arxiv.org/abs/1907.02859",
        source_type="technical_report",
        evidence_tier="binary_analysis_context",
        project_stage="roadmap",
        project_role="Motivates future trace interchange designs for binary-analysis tool outputs.",
        claim_boundary="GTIRB is roadmap context; current traces use the local JSON schema.",
    ),
    ScholarshipSource(
        key="ghidra",
        title="Ghidra Software Reverse Engineering Framework",
        authors="National Security Agency Research Directorate",
        year=2026,
        venue="Ghidra GitHub Repository",
        doi=None,
        url="https://github.com/NationalSecurityAgency/ghidra",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a future full-suite reverse-engineering adapter and comparison surface.",
        claim_boundary="No Ghidra-derived analysis is claimed until a project adapter emits provenance.",
    ),
    ScholarshipSource(
        key="qiling",
        title="Qiling Framework",
        authors="Qiling Framework Project",
        year=2026,
        venue="Qiling GitHub Repository",
        doi=None,
        url="https://github.com/qilingframework/qiling",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a higher-level emulation roadmap adapter built above Unicorn.",
        claim_boundary="Qiling is not an implemented backend in the current fixture run.",
    ),
    ScholarshipSource(
        key="miasm",
        title="Miasm Reverse Engineering Framework",
        authors="CEA SEC",
        year=2026,
        venue="Miasm GitHub Repository",
        doi=None,
        url="https://github.com/cea-sec/miasm",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a future Python-native disassembly, IR, and emulation adapter.",
        claim_boundary="Miasm facts are roadmap-only until generated by a checked adapter.",
    ),
    ScholarshipSource(
        key="dyninst",
        title="DyninstAPI: Tools for binary instrumentation, analysis, and modification",
        authors="Dyninst Project",
        year=2026,
        venue="Dyninst GitHub Repository",
        doi=None,
        url="https://github.com/dyninst/dyninst",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a future binary instrumentation and analysis adapter family.",
        claim_boundary="Dyninst is not used to produce the current trace or figures.",
    ),
    ScholarshipSource(
        key="retdec",
        title="RetDec: A retargetable machine-code decompiler based on LLVM",
        authors="Avast",
        year=2026,
        venue="RetDec GitHub Repository",
        doi=None,
        url="https://github.com/avast/retdec",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a future decompiler-derived observation channel and cautionary comparison surface.",
        claim_boundary="The manuscript does not claim decompilation or RetDec output in the current fixture pipeline.",
    ),
    ScholarshipSource(
        key="mcsema",
        title="McSema executable lifter",
        authors="Trail of Bits",
        year=2026,
        venue="McSema GitHub Repository",
        doi=None,
        url="https://github.com/lifting-bits/mcsema",
        source_type="software_repository",
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documents a future binary-lifting comparison route into LLVM bitcode.",
        claim_boundary="McSema is roadmap context and not a current evidence-producing package.",
    ),
    ScholarshipSource(
        key="parr2022activeinference",
        title="Active Inference: The Free Energy Principle in Mind, Brain, and Behavior",
        authors="Thomas Parr, Giovanni Pezzulo, and Karl J. Friston",
        year=2022,
        venue="MIT Press",
        doi=None,
        url="https://mitpress.mit.edu/9780262045368/active-inference/",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Canonical textbook grounding generative models, belief updating, and the free energy principle.",
        claim_boundary="Cited as conceptual grounding for the temporal and active layers, not as evidence specific to assembly traces.",
    ),
    ScholarshipSource(
        key="rabiner1989tutorial",
        title="A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition",
        authors="Lawrence R. Rabiner",
        year=1989,
        venue="Proceedings of the IEEE",
        doi="10.1109/5.18626",
        url="https://ieeexplore.ieee.org/document/18626",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Canonical reference for hidden Markov model inference (forward, Viterbi, forward-backward) underlying the temporal layer.",
        claim_boundary="Provides standard HMM methodology; the project's left-to-right phase structure is an instance of it, not a new algorithm.",
    ),
    ScholarshipSource(
        key="dempster1977em",
        title="Maximum Likelihood from Incomplete Data Via the EM Algorithm",
        authors="Arthur P. Dempster, Nan M. Laird, and Donald B. Rubin",
        year=1977,
        venue="Journal of the Royal Statistical Society: Series B (Methodological)",
        doi="10.1111/j.2517-6161.1977.tb01600.x",
        url="https://academic.oup.com/jrsssb/article/39/1/1/7027539",
        source_type="peer_reviewed_theory",
        evidence_tier="statistical_method",
        project_stage="implemented",
        project_role="Grounds the observation-only expectation-maximization estimator for incomplete latent-state assignments in the ordered HMM companion.",
        claim_boundary="The paper motivates the estimator; finite project outputs remain fixture-bound and do not establish unsupervised phase discovery, semantic identifiability, or population validity.",
    ),
    ScholarshipSource(
        key="gelman2013bda",
        title="Bayesian Data Analysis",
        authors="Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin",
        year=2013,
        venue="CRC Press",
        doi=None,
        url="https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955",
        source_type="methodology_textbook",
        evidence_tier="statistical_method",
        project_stage="implemented",
        project_role="Grounds the use of simulation-based generative adequacy checks as model diagnostics.",
        claim_boundary="The textbook motivates the diagnostic design; the generated intervals describe this fixed toy model and do not establish posterior or population uncertainty.",
    ),
    ScholarshipSource(
        key="guo2017calibration",
        title="On Calibration of Modern Neural Networks",
        authors="Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger",
        year=2017,
        venue="Proceedings of the 34th International Conference on Machine Learning",
        doi=None,
        url="https://proceedings.mlr.press/v70/guo17a.html",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Provides the reliability-diagram and expected-calibration-error context for decoder confidence reporting.",
        claim_boundary="Calibration metrics describe this labelled fixture corpus and do not establish deployment reliability.",
    ),
    ScholarshipSource(
        key="kass1995bayes",
        title="Bayes Factors",
        authors="Robert E. Kass and Adrian E. Raftery",
        year=1995,
        venue="Journal of the American Statistical Association",
        doi="10.1080/01621459.1995.10476572",
        url="https://doi.org/10.1080/01621459.1995.10476572",
        source_type="peer_reviewed_theory",
        evidence_tier="theory",
        project_stage="context",
        project_role="Provides the statistical context for interpreting log marginal likelihood and Bayes-factor-style model evidence.",
        claim_boundary="The evidence comparison is descriptive for fixed toy-model parameters, not a population-level model-selection claim.",
    ),
    ScholarshipSource(
        key="brown2001binomial",
        title="Interval Estimation for a Binomial Proportion",
        authors="Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta",
        year=2001,
        venue="Statistical Science",
        doi="10.1214/ss/1009213286",
        url="https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.pdf",
        source_type="peer_reviewed_statistics",
        evidence_tier="statistical_method",
        project_stage="context",
        project_role="Supports the Wilson score interval choice for small binomial hit counts and discourages Wald-interval overinterpretation.",
        claim_boundary="The interval is used descriptively for pooled labelled steps; it does not repair dependence, small-corpus scope, or external-validity limitations.",
    ),
    ScholarshipSource(
        key="holm1979multiple",
        title="A Simple Sequentially Rejective Multiple Test Procedure",
        authors="Sture Holm",
        year=1979,
        venue="Scandinavian Journal of Statistics",
        doi="10.2307/4615733",
        url="https://www.jstor.org/stable/4615733",
        source_type="peer_reviewed_statistics",
        evidence_tier="statistical_method",
        project_stage="implemented",
        project_role="Provides the primary reference for the preregistered Holm familywise multiplicity adjustment.",
        claim_boundary="The adjustment controls the declared comparison family; it does not turn a finite benchmark into population evidence or repair weak endpoint design.",
    ),
    ScholarshipSource(
        key="cameron2008clusterbootstrap",
        title="Bootstrap-Based Improvements for Inference with Clustered Errors",
        authors="A. Colin Cameron, Jonah B. Gelbach, and Douglas L. Miller",
        year=2008,
        venue="The Review of Economics and Statistics",
        doi="10.1162/rest.90.3.414",
        url="https://doi.org/10.1162/rest.90.3.414",
        source_type="peer_reviewed_statistics",
        evidence_tier="statistical_method",
        project_stage="implemented",
        project_role="Grounds the use of source-family resampling and the caution required when the effective cluster count is small.",
        claim_boundary="The project uses a transparent family-cluster bootstrap receipt; this citation does not validate the chosen fixture population or establish asymptotic validity for it.",
    ),
    ScholarshipSource(
        key="lindley1956information",
        title="On a Measure of the Information Provided by an Experiment",
        authors="Dennis V. Lindley",
        year=1956,
        venue="The Annals of Mathematical Statistics",
        doi="10.1214/aoms/1177728069",
        url="https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-4/On-a-Measure-of-the-Information-Provided-by-an-Experiment/10.1214/aoms/1177728069.full",
        source_type="peer_reviewed_statistics",
        evidence_tier="statistical_method",
        project_stage="context",
        project_role="Provides the classical expected-information framing for finite sensor-choice objectives.",
        claim_boundary="The project applies information gain only to a bounded model-derived sensor tree; the source does not support a universal active-sensing law or a physical runtime cost model.",
    ),
    ScholarshipSource(
        key="sangiorgi2011bisimulation",
        title="Introduction to Bisimulation and Coinduction",
        authors="Davide Sangiorgi",
        year=2011,
        venue="Cambridge University Press",
        doi="10.1017/CBO9780511777110",
        url="https://www.cambridge.org/core/books/introduction-to-bisimulation-and-coinduction/7C7D7791F1B4AE459177531C47932E42",
        source_type="methodology_textbook",
        evidence_tier="formal_verification",
        project_stage="context",
        project_role="Provides the formal behavioral-equivalence and bisimulation vocabulary used to delimit quotient-style claims.",
        claim_boundary="The book supplies theory and terminology; the finite quotient receipt is not a bisimulation proof for x86-64 or arbitrary programs.",
    ),
    ScholarshipSource(
        key="wilkinson2016fair",
        title="The FAIR Guiding Principles for scientific data management and stewardship",
        authors=(
            "Mark D. Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, "
            "Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, "
            "Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, "
            "Jildau Bouwman, Anthony J. Brookes, Tim Clark, Mercè Crosas, "
            "Ingrid Dillo, Olivier Dumon, Paul Groth, Carole Goble, Jeffrey S. Grethe, "
            "and collaborators"
        ),
        year=2016,
        venue="Scientific Data",
        doi="10.1038/sdata.2016.18",
        url="https://doi.org/10.1038/sdata.2016.18",
        source_type="peer_reviewed_reproducibility",
        evidence_tier="provenance_method",
        project_stage="context",
        project_role=(
            "Grounds the use of findable, accessible, interoperable, and reusable "
            "provenance metadata as a stewardship objective for the replication bundle."
        ),
        claim_boundary=(
            "FAIR principles motivate metadata and reuse design; a local manifest "
            "does not establish independent replication or scientific validity."
        ),
    ),
    ScholarshipSource(
        key="w3cprov2013",
        title="The PROV Ontology",
        authors="Timothy Lebo, Satya Sahoo, and Deborah McGuinness",
        year=2013,
        venue="W3C Recommendation",
        doi=None,
        url="https://www.w3.org/TR/2013/REC-prov-o-20130430/",
        source_type="formal_methods_documentation",
        evidence_tier="provenance_method",
        project_stage="context",
        project_role=(
            "Provides the authoritative provenance vocabulary for distinguishing generated "
            "entities, activities, and agents when describing artifact lineage."
        ),
        claim_boundary=(
            "PROV-O motivates the lineage vocabulary; this repository emits a compatible "
            "receipt contract but does not claim full PROV-O or RDF conformance."
        ),
    ),
    ScholarshipSource(
        key="rocrate2023",
        title="RO-Crate Metadata Specification 1.1",
        authors=(
            "Peter Sefton, Eoghan Ó Carragáin, Stian Soiland-Reyes, Oscar Corcho, "
            "Daniel Garijo, Raul Palma, Frederik Coppens, Carole Goble, and collaborators"
        ),
        year=2023,
        venue="Research Object Crate Recommendation",
        doi="10.5281/zenodo.7867028",
        url="https://www.researchobject.org/ro-crate/specification/1.1/",
        source_type="formal_methods_documentation",
        evidence_tier="provenance_method",
        project_stage="context",
        project_role=(
            "Informs the separation of machine-readable metadata, payload files, and "
            "human-readable reproduction guidance in a research package."
        ),
        claim_boundary=(
            "The project adopts selected packaging principles; its JSON receipts are not "
            "presented as a complete RO-Crate implementation or certification."
        ),
    ),
    ScholarshipSource(
        key="sandve2013repro",
        title="Ten Simple Rules for Reproducible Computational Research",
        authors="Geir Kjetil Sandve, Anton Nekrutenko, James Taylor, and Eivind Hovig",
        year=2013,
        venue="PLoS Computational Biology",
        doi="10.1371/journal.pcbi.1003285",
        url="https://doi.org/10.1371/journal.pcbi.1003285",
        source_type="peer_reviewed_reproducibility",
        evidence_tier="provenance_method",
        project_stage="context",
        project_role=(
            "Supports recording exact software versions, intermediate outputs, seeds, "
            "and source-to-result links as reproducibility design requirements."
        ),
        claim_boundary=(
            "These rules motivate the engineering contract; passing local gates does not "
            "establish independent replication or scientific validity."
        ),
    ),
    ScholarshipSource(
        key="acm2020artifact",
        title="Artifact Review and Badging - Current",
        authors="Association for Computing Machinery",
        year=2020,
        venue="ACM Publications Policy",
        doi=None,
        url="https://www.acm.org/publications/policies/artifact-review-and-badging-current",
        source_type="formal_methods_documentation",
        evidence_tier="provenance_method",
        project_stage="context",
        project_role=(
            "Provides the publication-practice distinction between author-run reproducibility "
            "and reproduction or replication by a separate person or team."
        ),
        claim_boundary=(
            "The policy is used to name evidence states carefully; it is not an external "
            "artifact-review result for this repository."
        ),
    ),
)


def scholarship_payload() -> dict[str, Any]:
    """Return the complete source ledger payload."""

    source_counts: dict[str, int] = {}
    evidence_tier_counts: dict[str, int] = {}
    project_stage_counts: dict[str, int] = {}
    for source in SCHOLARSHIP_SOURCES:
        source_counts[source.source_type] = source_counts.get(source.source_type, 0) + 1
        evidence_tier_counts[source.evidence_tier] = (
            evidence_tier_counts.get(
                source.evidence_tier,
                0,
            )
            + 1
        )
        project_stage_counts[source.project_stage] = (
            project_stage_counts.get(
                source.project_stage,
                0,
            )
            + 1
        )
    return {
        "source_count": len(SCHOLARSHIP_SOURCES),
        "source_type_counts": source_counts,
        "evidence_tier_counts": evidence_tier_counts,
        "project_stage_counts": project_stage_counts,
        "sources": [source.to_dict() for source in SCHOLARSHIP_SOURCES],
    }


def scholarship_keys() -> set[str]:
    """Return the citation keys required by the source ledger."""

    return {source.key for source in SCHOLARSHIP_SOURCES}


def write_scholarship_sources(output_dir: Path = OUTPUT_DIR) -> str:
    """Write source-ledger artifacts and return the JSON path relative to the project."""

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "scholarship_sources.json"
    write_json_artifact(
        json_path,
        scholarship_payload(),
        project_root=output_dir.parent,
        producer_path=Path(__file__),
    )
    markdown_path = data_dir / "scholarship_matrix.md"
    rows = [
        "# Scholarship Source Matrix",
        "",
        "| Key | Stage | Tier | Role | Boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in SCHOLARSHIP_SOURCES:
        rows.append(
            f"| `{source.key}` | {source.project_stage} | {source.evidence_tier} | "
            f"{source.project_role} | {source.claim_boundary} |"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    from .figures import plot_scholarship_evidence_matrix, plot_scholarship_source_roles

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_scholarship_source_roles(
        [source.to_dict() for source in SCHOLARSHIP_SOURCES],
        output_dir / "figures" / "scholarship_source_roles.png",
    )
    plot_scholarship_evidence_matrix(
        [source.to_dict() for source in SCHOLARSHIP_SOURCES],
        output_dir / "figures" / "scholarship_evidence_matrix.png",
    )
    return str(json_path.relative_to(PROJECT_ROOT))
