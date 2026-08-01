"""Scholarship source ledger for assembly analysis and active inference claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

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
        claim_boundary="Optional probe status does not mean angr facts are present in v0.1 outputs.",
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
        evidence_tier="roadmap_adapter",
        project_stage="roadmap",
        project_role="Documented as an optional object-dump baseline.",
        claim_boundary="A command probe is not a semantic agreement proof.",
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
        claim_boundary="Cited for ecosystem context, not as evidence that BAP is run in v0.1.",
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
        claim_boundary="Qiling is not an implemented backend in the v0.1 fixture run.",
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
        claim_boundary="The manuscript does not claim decompilation or RetDec output in v0.1.",
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
    json_path.write_text(
        json.dumps(scholarship_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
