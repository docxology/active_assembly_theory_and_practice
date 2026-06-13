# Claim Map

This project treats citations as bounded support for specific claims, not as
general endorsement of the whole scaffold.

| Claim | Evidence | Project artifact | Boundary |
| --- | --- | --- | --- |
| Assembly bytes can be decoded into auditable instruction events. | Capstone documentation and generated disassembly records. | `output/data/disassembly_trace.json` | The claim is fixture-specific and does not imply semantic decompilation. |
| Bounded execution can produce a reproducible dynamic register trace. | Unicorn documentation and generated emulation records. | `output/data/emulation_trace.json` | The bounded fixture is not a sandbox for arbitrary binaries. |
| Assembly events can be encoded as observations for a generative model. | Project feature code plus Active Inference/FEP theory sources. | `output/data/trace_bundle.json` | The observation vocabulary is hand-authored and does not prove program understanding. |
| RxInfer is the authoritative Bethe Free Energy backend in this project. | RxInfer JOSS paper, RxInfer static-inference docs, and generated JSONL records. | `output/rxinfer/free_energy_trace.jsonl` | pymdp diagnostics are comparison values, not RxInfer Bethe Free Energy. |
| Optional assembly-analysis packages are part of the roadmap. | Package documentation and provenance probes. | `output/data/trace_bundle.json` package provenance | Unavailable optional adapters do not contribute evidence streams. |
| Broader binary-analysis packages define expansion targets. | BAP, GTIRB, Ghidra, Qiling, Miasm, Dyninst, RetDec, and McSema sources. | `output/data/scholarship_sources.json` | These sources define possible adapters, not current facts about the fixture. |
| Generated analysis artifacts are tracked for freshness. | Manifested file sizes and SHA-256 hashes. | `output/data/reproducibility_manifest.json` | Hash agreement detects stale or replaced artifacts but does not establish scientific validity. |
| The manuscript is claim-bounded and renderable. | Hydrated manuscript, validation report, and rendered PDF/HTML. | `output/data/validation_report.json` | Passing validation is a project contract, not a peer-review substitute. |

The validator links this map to executable checks by enforcing trace semantics,
RxInfer artifact consistency, bibliography closure, source-stage partitioning,
figure nonblankness, artifact-hash freshness, and durable RedTeam remediation.
