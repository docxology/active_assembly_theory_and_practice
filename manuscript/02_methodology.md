# Methodology {#sec:methodology}

The demonstrator starts with a fixed x86-64 fixture whose instruction sequence is
known and auditable. Capstone decodes the bytes into address, mnemonic, operand,
and byte-span records. Unicorn then executes the same bytes under a bounded
instruction count and records the executed addresses plus pre/post snapshots of
all sixteen general-purpose registers, `RIP`, and `EFLAGS`. The Python reference
state additionally carries definedness for modeled flags, bounded byte memory,
bounded stack bounds, the decoded instruction, and an explicit execution status.
The final register state is retained as a summary, not as the only semantic
check.
These evidence streams are joined into trace schema `{{TRACE_SCHEMA_VERSION}}`
with sample hash `{{BINARY_SHA256}}`. The bundle records
`{{AVAILABLE_PACKAGE_COUNT}}` available package adapters out of
`{{TOTAL_PACKAGE_COUNT}}` total adapter slots, including unavailable optional
packages as explicit provenance rather than silent assumptions; the package
provenance is visualized in [@fig:package-provenance].

The feature layer maps instruction mnemonics into `{{OBSERVATION_CATEGORY_COUNT}}`
observation categories:
load immediate, arithmetic update, bitwise transform, comparison, control
transfer, and other. These observations are supplied to a discrete likelihood
matrix over `{{HIDDEN_STATE_COUNT}}` latent program-state classes: setup, accumulator update,
representation transform, guard check, and exit. The posterior update uses
the real pymdp Agent's state-inference interface, producing one belief vector
and one comparison free-energy diagnostic per observation step. This
identity-transition comparator retains a separate NumPy/softmax regression
reference; it is not the prospective sensing model.
The end-to-end observation pipeline is visualized in [@fig:observation-pipeline],
and the typed trace feature matrix in [@fig:trace-feature-matrix].

## Assembly evidence and notation {#sec:assembly-evidence}

### Executable model specifications

The canonical categorical specifications are `configs/gnn/passive_phase.md`
and `configs/gnn/fixture_sensor.md`. The configured Generalized Notation
Notation implementation [@friedman2026gnn] parses their numeric tensors before a strict local
compiler admits the supported finite profile. The compiler checks declared
axes, distributions, dependencies and policy enumeration; the upstream
revision, imported-source hashes and tensor digest accompany each study.
A is observation likelihood, B is action-conditioned transition, C contains
dimensionless utilities, D is the initial prior and E is the policy prior.
The first observation conditions D before any transition. The passive
categorical Python and native Julia paths consume the same parameter export;
the prospective sensor model executes through pymdp.

The same source requirement extends to every inference comparator and fitted
diagnostic. The generated catalog contains {{GNN_INFERENCE_MODEL_COUNT}} model
profiles and {{GNN_INFERENCE_INSTANCE_COUNT}} distinct numeric instances at
`{{GNN_INFERENCE_CATALOG_PATH}}`. Besides the two canonical specifications,
these profiles declare the phase HMM, identity comparator, Gaussian event-code
model, factorized and hierarchical likelihoods, finite sensor channels,
retrospective ordering weights, logistic and recurrent classifiers, Markov
classifier, Platt calibration, and linear cost calibration. A profile defines
an architecture; a numeric instance binds its current parameters and training
context. These counts describe software artifacts, not independent samples.

Each adapter consumes parameters returned by the configured upstream GNN
parser. Its declared axes and probability constraints are checked before
inference; fitted parameters are retained as specialized GNN sources. The
upstream graph renderer and a larger-label rendering of the same parsed
topology accompany every profile. Python and Julia adapters implement the
declared bounded architectures. This extends source binding across the study
without claiming universal GNN code generation or turning a discriminative
classifier into an active-inference agent.

For this fixed-parameter profile, each policy's negative expected free energy
is the finite-horizon sum of expected dimensionless utility and modality-wise
expected state information gain. With policy precision $\gamma$ and policy
prior $E$, the Agent uses

$$q(\pi)=\operatorname{softmax}_{\pi}\!\left[\gamma\,(-G(\pi))+\log E(\pi)\right].$$

Larger returned negative-EFE scores are favored; the reported EFE reverses
their sign. Parameter learning and parameter information gain are disabled.
This is the predictive policy objective exercised through pymdp, rather than
the adaptive dynamic-programming optimum of the separate finite sensing
experiment [@heins2022pymdp]. The executable software version and admitted
model-language declaration are recorded separately in the upstream receipt.

In the bounded sensor POMDP, the real Agent infers states and policies and
selects an action before the environment performs the selected static or
dynamic fixture acquisition. Evaluation identity is recorded after decisions;
missing acquisition and observed-but-unseen evidence are different outcomes.
Independent finite references and A/B/C/D/E interventions test implementation
behavior. These authored toy models do not establish learned semantics,
general reverse engineering or external validity. Their policy EFE scale is
distinct from both native Bethe free-energy diagnostics below.

### Instruction evidence and bounded state

The byte fixture is an instruction-semantics test case, not an opaque binary
sample. In x86-64, an instruction is a variable-length encoding whose bytes,
decoded operands, architectural effects, and dynamic execution are related but
not interchangeable. Intel's architecture manuals provide the reference
vocabulary for instruction encodings and register effects [@intel_sdm].
Capstone's detail interface exposes instruction identifiers, operand kinds,
implicit register reads/writes, and architecture-specific operand metadata
[@capstone;@quynh2014capstone]. Unicorn complements that static view with a
CPU-state transition interface and code/memory instrumentation hooks
[@unicorn;@quynh2015unicorn]. The implementation preserves those layers
separately: Capstone supplies the static decode, Unicorn supplies the bounded
address/register trace, and the feature mapper supplies the deliberately small
observation vocabulary. This separation is what makes a later semantic or
control-flow extension auditable rather than silently replacing decoded
assembly with a guessed high-level label.

Let $b_{0:L-1}$ be the fixture byte string and
$a_0={{ASSEMBLY_BASE_ADDRESS}}$
the load address. The static decoder is represented as

$$
\delta_{\mathrm{CS}}(b,a_0) = \bigl[(a_i,\ell_i,\mathrm{mnemonic}_i,\mathrm{operands}_i,\mathrm{bytes}_i)\bigr]_{i=0}^{N-1},
$$ {#eq:capstone-decode}

where $a_i$ is an instruction address, $\ell_i$ its decoded byte length,
and $N={{INSTRUCTION_COUNT}}$ for the selected fixture. The offsets are
therefore recovered from the instruction lengths; they are not inferred from a
fixed-width word assumption. The bounded dynamic record is

$$
\tau = \bigl(r_0,r_1,\ldots,r_T;\;a^{\mathrm{exec}}_0, a^{\mathrm{exec}}_1,\ldots,a^{\mathrm{exec}}_{T-1}\bigr),
$$ {#eq:unicorn-trace}

where $r_t$ is the captured register state and
$a^{\mathrm{exec}}_t$ is the address observed by Unicorn's code hook. The
join is address-based: a static instruction is marked executed exactly when
its address occurs in the bounded dynamic trace. That rule is intentionally
weaker than claiming complete path or control-flow recovery.

[@lst:assembly_fixture] is generated from these live Capstone and Unicorn
records for architecture `{{ASSEMBLY_ARCHITECTURE}}`; the machine-readable
source is `{{ASSEMBLY_LISTING_ARTIFACT}}`. The left fields preserve the
assembly evidence; the trailing fields are the categorical projection consumed
by the inference model.

```{#lst:assembly_fixture .asm caption="Source-bound x86-64 fixture listing; static decode and bounded dynamic annotations are generated from the canonical trace bundle."}
{{ASSEMBLY_LISTING}}
```

![{{FIGURE_ASSEMBLY_STATE_PLATE_CAPTION}}]({{FIGURE_ASSEMBLY_STATE_PLATE_PATH}}){#fig:assembly-state-plate width=100%}

The state plate ([@fig:assembly-state-plate]) is the visual companion to
[@lst:assembly_fixture]. It is
generated from the same listing and correspondence receipts, but it keeps the
evidence types visually separate: bytes and decoded operands are static
observations, register and program-counter snapshots are bounded dynamic
observations, and flag/memory annotations are included only when the contract
defines them. An unsupported or out-of-bounds transition is a status-bearing
result, not a fabricated state. This layout is useful for reviewing the
mechanics of a single step without suggesting that a short fixture is a
complete architectural model.

The fixture exercises four distinct pieces of x86-64 notation: an immediate
load into `EAX`, an arithmetic update, a shift-based representation transform,
and a second arithmetic update. A write to a 32-bit register such as `EAX`
also determines the corresponding 64-bit `RAX` state according to the x86-64
architectural rules [@intel_sdm]. Unicorn reports the resulting final register
state `RAX = {{FINAL_RAX}}`; this is a concrete execution check on the bytes,
not a claim that the mnemonic sequence constitutes a recovered program
meaning. The feature map then applies

$$
\begin{aligned}
o_i &= \phi(\mathrm{mnemonic}_i,\mathrm{operands}_i) \in \mathcal{O},\\
\mathcal{O} &= \{\text{load-immediate},\text{arithmetic-update},
\text{bitwise-transform},\\
&\qquad \text{comparison},\text{control-transfer},\text{other}\},
\end{aligned}
$$ {#eq:assembly-observation-map}

so the model sees a declared categorical channel rather than raw opcode bytes.
In particular, `add`, `sub`, and their related update mnemonics share the
`arithmetic_update` channel, while `shl`/`shr`-family instructions share the
`bitwise_transform` channel. This is a transparent pedagogical abstraction;
it does not encode flags, memory aliasing, calling conventions, or complete
instruction semantics.

### Typed evidence channels and factorized scoring {#sec:typed-observation-model}

The roadmap's first expansion retains the discarded evidence as a typed record
rather than silently changing the six-category benchmark. The schema version is
`{{TYPED_OBSERVATION_SCHEMA_VERSION}}` and contains `{{TYPED_FIELD_COUNT}}`
fields (`{{TYPED_STATIC_FIELD_COUNT}}` static fields plus the dynamic execution
status). The field families are `{{TYPED_FIELD_NAMES}}`. Instruction family and
operand class are syntactic classifications over the decoded mnemonic and
operand metadata; register read/write sets and flag effects are taken from
Capstone's detail API; memory addressing mode is a bounded classification of
explicit base/index/displacement metadata; branch target class is internal,
external, return, indirect, or none relative to the decoded span; and dynamic
status is the address-membership join to the bounded Unicorn code trace
[@intel_sdm;@capstone;@unicorn]. None of these fields is a phase label.

For instruction $i$, the typed record is

$$
\mathbf{x}_i = (f_i,o_i,R_i,W_i,\varphi_i,m_i,c_i,d_i),
$$ {#eq:typed-observation-record}

where $f_i$ is instruction family, $o_i$ operand class, $R_i/W_i$ the
canonicalized register read/write sets, $\varphi_i$ flag effect, $m_i$ memory
addressing mode, $c_i$ branch-target class, and $d_i$ dynamic status. The raw
mnemonic and address remain in the record for auditability but are not treated
as a high-cardinality semantic target. A missing value is encoded as
`__MISSING__`; a value not present in the training support is encoded as
`__UNKNOWN__`. This makes the missing-field policy an explicit model input,
rather than an implicit zero or an exception.

### Observation semantics and protected evidence {#sec:observation-semantics}

The observation contract makes absence and protection first-class data states.
An observed categorical value is different from a field that an action did not
produce; a value outside the training vocabulary is different from either of
those; and an instruction, target, or capability outside the declared contract
is not a measured numeric outcome. The serialized tokens `__MISSING__` and
`__UNKNOWN__` preserve the first two distinctions through feature construction,
while structured `unsupported` and `unavailable` statuses preserve the latter
two. A genuine numeric zero remains valid when, and only when, a producer
receipt measures zero. This prevents a convenient sentinel from silently
changing the estimand or making a capability gap appear to be a failed method.

The same principle governs the replay interface. Public rows expose only the
label-blind fields declared by the replication manifest; predictions return
with stable identifiers; and gold labels remain in a protected scoring
surface. Protocol and artifact hashes connect these surfaces without turning
protected labels into model inputs. The boundary is provenance-preserving, not
a cryptographic claim about every possible side channel. It establishes what
the model was permitted to observe and what the scorer was permitted to know.

![{{FIGURE_OBSERVATION_SEMANTICS_CAPTION}}]({{FIGURE_OBSERVATION_SEMANTICS_PATH}}){#fig:observation-semantics width=100%}

[@fig:observation-semantics] is therefore a schema and evidence-boundary
visual rather than a performance result. It is generated from the typed
observation, research-protocol, and replication receipts, and its exact status
table makes the semantics inspectable in grayscale and without relying on
color. The diagram does not assert that the current observation channel is
complete, that protected labels are recoverable, or that agreement under the
finite channel entails unrestricted program equivalence.

The typed records are evaluated as a companion to the collapsed channel across
`{{TYPED_FIXTURE_COUNT}}` complete labelled fixtures. For a training-only
support $V_f$ for field $f$, a smoothed field factor is

$$
\widehat{p}_f(v\mid s)=
\frac{n_{f,v,s}+\alpha}{\sum_{v'\in V_f}n_{f,v',s}+\lvert V_f\rvert\alpha},
$$ {#eq:typed-field-factor}

and the factorized evidence score is

$$
\ell_{\mathrm{fact}}(s\mid\mathbf{x}_i)=
\sum_{f\in\mathcal{F}}\log \widehat{p}_f(x_{i,f}\mid s).
$$ {#eq:factorized-observation-score}

The hierarchical companion groups fields into instruction semantics, register
context, control flow, and dynamic context. It averages log scores within each
active group and then averages groups before normalizing over states:

$$
\ell_{\mathrm{hier}}(s\mid\mathbf{x}_i)=
\frac{1}{|\mathcal{G}_i|}\sum_{g\in\mathcal{G}_i}
\frac{1}{|\mathcal{F}_g|}\sum_{f\in\mathcal{F}_g}
\log \widehat{p}_f(x_{i,f}\mid s).
$$ {#eq:hierarchical-observation-score}

This group-balanced score is intentionally called a composite score: it is not
presented as proof that the fields are conditionally independent, nor as a
normalized joint distribution over all assembly records. The existing
six-category likelihood remains the deliberately collapsed reference. The
complete schema is visualized in [@fig:typed-observation-matrix], while the
fixture-level comparison, field-family ablation, and masked-memory sensitivity
are shown in [@fig:factorized-observation-comparison].

![{{FIGURE_TYPED_OBSERVATION_MATRIX_CAPTION}}]({{FIGURE_TYPED_OBSERVATION_MATRIX_PATH}}){#fig:typed-observation-matrix width=100%}

The factorized evaluation uses `{{TYPED_FIELD_GROUP_COUNT}}` field groups and
`{{FACTORIZED_ABLATION_GROUP_COUNT}}` corresponding leave-one-group-out
ablations, while holding out complete fixtures. For each held-out fixture $f_k$, every field
support, conditional factor, transition count, and initial-state count is fit on
$\mathcal{F}_{-k}$; the excluded record is then decoded once. The artifact is
`{{FACTORIZED_OBSERVATION_ARTIFACT}}`. The resulting comparison is explicitly
diagnostic: collapsed mean accuracy is `{{FACTORIZED_COLLAPSED_MEAN_ACCURACY}}`,
static typed factorization is `{{FACTORIZED_STATIC_MEAN_ACCURACY}}`, full
factorization is `{{FACTORIZED_MEAN_ACCURACY}}`, and the hierarchical composite
score is `{{HIERARCHICAL_MEAN_ACCURACY}}`. Thus the full typed score changes the
collapsed reference by `{{FACTORIZED_MINUS_COLLAPSED}}` and changes the static
typed ablation by `{{FACTORIZED_MINUS_STATIC}}` on this tiny corpus. Masking the
memory-addressing field yields mean accuracy `{{MISSING_FIELD_MEAN_ACCURACY}}`
(`{{MISSING_FIELD_MEAN_DELTA}}` relative to full factorization), a useful null
result because the labelled fixtures contain little memory-mode variation.

![{{FIGURE_FACTORIZED_OBSERVATION_COMPARISON_CAPTION}}]({{FIGURE_FACTORIZED_OBSERVATION_COMPARISON_PATH}}){#fig:factorized-observation-comparison width=100%}

### Fixture-design coverage

The typed comparison is now evaluated on an expanded transparent corpus rather
than on register-only straight-line variants. The corpus contains
`{{CORPUS_FIXTURE_COUNT}}` complete fixtures. A source-bound coverage inventory
records `{{CORPUS_MEMORY_FIXTURE_COUNT}}` fixture with explicit memory addressing,
`{{CORPUS_ALTERNATE_PATH_FIXTURE_COUNT}}` fixture with a statically decoded but
unexecuted instruction, and `{{CORPUS_REPEATED_PHASE_FIXTURE_COUNT}}` fixtures
with repeated authored phase labels. The inventory also records
`{{CORPUS_UNIQUE_INSTRUCTION_FAMILY_COUNT}}` instruction families,
`{{CORPUS_UNIQUE_MEMORY_MODE_COUNT}}` non-null memory addressing mode(s), and
`{{CORPUS_UNIQUE_BRANCH_CLASS_COUNT}}` non-null branch-target class(es). These
counts are derived from the real Capstone and bounded Unicorn producer path;
they describe fixture design, not independent binary samples.

For fixture $k$, define the design vector

$$
\mathbf{c}_k=(N_k,E_k,R_k,M_k,B_k,U_k),
$$ {#eq:fixture-coverage-vector}

where $N_k$ is the number of statically decoded instructions, $E_k$ is the
number of distinct dynamically executed addresses, $R_k$ is the number of
contiguous authored phase runs, $M_k$ is the number of static memory-operand
instructions, $B_k$ is the number of observed non-null branch-target classes,
and $U_k$ is the number of typed instruction families. The coverage artifact
is `{{CORPUS_COVERAGE_ARTIFACT}}`; [@fig:corpus-coverage] visualizes the
quantitative $N_k/E_k$ distinction and the categorical design flags. The
artifact is a promotion gate for fixture diversity, not a semantic score: it
cannot show that a field is intrinsic to program meaning or that
`{{TEMPORAL_FIXTURE_COUNT}}` toy programs represent arbitrary binaries.

![{{FIGURE_CORPUS_COVERAGE_CAPTION}}]({{FIGURE_CORPUS_COVERAGE_PATH}}){#fig:corpus-coverage width=100%}

### Bounded transfer functions and control-flow formalism

The categorical map is supplemented by a small executable semantics layer.
Capstone detail provides register read/write sets and direct operand spelling.
A bounded transfer function executes the documented immediate-register,
arithmetic, bitwise, shift, comparison, and conditional-branch subset over the
dynamic path. The report is emitted as {{SYMBOLIC_SEMANTICS_ARTIFACT}}.

For the selected fixture, {{SYMBOLIC_SUPPORTED_COUNT}} instructions are covered
by the transfer-function subset and {{SYMBOLIC_UNSUPPORTED_COUNT}} executed
instructions are unsupported; the symbolic final value agrees with Unicorn
({{SYMBOLIC_FINAL_RAX_MATCH}}). Unsupported executed instructions fail closed
rather than receiving a guessed effect. This is an auditable explanation layer,
not a complete x86 lifter, symbolic solver, decompiler, or claim of semantic
program understanding [@intel_sdm;@capstone;@unicorn;@dasgupta2019x86semantics].
The distinction is substantive: complete formal x86-64 semantics are a
separate research object, whereas this package deliberately verifies only a
small transfer-function subset and fails closed outside it
[@dasgupta2019x86semantics].

Let q_i be the concrete register state before instruction i and T_i the
transfer function selected by its decoded mnemonic and operands. For an
executed supported instruction, the bounded execution is

$$
q_{i+1}=T_i(q_i), \qquad
e_{i+1}=\Psi_i(e_i,\mathrm{operands}_i),
$$ {#eq:bounded-transfer}

where e_i is the parallel readable symbolic expression. The independent
Unicorn check tests the concrete terminal projection,

$$
\pi_{RAX}(q_N^{\mathrm{symbolic}})
=q_N^{\mathrm{Unicorn}},
$$ {#eq:semantic-crosscheck}

not equality of the textual expression.

The stronger instrument contract now compares the bounded register projection
at every executed step. Let $u_i^-$ and $u_i^+$ be the pre- and post-step
Unicorn snapshots aligned to the decoded address, and let
$\widehat{q}_i^+=T_i(\widehat{q}_i^-)$ be the transfer-function update. The
stepwise `RAX` endpoint is

$$
I_i^{RAX}=\mathbb{1}\!\left[\pi_{RAX}(\widehat{q}_i^-)=\pi_{RAX}(u_i^-)
\;\land\;
\pi_{RAX}(\widehat{q}_i^+)=\pi_{RAX}(u_i^+)\right],
$$ {#eq:stepwise-rax-refinement}

and the report records {{SYMBOLIC_STEPWISE_RAX_AGREEMENT_COUNT}} agreements out
of {{SYMBOLIC_STEPWISE_RAX_COMPARABLE_COUNT}} comparable executed steps. For
architectural flags, the interpreter compares only bits it defines for a
particular operation: carry (`CF`), zero (`ZF`), sign (`SF`), and overflow
(`OF`). It records {{SYMBOLIC_STEPWISE_FLAG_AGREEMENT_COUNT}} agreements out of
{{SYMBOLIC_STEPWISE_FLAG_COMPARISON_COUNT}} defined bit comparisons. An
undefined or preserved flag is not silently treated as a match. These are
trace-refinement checks for a declared subset, not a proof of the complete
x86-64 transition relation [@intel_sdm;@dasgupta2019x86semantics]. The stepwise
refinements are visualized in [@fig:trace-refinement].

![{{FIGURE_TRACE_REFINEMENT_CAPTION}}]({{FIGURE_TRACE_REFINEMENT_PATH}}){#fig:trace-refinement width=96%}

![{{FIGURE_SEMANTIC_COVERAGE_MATRIX_CAPTION}}]({{FIGURE_SEMANTIC_COVERAGE_MATRIX_PATH}}){#fig:semantic-coverage-matrix width=100%}

The coverage matrix ([@fig:semantic-coverage-matrix]) should be interpreted as a
scope ledger. A check mark in
the Lean column is evidence about the checked formal fragment; a Capstone or
Unicorn mark is evidence about the corresponding adapter call on the recorded
fixture. These marks are not interchangeable and do not aggregate into a
percentage of the entire x86-64 architecture. Textual statuses and hatching
preserve the distinction between unsupported semantics, unavailable tooling,
and a cell that has simply not yet been observed.

### Mechanically checked closed-world semantics {#sec:formal-fragment}

The executable transfer functions are complemented by a small, independent
operational-semantics fragment in Lean. Its purpose is not to relabel the
Capstone trace as a proof of x86-64 correctness; it is to make the algebraic
equivalence witnesses used by the quotient-state program explicit and kernel
checked. The formal state is a modular word
$q\in\mathbb{W}_{64}=\{0,\ldots,2^{64}-1\}$, and the closed operation set is
`mov`, `add`, `sub`, `inc`, and `nop`. The formal step function is

$$
\begin{aligned}
\operatorname{step}(\operatorname{mov}\ v,q)&=v,&
\operatorname{step}(\operatorname{add}\ v,q)&=q+v \pmod{2^{64}},\\
\operatorname{step}(\operatorname{sub}\ v,q)&=q-v \pmod{2^{64}},&
\operatorname{step}(\operatorname{inc},q)&=q+1 \pmod{2^{64}},\\
\operatorname{step}(\operatorname{nop},q)&=q.&
\end{aligned}
$$ {#eq:formal-closed-step}

For a finite program $P$, define its observational relation over this formal
domain by

$$
P\equiv_{64}Q\quad\Longleftrightarrow\quad
\forall q\in\mathbb{W}_{64},\;\operatorname{run}(P,q)=\operatorname{run}(Q,q).
$$ {#eq:formal-observational-equivalence}

The checked receipt is `{{FORMAL_VERIFICATION_ARTIFACT}}`, visualized in
[@fig:formal-verification-receipt]: Lean toolchain
`{{FORMAL_TOOLCHAIN}}` builds `{{FORMAL_THEOREM_COUNT}}` named theorems and
reports status `{{FORMAL_VERIFICATION_STATUS}}`. The current theorem manifest
proves two `inc` operations equal one addition by two, proves that `nop` is
neutral in a finite program, proves two constant-addition programs equivalent
under every formal initial word, proves a changed subtraction is *not*
equivalent by an explicit initial-state witness, and proves that appending a
`nop` preserves the formal run. The trusted base is the Lean kernel and its
imported core/standard library; no external solver output is accepted as a
proof certificate. The generated bridge also serializes a transparent boundary
program from the Python reference path and checks its expected modular
`Word64` result in Lean. This shared fixture is a checked cross-language anchor,
while the broader Python/Capstone/Unicorn correspondence remains computational
evidence. This is a mechanically checked algebraic prerequisite for the local
quotient protocol, not a theorem about Capstone, Unicorn, x86 flags, memory,
exceptions, compiler correctness, or all assembly programs.
Lean 4's scholarly account describes the system as both an extensible theorem
prover and a programming language [@demoura2021lean4]; here, only the kernel
checked receipt for the declared local fragment is promoted.

![{{FIGURE_FORMAL_VERIFICATION_RECEIPT_CAPTION}}]({{FIGURE_FORMAL_VERIFICATION_RECEIPT_PATH}}){#fig:formal-verification-receipt width=96%}

### Cross-architecture adapter falsification surface {#sec:cross-architecture}

The first architecture extension is deliberately a runtime adapter benchmark,
not a portability claim. The same declared numeric operation is encoded as
transparent bytes for x86-64, AArch64, and RISC-V64. Capstone decodes three
instructions in each fixture; Unicorn executes exactly the two non-returning
instructions; and the final accumulator is compared with the declared value.
The receipt `{{CROSS_ARCHITECTURE_ARTIFACT}}` records the bytes, decoder
mnemonics, API provenance, bounded instruction count, and outcome for all
`{{CROSS_ARCHITECTURE_COUNT}}` architectures. All bounded executions passed
(`{{CROSS_ARCHITECTURE_ALL_PASSED}}`) and the common numeric result is
`{{CROSS_ARCHITECTURE_RESULT}}`; the benchmark is visualized in
[@fig:cross-architecture-benchmark].

The adapter contract is intentionally weaker than semantic equivalence. It
does not compare flags, exception behavior, calling conventions, memory
effects, ABI obligations, compiler transformations, or complete instruction
semantics across ISAs. Its scientific value is adversarial at the software
boundary: a real decoder/emulator path must execute before an architecture is
counted, and a stale hand-written result cannot satisfy the receipt. The
benchmark is therefore a falsification surface for adapter drift and a
starting point for source-family-disjoint cross-compiler work, not evidence of
cross-architecture latent-state invariance.

![{{FIGURE_CROSS_ARCHITECTURE_BENCHMARK_CAPTION}}]({{FIGURE_CROSS_ARCHITECTURE_BENCHMARK_PATH}}){#fig:cross-architecture-benchmark width=96%}

The static direct-branch partition produces {{CFG_BLOCK_COUNT}} basic blocks and
{{CFG_EDGE_COUNT}} CFG edges in the current bundle. A target outside the decoded
span remains an explicit external edge, so the representation records the
observed boundary without overstating it as whole-binary CFG recovery.

More formally, let each decoded instruction be

$$
u_i=(a_i,\ell_i,m_i,x_i,R_i,W_i),
$$ {#eq:instruction-event}

where $a_i$ and $\ell_i$ are its address and byte length, $m_i$ is the mnemonic,
$x_i$ is the operand string, and $R_i/W_i$ are Capstone-derived register
read/write sets. Let $L$ contain instruction zero, every in-range direct branch
target, and the instruction immediately after each branch or return. Sorting
$L=(\lambda_0,\ldots,\lambda_{K-1})$ yields the bounded basic-block partition

$$
\begin{gathered}
\mathcal{B}_j=\{u_i:\lambda_j\leq i<\lambda_{j+1}\},\qquad \lambda_K=N,\\
\bigsqcup_{j=0}^{K-1}\mathcal{B}_j=\{u_0,\ldots,u_{N-1}\}.
\end{gathered}
$$ {#eq:cfg-partition}

An edge $(\mathcal{B}_j,\mathcal{B}_k,\tau)$ is emitted for a direct jump,
conditional taken branch, or fall-through successor; an out-of-span target is
retained as an `external_*` endpoint. This is a finite CFG representation of the
fixture's decoded span, not whole-program reachability.

For the separate leakage-safe estimation path, let $f_k$ be the held-out fixture
and $\mathcal{F}_{-k}$ the remaining labelled fixtures. With positive smoothing
concentrations $\alpha_A,\alpha_B,\alpha_D$, the fitted categorical parameters
are

$$
\begin{gathered}
A_{o,s}^{(-k)}=\frac{n_{o,s}^{(-k)}+\alpha_A}
{\sum_{o'}n_{o',s}^{(-k)}+|\mathcal{O}|\alpha_A},\\
B_{s',s}^{(-k)}=\frac{n_{s',s}^{(-k)}+\alpha_B}
{\sum_{r}n_{r,s}^{(-k)}+|\mathcal{S}|\alpha_B},\\
D_s^{(-k)}=\frac{n_s^{(-k)}+\alpha_D}
{\sum_r n_r^{(-k)}+|\mathcal{S}|\alpha_D}.
\end{gathered}
$$ {#eq:smoothed-hmm-fit}

The superscript $(-k)$ is an exclusion contract: no observation, phase label,
or transition count from $f_k$ enters its fitted tensors. The resulting model is
then scored once on $f_k$ and written to {{HELD_OUT_MODEL_ARTIFACT}}. This is
supervised fixture-level estimation, not unsupervised phase discovery.

The branch-bearing CFG evidence is retained in {{CFG_EVIDENCE_ARTIFACT}} and
visualized in [@fig:assembly-cfg]. It makes the distinction between a decoded
direct branch, a fall-through successor, and an out-of-span target inspectable
without importing a larger reverse-engineering framework.

![{{FIGURE_ASSEMBLY_CFG_CAPTION}}]({{FIGURE_ASSEMBLY_CFG_PATH}}){#fig:assembly-cfg width=100%}

### Memory operands and dynamic access provenance

Register traces alone are insufficient for an assembly-to-inference interface:
an instruction can read or write memory without changing the final accumulator,
and a static memory operand does not establish that the access occurred. The
method therefore includes a transparent `x86_memory` fixture in the expanded
labelled corpus. Capstone
records the base register, index register, scale, displacement, and operand
width for every memory operand; Unicorn memory hooks record the access type,
effective address, width, value, triggering instruction address, and dynamic
event index. The memory-evidence artifact remains a separately named provenance
projection even though the same fixture now participates in the typed
leave-one-fixture-out evaluation; this keeps the store/load join inspectable
without silently changing the meaning of the collapsed channel.

The static operand relation is written as

$$
m_i=(b_i,j_i,\sigma_i,d_i,w_i),
\qquad
\widehat{a}_{i}=R[b_i]+\sigma_i R[j_i]+d_i,
$$ {#eq:memory-effective-address}

where $b_i$ and $j_i$ are the base and index registers (either may be absent),
$\sigma_i$ is the scale, $d_i$ is the signed displacement, and $w_i$ is the
operand width. A dynamic memory event is

$$
\mu_k=(k,\rho_k,a_k,w_k,v_k,i_k),
$$ {#eq:memory-event}

with event order $k$, access relation $\rho_k\in\{\mathrm{read},
\mathrm{write}\}$, effective address $a_k$, width $w_k$, observed value
$v_k$, and triggering instruction index $i_k$. The contract joins the two
streams only when $i_k$ has a declared memory operand and the observed address
equals the bounded Unicorn register state plus the decoded displacement. It
does not perform alias analysis, recover heap objects, or infer memory meaning
from an address alone.

The current provenance artifact is {{MEMORY_EVIDENCE_ARTIFACT}}, visualized in
[@fig:assembly-memory-timeline]. It contains
{{MEMORY_STATIC_INSTRUCTION_COUNT}} static memory instructions and
{{MEMORY_DYNAMIC_EVENT_COUNT}} dynamic events: {{MEMORY_WRITE_COUNT}} write and
{{MEMORY_READ_COUNT}} read event(s) at {{MEMORY_DISTINCT_ADDRESS_COUNT}}
distinct effective address(es). The store and load observe the same fixture
value, providing a minimal end-to-end check that the static operand, effective
address, dynamic hook, and terminal register result agree. This is an
instrumentation demonstration, not evidence of general memory semantics.

![{{FIGURE_ASSEMBLY_MEMORY_TIMELINE_CAPTION}}]({{FIGURE_ASSEMBLY_MEMORY_TIMELINE_PATH}}){#fig:assembly-memory-timeline width=100%}

The native RxInfer layer is invoked through `julia --project` and reads the
same trace bundle. The current model maps observation IDs into numeric event
codes and runs a conjugate Gaussian observation model with `free_energy=true`.
Every exposed `result.free_energy` value is written to
`output/rxinfer/free_energy_trace.jsonl` with data and configuration hashes,
following the RxInfer static-inference interface [@rxinfer_static_docs]. The
Python-to-RxInfer integration is a native Julia CLI call because the RxInfer
documentation identifies the server/client route as third-party and not
maintained by the core team [@rxinfer_python_docs]. This Gaussian trajectory is
an explicitly retained numeric event-code diagnostic. The same package now
also runs a validated, namespaced categorical companion,
`{{RXINFER_CATEGORICAL_MODEL_NAME}}`, over one-hot observation categories and
fixed left-to-right latent phases. Its diagnostics are written to
`{{RXINFER_CATEGORICAL_ARTIFACT}}` and contain
`{{RXINFER_CATEGORICAL_FREE_ENERGY_COUNT}}` finite values, with final value
`{{RXINFER_CATEGORICAL_FINAL_FREE_ENERGY}}`. The fixed-chain iteration values
are already stationary after the first sweep. The prefix-evidence and
conditional-surprisal panels in [@fig:rxinfer-categorical-free-energy] instead
show native inference on successively longer observed sequences, checked
against an independent forward calculation. They are evidence accumulation
diagnostics, not learning or convergence curves
[@rxinfer54_bethe_docs;@rxinfer54_message_passing]. The Gaussian and categorical
free-energy values are not compared on an absolute scale: they are distinct
factor graphs with different variational objectives.

The categorical companion makes the intended phase model explicit. For latent
states $z_t$ and categorical observations $x_t$,

$$
p(z_{1:T},x_{1:T})=p(z_1)
\prod_{t=2}^{T}p(z_t\mid z_{t-1})
\prod_{t=1}^{T}p(x_t\mid z_t),
$$ {#eq:categorical-rxinfer-hmm}

where the transition and emission matrices are recorded in the companion
configuration artifact. RxInfer executes this model through its native
`Categorical` and `DiscreteTransition` factors; the Julia package tests assert
finite, correctly namespaced free-energy and posterior artifacts. The companion
is therefore an implemented method path, while its fixed parameters and small
fixture input remain insufficient for claims of learned program semantics.

![{{FIGURE_RXINFER_CATEGORICAL_FREE_ENERGY_CAPTION}}]({{FIGURE_RXINFER_CATEGORICAL_FREE_ENERGY_PATH}}){#fig:rxinfer-categorical-free-energy width=88%}

![{{FIGURE_OBSERVATION_PIPELINE_CAPTION}}]({{FIGURE_OBSERVATION_PIPELINE_PATH}}){#fig:observation-pipeline width=95%}

![{{FIGURE_PACKAGE_PROVENANCE_CAPTION}}]({{FIGURE_PACKAGE_PROVENANCE_PATH}}){#fig:package-provenance width=90%}

![{{FIGURE_TRACE_FEATURE_MATRIX_CAPTION}}]({{FIGURE_TRACE_FEATURE_MATRIX_PATH}}){#fig:trace-feature-matrix width=95%}

## Temporal, active, and evaluated inference {#sec:advanced-methods}

The identity-transition comparison filter carries the previous posterior
forward unchanged and therefore omits state dynamics — implicitly assuming an
identity transition over latent phases. The temporal model adds a left-to-right
transition matrix over the `{{HIDDEN_STATE_COUNT}}` program phases, yielding a genuine discrete
hidden Markov model. Forward filtering alternates a predict step (the transition
matrix applied to the previous posterior) with a categorical likelihood update;
Viterbi decoding recovers the most probable joint phase path; and
forward–backward smoothing refines the per-step beliefs
[@friston2017activeinference;@rabiner1989tutorial]. Contiguous program phases are then segmented
directly from the smoothed beliefs.

The complete Bayesian graph of this model is drawn in [@fig:generative-model]:
the hidden phase states form the left-to-right Markov
backbone, each emits a categorical observation through the likelihood `A`, the
initial belief is fixed by the prior `D`, and the active layer scores candidate
observations through the expected free energy.

### Observation-only parameter learning {#sec:observation-only-learning}

The fixed matrices are complemented by an explicitly separated parameter-learning
path. For each complete-fixture holdout, the training input is the sequence of
observation-category IDs from the other fixtures; the hand-authored phase labels
are not passed to fitting. This is an observation-only estimator over a declared
ordered-state scaffold, not a discovery algorithm. Because categorical latent
states are label-non-identifiable without a constraint, the fit is initialized
from the inspectable reference model and retains its ordered phase names only as
an architectural coordinate system. The held-out labels are introduced exactly
once, in the scoring layer.

For observation sequence $o_{1:T}$, the E step computes smoothed responsibilities

$$
\gamma_t(s)=P(s_t=s\mid o_{1:T}),\qquad
\xi_t(s',s)=P(s_t=s,s_{t+1}=s'\mid o_{1:T}),
$$ {#eq:em-responsibilities}

using the scaled forward--backward recursions. With positive Dirichlet
pseudocounts, the M step updates the column-stochastic emission and transition
parameters as

$$
\widehat A_{o,s}=
\frac{\alpha_A+\sum_t\mathbf{1}[o_t=o]\gamma_t(s)}
{|\mathcal O|\alpha_A+\sum_t\gamma_t(s)},
$$ {#eq:em-emission-update}

and, for the ordered support
$\mathcal E=\{(s,s),(s+1,s)\}$,

$$
\widehat B_{s',s}\propto
\epsilon + \mathbf{1}[(s',s)\in\mathcal E]
\left(\alpha_B+\sum_t\xi_t(s',s)\right),
\qquad
\widehat D_s\propto\alpha_D+\gamma_1(s).
$$ {#eq:em-transition-update}

Here $\epsilon$ is a strictly positive transition floor, which keeps the
shared log-space inference contract valid while making the self/next-state
prior soft rather than a hard zero. Each training complement is iterated for at
most `{{PARAMETER_LEARNING_MAX_ITERATIONS}}` updates, and the fitted model is
scored on the excluded fixture without further adaptation. The complete rows,
fitted tensors, evidence histories, and fit-input contract are emitted in
`{{PARAMETER_LEARNING_ARTIFACT}}`. The estimator follows the incomplete-data
maximum-likelihood logic of the EM algorithm [@dempster1977em] and standard HMM
forward--backward inference [@rabiner1989tutorial]; the implementation remains
fixture-bound and does not turn the phase scaffold into learned semantics.

Robustness is not summarized by selecting the most favorable start. A fixed-
penalty Dirichlet-MAP comparator is fit from the same observation-only inputs,
and its held-out likelihood remains separate from the primary ordered-EM score.
For each held-out sequence, a label-free posterior-predictive check simulates
same-length categorical sequences and compares observation entropy, category
support, and adjacent-change rate with central 95% simulation intervals. The
model-selection ledger records that any reference choice uses training evidence
only; held-out phase labels enter only after fitting. Non-convergence, weak
local optima, and seed spread remain visible in the receipt rather than being
filtered from the figure.

To make initialization sensitivity part of the estimand rather than an invisible
implementation choice, each holdout is fit from five frozen starts: the
reference parameterization, a uniform parameterization, and three deterministic
random starts with seeds recorded in the receipt. Let
$\mathcal I$ denote these starts and let $\widehat y_{h,i}$ be the held-out
sequence predicted for holdout $h$ from start $i$. We retain the within-holdout
accuracy range

$$
R_h=\max_{i\in\mathcal I}\operatorname{Acc}(\widehat y_{h,i},y_h)
     -\min_{i\in\mathcal I}\operatorname{Acc}(\widehat y_{h,i},y_h),
$$ {#eq:multi-start-range}

and the mean pairwise sequence agreement across starts. These are robustness
diagnostics, not uncertainty intervals over fixtures: a large $R_h$ or low
agreement is evidence that the latent fit is initialization-sensitive. Start
selection is not performed using held-out labels, and every start—including
weak or unstable fits—is retained in `{{PARAMETER_LEARNING_ARTIFACT}}`.

![{{FIGURE_GENERATIVE_MODEL_CAPTION}}]({{FIGURE_GENERATIVE_MODEL_PATH}}){#fig:generative-model width=95%}

## Formal description

::: {.definition #def:genmodel title="Discrete generative model"}
A discrete temporal generative model is the triple `(D, B, A)` where `D` is a
prior over the hidden phase `s_1`, `B` is a column-stochastic transition matrix
`B[s', s] = P(s_{t+1} = s' | s_t = s)`, and `A` is a column-stochastic
likelihood `A[o, s] = P(o_t = o | s_t = s)`. An observation sequence
`o_{1:T}` arises from the phases following the left-to-right Markov chain, each
phase emitting one observation.
:::

::: {.definition #def:expected-free-energy title="Retrospective ordering score"}
For an already decoded candidate category `o`, the ordering score is the negative
sum of its realized phase-entropy reduction and a log-probability preference
`ln r(o)`. The diagnostic selects the candidate minimizing that score. A flat
probability preference `r` makes the pragmatic term constant across candidates.
This retrospective preference is distinct from the dimensionless utility `C`
in the canonical prospective sensor POMDP specified above.
:::

::: {.definition #def:model-evidence title="Log marginal likelihood (model evidence)"}
The model evidence of the observed traces under a generative model is the log
marginal likelihood `sum_t ln P(o_t | o_{1:t-1})`, where each conditional is the
forward-filter normalizer at step `t`.
:::

::: {.proposition #prop:forward-normalizer title="Forward-filter normalizer is the evidence"}
At each step the forward filter's normalizer equals the conditional marginal
`P(o_t | o_{1:t-1})`, so the cumulative log of the normalizers is exactly the
model's log marginal likelihood, as defined above.
:::

::: {.proposition #prop:transition-collapse title="Uniform transitions collapse temporal to memoryless"}
When the transition matrix is uniform, `B @ posterior` is a uniform vector, so the
temporal forward update reduces exactly to the per-step memoryless posterior
`P(s_t | o_t) ∝ A[o_t]`; the temporal and memoryless decoders then coincide.
:::

![{{FIGURE_TEMPORAL_BELIEFS_X86_ACCUMULATOR_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_ACCUMULATOR_PATH}}){#fig:temporal-beliefs-x86-accumulator width=90%}

The companion heatmaps in [@fig:temporal-beliefs-x86-accumulator],
[@fig:temporal-beliefs-x86-bitwise], [@fig:temporal-beliefs-x86-guard],
[@fig:temporal-beliefs-x86-long-accumulator], [@fig:temporal-beliefs-x86-memory],
and [@fig:temporal-beliefs-x86-guard-taken] expose the same filtering contract
across the labelled fixtures. They are supporting diagnostics rather than additional
independent experiments: row labels are observation IDs, colors are posterior
probabilities, and the hand-authored phase labels are not discovered from the
bytes. Four further panels make the expanded fixture design explicit: the
call/ret panel ([@fig:temporal-beliefs-x86-call-return]) shows an in-span call
and `ret` roundtrip whose return address is pushed and popped inside the
decoded span; the not-taken-guard panel ([@fig:temporal-beliefs-x86-branch-not-taken])
holds the guard phase across consecutive comparisons whose conditional branches
are statically decoded but not taken; the stack-local panel
([@fig:temporal-beliefs-x86-stack-local]) interleaves accumulator arithmetic
with bounded stack-local state in repeated phase runs; and the
optimization-rewrite panel ([@fig:temporal-beliefs-x86-optimization-rewrite])
holds the add/inc/lea optimization-sensitive motif, in which the `lea` rewrite
lands in the residual `other` observation category even though its latent phase
is accumulator update — a designed ambiguity retained as data, not smoothed
away. The earlier two expanded panels retain a bounded stack-memory operand and
a taken branch whose static fall-through instruction is not executed.

![{{FIGURE_TEMPORAL_BELIEFS_X86_CALL_RETURN_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_CALL_RETURN_PATH}}){#fig:temporal-beliefs-x86-call-return width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_BRANCH_NOT_TAKEN_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_BRANCH_NOT_TAKEN_PATH}}){#fig:temporal-beliefs-x86-branch-not-taken width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_STACK_LOCAL_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_STACK_LOCAL_PATH}}){#fig:temporal-beliefs-x86-stack-local width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_OPTIMIZATION_REWRITE_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_OPTIMIZATION_REWRITE_PATH}}){#fig:temporal-beliefs-x86-optimization-rewrite width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_BITWISE_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_BITWISE_PATH}}){#fig:temporal-beliefs-x86-bitwise width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_GUARD_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_GUARD_PATH}}){#fig:temporal-beliefs-x86-guard width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_LONG_ACCUMULATOR_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_LONG_ACCUMULATOR_PATH}}){#fig:temporal-beliefs-x86-long-accumulator width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_MEMORY_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_MEMORY_PATH}}){#fig:temporal-beliefs-x86-memory width=90%}

![{{FIGURE_TEMPORAL_BELIEFS_X86_GUARD_TAKEN_CAPTION}}]({{FIGURE_TEMPORAL_BELIEFS_X86_GUARD_TAKEN_PATH}}){#fig:temporal-beliefs-x86-guard-taken width=90%}

The retrospective ordering heuristic defined above scores each already decoded
candidate by an epistemic term and a pragmatic term (a log-probability preference
over observation categories), then selects the candidate with the lowest score
[@friston2017activeinference]. This yields a category-informed instruction
ordering compared with left-to-right scanning. Its acquisition boundary differs
from that of the prospective sensor Agent specified above.

We retain two explicitly named epistemic quantities. The primary corpus policy
uses the *realized* information gain of the category already decoded for a
candidate. Its predictive companion weights that entropy reduction by
P(o)=A[o,:]D, the prior predictive probability of the category. Therefore

$$
I(S;O)=\sum_o P(o)\left[H(D)-H(P(S\mid o))\right],
$$ {#eq:predictive-information-gain}

and the per-category weighted terms sum to the one-step expected information
gain {{EXPECTED_OBSERVATION_INFO_GAIN}} nats for the authored model. This
predictive quantity is non-negative in aggregate and provides an honest
calibration diagnostic. It does not create an unmodelled stochastic sensor:
candidate categories are still decoded before selection, so the active
comparison remains a bounded demonstrator rather than a claim of deployment-
ready active binary acquisition.

### Action-conditioned sensing

The next active-inference tranche changes the information-acquisition contract.
Rather than scoring a category that is already visible, a sensing action
`a=(t,f)` requests field `f` from instruction step `t`. The action policy sees
only a training-only field likelihood and the current phase belief before it
chooses. The held-out typed value is revealed by the bounded fixture
environment only after the choice. For a field support `V_f`, the predictive
outcome distribution is

$$
P(v\mid b_t,a)=\sum_s \widehat p_f(v\mid s)b_t(s),
$$ {#eq:action-predictive-outcome}

and the action's expected epistemic value is the mutual-information term

$$
\operatorname{EIG}(a\mid b_t)=
\sum_{v\in V_f}P(v\mid b_t,a)\left[H(b_t)-H(b_{t+1}^{(v)})\right].
$$ {#eq:action-expected-information}

The demonstrator uses a common normalized action cost `c(a)=1` and ranks
actions by

$$
G_{\mathrm{sense}}(a\mid b_t)=
-\operatorname{EIG}(a\mid b_t)+\lambda c(a),
\qquad \lambda={{ACTION_SENSING_COST_WEIGHT}}.
$$ {#eq:action-sensing-efe}

Here `lambda` is a fixed configured weight, not a fitted utility or a physical
cost conversion; the generated artifact also records the active-minus-fixed
action-count contrast as an outcome. After selection, the observed field value
updates the belief with the corresponding factor likelihood. The evaluation
compares this predictive policy with source-order fixed sensing, a seeded-random
permutation, and an explicitly inadmissible oracle that is allowed to inspect
the realized outcome before choosing. All policies share the same fixture-level
holdout, threshold, and action budget. This separation makes the no-peeking
boundary testable and prevents a favorable realized outcome from being used to
retroactively justify the active choice.

The action-conditioned output is a bounded sensor-choice mechanism experiment,
not a claim about autonomous binary analysis. Costs are normalized action units
rather than wall-clock measurements, and field values remain syntactic
Capstone/Unicorn evidence rather than discovered program semantics.

### Exact finite-horizon information-per-cost analysis {#sec:information-frontier}

The action-conditioned experiment is paired with a smaller mechanism analysis
whose optimization problem is exact within its declared finite support. The
label-blind factorized model supplies one categorical sensor for each selected
typed field. For sensor $j$ with outcomes $v$ and state-conditioned channel
$L_j(v\mid s)$, belief $b$ yields predictive probability

$$
p_j(v\mid b)=\sum_s L_j(v\mid s)b(s),\qquad
b^{(v)}(s)=\frac{L_j(v\mid s)b(s)}{p_j(v\mid b)}.
$$ {#eq:finite-sensor-posterior}

Let $\mathcal{J}_{-j}$ denote the remaining sensors in $\mathcal{J}$
after querying and removing $j$. The action objective $Q_h$ and finite-horizon value $J_h$
retain that remaining-set argument explicitly:

$$
\begin{gathered}
Q_h(b,j;\mathcal{J})=\lambda c_j+
\sum_{v:p_j(v\mid b)>0}p_j(v\mid b)
J_{h-1}(b^{(v)};\mathcal{J}_{-j}),\\
J_h(b;\mathcal{J})=\min_{j\in\mathcal{J}}Q_h(b,j;\mathcal{J}).
\end{gathered}
$$ {#eq:finite-sensor-dynamic-program}

The terminal values are $J_0(b;\mathcal{J})=H(b)$ and
$J_h(b;\varnothing)=H(b)$, with declared sensor costs
$c_j>0$. Posterior updates are defined only for positive-probability outcomes.
The implementation enumerates all positive-probability outcome branches and
memoizes normalized beliefs, remaining sensors, horizon, and cost weight. It
therefore supplies an exact result for the finite sensor family and horizons
in the receipt, while making no claim about an infinite-horizon policy,
continuous sensing, or operational acquisition time.

The source-bound artifact is `{{ACTIVE_INFORMATION_ARTIFACT}}`, with
`{{ACTIVE_INFORMATION_SENSOR_COUNT}}` sensors and
`{{ACTIVE_INFORMATION_FRONTIER_ROW_COUNT}}` frontier rows. Exact branch
enumeration is recorded as `{{ACTIVE_INFORMATION_EXACT}}`; the frontier is
visualized in [@fig:active-information-frontier]. That information
frontier uses model-derived unit costs and is therefore a
mechanism result, not a wall-clock law. The measured-cost receipt in
the measured-sensing section below is a separate, deterministic/calibrated layer;
neither artifact supports a universal cost claim or a policy guarantee outside
the finite declared family.

![{{FIGURE_ACTIVE_INFORMATION_FRONTIER_CAPTION}}]({{FIGURE_ACTIVE_INFORMATION_FRONTIER_PATH}}){#fig:active-information-frontier width=96%}

![{{FIGURE_POLICY_CERTIFICATE_FRONTIER_CAPTION}}]({{FIGURE_POLICY_CERTIFICATE_FRONTIER_PATH}}){#fig:policy-certificate-frontier width=100%}

The certificate frontier ([@fig:policy-certificate-frontier]) makes the cost
model auditable at the same time as
it makes the optimization boundary visible. Exact dynamic programming supplies
the finite oracle and its regret/optimality gap; any timing overlay is a local
calibration with its own machine receipt. The visual must therefore be read as
two linked coordinate systems, not as a universal exchange rate between
information and hardware time.

A labelled corpus of `{{TEMPORAL_FIXTURE_COUNT}}` transparent byte fixtures, each
carrying a hand-authored ground-truth phase sequence aligned to its decoded
instructions, supports the project's first quantitative evaluation: decoding
accuracy, calibration, a free-energy convergence summary, and an
identity-transition, temporal, and Viterbi comparison, each guarded by shuffled-label
and uniform-belief negative controls so that a reported gain cannot be vacuous
[@heins2022pymdp].

The corpus is a deliberately small, hand-labelled demonstrator rather than a
sample of arbitrary binaries. The confidence intervals reported below are
descriptive Wilson score intervals around pooled per-step hit counts; steps
within a fixture are dependent. In addition to the transparent hand-authored
model, the project now reports a leakage-safe supervised fit: each
leave-one-fixture-out row estimates A, B, and D on the other fixtures only, then
scores the excluded fixture. The calibration summaries follow the
reliability-diagram and ECE framing of [@guo2017calibration], while remaining
fixture-specific. Wilson score intervals are used because the displayed
quantities are binomial hit-rate summaries and the Wilson construction behaves
more reliably than a Wald approximation for small denominators
[@brown2001binomial]. This choice does not remove the dependence among steps
or broaden the corpus into a population sample.

The training/development/held-out assignment is a declared fixture inventory.
The temporal estimation reports below use leave-one-fixture-out folds: each
fold excludes its scored fixture but may train on fixtures assigned to the
inventory's other partitions. They do not report a model trained once on the
fixed training partition and evaluated once on its held-out partition.
Related authored fixtures are also not independent source lineages. The
fold results therefore support a complete-fixture exclusion diagnostic,
with no claim of lineage-disjoint generalization.

The corpus expands the original six authored programs with four designed
fixtures: an in-span call/`ret` roundtrip (the call pushes a return address
inside the decoded span and the `ret` pads it back), a not-taken guard pair
(two conditional branches whose guards both fall through, so every statically
decoded instruction executes), a stack-local fixture (a push/pop pair keeps
bounded scratch state below the stack pointer while accumulator arithmetic
continues), and an optimization-rewrite fixture (the add/immediate, `inc`, and
`lea` rewrite motif in which `add` and `inc` share one arithmetic observation
category and the `lea` rewrite lands in the residual `other` category despite
its accumulator-update latent phase). The temporal family split
(`{{TEMPORAL_SPLIT_SUMMARY}}` training/development/held_out) was frozen in the
protocol before any scoring under protocol v2.1, with each fixture assigned
wholesale to exactly one partition and no family overlap.

The expanded experiments are governed by one versioned protocol rather than by
independently edited scripts. The protocol receipt is
`{{RESEARCH_PROTOCOL_ARTIFACT}}`, identified as
`{{RESEARCH_PROTOCOL_ID}}` and hashed as
`{{RESEARCH_PROTOCOL_HASH}}`. Every new scientific receipt records that hash,
its own schema version, a producer-source digest, input hashes or source
manifests, and the available toolchain versions. A generated number therefore
has two obligations: it must be reproducible from its producer, and it must
remain within the frozen scientific contract. A byte-for-byte replay of an
unfrozen protocol would only reproduce an implementation accident.

The protocol freezes the observational state, interventions, missing/unseen
tokens, supported instruction boundary, family-level partitions, model
families, seeds, primary metrics, clustered uncertainty, multiplicity policy,
deterministic sensing costs, and the external-source safety boundary. The
public input directory in the local replication bundle is label-blind. A
missing field is encoded as `__MISSING__`; a value absent from training support
is encoded as `__UNKNOWN__`. Neither is replaced by zero, an empty string, or a
silently dropped row. Unsupported instructions, unavailable compilers, build
failures, and underpowered strata remain represented as structured statuses.

### Protocol-v2 artifact envelope and manuscript closure {#sec:artifact-receipt-contract}

The protocol contract is layered rather than destructive. Every project-owned
JSON artifact currently carries the common receipt schema
`{{ARTIFACT_RECEIPT_SCHEMA_VERSION}}` above its domain-specific `schema_version`.
The common envelope requires `{{ARTIFACT_RECEIPT_REQUIRED_FIELD_COUNT}}` fields:
the protocol identifier and digest, producer revision, input hashes, toolchain
versions, evidence status, limitations, unsupported cases, and underpowered
cases. This separation keeps a trace-schema migration distinguishable from a
provenance-contract migration and allows domain validators to retain their
existing interpretation.

The current generated surface contains `{{ARTIFACT_RECEIPT_COUNT}}` JSON
artifacts, of which `{{ARTIFACT_RECEIPT_ERROR_COUNT}}` fail the receipt audit.
Producers write through an atomic temporary-file replacement, so a failed
serialization cannot expose a partially written receipt. Legacy payloads may
be migrated only by an explicit producer-stage command; validation is
read-only and fails closed when a payload bypasses that stage. The canonical
audit command is
`{{ARTIFACT_RECEIPT_AUDIT_COMMAND}}`. This is a provenance and publication-
integrity mechanism informed by PROV-O, RO-Crate, and reproducible-computing
practice [@w3cprov2013;@rocrate2023;@sandve2013repro]. Its receipt count is not
a scientific sample size, and its passing state cannot promote semantic
identifiability, external validity, or independent replication. The local
versus independently reproduced distinction follows the terminology used by
the ACM artifact-review policy [@acm2020artifact].

## Assembly state, normalized intermediate representation, and flags {#sec:assembly-formalism}

The machine-facing formalism is a normalized instruction intermediate
representation (IR), not a claim that a mnemonic string is a semantic object.
For a decoded instruction $i$, the IR records

$$
I_i=(\mathrm{pc}_i,\mathrm{op}_i,\mathrm{dst}_i,\mathrm{src}_i,
\mathrm{imm}_i,\mathrm{mem}_i,\mathrm{reads}_i,\mathrm{writes}_i,
\mathrm{flags}^{r}_i,\mathrm{flags}^{w}_i,\mathrm{target}_i),
$$ {#eq:normalized-assembly-ir}

where `pc` is the instruction address, `op` is a normalized operation family,
`dst/src` are canonical register or memory terms, and the remaining fields
preserve explicit register, flag, memory, and control-flow effects. The
canonicalization is conservative: it can identify a bounded class of equivalent
operand descriptions, but it does not infer calling conventions, alias sets,
compiler intent, or source-level names.

The common bounded state is

$$
\sigma=(pc,G,F,M,K,i,x),
$$ {#eq:assembly-machine-state}

with program counter $pc$, modeled general-purpose registers $G$, condition
flags $F$, bounded byte-addressable memory $M$, bounded stack slice $K$,
decoded IR record $i$, and an explicit exceptional/unsupported marker $x$. The
modeled flags are those read or written by the supported operation; they are
not treated as globally known architectural state when an instruction does not
define them. A supported small-step transition has the form

::: {.definition #def:normalized-machine-state title="Normalized bounded machine state"}
The register component is the explicit 16-register file
$G=(g_0,\ldots,g_{15})\in(\mathbb{W}_{64})^{16}$, with `RIP` represented by
$pc$ rather than hidden in memory. A byte-addressable memory is a finite partial
map $M:\mathbb{W}_{64}\rightharpoonup\mathbb{W}_{8}$ whose declared domain is
the bounded memory interval; the stack slice $K$ is a named subinterval of that
domain. Each flag is a pair `(value, definedness)`, so a preserved or undefined
flag is never coerced to a Boolean observation. The state also records the
instruction index, the normalized instruction, and one of `running`,
`halted`, `unsupported`, `exceptional`, or `bound_exceeded`.
:::

Operand width is part of the transition, not an annotation added after the
fact. For a value $v$ written to a register whose current value is $g$, the
write operator is

$$
\operatorname{write}_w(g,v)=
\begin{cases}
v\bmod 2^{64}, & w=64,\\
v\bmod 2^{32}, & w=32,\\
(g\mathbin{\&}\neg(2^w-1))\mathbin{|}(v\bmod 2^w), & w\in\{8,16\}.
\end{cases}
$$ {#eq:register-width-write}

Thus a 32-bit write zero-extends into the 64-bit architectural register, while
8- and 16-bit writes preserve their untouched upper bits. For a memory operand
with base $b$, index $j$, scale $s\in\{1,2,4,8\}$, and signed displacement
$d$, the effective address is

$$
a=(b+s\,j+d)\bmod 2^{64},\qquad
\operatorname{valid}(a,n)\iff [a,a+n-1]\subseteq\operatorname{dom}(M).
$$ {#eq:bounded-effective-address}

Loads and stores are admitted only when `valid` holds for every byte. The
reference interpreter does not turn a failed bounds check into a wrapped or
zero-filled value; it returns a structured bound or unsupported result.

For $w$-bit operands $a,b$ and result $r$, the arithmetic flags are defined by

$$
\begin{aligned}
CF_{+}&=\mathbf{1}[a+b\ge 2^w], & CF_{-}&=\mathbf{1}[a<b],\\
ZF&=\mathbf{1}[r=0], & SF&=\operatorname{bit}_{w-1}(r),\\
OF_{+}&=\mathbf{1}[\operatorname{sign}(a)=\operatorname{sign}(b)\ne\operatorname{sign}(r)],\\
OF_{-}&=\mathbf{1}[\operatorname{sign}(a)\ne\operatorname{sign}(b)\land
\operatorname{sign}(r)\ne\operatorname{sign}(a)].
\end{aligned}
$$ {#eq:integer-flag-semantics}

`ADC` and `SBB` include the incoming defined `CF` in the arithmetic before
computing their outgoing flags. `INC` and `DEC` use the corresponding
add/subtract result but preserve the incoming `CF`; `CMP` uses subtraction to
set flags without writing its destination, and `TEST` uses a logical AND
without writing either operand. Logical and shift operations mark only the
flags explicitly defined by the contract. These cases are separately exercised
at zero, all-ones, sign-transition, carry/borrow, and shift-boundary values.

::: {.proposition #prop:bounded-step-determinism title="Determinism of a supported bounded step"}
For a normalized instruction whose operands, defined flag reads, memory range,
and control-flow target satisfy the protocol bounds, the reference transition
$\langle\sigma,I_i\rangle\to\sigma'$ is deterministic. If any precondition is
not satisfied, the transition returns one structured non-running status rather
than an architectural value. This proposition is about the implemented finite
interpreter; it is not a completeness or total-correctness theorem for x86-64.
:::

$$
\langle\sigma,I_i\rangle\longrightarrow\sigma'
$$ {#eq:assembly-small-step}

and is defined only when operand widths, address bounds, flag dependencies, and
exceptional behavior are within the declared contract. For example, a 64-bit
addition updates the destination modulo $2^{64}$ and computes its declared
carry, zero, sign, and overflow bits; a compare updates flags without writing
the destination; and a conditional branch reads only the flags listed by its
condition. `movzx`, `movsx`, `lea`, bounded loads/stores, stack operations,
calls, returns, and branches are admitted only when their width and target
behavior fit the bounded state. SIMD, floating point, privileged/system state,
self-modifying code, unmodeled exceptions, undefined behavior, and unbounded
aliasing are rejected with an unsupported reason.

The assembly listing in [@lst:assembly_fixture] remains the human-readable
projection of this IR. The new correspondence receipt compares the pre-state,
decoded address, post-state, modeled flags, memory effects, and control-flow
target at every bounded step; terminal equality alone is not accepted as a
trace proof. The Intel manuals specify architectural behavior, Capstone
supplies a decoding/API boundary, Unicorn supplies bounded execution, and the
Lean fragment supplies independently checked algebra for a much smaller closed
subset [@intel_sdm;@capstone;@unicorn;@lean_reference].

```{#lst:normalized_ir .text caption="Illustrative normalized instruction records; the generated receipt retains the full source-bound correspondence and unsupported boundary."}
0x1000  MOV  RAX <- 1
  flags: preserved
  memory/target: none
0x1005  ADD  RAX <- RAX + 2
  flags written: CF, PF, AF, ZF, SF, OF
  memory/target: none
0x1009  JCC  if ZF: target
  flags read: ZF; flags preserved
  memory/target: bounded internal target
```

## Formal correspondence and transfer receipts {#sec:formal-correspondence}

The Python reference interpreter is executed on transparent repository-owned
fixtures and compared with real Capstone decoding and bounded Unicorn traces.
The receipt `{{FORMAL_CORRESPONDENCE_ARTIFACT}}` covers
`{{FORMAL_CORRESPONDENCE_FIXTURE_COUNT}}` fixtures under
`{{FORMAL_CORRESPONDENCE_INTERVENTION_COUNT}}` declared interventions each,
with `{{FORMAL_CORRESPONDENCE_STEP_COUNT}}` reference step rows. Its status is
`{{FORMAL_CORRESPONDENCE_STATUS}}`, and the comparison is visualized in
[@fig:formal-correspondence]. The comparison is all-step and fail-closed:
an unsupported row, environment error, or mismatch cannot be converted into a
terminal-only success. The Lean prerequisite is reported separately in
`{{FORMAL_VERIFICATION_ARTIFACT}}`; a generated Lean bridge now serializes one
shared bounded state with all 16 general-purpose registers, `RIP`, modeled flag
definedness, memory and stack bounds, step index, and running status. The
bridge covers `{{LEAN_BRIDGE_FIXTURE_COUNT}}` transparent fixtures with
`{{LEAN_BRIDGE_THEOREM_COUNT}}` generated machine-state theorems, all
kernel-checked (`{{LEAN_BRIDGE_KERNEL_CHECKED}}`) under the
`{{FORMAL_TOOLCHAIN}}` toolchain, alongside `{{FORMAL_THEOREM_COUNT}}` formal
theorems with an axiom report that is clean
(`{{FORMAL_AXIOM_REPORT_CLEAN}}`). The kernel-checked fixture invariants cover
the 64-bit `RAX` increment, preservation of the other registers and carry
flag, step advancement, and bound preservation. A compiled Lean theorem is not
silently promoted to a proof of the Python lifter, the Capstone/Unicorn path,
or the complete x86-64 ISA.

![{{FIGURE_FORMAL_CORRESPONDENCE_CAPTION}}]({{FIGURE_FORMAL_CORRESPONDENCE_PATH}}){#fig:formal-correspondence width=100%}

The transfer track is deliberately separate from the transparent-execution
track. Portable source families are compiled with the exact locally available
Clang target command, object and disassembly hashes are retained, and emitted
objects are statically decoded without execution. The Clang manual defines the
compiler-driver and target-selection interface used for this acquisition
[@clang_users_manual]. Optional NASM and Rust cells use their documented
object-format and static-emission interfaces [@nasm;@rustc_manual]; a
missing compiler or target remains an unavailable cell. The design is provenance-oriented rather than a
compiler-correctness claim; translation validation is the relevant precedent
for treating each produced translation as an individually checkable artifact
[@necula2000translationvalidation]. The current receipt is
`{{COMPILER_TRANSFER_ARTIFACT}}`: a `{{COMPILER_TRANSFER_ROW_COUNT}}`-cell
matrix over the compilers `{{COMPILER_TRANSFER_COMPILERS}}`, the object
formats `{{COMPILER_TRANSFER_OBJECT_FORMATS}}`, and the architectures
`{{COMPILER_TRANSFER_ARCHITECTURES}}`, of which
`{{COMPILER_TRANSFER_VALIDATED_COUNT}}` cells are validated static objects and
`{{COMPILER_TRANSFER_UNAVAILABLE_COUNT}}` are explicit unavailable rows
retaining their compiler diagnostics. The
separate licensed-source receipt is `{{EXTERNAL_SOURCE_STATUS}}`; it therefore
cannot support an application-facing effectiveness claim. A target absent from
the local compiler is a missing capability, not evidence about that
architecture. This is the correct interpretation of the compiler capability summary in
[@fig:compiler-transfer].

The separate licensed-source receipt is `{{EXTERNAL_SOURCE_ARTIFACT}}` with
status `{{EXTERNAL_SOURCE_STATUS}}`, `{{EXTERNAL_SOURCE_PROJECT_COUNT}}`
manifested projects, and `{{EXTERNAL_SOURCE_LABELABLE_EXAMPLE_COUNT}}`
labelable examples. The pinned manifest records `{{EXTERNAL_STATIC_ROW_COUNT}}`
source rows spanning `{{EXTERNAL_SOURCE_PROJECT_COUNT}}` projects and two
revisions per project. Static coverage and source-bound
labels are separate evidence surfaces. The static receipt
`{{EXTERNAL_STATIC_ARTIFACT}}` is `{{EXTERNAL_STATIC_STATUS}}`: it retains
`{{EXTERNAL_STATIC_VALIDATED_COUNT}}` successfully compiled-and-decoded rows
of `{{EXTERNAL_STATIC_ROW_COUNT}}`; no static object is treated as an
equivalence label. The dedicated label receipt `{{EXTERNAL_LABEL_ARTIFACT}}`
is `{{EXTERNAL_LABEL_STATUS}}` and contains `{{EXTERNAL_LABEL_ROW_COUNT}}`
accepted rows: `{{EXTERNAL_LABEL_EQUIVALENT_COUNT}}`
lexical-equivalence witnesses, `{{EXTERNAL_LABEL_NEAR_MISS_COUNT}}`
contract-directed one-change near misses, and
`{{EXTERNAL_LABEL_NEGATIVE_COUNT}}` identity negative controls, with
`{{EXTERNAL_LABEL_EXCLUSION_COUNT}}` explicit exclusions. The `{{EXTERNAL_LABEL_TEST_PROJECTS}}`
project is reserved for the held-out split. A trailing comment is a
preprocessing-token witness, not a universal semantic-equivalence theorem;
near-miss rows are source-contract controls and are not executed behavioral
labels. Every accepted row binds an immutable revision, license-file digest,
archive digest, compiler argument vector, target triple, source edit and
argument, base/pair source bytes, object hashes, canonical disassembly hashes,
and Capstone coverage; emitted objects are not executed by the canonical
pipeline.

The external-source gate is therefore a safety and provenance interface, not a
permission to execute downloaded software. Acquisition requires the explicit
network opt-in command and a pinned manifest; replay, scoring, and manuscript
generation remain offline. The current receipt records
`{{EXTERNAL_SOURCE_GATE_COUNT}}` gate fields and keeps failed static rows,
source-contract exclusions, and the missing separate replay as `underpowered`
or `not_yet_observed` evidence rather than inventing a denominator. The local
compiler capability receipt is `{{EXTERNAL_CAPABILITIES_ARTIFACT}}` with
status `{{EXTERNAL_CAPABILITIES_STATUS}}` and `{{EXTERNAL_CAPABILITY_TARGET_COUNT}}`
target probes that produced a static object. A compiler-advertised target is
not counted unless this empirical probe succeeds.

The acquisition attempt itself is separately recorded in
`{{EXTERNAL_ACQUISITION_ARTIFACT}}` with status
`{{EXTERNAL_ACQUISITION_STATUS}}`, `{{EXTERNAL_ACQUISITION_ROW_COUNT}}` retained
source rows, and recorded network use `{{EXTERNAL_ACQUISITION_NETWORK_USED}}`.
That receipt binds the manifest and producer hashes, preserves one row per
pinned source, and distinguishes a verified local-cache reuse from a network
download. Downloads are published only after archive-digest verification;
extraction is staged, archive links are rejected, and the declared license
bytes are checked before publication. The validator repeats those checks
without network access. This is a freshness and safety improvement for the
source-bound corpus, not evidence that the external build was independently
replayed or that any source-level contract is a universal behavioral-
equivalence label.

![{{FIGURE_COMPILER_TRANSFER_CAPTION}}]({{FIGURE_COMPILER_TRANSFER_PATH}}){#fig:compiler-transfer width=100%}

![{{FIGURE_STATISTICAL_HIERARCHY_CAPTION}}]({{FIGURE_STATISTICAL_HIERARCHY_PATH}}){#fig:statistical-hierarchy width=100%}

The hierarchy ([@fig:statistical-hierarchy]) is the reason the analysis reports
family-level uncertainty.
Transformations, interventions, traces, and instruction steps are useful
observations, but they share source lineage and cannot be treated as a flat
independent sample. Fixing the family split before fitting also prevents the
development and test boundaries from being reconstructed from row-level
features or source order.

## Quotient classes and co-primary classifier portfolio {#sec:quotient-classifier}

The quotient endpoint is pairwise by design. A pair is labelled positive when
the two transparent programs have equal terminal `RAX` for every intervention
in the frozen catalog; it is negative when at least one intervention separates
the terminal outputs. The relation is generated before model fitting and is
not a mnemonic label, family identifier, transformation name, or model output.
This is a bounded observational quotient, not complete program equivalence. In
formal-methods terms it is an observation relation over a finite intervention
catalog, not a bisimulation relation preserved by every transition
[@sangiorgi2011bisimulation].

The classifier receipt is `{{QUOTIENT_CLASSIFIER_ARTIFACT}}` with status
`{{QUOTIENT_CLASSIFIER_STATUS}}`; its held-out portfolio comparison is visualized
in [@fig:quotient-classifier]. It uses `{{QUOTIENT_CLASSIFIER_FAMILY_COUNT}}`
families and
`{{QUOTIENT_CLASSIFIER_ROW_COUNT}}` pair rows, split into
`{{QUOTIENT_CLASSIFIER_TRAINING_FAMILY_COUNT}}` training,
`{{QUOTIENT_CLASSIFIER_DEVELOPMENT_FAMILY_COUNT}}` development, and
`{{QUOTIENT_CLASSIFIER_HELD_OUT_FAMILY_COUNT}}` held-out families. No family
occurs in more than one partition. The calibrated interpretable model is a
regularized binary logistic classifier over typed trace summaries; the temporal
model is a class-conditional first-order Markov likelihood over typed token
sequences; and the recurrent model is a fixed one-layer NumPy tanh recurrent
network. The three are co-primary. A typed-trace summary is retained as the
historical surface comparator, and six additional raw surface baselines are fit
independently from training-only vocabularies: opcode histograms, hashed byte
bigrams, mnemonic counts, instruction-length summaries, control-flow summaries,
and memory-event counts. These baselines are not collapsed into one favorable
“surface” score; each receives the same held-out predictions, family-level
bootstrap, and development/test separation as the model portfolio.

The fixed iteration budgets and stopping rules are part of the protocol, as is
the eight-state intervention catalog (`{{QUOTIENT_CLASSIFIER_INTERVENTION_COUNT}}`
initial register values): the
logistic and recurrent fits do not inspect the held-out test set, while the
Markov fit is a smoothed closed-form count estimator. Five deterministic seeds
are rerun for each stochastic family and their range is recorded separately
from the source-family bootstrap intervals. The recurrent implementation also
emits a finite-difference gradient receipt; the checked derivative error is a
software correctness diagnostic, not a performance interval.

For primary metric $m$ and model $k$, balanced accuracy is the mean of positive
and negative recall, while macro-F1 averages the F1 values for both quotient
classes. Let $u$ index held-out transformation-family identifiers and let $\widehat\theta_{k,m}^{(b)}$
be the metric after resampling whole families with replacement. Resampling the
family identifier, rather than treating repeated intervention rows as independent,
is the declared dependence safeguard; the small-cluster caveat remains explicit
[@cameron2008clusterbootstrap]. These named transformation families share
authored template lineages; disjoint identifiers do not establish independent
source sampling. The intervals and prediction swaps are conditional diagnostics
on the finite inventory, with no population-generalization coverage claim.
The reported interval is the 2.5th--97.5th
percentile of the frozen
`{{CI_LEVEL_PERCENT}}%` cluster bootstrap:

$$
\mathrm{CI}_{k,m}=\left[Q_{0.025}(\widehat\theta_{k,m}^{(b)}),
Q_{0.975}(\widehat\theta_{k,m}^{(b)})\right].
$$ {#eq:quotient-cluster-interval}

Brier score and fixed-bin expected calibration error are retained alongside the
primary metrics. The five-percentage-point practical margin is retained. Protocol v2.2 separates
percentile bootstrap intervals from significance testing: an uncentered
bootstrap fraction below zero is a descriptive tail fraction, not a null
p-value. The paired randomization diagnostic swaps all predictions in each
held-out family between the model and baseline, recomputes both metrics, and
uses inclusive ties with $(b+1)/(B+1)$ for $B$ Monte Carlo draws
[@phipson2010permutation]. Its null assumes prediction-block exchangeability
within families, conditional on the fitted models. It does not include
model-refitting uncertainty. This is a retrospective methods correction;
superiority claims are withheld pending a prospectively frozen independent
comparison. Holm adjustment [@holm1979multiple] spans the
co-primary model-by-baseline-by-metric comparison family. The current held-out values
are {{QUOTIENT_CLASSIFIER_INTERPRETABLE_BALANCED_ACCURACY}} /
{{QUOTIENT_CLASSIFIER_INTERPRETABLE_MACRO_F1}} for the interpretable model,
{{QUOTIENT_CLASSIFIER_TEMPORAL_BALANCED_ACCURACY}} /
{{QUOTIENT_CLASSIFIER_TEMPORAL_MACRO_F1}} for the temporal model,
{{QUOTIENT_CLASSIFIER_NEURAL_BALANCED_ACCURACY}} /
{{QUOTIENT_CLASSIFIER_NEURAL_MACRO_F1}} for the recurrent model, and
{{QUOTIENT_CLASSIFIER_BASELINE_BALANCED_ACCURACY}} /
{{QUOTIENT_CLASSIFIER_BASELINE_MACRO_F1}} for the surface baseline. These
numbers are reported as a portfolio: overlapping clustered intervals,
development/test differences, and the small number of declared families
do not support a superiority claim. The retrospective correction withholds
all such claims; a future prospectively frozen comparison must satisfy the
practical margin and inference rule on independent evidence. The receipt retains
`{{QUOTIENT_CLASSIFIER_RAW_BASELINE_COUNT}}` raw baseline models and
`{{QUOTIENT_CLASSIFIER_RAW_COMPARISON_COUNT}}` model-by-baseline comparison rows;
no post-hoc baseline winner is promoted.

![{{FIGURE_QUOTIENT_CLASSIFIER_CAPTION}}]({{FIGURE_QUOTIENT_CLASSIFIER_PATH}}){#fig:quotient-classifier width=100%}

![{{FIGURE_QUOTIENT_CONSTRUCTION_CAPTION}}]({{FIGURE_QUOTIENT_CONSTRUCTION_PATH}}){#fig:quotient-construction width=100%}

The construction diagram ([@fig:quotient-construction]) separates three
operations that are often conflated:
declaring a source-level pair role, applying the finite intervention catalog,
and assigning an observational label from the resulting normalized traces.
Only the last operation supplies the classifier target. The relation is useful
for a controlled benchmark because it is reproducible; it is not a claim that
two programs agree under every input, architecture, exception, or calling
convention.

### Exact intervention power and finite separating sets {#sec:intervention-power}

The quotient relation is only as informative as its declared intervention
catalog. We therefore compute intervention power directly from the source-bound
terminal-output matrix, before any classifier is fitted. For the separate
representation-invariance pilot, let $\mathcal U_{\mathrm{pilot}}$ be its four
initial-state interventions and let $e_{p,u}\in\{0,1\}$ indicate
whether pair $p$ has equal terminal output under intervention $u$. For every
non-empty subset $S\subseteq\mathcal U_{\mathrm{pilot}}$, the near-miss separation and
equivalent-pair consistency are

$$
\begin{gathered}
\operatorname{Sep}(S)=
\frac{1}{|\mathcal N|}\sum_{p\in\mathcal N}
\mathbf 1[\exists u\in S:e_{p,u}=0],\\
\operatorname{Cons}(S)=
\frac{1}{|\mathcal E|}\sum_{p\in\mathcal E}
\mathbf 1[\forall u\in S:e_{p,u}=1].
\end{gathered}
$$ {#eq:intervention-power}

where $\mathcal E$ and $\mathcal N$ are the declared equivalent and near-miss
sets. A subset is *fully separating* only when both quantities equal one. The
producer enumerates all `{{INTERVENTION_POWER_SUBSET_COUNT}}` non-empty subsets
over `{{INTERVENTION_POWER_PAIR_COUNT}}` pair outcomes, retains the tied
minimum separating sets, and keeps the one-state negative control visible. The
subset enumeration is visualized in [@fig:intervention-power]. The
receipt is `{{INTERVENTION_POWER_ARTIFACT}}`; its minimum fully separating
cardinality is `{{INTERVENTION_POWER_MINIMAL_CARDINALITY}}` with
`{{INTERVENTION_POWER_MINIMAL_SET_COUNT}}` tied set(s). Under the deliberately
insufficient one-state control, separation is
`{{INTERVENTION_POWER_SINGLE_STATE_SEPARATION}}` and
`{{INTERVENTION_POWER_SINGLE_STATE_UNRESOLVED_COUNT}}` near-miss families remain
unresolved.

This construction is exact for the finite matrix and cannot by itself identify
the underlying semantic quotient. In particular, a singleton separator may be
an artifact of the chosen interventions, while an unresolved control can be a
useful counterexample to an underpowered catalog. No intervention subset is
allowed to be chosen after inspecting a model prediction, and the promotion
status remains fail closed.

![{{FIGURE_INTERVENTION_POWER_CAPTION}}]({{FIGURE_INTERVENTION_POWER_PATH}}){#fig:intervention-power width=100%}

## Measured sensing costs and finite policy certification {#sec:measured-sensing}

The active-sensing cost model has two layers. The primary layer is a
deterministic vector

$$
\begin{gathered}
c(a)=(n_{\mathrm{invoke}},n_{\mathrm{decode}},n_{\mathrm{step}},n_{\mathrm{reg/flag}},\\
n_{\mathrm{branch}},n_{\mathrm{memory}},n_{\mathrm{bytes}},n_{\mathrm{unsupported}}).
\end{gathered}
$$ {#eq:deterministic-sensor-cost}

recorded for every query. It is machine-independent and auditable. The
optional calibration layer executes a fixed matrix operation after warm-up and
records median, IQR, 95th percentile, minimum, maximum, and raw repetition
values together with fixed-loop overhead, a deterministic-vector regression,
and Python, operating-system, and package receipts. Wall-clock observations are
not compared across machines without a matching calibration receipt. The
source-bound receipt is `{{SENSOR_COST_ARTIFACT}}`. The current sensor receipt
contains `{{SENSOR_COST_SENSOR_COUNT}}` sensors and
`{{SENSOR_CERTIFICATE_COUNT}}` exact finite certificates; its maximum reported
optimality gap is `{{SENSOR_CERTIFICATE_MAX_GAP}}`.

For a finite state/action problem, the oracle value is obtained by exhaustive
dynamic programming over every positive-probability sensor outcome. The
expected-information component follows the classical view of an experiment as
changing uncertainty under a stated prior, while the cost layer is separately
audited [@lindley1956information]. The implemented policy is certified only when its objective equals the oracle
within the declared five-percent tolerance. The current policy is itself
the exact finite dynamic-programming oracle, so its zero gap is a certificate
of the enumerated benchmark—not of an infinite-horizon policy, a physical
instrument, or a universal active-sensing law. The deterministic/calibrated
comparison is shown in [@fig:sensor-costs].
The same receipt evaluates fixed-first, greedy-information, seeded-random, and
abstaining policies under the same finite outcome tree. Their regret and
deterministic cost vectors are comparison diagnostics; the oracle is an
inadmissible reference and the non-oracle policies receive no certificate
unless they independently satisfy the declared tolerance.

![{{FIGURE_SENSOR_COSTS_CAPTION}}]({{FIGURE_SENSOR_COSTS_PATH}}){#fig:sensor-costs width=100%}

## Frozen blind replication harness {#sec:replication-harness}

The local replication interface is a CLI rather than a notebook state. Its
metadata follows the stewardship aim of making research objects findable,
accessible, interoperable, and reusable [@wilkinson2016fair], while the local
receipt `{{REPLICATION_ARTIFACT}}` makes the narrower claim of deterministic repository replay. The
generator creates the deterministic public fixture/source input, validation
checks all hashes and the protocol digest, each model emits predictions, and a
separate scoring mode joins predictions to the gold labels. The public manifest
contains `record_id`, typed sequence, and feature values only, in deterministic
digest order; the gold file
contains labels and family/split metadata and is held outside the public input
directory. The bundle currently contains `{{REPLICATION_PUBLIC_COUNT}}` public
rows and `{{REPLICATION_GOLD_COUNT}}` gold rows, with label separation reported
as `{{REPLICATION_LABEL_SEPARATED}}`. The public protocol, toolchain receipt,
and offline instructions are part of the hashed public bundle
(`{{REPLICATION_PUBLIC_BUNDLE_SHA256}}`). Replay is local and no-network; the
protected receipt separately records the current endpoint as
`{{INDEPENDENT_REPLAY_STATUS}}` in the `{{INDEPENDENT_REPLAY_CONTROL_SCOPE}}`
scope. The receipt retains separate model (`{{INDEPENDENT_REPLAY_MODEL}}`),
protected-score row-count (`{{INDEPENDENT_REPLAY_ROW_COUNT}}`), and
network-disabled (`{{INDEPENDENT_REPLAY_NETWORK_DISABLED}}`) fields. A completed
receipt requires observed values for these fields and a separately controlled-run
attestation. Replay of the frozen interface is evidence about the frozen interface,
not an independent physical-machine run or an independent reimplementation.
The separation contract is stronger than omitting a visible `label` key. Public
record identifiers are derived only from public sequence/features, and the
validator rejects any forbidden family, split, variant, or quotient field in
the public rows. Exact public/gold row counts, protocol hash, input hashes, and
prediction identifiers must match. A changed source or protocol hash therefore
invalidates the receipt rather than silently producing a new benchmark under an
old claim. The stable commands are `generate-public`, `verify-public`,
`run-model`, `score --gold-path`, `verify-separation`, `replay --offline`,
`write-replay-receipt`, `ingest-replay`, and `write-receipts`; the label-blind
model path never opens the private gold directory. Receipt ingestion requires
an explicit separately controlled-run attestation and a protected scoring
boundary, so a local rerun cannot promote itself. The data flow is summarized
in [@fig:replication-manifest].

![{{FIGURE_REPLICATION_MANIFEST_CAPTION}}]({{FIGURE_REPLICATION_MANIFEST_PATH}}){#fig:replication-manifest width=100%}

![{{FIGURE_REPLAY_FAILURE_AUDIT_CAPTION}}]({{FIGURE_REPLAY_FAILURE_AUDIT_PATH}}){#fig:replay-failure-audit width=100%}

The controlled replay audit ([@fig:replay-failure-audit]) is included because a
reproducibility package must make failure inspectable. When an independent
receipt is ingested, the protected score is computed once over the frozen
interface, and the audit retains whatever discrimination it shows as a bounded
endpoint rather than converting it into a favorable claim by changing the model,
labels, or endpoint. The guest boundary attests to the separately controlled-run
procedure required for ingestion, while the protected scoring boundary confirms
where labels were available. The recorded replay status is
`{{INDEPENDENT_REPLAY_STATUS}}`; controlled replay evidence is not independent
physical replication or independent reimplementation.
