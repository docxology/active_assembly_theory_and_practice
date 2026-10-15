# Getting started

This repository is a transparent assembly/inference demonstrator and technical
monograph. It provides bounded local computation; independent replay,
reimplementation, external validity and broad semantics/theorem promotion
remain separate evidence gates. See [architecture](architecture.md) and
[statistics](statistics.md) for the current computation and interpretation.

## Locked environments

From this checkout, install the committed Python environment:

```bash
uv sync --locked --extra dev
uv run --locked python scripts/00_preflight.py
```

`.python-version` pins Python 3.13; `pyproject.toml` supports Python >=3.11.
Use `uv.lock` rather than selecting newer dependencies incidentally. The
preflight imports real Capstone, Unicorn and pymdp and validates the Julia
manifest; successful imports alone do not run the scientific or document gates.
Keep the environment, base interpreter, caches and temporary files on mounted
GitVault. Set `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, `TMPDIR` and
`UV_PYTHON_INSTALL_DIR` to external locations and use `UV_LINK_MODE=copy`.
Check both `sys.prefix` and `sys.base_prefix` before source work: an external
environment can still point to an internal interpreter. Configure `GNN_ROOT`
as described in [executable GNN models](gnn_models.md).

| Tool | Contract |
| --- | --- |
| Python | Committed 3.13 interpreter pin and locked development extra. |
| Julia / RxInfer | Julia 1.12.7 and RxInfer 5.4.0 in the committed Julia manifest; set `JULIA` to the exact executable if discovery selects another version. |
| Lean | `formal/lean-toolchain` pins v4.28.0; `lake build` checks the declared fragment. |
| Template | A configured `docxology/template` checkout with its own prepared runtime and a working-lane link to this project. |
| Document tools | Pandoc/crossref and the template's Office filters; native LuaLaTeX with tagged PDF support, `seqsplit` and `selnolig`; qpdf and VeraPDF with UA-2 support for post-render checking. |
| Optional capabilities | External compilers, architecture targets and Lima are explicitly probed; unavailable rows remain visible. They are not generic prerequisites for every Python unit test. |

Install native toolchains during environment preparation. Replay and rendering
then use local inputs offline. If the pinned Julia manifest cannot instantiate
because a local General registry is stale, refresh that registry while keeping
the manifest; do not resolve to a replacement version or fabricate a receipt.
Native Julia tests use `allow_reresolve=false`.

## Template handoff

Set `TEMPLATE_REPO_ROOT` to the canonical template checkout. The resolver also
recognizes a sibling `template/` and GitVault's `Public/template/` layout.
Set `TEMPLATE_PYTHON` to its prepared external Python environment. The
project renderer passes the explicit current root to the template's native
rendering APIs and checks fresh output bytes:

```bash
uv run --locked python scripts/render_manuscript.py \
  --project-root "$PWD" --skip-manuscript-hydration
```

The renderer preserves shared template links. Its default runtime is the
template's `.venv/bin/python` (Windows:
`.venv/Scripts/python.exe`); set `TEMPLATE_PYTHON` to a prepared Python
executable if the template uses a different environment. It never changes the
template's dependencies implicitly.

Qualify the selected template revision before the full scientific build. It
must provide the shared manuscript-composition and native Office-projection
APIs, including `infrastructure/rendering/office_native_projection.py`.
The clean committed revision `192c8bda16c08d06ae2fee4eac71d56901d3ad22`
provides this contract; the older `55dbf35` branch does not. Use an isolated
checkout when the shared template has another active owner. A prepared Python
environment alone does not establish renderer compatibility.

## First canonical build

```bash
scripts/run_full_verification.sh --analysis-only --julia
uv run --locked python scripts/render_manuscript.py
scripts/run_full_verification.sh --post-render-only
```

The first command regenerates analysis, runs Python/coverage/lint/type and
native gates, and authoritatively hydrates the manuscript. The second exports
that exact tree through the canonical template with hydration skipped. The
last completes the configured DOCX projection and checks existing outputs
strictly, without refitting models or replacing scientific figures. For one
complete run, use `scripts/run_full_verification.sh --julia`.

See [verification](verification.md) for each mode and [document rendering](document_rendering.md)
for receipts, paths and manual visual review. Generate before checking a fresh
checkout: tests that consume canonical artifacts need current outputs.

## Failure diagnosis

| Symptom | Next check |
| --- | --- |
| Julia version or manifest mismatch | Resolve the exact 1.12.7 binary via `JULIA`, preserve the manifest, rerun the native gate. |
| Missing template/runtime | Check `TEMPLATE_REPO_ROOT`, `TEMPLATE_PYTHON` and both external Python prefixes. |
| Stale composition or hydrated-source digest | Hydrate before the template export; then complete DOCX projection and regenerate document receipts. |
| Blank/missing/stale figure | Rebuild its source receipt and canonical figure bundle before hydration; never repair a receipt by editing its hash. |
| Strict check deferred or unresolved | Use `--post-render-only`, which forces `ACTIVE_ASSEMBLY_DOCX_PRE_RENDER=0`. |
| Missing external archive/compiler/target | Preserve the explicit failure or unavailable status; use the separately authorized pinned acquisition procedure only when needed. |

Generated `output/` is ignored by Git. A push carries authored source,
contracts and documented examples; it does not upload a rendered PDF/DOCX or
prove that another operator reproduced the result.
