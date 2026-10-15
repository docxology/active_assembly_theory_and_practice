# Manuscript and document rendering

The project owns scientific generation and hydration; the configured sibling
`docxology/template` owns native multi-format rendering. Preparation and
registration are in [getting started](getting_started.md); gate selection is
in [verification](verification.md). Generated files must never be edited to
agree with a narrative or a receipt.

## Ordered handoff

1. Generate source-bound receipts, figures, registries and variables.
2. Run `scripts/z_generate_manuscript_variables.py`: it invokes the project
   hydration gate, checks every expected section and substitution, and reaches
   the source-validation summary fixpoint.
3. Run `scripts/render_manuscript.py`: it validates that hydration again,
   passes the explicit current project root to the real template rendering
   APIs with `--skip-manuscript-hydration`. It verifies fresh nonempty
   derivatives and unchanged source/hydrated bytes without changing shared links.
4. Complete the native DOCX projection with `scripts/render_docx_cover.py`.
5. Audit DOCX, PDF structure, qpdf and VeraPDF; validate existing scientific
   outputs with pre-render deferral disabled.
6. Visually inspect the exported editions at page scale and dense figures or
   equations at full size. Machine gates do not replace that review.

`run_scientific_program.py render-manuscript` is a legacy command name for
**hydration only**. It does not invoke the template export. Prefer the explicit
project scripts above. `scripts/run_full_verification.sh` runs the entire
sequence; `--analysis-only` stops before export and `--post-render-only`
checks an existing template edition without regenerating its scientific inputs.

For a manual handoff from this checkout, the explicit-root call is:

```bash
uv run python scripts/render_manuscript.py \
  --project-root "$PWD" \
  --skip-manuscript-hydration
```

Keep source, protocol and producer code stable from hydration through the final
audits. If they change, regenerate and render again. A documentation-only edit
can require refreshed build lineage without changing the scientific bytes;
the existing-output gate checks that relationship explicitly.

## Generated edition and receipt map

| Surface | Path | Contract |
| --- | --- | --- |
| Hydrated sections | `output/manuscript/[0-9][0-9]_*.md` | Exact source-token substitution, expected section set and no unresolved tokens. |
| Combined source | `output/web/_combined_manuscript.md` | Canonical composition in `output/reports/manuscript_composition.json` binds current numbered sections and combined bytes. |
| PDF | `output/pdf/active_assembly_theory_and_practice_combined.pdf` | Native tagged PDF and real tag-tree/UA-2 audits. |
| DOCX | `output/docx/active_assembly_theory_and_practice_combined.docx` | Native Office export, configured cover, scientific media and embedded hydrated-source digest. |
| PDF receipt | `output/data/pdf_accessibility_report.json` | Actual structure, alternatives, formula associated files, links and table parity. |
| DOCX receipt | `output/data/docx_render_receipt.json` | OOXML integrity, source/adapter/media lineage and image/caption/cover contract. |
| Build receipt | `output/data/build_provenance.json` | Current source, template, generated, hydrated and rendered lineage. |
| Explicit-root export receipt | `output/data/project_render_receipt.json` | Exact selected root, template implementation, frozen hydration and freshly emitted derivatives. |

## DOCX projection

The template's current Office lane omits this project's configured cover.
`src/document_rendering.py` uses that same native Pandoc renderer, citations,
crossref and Office metadata. The project Lua filter runs after scientific
numbering, inserts the uncaptioned raster cover with its configured alternative
and page break, and resolves scientific references to absolute canonical
`output/figures/` paths. This prevents committed `figures/` examples from
shadowing current images through resource-directory search order. Conversion
stages a new file and replaces the edition only after success.

The schema-1.1 DOCX receipt compares all digest-bound canonical scientific
media with embedded bytes and binds the Python adapter and Lua filter. The two
validator-summary figures (`artifact_hash_manifest` and `validation_summary`)
are excluded from that aggregate digest to avoid a self-referential cycle;
their nonblankness, alternative, caption and count checks still apply. Figure
and caption totals come from the live registry, with one additional image for
the unnumbered cover.

The DOCX core description stores `active-assembly-hydrated-sha256:` plus the
hydrated-source digest at conversion time. A changed source cannot be certified
by merely writing a new receipt around old Word text. The adapter also rejects
an obsolete combined composition before conversion. A passing receipt does
not certify visual reading order or PDF/UA compliance.

## PDF and visual review

The abstract metadata declares PDF 2.0, tagging, language and the UA-2 target
before the document class; the preamble requests display of the document title.
The exporter hashes `preamble.md` as an auxiliary input. Reader, slide and DOCX
composition inventories contain the manuscript sections; PDF rendering consumes
the preamble separately.
These declarations must be checked against the actual file. Run the project
structure audit, `qpdf --check` and `verapdf --flavour ua2`; inspect VeraPDF's
reported compliance, not merely its process exit status. The PDF audit checks
actual native boolean values, heading structure, figure alternatives/captions,
MathML/TeX formula associated files, structured links and table-tag parity. The dedicated
`scripts/validate_pdf_ua.py` runs VeraPDF in JSON mode, requires one normal
UA-2 result with `compliant=true`, zero failures and positive passed counts,
and retains the PDF-digest-bound result in `output/reports/verapdf_ua2.json`.
A successful process exit alone cannot clear this gate.

Review every final PDF page for clipping, equation completeness, contrast,
figure legibility, caption attachment and reading order. For DOCX, render with
the bundled document runtime and inspect each final page; do not substitute a
machine integrity receipt for complete visual or accessibility certification.
Any earlier dated review applies only to the artifact bytes recorded there.

The two post-render receipts stay outside the self-referential scientific byte
manifest. `validate_analysis_outputs.py --existing-only` writes validation JSON
and refreshes build lineage without replacing figures, figure metadata or
hydrated sections. Its default regeneration mode belongs before export.
