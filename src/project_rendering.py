"""Render a verified hydrated manuscript with the real template's lower APIs."""

from __future__ import annotations

from dataclasses import replace
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from .template_hydration import (
    SOURCE_EXCLUDES,
    resolve_template_root,
    validate_hydrated_tree,
)


def _hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def render_project(
    project_root: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "html", "docx"),
    project_name: str | None = None,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Render without leaking the template's import search path or scripts package."""
    original_path = sys.path[:]
    original_bytecode = sys.dont_write_bytecode
    original_scripts = {
        name: module
        for name, module in sys.modules.items()
        if name == "scripts" or name.startswith("scripts.")
    }
    try:
        return _render_project(
            project_root,
            formats=formats,
            template_root=template_root,
            project_name=project_name,
        )
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts."):
                if name not in original_scripts:
                    del sys.modules[name]
        sys.modules.update(original_scripts)


def _render_project(
    project_root: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "html", "docx"),
    project_name: str | None = None,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Consume existing hydration, render explicit paths, and retain their identity.

    No project discovery/linking CLI or hydration producer is invoked here.
    A returned path must be expected, nonempty, and freshly written even when
    a template helper catches its own exception and leaves an older output.
    """
    root = project_root.resolve(strict=True)
    name = project_name or root.name
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("RENDER_PROJECT_NAME_INVALID")
    if (
        not formats
        or len(set(formats)) != len(formats)
        or set(formats) - {"pdf", "html", "docx", "slides"}
    ):
        raise ValueError("RENDER_FORMAT_INVALID: request pdf, html and/or docx")
    if "docx" in formats and not {"pdf", "html"}.intersection(formats):
        raise ValueError("RENDER_DOCX_REQUIRES_COMPOSITION: request pdf or html too")
    for relative in (
        "manuscript",
        "output",
        "output/manuscript",
        "output/pdf",
        "output/web",
        "output/docx",
        "output/slides",
    ):
        if not (root / relative).resolve().is_relative_to(root):
            raise RuntimeError(f"RENDER_PATH_ESCAPE: {relative}")
    template = (template_root or resolve_template_root(root)).resolve(strict=True)
    implementation = template / "infrastructure/rendering/core.py"
    if not implementation.is_file():
        raise RuntimeError("RENDER_TEMPLATE_INVALID: missing lower rendering API")
    hydrated = root / "output/manuscript"
    from .artifact_contract import artifact_domain_view

    variables = artifact_domain_view(
        json.loads((root / "output/data/manuscript_variables.json").read_text())
    )
    if (
        not isinstance(variables, dict)
        or not variables
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in variables.items()
        )
    ):
        raise ValueError("RENDER_VARIABLES_INVALID: expected a nonempty string mapping")
    # The artifact envelope adds non-string metadata; the domain mapping is
    # consumed here without relaxing checks on the actual token values.
    validate_hydrated_tree(hydrated, variables, project_root=root)
    sources = sorted(
        path for path in hydrated.glob("*.md") if path.name not in SOURCE_EXCLUDES
    )
    source_paths = sorted(
        path for path in (root / "manuscript").glob("*") if path.is_file()
    )
    hydrated_paths = sorted(path for path in hydrated.glob("*") if path.is_file())
    before_sources = _hashes(source_paths, root)
    before_hydrated = _hashes(hydrated_paths, root)
    for source in source_paths:
        if source.suffix in {".yaml", ".bib"} and (
            not (hydrated / source.name).is_file()
            or source.read_bytes() != (hydrated / source.name).read_bytes()
        ):
            raise RuntimeError(
                f"RENDER_HYDRATION_STALE: auxiliary source {source.name}"
            )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(template))
    rendering = importlib.import_module("infrastructure.rendering")
    rendering_config = importlib.import_module("infrastructure.rendering.config")
    combined_exports = importlib.import_module(
        "infrastructure.rendering._combined_exports"
    )
    diagnostic = importlib.import_module("infrastructure.core.logging.diagnostic")
    rendering_core = importlib.import_module("infrastructure.rendering.core")
    if (
        rendering_core.__file__ is None
        or Path(rendering_core.__file__).resolve() != implementation.resolve()
    ):
        raise RuntimeError(
            "RENDER_TEMPLATE_IDENTITY: another template is already imported"
        )
    config_path = hydrated / "config.yaml"
    project_config = (
        yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    )
    config = replace(
        rendering_config.RenderingConfig.from_project_config(project_config or {}),
        manuscript_dir=str(hydrated),
        figures_dir=str(root / "output/figures"),
        output_dir=str(root / "output"),
        pdf_dir=str(root / "output/pdf"),
        web_dir=str(root / "output/web"),
        slides_dir=str(root / "output/slides"),
        docx_dir=str(root / "output/docx"),
        epub_dir=str(root / "output/epub"),
        template_dir=str(template / "infrastructure/rendering/templates"),
        enable_pdf="pdf" in formats,
        enable_html="html" in formats,
        enable_docx="docx" in formats,
        enable_slides="slides" in formats,
        enable_epub=False,
    )
    manager = rendering.RenderManager(
        config, manuscript_dir=hydrated, figures_dir=root / "output/figures"
    )
    expected = {
        "pdf": root / "output/pdf" / f"{name}_combined.pdf",
        "html": root / "output/web/index.html",
        "docx": root / "output/docx" / f"{name}_combined.docx",
    }
    selected = {kind: path for kind, path in expected.items() if kind in formats}
    if "slides" in formats:
        for source in sources:
            selected[f"slides:{source.stem}"] = (
                root / "output/slides" / f"{source.stem}_slides.pdf"
            )
    previous = {
        kind: path.stat().st_mtime_ns if path.exists() else None
        for kind, path in selected.items()
    }
    for path in selected.values():
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f"RENDER_PATH_ESCAPE: {path}")
    started = time.time_ns()
    rendered: list[Path] = []
    if "pdf" in formats:
        rendered.append(manager.render_combined_pdf(sources, hydrated, name))
    if "html" in formats:
        rendered.append(manager.render_combined_web(sources, hydrated, name))
    if "docx" in formats:
        combined = root / "output/pdf/_combined_manuscript.md"
        if "pdf" not in formats:
            combined = root / "output/web/_combined_manuscript.md"
        combined_exports.render_combined_docx(
            manager,
            hydrated,
            name,
            diagnostic.DiagnosticReporter(name),
            combined_md=combined,
        )
        rendered.append(expected["docx"])
    if "slides" in formats:
        for source in sources:
            rendered.append(manager.render_slides(source))
    for kind, path in selected.items():
        if path not in rendered or not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"RENDER_OUTPUT_MISSING: {kind} {path}")
        if (
            path.stat().st_mtime_ns < started
            or path.stat().st_mtime_ns == previous[kind]
        ):
            raise RuntimeError(f"RENDER_OUTPUT_STALE: {kind} {path}")
        if (
            path.resolve().parent != path.parent.resolve()
            or not path.resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"RENDER_OUTPUT_ESCAPE: {path}")
    if (
        _hashes(source_paths, root) != before_sources
        or _hashes(hydrated_paths, root) != before_hydrated
    ):
        raise RuntimeError(
            "RENDER_INPUT_CHANGED: source/hydrated bytes changed during rendering"
        )
    revision = subprocess.run(
        ("git", "-C", str(template), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    receipt = {
        "project_root": str(root),
        "template_root": str(template),
        "template_revision": revision,
        "source_hashes": before_sources,
        "hydrated_hashes": before_hydrated,
        "rendered_hashes": _hashes(rendered, root),
        "formats": list(formats),
        "skip_manuscript_hydration": True,
    }
    from .build_provenance import _template_implementation_hashes

    receipt["template_implementation_hashes"] = _template_implementation_hashes(
        template
    )
    receipt["variables_sha256"] = hashlib.sha256(
        (root / "output/data/manuscript_variables.json").read_bytes()
    ).hexdigest()
    from .artifact_contract import write_json_atomic

    write_json_atomic(root / "output/reports/project_render.json", receipt)
    return receipt
