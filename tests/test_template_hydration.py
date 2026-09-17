"""Unit coverage for the template hydration contract in src/template_hydration.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.template_hydration as template_hydration
from src.template_hydration import (
    SOURCE_EXCLUDES,
    resolve_template_root,
    validate_hydrated_tree,
)

VARIABLES = {"PROJECT_TITLE": "Active Assembly Theory and Practice"}


def _make_source_tree(project_root: Path) -> None:
    source_dir = project_root / "manuscript"
    source_dir.mkdir(parents=True)
    (source_dir / "00_abstract.md").write_text(
        f"Title: {{{{PROJECT_TITLE}}}}", encoding="utf-8"
    )
    (source_dir / "01_body.md").write_text("Body without tokens", encoding="utf-8")
    # Excluded source files never participate in the hydration contract.
    (source_dir / "AGENTS.md").write_text("agent guidance", encoding="utf-8")


def _hydrate(project_root: Path, out_dir: Path) -> None:
    for path in (project_root / "manuscript").glob("*.md"):
        if path.name in SOURCE_EXCLUDES:
            continue
        text = path.read_text(encoding="utf-8")
        for key, value in VARIABLES.items():
            text = text.replace("{{" + key + "}}", value)
        (out_dir / path.name).write_text(text, encoding="utf-8")


def test_validate_hydrated_tree_accepts_exact_hydration(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _hydrate(tmp_path, out_dir)

    validate_hydrated_tree(out_dir, VARIABLES, project_root=tmp_path)


def test_validate_hydrated_tree_rejects_missing_file(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _hydrate(tmp_path, out_dir)
    (out_dir / "00_abstract.md").unlink()

    with pytest.raises(RuntimeError) as excinfo:
        validate_hydrated_tree(out_dir, VARIABLES, project_root=tmp_path)
    assert "missing=['00_abstract.md']" in str(excinfo.value)


def test_validate_hydrated_tree_rejects_extra_file(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _hydrate(tmp_path, out_dir)
    (out_dir / "99_extra.md").write_text("not in the source tree", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        validate_hydrated_tree(out_dir, VARIABLES, project_root=tmp_path)
    assert "extra=['99_extra.md']" in str(excinfo.value)


def test_validate_hydrated_tree_rejects_unresolved_token(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _hydrate(tmp_path, out_dir)
    (out_dir / "00_abstract.md").write_text(
        "Title: {{PROJECT_TITLE}}", encoding="utf-8"
    )

    with pytest.raises(RuntimeError) as excinfo:
        validate_hydrated_tree(out_dir, VARIABLES, project_root=tmp_path)
    assert "unresolved=['00_abstract.md']" in str(excinfo.value)


def test_validate_hydrated_tree_rejects_mismatched_substitution(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _hydrate(tmp_path, out_dir)
    (out_dir / "00_abstract.md").write_text("Title: Wrong Title", encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        validate_hydrated_tree(out_dir, VARIABLES, project_root=tmp_path)
    assert "mismatched=['00_abstract.md']" in str(excinfo.value)


def test_resolve_template_root_prefers_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A discoverable sibling template checkout exists, but the explicit
    # environment override must win.
    workspace = tmp_path / "workspace"
    project_root = workspace / "project"
    sibling = workspace / "template"
    for base in (project_root, sibling / "infrastructure/rendering"):
        base.mkdir(parents=True)
    (sibling / "infrastructure/rendering/manuscript_injection.py").touch()
    monkeypatch.setattr(template_hydration, "PROJECT_ROOT", project_root)

    explicit = workspace / "explicit"
    (explicit / "infrastructure/rendering").mkdir(parents=True)
    (explicit / "infrastructure/rendering/manuscript_injection.py").touch()
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(explicit))

    assert resolve_template_root() == explicit.resolve()


def test_resolve_template_root_discovers_sibling_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEMPLATE_REPO_ROOT", raising=False)
    workspace = tmp_path / "workspace"
    project_root = workspace / "project"
    project_root.mkdir(parents=True)
    sibling = workspace / "template"
    (sibling / "infrastructure/rendering").mkdir(parents=True)
    (sibling / "infrastructure/rendering/manuscript_injection.py").touch()
    monkeypatch.setattr(template_hydration, "PROJECT_ROOT", project_root)

    assert resolve_template_root() == sibling.resolve()


def test_resolve_template_root_rejects_incomplete_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="TEMPLATE_REPO_ROOT does not contain"):
        resolve_template_root()
