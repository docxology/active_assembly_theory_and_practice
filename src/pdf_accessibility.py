"""Structural accessibility audit for the rendered manuscript PDF.

The template renderer owns PDF authoring and tagging.  This module is a
project-side, post-render audit: it inspects the actual PDF structure tree and
compares it with the project's generated figure registry and source manuscript
surface.  A ``Tagged: yes`` metadata flag or a VeraPDF pass is necessary but
not sufficient for this audit; the semantic roles, alternatives, associated
math files, link structure, and source/rendered table contract are checked
explicitly here.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from pypdf import PdfReader
from pypdf.generic import BooleanObject

from .artifact_contract import write_json_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = (
    PROJECT_ROOT / "output/pdf/active_assembly_theory_and_practice_combined.pdf"
)
DEFAULT_REPORT = PROJECT_ROOT / "output/data/pdf_accessibility_report.json"


def rendered_pdf_text_errors(pdf_path: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-bbox", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    namespace = {"x": "http://www.w3.org/1999/xhtml"}
    pages = ET.fromstring(result.stdout).findall(".//x:page", namespace)
    if not pages:
        raise ValueError(f"PDF text extraction returned no pages: {pdf_path}")
    errors = []
    for number, page in enumerate(pages, 1):
        width = float(page.attrib["width"])
        height = float(page.attrib["height"])
        words = page.findall(".//x:word", namespace)
        for word in words:
            if (
                float(word.attrib["xMin"]) < -1
                or float(word.attrib["yMin"]) < -1
                or float(word.attrib["xMax"]) > width + 1
                or float(word.attrib["yMax"]) > height + 1
            ):
                errors.append(f"page {number}: word outside paper: {word.text}")
        text = " ".join(word.text or "" for word in words)
        for match in re.finditer(r"\b(?:sec|def|prop|thm|lem):[\w-]+", text):
            errors.append(f"page {number}: unresolved reference: {match.group()}")
    return errors


def _deref(value: Any) -> Any:
    """Resolve a pypdf indirect object without assuming every value is one."""

    try:
        return value.get_object()
    except AttributeError:
        return value


def _children(node: Any) -> list[Any]:
    """Return structure-tree children as a normalized list."""

    if not hasattr(node, "get"):
        return []
    value = _deref(node.get("/K"))
    if value is None:
        return []
    if isinstance(value, list):
        return [_deref(child) for child in value]
    return [_deref(value)]


def _walk_structure(root: Any) -> Iterable[Any]:
    """Yield dictionary-like structure nodes once in reading-tree order."""

    stack = [_deref(root)]
    visited: set[int] = set()
    while stack:
        node = _deref(stack.pop())
        if not hasattr(node, "get") or id(node) in visited:
            continue
        visited.add(id(node))
        yield node
        children = _children(node)
        stack.extend(reversed(children))


def _role_name(value: Any) -> str:
    """Normalize a PDF name object to its slash-prefixed string."""

    return str(value) if value is not None else ""


def _resolve_role(role: str, role_map: Any) -> str:
    """Resolve a custom structure role through the PDF ``RoleMap``."""

    mapping = _deref(role_map)
    seen: set[str] = set()
    current = role
    while current and current not in seen and hasattr(mapping, "get"):
        seen.add(current)
        replacement = mapping.get(current)
        if replacement is None:
            break
        current = _role_name(replacement)
    return current


def _number_tree_keys(node: Any) -> set[int]:
    """Collect keys from a PDF number tree (used by the ParentTree)."""

    node = _deref(node)
    if not hasattr(node, "get"):
        return set()
    keys: set[int] = set()
    numbers = _deref(node.get("/Nums"))
    if isinstance(numbers, list):
        for index in range(0, len(numbers) - 1, 2):
            try:
                keys.add(int(numbers[index]))
            except (TypeError, ValueError):
                continue
    children = _deref(node.get("/Kids"))
    if isinstance(children, list):
        for child in children:
            keys.update(_number_tree_keys(child))
    elif children is not None:
        keys.update(_number_tree_keys(children))
    return keys


def _check(name: str, passed: bool, detail: str, **metrics: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "passed": bool(passed), "detail": detail}
    if metrics:
        row["metrics"] = metrics
    return row


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_source_tables(project_root: Path) -> int:
    """Count authored table environments in the template-produced TeX source."""

    tex_path = project_root / "output/pdf/_combined_manuscript.tex"
    if not tex_path.is_file():
        return 0
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"\\begin\{(?:table|tabular|longtable)\}", text))


def _annotation_links(reader: PdfReader) -> tuple[int, list[int], int]:
    """Return total links, unstructured link StructParents, and action count."""

    total = 0
    missing_struct_parent: list[int] = []
    with_actions = 0
    for page_number, page in enumerate(reader.pages, start=1):
        annotations = _deref(page.get("/Annots"))
        if not isinstance(annotations, list):
            continue
        for annotation_ref in annotations:
            annotation = _deref(annotation_ref)
            if _role_name(annotation.get("/Subtype")) != "/Link":
                continue
            total += 1
            if annotation.get("/StructParent") is None:
                missing_struct_parent.append(page_number)
            if annotation.get("/A") is not None or annotation.get("/Dest") is not None:
                with_actions += 1
    return total, missing_struct_parent, with_actions


def audit_pdf(
    pdf_path: Path = DEFAULT_PDF,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Audit one rendered PDF and return a JSON-serializable report.

    The audit intentionally fails when the configured source-bound registries
    are absent.  That prevents a structurally valid but stale or untraceable
    PDF from being reported as complete.
    """

    pdf_path = Path(pdf_path)
    project_root = Path(project_root)
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"pdf": str(pdf_path)}

    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        checks.append(_check("pdf_present", False, "rendered PDF is missing or empty"))
        return {
            "schema_version": "1.0",
            "pdf_path": str(pdf_path),
            "checks": checks,
            "summary": {"passed": 0, "failed": 1, "all_passed": False},
            "metrics": metrics,
        }
    checks.append(
        _check(
            "pdf_present",
            True,
            "rendered PDF exists and is non-empty",
            bytes=pdf_path.stat().st_size,
        )
    )

    try:
        reader = PdfReader(str(pdf_path))
        catalog = _deref(reader.trailer["/Root"])
    except (
        Exception
    ) as exc:  # pragma: no cover - malformed-PDF path is integration-only
        checks.append(_check("pdf_opened", False, f"PdfReader failed: {exc}"))
        return {
            "schema_version": "1.0",
            "pdf_path": str(pdf_path),
            "checks": checks,
            "summary": {"passed": 1, "failed": 1, "all_passed": False},
            "metrics": metrics,
        }

    checks.append(
        _check(
            "pdf_opened", True, "pypdf opened the rendered PDF", pages=len(reader.pages)
        )
    )
    struct_root = _deref(catalog.get("/StructTreeRoot"))
    parent_tree = (
        _deref(struct_root.get("/ParentTree")) if hasattr(struct_root, "get") else None
    )
    mark_info = _deref(catalog.get("/MarkInfo"))
    lang = str(catalog.get("/Lang", "")).strip()
    marked_value = mark_info.get("/Marked") if hasattr(mark_info, "get") else None
    # PDF BooleanObject(False) is truthy as a Python object; read its value.
    marked = marked_value is True or (
        isinstance(marked_value, BooleanObject) and marked_value.value is True
    )

    configured_language = "en"
    config_path = project_root / "manuscript/config.yaml"
    if config_path.is_file():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            configured_language = str(config.get("metadata", {}).get("language", "en"))
        except Exception:
            configured_language = "en"
    checks.extend(
        [
            _check(
                "document_language",
                lang.lower() == configured_language.lower(),
                f"catalog Lang={lang!r}; configured={configured_language!r}",
            ),
            _check("marked_pdf", marked, f"MarkInfo.Marked={marked}"),
            _check(
                "structure_tree",
                hasattr(struct_root, "get"),
                "StructTreeRoot is present",
            ),
            _check(
                "parent_tree",
                hasattr(parent_tree, "get"),
                "StructTreeRoot.ParentTree is present",
            ),
        ]
    )

    nodes = list(_walk_structure(struct_root)) if hasattr(struct_root, "get") else []
    raw_roles = Counter(
        _role_name(node.get("/S")) for node in nodes if node.get("/S") is not None
    )
    role_map = (
        _deref(struct_root.get("/RoleMap")) if hasattr(struct_root, "get") else None
    )
    resolved_roles = Counter(
        _resolve_role(_role_name(node.get("/S")), role_map)
        for node in nodes
        if node.get("/S") is not None
    )
    required_role_mappings = {
        "/section": "/H1",
        "/subsection": "/H2",
        "/subsubsection": "/H3",
    }
    role_mapping_errors = [
        f"{source}->{target}"
        for source, target in required_role_mappings.items()
        if _resolve_role(source, role_map) != target
    ]
    checks.append(
        _check(
            "role_map",
            hasattr(role_map, "get") and not role_mapping_errors,
            "custom heading roles resolve to standard H1/H2/H3 roles"
            if not role_mapping_errors
            else f"missing mappings: {', '.join(role_mapping_errors)}",
            required=required_role_mappings,
        )
    )

    heading_levels = [
        int(resolved_role[2:])
        for node in nodes
        for resolved_role in [_resolve_role(_role_name(node.get("/S")), role_map)]
        if re.fullmatch(r"/H[1-6]", resolved_role)
    ]
    heading_errors: list[str] = []
    if heading_levels and heading_levels[0] != 1:
        heading_errors.append(f"first heading is H{heading_levels[0]}")
    for previous, current in zip(heading_levels, heading_levels[1:]):
        if current > previous + 1:
            heading_errors.append(f"heading jump H{previous}->H{current}")
    checks.append(
        _check(
            "heading_hierarchy",
            bool(heading_levels) and not heading_errors,
            "heading levels are present and do not skip a level"
            if not heading_errors
            else "; ".join(heading_errors),
            sequence=[f"H{level}" for level in heading_levels],
        )
    )

    registry_path = project_root / "output/data/figure_registry.json"
    cover_alt = ""
    registry_alts: set[str] = set()
    registry_count = 0
    if registry_path.is_file():
        registry = _load_json(registry_path)
        rows = registry.get("figures", [])
        registry_count = len(rows)
        registry_alts = {
            str(row.get("alt_text", "")).strip() for row in rows if row.get("alt_text")
        }
    config_path = project_root / "manuscript/config.yaml"
    if config_path.is_file():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            cover_alt = str(
                config.get("paper", {}).get("cover", {}).get("alt", "")
            ).strip()
        except Exception:
            cover_alt = ""

    figure_nodes = [node for node in nodes if _role_name(node.get("/S")) == "/Figure"]
    figure_alts = [str(node.get("/Alt", "")).strip() for node in figure_nodes]
    observed_alts = {alt for alt in figure_alts if alt}
    missing_registry_alts = sorted(registry_alts - observed_alts)
    missing_cover_alt = bool(cover_alt) and cover_alt not in observed_alts
    caption_count = sum(1 for node in nodes if _role_name(node.get("/S")) == "/Caption")
    checks.extend(
        [
            _check(
                "figure_alternatives",
                bool(registry_alts) and not missing_registry_alts,
                "every generated figure-registry alternative appears in the PDF tag tree"
                if not missing_registry_alts
                else f"missing alternatives: {missing_registry_alts}",
                registry_count=registry_count,
                figure_tag_count=len(figure_nodes),
                figure_tags_with_alt=sum(bool(alt) for alt in figure_alts),
            ),
            _check(
                "cover_alternative",
                bool(cover_alt) and not missing_cover_alt,
                "configured cover alternative appears in a Figure tag"
                if not missing_cover_alt and cover_alt
                else "configured cover alternative is missing from the PDF tag tree",
            ),
            _check(
                "figure_captions",
                caption_count >= registry_count,
                f"caption tags={caption_count}; generated figure rows={registry_count}",
            ),
        ]
    )

    formula_nodes = [node for node in nodes if _role_name(node.get("/S")) == "/Formula"]
    formula_failures: list[str] = []
    formula_relationships: Counter[str] = Counter()
    for index, node in enumerate(formula_nodes, start=1):
        associations = _deref(node.get("/AF"))
        associations = associations if isinstance(associations, list) else []
        relationships: set[str] = set()
        embedded = 0
        for association_ref in associations:
            association = _deref(association_ref)
            relation = _role_name(association.get("/AFRelationship"))
            relationships.add(relation)
            formula_relationships[relation] += 1
            if hasattr(_deref(association.get("/EF")), "get"):
                embedded += 1
        if (
            "/Supplement" not in relationships
            or "/Source" not in relationships
            or embedded < 2
        ):
            formula_failures.append(
                f"Formula {index}: AF relationships={sorted(relationships)} embedded={embedded}"
            )
    checks.append(
        _check(
            "formula_associated_files",
            not formula_failures,
            "every Formula tag has embedded MathML supplement and TeX source associated files"
            if not formula_failures
            else "; ".join(formula_failures[:5]),
            formula_count=len(formula_nodes),
            relationships=dict(sorted(formula_relationships.items())),
        )
    )

    link_count, missing_link_struct_parent, action_count = _annotation_links(reader)
    parent_keys = (
        _number_tree_keys(parent_tree) if hasattr(parent_tree, "get") else set()
    )
    struct_parent_values: set[int] = set()
    for page in reader.pages:
        annotations = _deref(page.get("/Annots"))
        if not isinstance(annotations, list):
            continue
        for ref in annotations:
            annotation = _deref(ref)
            if (
                _role_name(annotation.get("/Subtype")) == "/Link"
                and annotation.get("/StructParent") is not None
            ):
                try:
                    struct_parent_values.add(int(annotation.get("/StructParent")))
                except (TypeError, ValueError):
                    pass
    missing_parent_keys = sorted(struct_parent_values - parent_keys)
    checks.append(
        _check(
            "structured_links",
            link_count > 0
            and not missing_link_struct_parent
            and not missing_parent_keys,
            f"{link_count} link annotations have StructParent entries resolved through ParentTree"
            if not missing_link_struct_parent and not missing_parent_keys
            else f"unstructured links={len(missing_link_struct_parent)}; missing ParentTree keys={missing_parent_keys[:10]}",
            link_count=link_count,
            action_count=action_count,
            parent_tree_key_count=len(parent_keys),
        )
    )

    source_table_count = _expected_source_tables(project_root)
    pdf_table_count = sum(1 for node in nodes if _role_name(node.get("/S")) == "/Table")
    checks.append(
        _check(
            "table_structure",
            pdf_table_count == source_table_count,
            f"PDF Table tags={pdf_table_count}; source table environments={source_table_count}",
        )
    )

    # Expose both raw and resolved roles so a reviewer can inspect the exact
    # semantic surface without reparsing the PDF.
    role_map_object: Any = role_map
    role_map_items: Any = (
        role_map_object.items() if hasattr(role_map_object, "items") else []
    )
    metrics.update(
        {
            "page_count": len(reader.pages),
            "raw_structure_roles": dict(raw_roles),
            "resolved_structure_roles": dict(resolved_roles),
            "figure_tag_count": len(figure_nodes),
            "caption_tag_count": caption_count,
            "formula_tag_count": len(formula_nodes),
            "link_annotation_count": link_count,
            "source_table_count": source_table_count,
            "pdf_table_tag_count": pdf_table_count,
            "role_map": {str(key): str(value) for key, value in role_map_items},
        }
    )
    passed = sum(bool(row["passed"]) for row in checks)
    failed = len(checks) - passed
    return {
        "schema_version": "1.0",
        "pdf_path": str(pdf_path),
        "checks": checks,
        "summary": {"passed": passed, "failed": failed, "all_passed": failed == 0},
        "metrics": metrics,
    }


def write_pdf_accessibility_report(
    pdf_path: Path = DEFAULT_PDF,
    report_path: Path = DEFAULT_REPORT,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Audit the rendered PDF and write the machine-readable report."""

    report = audit_pdf(pdf_path, project_root=project_root)
    report_path = Path(report_path)
    write_json_artifact(
        report_path,
        report,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return report_path


__all__ = [
    "DEFAULT_PDF",
    "DEFAULT_REPORT",
    "audit_pdf",
    "write_pdf_accessibility_report",
]
