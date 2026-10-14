"""Offline structural closure for authored Markdown documentation.

This checks the project's Markdown subset: fenced blocks, ATX headings,
inline local links and Pandoc citation keys. It does not fetch remote URLs or
certify prose, statistical claims, or the contents of generated artifacts.
"""

from collections import Counter
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

_LINK = re.compile(r"!?\[[^\]\n]*\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_CITATION = re.compile(r"(?<![\w])@([A-Za-z][\w:-]+)")


def _prose(path: Path) -> tuple[list[tuple[int, str]], list[str]]:
    lines: list[tuple[int, str]] = []
    errors: list[str] = []
    fence = ""
    start = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _FENCE.match(line)
        if fence:
            if (
                match
                and match[1][0] == fence[0]
                and len(match[1]) >= len(fence)
                and not match[2].strip()
            ):
                fence = ""
            continue
        if match:
            fence, start = match[1], number
        else:
            lines.append((number, line))
    if fence:
        errors.append(f"{path.name}:{start}: unclosed code fence")
    return lines, errors


def _anchors(lines: list[tuple[int, str]]) -> set[str]:
    result: set[str] = set()
    counts: Counter[str] = Counter()
    for _, line in lines:
        match = _HEADING.match(line)
        if not match:
            continue
        title = re.sub(r"<[^>]*>", "", match[2])
        title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        result.add(slug + suffix)
        counts[slug] += 1
    return result


def validate_documentation(project_root: Path) -> list[str]:
    """Check authored docs and README without reading generated outputs."""
    root = Path(project_root).resolve()
    paths = sorted({root / "docs/README.md", *(root / "docs").glob("*.md")}) + [
        root / "README.md"
    ]
    errors: list[str] = []
    bibliography = root / "manuscript/references.bib"
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography.read_text()))
    indexed: set[Path] = set()
    for path in paths:
        if not path.is_file():
            errors.append(f"{path.name}: missing documentation")
            continue
        lines, fence_errors = _prose(path)
        errors.extend(f"{path.relative_to(root)}: {error}" for error in fence_errors)
        headings = [_HEADING.match(line) for _, line in lines]
        if sum(bool(match and len(match[1]) == 1) for match in headings) != 1:
            errors.append(f"{path.relative_to(root)}: expected one level-one title")
        for number, line in lines:
            prefix = f"{path.relative_to(root)}:{number}"
            for key in _CITATION.findall(line):
                if key not in keys:
                    errors.append(f"{prefix}: unknown bibliography key {key}")
            for match in _LINK.finditer(line):
                reference = urlsplit(match[1])
                if reference.scheme or reference.netloc:
                    continue
                target = (
                    (path.parent / unquote(reference.path)).resolve()
                    if reference.path
                    else path
                )
                if not target.is_relative_to(root):
                    errors.append(f"{prefix}: local link escapes project: {match[1]}")
                elif not target.exists():
                    errors.append(f"{prefix}: missing local link: {match[1]}")
                else:
                    if path == root / "docs/README.md":
                        indexed.add(target)
                    if reference.fragment and target.suffix == ".md":
                        target_lines, _ = _prose(target)
                        if unquote(reference.fragment) not in _anchors(target_lines):
                            errors.append(
                                f"{prefix}: missing heading anchor: {match[1]}"
                            )
    for path in sorted((root / "docs").glob("*.md")):
        if (
            path.name not in {"AGENTS.md", "README.md"}
            and path.resolve() not in indexed
        ):
            errors.append(f"docs/README.md: unindexed topic {path.name}")
    return errors
