"""Real Markdown files exercise authored-document closure and malformed inputs."""

from pathlib import Path

from src.docs_validation import validate_documentation


def _project(root: Path) -> Path:
    (root / "docs").mkdir()
    (root / "manuscript").mkdir()
    (root / "manuscript/references.bib").write_text("@article{known, title={Known}}")
    (root / "README.md").write_text("# Project\n\n[Docs](docs/README.md)\n")
    (root / "docs/README.md").write_text("# Index\n\n[Topic](topic.md)\n")
    (root / "docs/topic.md").write_text("# Topic\n\nKnown citation [@known].\n")
    return root


def test_actual_authored_docs_are_closed():
    assert validate_documentation(Path(__file__).resolve().parents[1]) == []


def test_valid_local_links_fragments_duplicates_and_encoded_paths(tmp_path):
    root = _project(tmp_path)
    (root / "docs/topic.md").write_text(
        "# Topic\n\n## Same *heading*\n## Same *heading*\n"
        "[One](#same-heading) [Two](#same-heading-1)\n"
        '[Encoded](<extra%20file.txt> "Title")\n'
        "[Outside](https://example.invalid/does-not-exist)\n"
        "```python\n[ignored](missing.md) @unknown\n# ignored\n```\n"
        "~~~text\n~~~still-fenced\n~~~\n"
    )
    (root / "docs/extra file.txt").write_text("Text")
    assert validate_documentation(root) == []


def test_unclosed_fence_unknown_key_missing_links_and_anchors(tmp_path):
    root = _project(tmp_path)
    (root / "docs/topic.md").write_text(
        "# Topic\n\n[@absent] [Lost](lost.md) [Bad](#absent)\n"
        "[Escape](../../outside.md)\n```python\nunfinished\n"
    )
    errors = validate_documentation(root)
    assert any("unclosed code fence" in row for row in errors)
    assert any("unknown bibliography key absent" in row for row in errors)
    assert any("missing local link: lost.md" in row for row in errors)
    assert any("missing heading anchor: #absent" in row for row in errors)
    assert any("local link escapes project" in row for row in errors)


def test_missing_index_entry_and_title_drift_are_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "docs/new.md").write_text("No title\n")
    (root / "docs/topic.md").write_text("# First\n# Second\n")
    errors = validate_documentation(root)
    assert sum("expected one level-one title" in row for row in errors) == 2
    assert any("unindexed topic new.md" in row for row in errors)
    (root / "README.md").unlink()
    assert any("missing documentation" in row for row in validate_documentation(root))


def test_missing_folder_index_is_always_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "docs/README.md").unlink()
    (root / "README.md").write_text("# Project\n")
    errors = validate_documentation(root)
    assert any("README.md: missing documentation" in row for row in errors)
    assert any("unindexed topic topic.md" in row for row in errors)
