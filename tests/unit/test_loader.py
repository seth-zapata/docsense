"""Tests for the markdown document loader."""

import pytest

from docsense.ingestion.loader import (
    _extract_title,
    _strip_frontmatter,
    load_markdown_directory,
)


class TestStripFrontmatter:
    def test_with_frontmatter(self):
        text = "---\ntitle: My Doc\nlicense: apache-2.0\n---\n# Hello\nBody text."
        content, meta = _strip_frontmatter(text)
        assert content.strip() == "# Hello\nBody text."
        assert meta["title"] == "My Doc"
        assert meta["license"] == "apache-2.0"

    def test_without_frontmatter(self):
        text = "# Hello\nBody text."
        content, meta = _strip_frontmatter(text)
        assert content == text
        assert meta == {}

    def test_empty_frontmatter(self):
        text = "---\n\n---\n# Hello"
        content, meta = _strip_frontmatter(text)
        assert content.strip() == "# Hello"
        assert meta == {}

    def test_frontmatter_with_colons_in_value(self):
        text = "---\ntitle: My Doc: A Subtitle\n---\nBody."
        content, meta = _strip_frontmatter(text)
        assert meta["title"] == "My Doc: A Subtitle"


class TestExtractTitle:
    def test_h1_title(self):
        assert _extract_title("# My Document\nSome text") == "My Document"

    def test_no_title(self):
        assert _extract_title("Just some text\nwith no heading") is None

    def test_h2_not_matched(self):
        assert _extract_title("## Not a title\nSome text") is None

    def test_title_with_whitespace(self):
        assert _extract_title("  # Spaced Title  \nBody") == "Spaced Title"


class TestLoadMarkdownDirectory:
    def test_loads_files(self, tmp_path):
        (tmp_path / "doc1.md").write_text("---\ntitle: Doc One\n---\n# Doc One\nContent here.")
        (tmp_path / "doc2.md").write_text("# Doc Two\nMore content.")

        docs = load_markdown_directory(tmp_path)
        assert len(docs) == 2
        assert docs[0].metadata["title"] == "Doc One"
        assert docs[1].metadata["title"] == "Doc Two"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.md").write_text("---\ntitle: Empty\n---\n")
        (tmp_path / "real.md").write_text("# Real\nContent.")

        docs = load_markdown_directory(tmp_path)
        assert len(docs) == 1
        assert docs[0].source == "real.md"

    def test_loads_nested_directories(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested\nNested content.")

        docs = load_markdown_directory(tmp_path)
        assert len(docs) == 1
        assert docs[0].source == "subdir/nested.md"

    def test_doc_id_is_relative_path(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\nContent.")
        docs = load_markdown_directory(tmp_path)
        assert docs[0].doc_id == "doc.md"

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_markdown_directory(tmp_path / "nonexistent")

    def test_frontmatter_stripped_from_content(self, tmp_path):
        (tmp_path / "doc.md").write_text("---\ntitle: Test\n---\n# Test\nBody.")
        docs = load_markdown_directory(tmp_path)
        assert not docs[0].content.startswith("---")
        assert "Body." in docs[0].content
