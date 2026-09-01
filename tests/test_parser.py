"""
Unit tests for RepoViva Milestone 2: Intelligent Code & Documentation Chunking.
Uses Python standard library `unittest`.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import SourceFile
from src.parser import CodeChunker, Document, parse_and_chunk_files


class TestParserChunker(unittest.TestCase):

    def test_document_structure(self):
        """Test standard Document class compatibility."""
        doc = Document(
            page_content="def add(a, b):\n    return a + b",
            metadata={"language": "Python", "start_line": 1, "end_line": 2}
        )
        self.assertEqual(doc.page_content, "def add(a, b):\n    return a + b")
        self.assertEqual(doc.metadata["language"], "Python")
        self.assertEqual(doc.metadata["start_line"], 1)
        self.assertEqual(doc.metadata["end_line"], 2)
        
        doc_dict = doc.to_dict()
        self.assertIn("page_content", doc_dict)
        self.assertIn("metadata", doc_dict)

    def test_chunker_validation(self):
        """Test parameter validation for CodeChunker initialization."""
        with self.assertRaises(ValueError):
            CodeChunker(chunk_size=0)
            
        with self.assertRaises(ValueError):
            CodeChunker(chunk_size=500, chunk_overlap=500)

    def test_python_file_chunking(self):
        """Test splitting a Python source file into multiple line-aware chunks."""
        py_content = (
            "class MathUtils:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n\n"
            "class StringUtils:\n"
            "    def concatenate(self, s1, s2):\n"
            "        return s1 + s2\n"
        )
        source_file = SourceFile(
            relative_path="src/utils.py",
            file_name="utils.py",
            extension=".py",
            language="Python",
            line_count=10,
            content=py_content
        )

        chunker = CodeChunker(chunk_size=120, chunk_overlap=30)
        docs = chunker.chunk_source_file(source_file)

        self.assertGreater(len(docs), 1, "File should be split into multiple chunks")
        
        for doc in docs:
            self.assertEqual(doc.metadata["relative_path"], "src/utils.py")
            self.assertEqual(doc.metadata["language"], "Python")
            self.assertIn("start_line", doc.metadata)
            self.assertIn("end_line", doc.metadata)
            self.assertGreaterEqual(doc.metadata["end_line"], doc.metadata["start_line"])

    def test_markdown_chunking(self):
        """Test splitting Markdown documentation."""
        md_content = (
            "# Main Title\n\n"
            "This is the intro paragraph.\n\n"
            "## Section 1\n\n"
            "Content for section 1 goes here.\n\n"
            "## Section 2\n\n"
            "Content for section 2 goes here.\n"
        )
        source_file = SourceFile(
            relative_path="docs/README.md",
            file_name="README.md",
            extension=".md",
            language="Markdown",
            line_count=11,
            content=md_content
        )

        chunker = CodeChunker(chunk_size=80, chunk_overlap=15)
        docs = chunker.chunk_source_file(source_file)

        self.assertGreaterEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["language"], "Markdown")
        self.assertEqual(docs[0].metadata["chunk_index"], 0)

    def test_empty_file_chunking(self):
        """Test that empty or whitespace-only files produce zero chunks."""
        empty_sf = SourceFile(
            relative_path="empty.py",
            file_name="empty.py",
            extension=".py",
            language="Python",
            line_count=0,
            content=""
        )
        chunker = CodeChunker()
        docs = chunker.chunk_source_file(empty_sf)
        self.assertEqual(len(docs), 0)

    def test_parse_and_chunk_files_batch(self):
        """Test batch processing of multiple files."""
        sf1 = SourceFile(
            relative_path="f1.py", file_name="f1.py", extension=".py",
            language="Python", line_count=2, content="def f1():\n    pass\n"
        )
        sf2 = SourceFile(
            relative_path="f2.js", file_name="f2.js", extension=".js",
            language="JavaScript", line_count=2, content="function f2() {\n  return;\n}\n"
        )
        docs = parse_and_chunk_files([sf1, sf2])
        self.assertEqual(len(docs), 2)
        languages = {d.metadata["language"] for d in docs}
        self.assertEqual(languages, {"Python", "JavaScript"})


if __name__ == "__main__":
    unittest.main()
