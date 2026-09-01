"""
Unit and integration tests for RepoViva Milestone 4: RAG Pipeline.
Uses Python standard library `unittest` and temporary directories.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import Document
from src.vector_store import RepoVectorStore
from src.rag_pipeline import (
    RepoRAGPipeline,
    RAGResult,
    format_retrieved_context,
    extract_citations
)


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        """Set up temporary vector store for RAG pipeline testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vstore = RepoVectorStore(persist_directory=self.temp_dir.name)
        self.test_collection = "test_rag_collection"

        # Seed vector store with sample code chunks
        self.sample_docs = [
            Document(
                page_content="def initialize_database():\n    db = connect('db.sqlite')\n    return db",
                metadata={"relative_path": "src/db.py", "file_name": "db.py", "language": "Python", "start_line": 1, "end_line": 3}
            ),
            Document(
                page_content="def authenticate_jwt(token):\n    # Validate JWT signature\n    return decode(token)",
                metadata={"relative_path": "src/auth.py", "file_name": "auth.py", "language": "Python", "start_line": 15, "end_line": 18}
            )
        ]
        self.vstore.add_documents(self.sample_docs, collection_name=self.test_collection)

    def tearDown(self):
        """Clean up temporary directory."""
        self.vstore.delete_collection(self.test_collection)
        self.temp_dir.cleanup()

    def test_format_retrieved_context(self):
        """Test formatting of retrieved document chunks for prompt grounding."""
        formatted = format_retrieved_context(self.sample_docs)
        self.assertIn("--- Snippet #1 | File: src/db.py (Lines 1-3) | Language: Python ---", formatted)
        self.assertIn("initialize_database", formatted)
        self.assertIn("--- Snippet #2 | File: src/auth.py (Lines 15-18) | Language: Python ---", formatted)
        self.assertIn("authenticate_jwt", formatted)

    def test_extract_citations(self):
        """Test citation metadata extraction."""
        citations = extract_citations(self.sample_docs)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["relative_path"], "src/db.py")
        self.assertEqual(citations[0]["start_line"], 1)
        self.assertEqual(citations[0]["end_line"], 3)
        self.assertEqual(citations[1]["relative_path"], "src/auth.py")

    def test_empty_query(self):
        """Test RAG execution on empty query."""
        pipeline = RepoRAGPipeline(vector_store=self.vstore)
        res = pipeline.answer_question("", collection_name=self.test_collection)
        self.assertEqual(res.answer, "Query cannot be empty.")
        self.assertEqual(len(res.citations), 0)

    def test_rag_pipeline_execution(self):
        """Test full RAG question answering pipeline against populated vector store."""
        pipeline = RepoRAGPipeline(vector_store=self.vstore)
        res: RAGResult = pipeline.answer_question(
            query="how does JWT token authentication work",
            collection_name=self.test_collection,
            k=1
        )

        self.assertIsInstance(res, RAGResult)
        self.assertEqual(res.query, "how does JWT token authentication work")
        self.assertGreaterEqual(len(res.citations), 1)
        self.assertEqual(res.citations[0]["relative_path"], "src/auth.py")
        self.assertEqual(res.citations[0]["start_line"], 15)
        self.assertEqual(res.citations[0]["end_line"], 18)
        self.assertIsNotNone(res.answer)

    def test_rag_result_to_dict(self):
        """Test serialization of RAGResult."""
        res = RAGResult(
            query="test query",
            answer="test answer",
            retrieved_documents=self.sample_docs,
            citations=[{"relative_path": "src/db.py", "start_line": 1, "end_line": 3}]
        )
        res_dict = res.to_dict()
        self.assertEqual(res_dict["query"], "test query")
        self.assertEqual(res_dict["answer"], "test answer")
        self.assertEqual(len(res_dict["retrieved_documents"]), 2)
        self.assertEqual(len(res_dict["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
