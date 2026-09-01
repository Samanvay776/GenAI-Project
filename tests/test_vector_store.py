"""
Unit and integration tests for RepoViva Milestone 3: Vector Store & Embeddings.
Uses Python standard library `unittest` and temporary directories.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import Document
from src.vector_store import EmbeddingManager, RepoVectorStore, SimpleFallbackEmbeddings


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        """Set up temporary directory for ChromaDB / vector storage."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.embedding_manager = EmbeddingManager()
        self.vstore = RepoVectorStore(
            persist_directory=self.temp_dir.name,
            embedding_manager=self.embedding_manager
        )
        self.test_collection = "test_repoviva_collection"

    def tearDown(self):
        """Clean up temporary directory and collection."""
        self.vstore.delete_collection(self.test_collection)
        self.temp_dir.cleanup()

    def test_embedding_manager(self):
        """Test embedding generation."""
        embeddings = self.embedding_manager.embed_documents(["def hello(): pass", "import os"])
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)

        query_emb = self.embedding_manager.embed_query("search for function")
        self.assertEqual(len(query_emb), len(embeddings[0]))

    def test_add_and_count_documents(self):
        """Test adding documents to vector store and counting."""
        docs = [
            Document(
                page_content="def authenticate_user(username, password):\n    return True",
                metadata={"relative_path": "src/auth.py", "language": "Python", "start_line": 1, "end_line": 2}
            ),
            Document(
                page_content="class DatabaseConnection:\n    def connect(self):\n        pass",
                metadata={"relative_path": "src/db.py", "language": "Python", "start_line": 1, "end_line": 3}
            )
        ]
        ids = self.vstore.add_documents(docs, collection_name=self.test_collection)
        self.assertEqual(len(ids), 2)

        count = self.vstore.get_collection_count(collection_name=self.test_collection)
        self.assertEqual(count, 2)

    def test_similarity_search(self):
        """Test semantic similarity retrieval from vector store."""
        docs = [
            Document(
                page_content="def login_user(credentials):\n    # Verify user token and establish session\n    return token",
                metadata={"relative_path": "src/auth.py", "language": "Python", "start_line": 10, "end_line": 15}
            ),
            Document(
                page_content="def render_dashboard_chart(data):\n    # Draw canvas line chart for analytics\n    pass",
                metadata={"relative_path": "src/ui.py", "language": "Python", "start_line": 5, "end_line": 10}
            )
        ]
        self.vstore.add_documents(docs, collection_name=self.test_collection)

        # Search query related to user login authentication
        results = self.vstore.similarity_search(query="login credentials user token session", k=1, collection_name=self.test_collection)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["relative_path"], "src/auth.py")
        self.assertIn("login_user", results[0].page_content)

    def test_similarity_search_with_score(self):
        """Test semantic search returning similarity distance scores."""
        docs = [
            Document(
                page_content="SELECT * FROM users WHERE active = true;",
                metadata={"relative_path": "queries.sql", "language": "SQL", "start_line": 1, "end_line": 1}
            )
        ]
        self.vstore.add_documents(docs, collection_name=self.test_collection)

        results_with_scores = self.vstore.similarity_search_with_score(
            query="database SQL query active users",
            k=1,
            collection_name=self.test_collection
        )
        self.assertEqual(len(results_with_scores), 1)
        doc, score = results_with_scores[0]
        self.assertEqual(doc.metadata["language"], "SQL")
        self.assertIsInstance(score, float)

    def test_simple_fallback_embeddings(self):
        """Test fallback embedding generator directly."""
        embedder = SimpleFallbackEmbeddings(dim=32)
        vecs = embedder.embed_documents(["hello world", "foo bar"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 32)


if __name__ == "__main__":
    unittest.main()
