"""
Unit and integration tests for RepoViva Milestone 5: Technical Interview Question Generator.
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
from src.rag_pipeline import RepoRAGPipeline
from src.interview_generator import InterviewQuestion, InterviewQuestionGenerator


class TestInterviewGenerator(unittest.TestCase):

    def setUp(self):
        """Set up temporary vector store and generator."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vstore = RepoVectorStore(persist_directory=self.temp_dir.name)
        self.test_collection = "test_interview_collection"
        
        self.rag_pipeline = RepoRAGPipeline(vector_store=self.vstore)
        self.generator = InterviewQuestionGenerator(rag_pipeline=self.rag_pipeline)

    def tearDown(self):
        """Clean up temporary directory and collection."""
        self.vstore.delete_collection(self.test_collection)
        self.temp_dir.cleanup()

    def test_interview_question_serialization(self):
        """Test InterviewQuestion dataclass to_dict conversion."""
        q = InterviewQuestion(
            question_id=1,
            question_text="How does auth work?",
            topic="Security",
            target_files=["src/auth.py"],
            expected_concepts=["JWT", "Tokens"]
        )
        d = q.to_dict()
        self.assertEqual(d["question_id"], 1)
        self.assertEqual(d["question_text"], "How does auth work?")
        self.assertEqual(d["topic"], "Security")
        self.assertEqual(d["target_files"], ["src/auth.py"])
        self.assertEqual(d["expected_concepts"], ["JWT", "Tokens"])

    def test_generate_questions_empty_store(self):
        """Test fallback question generation when collection is empty."""
        questions = self.generator.generate_questions(collection_name="empty_coll", num_questions=3)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].topic, "Architecture & System Design")

    def test_generate_questions_with_documents(self):
        """Test question generation from populated codebase documents."""
        docs = [
            Document(
                page_content="class Pipeline:\n    def run(self):\n        try:\n            pass\n        except Exception as e:\n            raise e",
                metadata={"relative_path": "src/pipeline.py", "file_name": "pipeline.py", "language": "Python", "start_line": 1, "end_line": 6}
            ),
            Document(
                page_content="def parse_config(filepath):\n    import yaml\n    return yaml.safe_load(filepath)",
                metadata={"relative_path": "src/config.py", "file_name": "config.py", "language": "Python", "start_line": 1, "end_line": 3}
            )
        ]
        self.vstore.add_documents(docs, collection_name=self.test_collection)

        questions = self.generator.generate_questions(collection_name=self.test_collection, num_questions=3)
        self.assertEqual(len(questions), 3)

        for q in questions:
            self.assertIsInstance(q, InterviewQuestion)
            self.assertGreater(len(q.target_files), 0)
            self.assertGreater(len(q.expected_concepts), 0)
            self.assertIn("src/", q.target_files[0])

    def test_zero_questions_count(self):
        """Test requesting 0 questions returns empty list."""
        questions = self.generator.generate_questions(collection_name=self.test_collection, num_questions=0)
        self.assertEqual(len(questions), 0)


if __name__ == "__main__":
    unittest.main()
