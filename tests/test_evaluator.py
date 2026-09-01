"""
Unit and integration tests for RepoViva Milestone 6: Answer Evaluation & Adaptive Follow-up.
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
from src.interview_generator import InterviewQuestion
from src.evaluator import AnswerEvaluation, AnswerEvaluator


class TestAnswerEvaluator(unittest.TestCase):

    def setUp(self):
        """Set up temporary vector store and evaluator."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vstore = RepoVectorStore(persist_directory=self.temp_dir.name)
        self.test_collection = "test_eval_collection"
        
        self.rag_pipeline = RepoRAGPipeline(vector_store=self.vstore)
        self.evaluator = AnswerEvaluator(rag_pipeline=self.rag_pipeline)

        # Seed vector store
        self.docs = [
            Document(
                page_content="def process_payment(amount):\n    try:\n        charge(amount)\n    except PaymentError as e:\n        log_error(e)\n        raise",
                metadata={"relative_path": "src/payment.py", "file_name": "payment.py", "language": "Python", "start_line": 1, "end_line": 6}
            )
        ]
        self.vstore.add_documents(self.docs, collection_name=self.test_collection)

        self.sample_question = InterviewQuestion(
            question_id=1,
            question_text="How does src/payment.py handle payment processing and error logging?",
            topic="Error Handling",
            target_files=["src/payment.py"],
            expected_concepts=["Function Definitions", "Exception Handling", "Data Flow"]
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.vstore.delete_collection(self.test_collection)
        self.temp_dir.cleanup()

    def test_evaluation_dataclass_dict(self):
        """Test AnswerEvaluation serialization to dict."""
        eval_obj = AnswerEvaluation(
            question_id=1,
            question_text="Test Question",
            user_answer="Test Answer",
            score=85,
            is_correct=True,
            feedback="Good",
            strengths=["Exception Handling"],
            missed_concepts=[],
            citations=[],
            follow_up_question="Follow up?"
        )
        d = eval_obj.to_dict()
        self.assertEqual(d["question_id"], 1)
        self.assertEqual(d["score"], 85)
        self.assertTrue(d["is_correct"])

    def test_evaluate_empty_answer(self):
        """Test evaluation when candidate provides no answer."""
        res = self.evaluator.evaluate_answer(
            question=self.sample_question,
            user_answer="",
            collection_name=self.test_collection
        )
        self.assertEqual(res.score, 0)
        self.assertFalse(res.is_correct)
        self.assertIn("No answer was provided", res.feedback)

    def test_evaluate_strong_answer(self):
        """Test evaluation when candidate provides a complete answer."""
        answer = "The function definitions handle payment charges and use try-except blocks for exception handling and logging data flow."
        res = self.evaluator.evaluate_answer(
            question=self.sample_question,
            user_answer=answer,
            collection_name=self.test_collection
        )
        self.assertGreaterEqual(res.score, 70)
        self.assertTrue(res.is_correct)
        self.assertGreaterEqual(len(res.strengths), 2)
        self.assertIsNotNone(res.follow_up_question)

    def test_evaluate_partial_answer(self):
        """Test evaluation when candidate provides a partial answer missing concepts."""
        answer = "It defines a function."
        res = self.evaluator.evaluate_answer(
            question=self.sample_question,
            user_answer=answer,
            collection_name=self.test_collection
        )
        self.assertLess(res.score, 70)
        self.assertFalse(res.is_correct)
        self.assertGreater(len(res.missed_concepts), 0)
        self.assertIn("src/payment.py", res.follow_up_question)


if __name__ == "__main__":
    unittest.main()
